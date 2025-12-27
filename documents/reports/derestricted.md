# GLM-4.5 Air Derestricted FP8 Investigation

## Overview

Investigation into the feasibility of running the `glm-4.5-air-derestricted-fp8` model on vLLM with an RTX PRO 6000 Blackwell (sm_120) GPU.

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 6000 Blackwell |
| Compute Capability | sm_120 |
| CUDA Version Required | 12.8+ |
| VRAM | 96GB per card |

## Configuration Comparison

### File References
- **Working Model**: [`documents/model-configs/glm-4.5-air-fp8.json`](../model-configs/glm-4.5-air-fp8.json)
- **Derestricted Model**: [`documents/model-configs/glm-4.5-air-derestricted-fp8.json`](../model-configs/glm-4.5-air-derestricted-fp8.json)

### Key Differences

| Parameter | glm-4.5-air-fp8 (working) | glm-4.5-air-derestricted-fp8 |
|-----------|---------------------------|------------------------------|
| `hidden_size` | 5120 | **4096** |
| `intermediate_size` | 12288 | **10944** |
| `moe_intermediate_size` | 1536 | **1408** |
| `num_hidden_layers` | 92 | **46** |
| `n_routed_experts` | 160 | **128** |
| `routed_scaling_factor` | 2.5 | **1.0** |
| `first_k_dense_replace` | 3 | **1** |
| `use_qk_norm` | true | **false** |
| `dtype` | torch_dtype | **"bfloat16"** |
| **Ignore List Size** | ~900 layers | **1 layer** |

### Model Architecture: Glm4MoeForCausalLM

Source: [`vllm/model_executor/models/glm4_moe.py`](../../vllm/model_executor/models/glm4_moe.py)

```python
# Line 240: use_qk_norm parameter
use_qk_norm: bool = False,

# Lines 302-304: QK norm layer creation
if self.use_qk_norm:
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

# Lines 313-318: QK norm application in forward
if self.use_qk_norm:
    q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(
        q.shape
    )
    k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(
        k.shape
    )
```

## Quantization Configuration Analysis

### Working Model Ignore List (Excerpt)

```json
"ignore": [
  "model.layers.12.input_layernorm",
  "model.layers.48.input_layernorm",
  "model.layers.74.input_layernorm",
  "model.layers.5.self_attn.q_norm",
  "model.layers.11.self_attn.k_proj.bias",
  "model.layers.3.self_attn.k_proj.bias",
  "model.layers.53.self_attn.q_norm",
  "model.layers.52.mlp.gate",
  "model.layers.29.mlp.gate.e_score_correction_bias",
  // ... ~900 total entries
  "lm_head"
]
```

### Derestricted Model Ignore List

```json
"ignore": [
  "lm_head"
]
```

## GPU Capability Requirements

Source: [`vllm/model_executor/layers/quantization/compressed_tensors/schemes/`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/)

| Quantization Scheme | Min Capability | Source File | RTX PRO 6000 (sm_120) |
|---------------------|----------------|-------------|----------------------|
| W8A8FP8 | 89 | [`compressed_tensors_w8a8_fp8.py:81-83`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L81-L83) | ✅ 120 |
| W4A8FP8 | 90 | [`compressed_tensors_w4a8_fp8.py:68-70`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L68-L70) | ✅ 120 |
| W8A16FP8 | 80 | [`compressed_tensors_w8a16_fp8.py:36-38`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py#L36-L38) | ✅ 120 |
| FP24 | 90 | [`compressed_tensors_24.py:74-76`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_24.py#L74-L76) | ✅ 120 |

**Conclusion**: The RTX PRO 6000 Blackwell (sm_120) is fully compatible with all FP8 quantization schemes used by the model.

## Compressed Tensors Implementation

### Ignore Layer Logic

Source: [`vllm/model_executor/layers/quantization/compressed_tensors/utils.py:22-37`](../../vllm/model_executor/layers/quantization/compressed_tensors/utils.py#L22-L37)

```python
def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str] = tuple(),
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    if layer_name is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = layer_name.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
```

### Quantization Config Parsing

Source: [`vllm/config/model.py:836-890`](../../vllm/config/model.py#L836-L890)

```python
def _parse_quant_hf_config(...):
    # compressed-tensors uses "compression_config" as the key
    # Library name normalized from "compressed_tensors" to "compressed-tensors"
    # Config is passed to quantization method as quant_method
```

### CompressedTensorsConfig Class

Source: [`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:91`](../../vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py#L91)

```python
class CompressedTensorsConfig:
    def __init__(self, ...):
        # Line 91
        self.ignore = ignore  # List of layer names to skip quantization
        self.target_scheme_map = ...  # Maps layer targets to quantization schemes
        self.format = ...  # Global quantization format (e.g., "float_quantized")
```

## Quantization Scheme Details

### W8A8FP8 Scheme

Source: [`vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py)

```python
@classmethod
def get_min_capability(cls) -> int:
    # lovelace and up
    return 89
```

**Features**:
- Static or dynamic activation quantization
- Per-tensor or per-channel scaling
- Block quantization support for newer GPUs

### W4A8FP8 Scheme

Source: [`vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py)

```python
@classmethod
def get_min_capability(cls) -> int:
    # hopper
    return 90

# Constraints (lines 54-57)
if self.group_size != 128 or self.strategy != "group":
    raise ValueError(
        "W4A8 kernels require group quantization with group size 128"
    )
```

## Potential Issues

### 1. Minimal Ignore List

The derestricted model's minimal ignore list (`["lm_head"]`) means many more modules will be quantized:

- **Layer norms** - Quantizing normalization layers can cause numerical instability
- **Attention biases** - May affect attention computation accuracy
- **MoE gate correction biases** - Could impact expert routing

### 2. QK Normalization Disabled

The derestricted model has `use_qk_norm: false`, while the working model has `true`. This affects:
- Query/Key normalization layers not being created
- Different attention computation behavior

### 3. Different Model Variant

The derestricted model appears to be a **smaller variant**:
- Fewer layers (46 vs 92)
- Smaller hidden dimensions (4096 vs 5120)
- Fewer experts (128 vs 160)

## Recommended Action Plan

### Option 1: Direct Load Attempt

```python
from vllm import LLM

model = LLM(
    model="path/to/derestricted-model",
    trust_remote_code=True,
    dtype="bfloat16"
)
```

### Option 2: Modified Config with Safer Ignore List

If Option 1 fails, create a modified config with additional exclusions:

```json
{
  "ignore": [
    "lm_head",
    "embed_tokens",
    "model.norm"
  ]
}
```

### Option 3: Comprehensive Ignore List

For maximum compatibility, adapt the working model's ignore list pattern for the 46-layer architecture:

```json
{
  "ignore": [
    "lm_head",
    "embed_tokens",
    "model.norm",
    // Add pattern-based exclusions:
    "re:.*\\.input_layernorm$",
    "re:.*\\.post_attention_layernorm$",
    "re:.*\\.self_attn\\..*_norm$",
    "re:.*\\.mlp\\.gate\\.e_score_correction_bias$"
  ]
}
```

## Testing Script

```python
import torch
from vllm import LLM

# Test basic loading
try:
    model = LLM(
        model="path/to/derestricted-model",
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9
    )
    print("Model loaded successfully")

    # Test inference
    outputs = model.generate("Hello, world!", max_tokens=10)
    print(outputs)

except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
```

## Conclusion

**GPU Compatibility**: ✅ The RTX PRO 6000 Blackwell (sm_120) fully supports the required FP8 quantization schemes.

**Model Compatibility**: ⚠️ The derestricted model has significant configuration differences:
- Smaller model variant (46 layers vs 92)
- Minimal quantization ignore list
- Disabled QK normalization

**Recommendation**: Try direct loading first. If issues arise, create a modified config with a safer ignore list based on the working model's pattern.

## Related Files

- vLLM Model Implementation: [`vllm/model_executor/models/glm4_moe.py`](../../vllm/model_executor/models/glm4_moe.py)
- Compressed Tensors Config: [`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`](../../vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)
- Utility Functions: [`vllm/model_executor/layers/quantization/compressed_tensors/utils.py`](../../vllm/model_executor/layers/quantization/compressed_tensors/utils.py)
- W8A8FP8 Scheme: [`vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`](../../vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py)
- Config Parsing: [`vllm/config/model.py`](../../vllm/config/model.py)

## References

- Hardware Documentation: [`documents/hardware.md`](../hardware.md)
- Working Model Config: [`documents/model-configs/glm-4.5-air-fp8.json`](../model-configs/glm-4.5-air-fp8.json)
- Derestricted Model Config: [`documents/model-configs/glm-4.5-air-derestricted-fp8.json`](../model-configs/glm-4.5-air-derestricted-fp8.json)
- DeepGEMM FP8 Issues Report: [`documents/reports/deepgemm.md`](./deepgemm.md)
