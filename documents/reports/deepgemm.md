# DeepGEMM SM120 (Blackwell) Support Investigation

## Problem Statement

FP8 quantization fails on SM120 (Blackwell) architecture for models like Devstral Small 2 (24B), while AWQ works fine.

```
Error: DeepGemm or similar FP8 kernel error on SM120
```

## Root Cause Analysis

### 1. DeepGEMM Architecture Support

The original DeepGEMM library from DeepSeek is **SM90 (Hopper) only**.

**File**: [vllm/utils/deep_gemm.py:66-74](../vllm/utils/deep_gemm.py#L66)

```python
@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    is_supported_arch = current_platform.is_cuda() and (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
    )
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch
```

Note: Although the code claims Blackwell support, the actual DeepGEMM kernels require `tcgen05.fence` which is **only available on SM90**.

### 1.1 vLLM DeepGEMM Detection Bug

**File**: `vllm/utils/deep_gemm.py:25-29`

```python
is_supported_arch = current_platform.is_cuda() and (
    current_platform.is_device_capability(90)      # SM90 (Hopper)
    or current_platform.is_device_capability_family(100)  # SM100 family
)
```

| Check | Returns True For | DeepGEMM Works? |
|-------|------------------|-----------------|
| `is_device_capability(90)` | SM90 (Hopper H100) | ✅ Yes |
| `is_device_capability_family(100)` | SM100, SM120 | ❌ SM120 No |

**The Bug**: `is_device_capability_family(100)` returns `True` for **both** SM100 (Hopper) and SM120 (Blackwell), but DeepGEMM only supports SM100.

**Impact**: When loading Devstral Small 2 with FP8 on SM120:
1. `is_deep_gemm_supported()` returns `True` (wrong!)
2. vLLM attempts to use DeepGEMM kernels
3. DeepGEMM fails due to missing `tcgen05.fence` instruction
4. Falls back to CUTLASS or Triton (slower)

**This is a bug in vLLM's architecture detection**, not a model-specific issue. The fix would be to exclude SM120 from the DeepGEMM support check.

### 2. DeepGemmQuantScaleFMT Difference

**File**: `vllm/utils/deep_gemm.py:36-55`

```python
@classmethod
def init_oracle_cache(cls) -> None:
    """Initialize the oracle decision and store it in the class cache"""
    cached = getattr(cls, "_oracle_cache", None)
    if cached is not None:
        return

    use_e8m0 = (
        envs.VLLM_USE_DEEP_GEMM_E8M0
        and is_deep_gemm_supported()
        and (_fp8_gemm_nt_impl is not None)
    )
    if not use_e8m0:
        cls._oracle_cache = cls.FLOAT32
        return

    cls._oracle_cache = (  # type: ignore
        cls.UE8M0
        if current_platform.is_device_capability_family(100)
        else cls.FLOAT32_CEIL_UE8M0
    )
```

| Architecture | Capability | Scale Format |
|--------------|------------|--------------|
| SM100 (Hopper) | `is_device_capability_family(100)` | UE8M0 (fully optimized) |
| SM120 (Blackwell) | `else` path | FLOAT32_CEIL_UE8M0 (less optimized) |

### 3. Dense Model DeepGemm Decision

**File**: `vllm/utils/deep_gemm.py:382-401`

```python
def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight: torch.Tensor,
    supports_deep_gemm: bool | None = None,
):
    if supports_deep_gemm is None:
        supports_deep_gemm = is_deep_gemm_supported()

    # Verify DeepGEMM N/K dims requirements
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    return (
        supports_deep_gemm
        and output_dtype == torch.bfloat16
        and weight.shape[0] % N_MULTIPLE == 0
        and weight.shape[1] % K_MULTIPLE == 0
    )
```

**Requirements**:
- `output_dtype == torch.bfloat16`
- `weight.shape[0] % 64 == 0` (N dimension)
- `weight.shape[1] % 128 == 0` (K dimension)

### 4. FP8 Block Linear Operation

**File**: `vllm/model_executor/layers/quantization/utils/fp8_utils.py:246-268`

```python
def __init__(
    self,
    weight_group_shape: GroupShape,
    act_quant_group_shape: GroupShape,
    cutlass_block_fp8_supported: bool = CUTLASS_BLOCK_FP8_SUPPORTED,
    use_aiter_and_is_supported: bool = False,
):
    self.weight_group_shape = weight_group_shape
    self.act_quant_group_shape = act_quant_group_shape
    self.is_deep_gemm_supported = is_deep_gemm_supported()
    self.is_hopper = current_platform.is_device_capability(90)
    self.use_deep_gemm_e8m0 = is_deep_gemm_e8m0_used()
```

### 5. MoE Backend Selection

**File**: `vllm/model_executor/layers/quantization/fp8.py:123-199`

```python
def get_fp8_moe_backend(
    block_quant: bool,
    moe_parallel_config: FusedMoEParallelConfig,
    with_lora_support: bool,
) -> Fp8MoeBackend | None:
    # Priority order:
    # FLASHINFER_TRTLLM > FLASHINFER_CUTLASS > MARLIN > DEEPGEMM > AITER > TRITON

    if envs.VLLM_USE_DEEP_GEMM and moe_use_deep_gemm and block_quant:
        if not has_deep_gemm():
            logger.warning_once(
                "DeepGEMM backend requested but not available.", scope="local"
            )
        elif is_deep_gemm_supported():
            logger.info_once("Using DeepGEMM backend for FP8 MoE", scope="local")
            return Fp8MoeBackend.DEEPGEMM
```

## Devstral Small 2 Configuration Analysis

```json
{
  "architectures": ["Mistral3ForConditionalGeneration"],
  "dtype": "bfloat16",
  "text_config": {
    "hidden_size": 5120,
    "intermediate_size": 32768,
    "num_attention_heads": 32,
    "num_hidden_layers": 40
  },
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "static"
  }
}
```

**Dimension Check**:
| Dim | Value | Required | Status |
|-----|-------|----------|--------|
| hidden_size (N) | 5120 | % 64 == 0 | ✅ Pass |
| intermediate_size (K) | 32768 | % 128 == 0 | ✅ Pass |

The model meets all dimension requirements, but DeepGEMM kernels fail due to SM120 incompatibility.

## Workarounds

### Option 1: Use AWQ Quantization

```bash
vllm serve mistralai/Devstral-Small-2-24B --quantization awq
```

AWQ does not use DeepGEMM and works on all architectures.

### Option 2: Disable DeepGEMM

```bash
export VLLM_USE_DEEP_GEMM=0
```

This forces fallback to Triton or CUTLASS backends.

### Option 3: Use SGLang's DeepGEMM Fork

The SGLang project maintains a fork with multi-architecture support:

```bash
pip install git+https://github.com/sgl-project/DeepGEMM.git
```

This fork includes:
- SM90 (Hopper) support
- SM100a support
- SM120a (Blackwell) support

### Option 4: Wait for Upstream Fix

vLLM is tracking SM120 support in:
- Issue #31085: Add SM120 (RTX 6000/5000 Blackwell) support for native NVFP4 MoE kernels

## Related Files

| File | Purpose |
|------|---------|
| `vllm/utils/deep_gemm.py` | DeepGEMM wrapper and compatibility |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | FP8 block linear operations |
| `vllm/model_executor/layers/quantization/fp8.py` | FP8 configuration and MoE backend selection |
| `vllm/model_executor/layers/fused_moe/deep_gemm_moe.py` | MoE-specific DeepGEMM kernels |

## External References

- [DeepSeek DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)
- [SGLang DeepGEMM Fork](https://github.com/sgl-project/DeepGEMM)
- [NVIDIA CUTLASS SM120 Support](https://github.com/NVIDIA/cutlass/tree/master/examples/79_blackwell_geforce_gemm)
- [vLLM Issue #31085: SM120 NVFP4 MoE kernels](https://github.com/vllm-project/vllm/issues/31085)

## Conclusion

There are **two issues** causing FP8 to fail on SM120:

1. **vLLM Bug**: `is_deep_gemm_supported()` incorrectly returns `True` for SM120 due to overly permissive architecture check (`is_device_capability_family(100)` matches both SM100 and SM120).

2. **DeepGEMM Design**: The original DeepGEMM library uses `tcgen05.fence` which is only available on SM90/SM100, not SM120.

**Recommended solutions for SM120 users:**
1. Use AWQ (Marlin kernel - fastest working option)
2. Use SGLang's fork of DeepGEMM
3. Wait for vLLM to fix the architecture detection bug
