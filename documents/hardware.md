# Hardware Specifications

This document details the hardware specifications for the vLLM development and testing system.

## System Overview

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen Threadripper PRO 9975WX |
| RAM | DDR5 64GB x 8 (512GB total) |
| GPU | NVIDIA RTX PRO 6000 Blackwell x 2 |

---

## CPU: AMD Ryzen Threadripper PRO 9975WX

### Core Configuration
- **Cores/Threads:** 32 cores / 64 threads
- **Architecture:** Zen 5 (4nm process)
- **Socket:** sTR5

### Clock Speeds
- **Base Clock:** 4.0 GHz
- **Maximum Boost Clock:** Up to 5.4 GHz

### Cache
- **L3 Cache:** 128 MB

### Memory Support
- **Memory Type:** 8-channel DDR5
- **Maximum Memory Support:** DDR5-6400

### Platform Features
- **PCIe Lanes:** 128 PCIe 5.0 lanes
- **TDP:** 350W
- **Instruction Sets:** AVX-512 support for AI & FEA applications

### Links
- [AMD Official Page](https://www.amd.com/en/products/processors/workstations/ryzen-threadripper/9000-wx-series/amd-ryzen-threadripper-pro-9975wx.html)
- [TechPowerUp Specs](https://www.techpowerup.com/cpu-specs/ryzen-threadripper-pro-9975wx.c4165)

---

## GPU: NVIDIA RTX PRO 6000 Blackwell (Dual)

### Architecture
- **GPU Architecture:** NVIDIA Blackwell
- **Compute Capability:** SM_120

### Key Specifications

| Feature | Specification |
|---------|---------------|
| **CUDA Cores** | 24,064 |
| **Tensor Cores** | 5th Generation |
| **RT Cores** | 4th Generation |
| **GPU Memory** | 96GB GDDR7 with ECC per card |
| **Memory Interface** | 512-bit |
| **Total VRAM (2x GPUs)** | 192GB |
| **Interface** | PCIe Gen 5.0 x16 |
| **Max Power Consumption** | 600W (Workstation Edition) |

### Connectivity
- 4x DisplayPort 2.1 per card
- NVIDIA RTX PRO SYNC Compatible

### Features
- Universal MIG (Multi-Instance GPU) support
- ECC memory support
- Designed for AI workloads, massive datasets, and multi-billion-parameter models

### CUDA & Driver Support
| Requirement | Minimum Version |
|-------------|-----------------|
| **CUDA Version** | 12.8+ |
| **NVIDIA Driver** | 575+ (recommended 580+) |
| **Compute Capability** | 12.0 (sm_120) |
| **PyTorch Support** | 2.7.0+ (with CUDA 12.8+) |

**Note:** The sm_120 (Blackwell) compute capability is newly introduced. Framework support is still maturing - ensure you use CUDA 12.8+ and PyTorch 2.7.0+ for compatibility.

### Multi-GPU Configuration
- **NVLink Support:** NO - RTX PRO 6000 Blackwell does NOT support NVLink/NVSwitch
- **P2P Support:** PCIe-based P2P (Peer-to-Peer) communication available but with limitations
- **Interconnect:** Standard PCIe Gen 5.0 x16 per GPU
- **Multi-GPU:** Supported via CUDA but without high-speed NVLink interconnect
- **Bandwidth:** GPUs communicate through system PCIe fabric

**Important Limitations:**
- Unlike datacenter GPUs (A100/H100/B200), the RTX PRO 6000 Blackwell lacks NVLink
- PCIe-based P2P communication has known stability issues:
  - NCCL P2P may experience hangs on dual-GPU workstation configurations
  - CUDA samples show failing P2P-related tests with 2× Blackwell GPUs
  - Higher latency and bandwidth limitations compared to NVLink
  - Driver-level limitations may impact multi-GPU scaling performance
- Cross-GPU access rides on PCIe 5.0 peer-to-peer, which adds latency and makes scaling unreliable

### Links
- [NVIDIA Official - Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [NVIDIA Official - Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4272)
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA Developer Forum - Driver Support](https://forums.developer.nvidia.com/t/rtx-pro-6000-blackwell-workstation-edition-driver-support/332701)
- [CUDA 12.8 sm_120 Discussion](https://forums.developer.nvidia.com/t/cuda-toolkit-12-8-what-gpu-is-sm_120/322128)
- [PyTorch sm_120 Support Issue](https://github.com/pytorch/pytorch/issues/159207)
- [Level1Techs - Dual RTX PRO 6000 P2P/NCCL Issues](https://forum.level1techs.com/t/dual-rtx-pro-6000-blackwell-max-q-how-to-make-p2p-nccl-work/242403)
- [NVIDIA Dev Forum - P2P Issues with RTX 5090](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776)
- [CUDA Samples - P2P Tests Failing with 2x Blackwell GPUs](https://github.com/NVIDIA/cuda-samples/issues/390)
- [Google Cloud - P2P Fabric on G4 VMs](https://cloud.google.com/blog/products/compute/g4-vms-p2p-fabric-boosts-multi-gpu-workloads/)
- [CSDN Blog - RTX 6000 Blackwell vs Data Center GPUs (Chinese)](https://blog.csdn.net/qq_35082030/article/details/150337655)
- [Chiphell Forum - RTX PRO 6000 SLI/P2P Discussion (Chinese)](https://www.chiphell.com/thread-2733592-1-1.html)
- [Level1Techs - GPUDirect RDMA on RTX PRO 6000](https://forum.level1techs.com/t/gpudirect-rdma-on-nvidia-rtx-pro-6000-blackwell-max-q/237059)

---

## Memory Configuration

### System RAM
- **Type:** DDR5
- **Configuration:** 8 x 64GB modules
- **Total:** 512GB
- **Channels:** 8-channel (optimized for Threadripper)
- **Speed:** Up to DDR5-6400 support

### GPU Memory
- **Per GPU:** 96GB GDDR7 with ECC
- **Total (2x GPUs):** 192GB
- **Interface:** 512-bit per card

---

## Platform Capabilities

### PCIe Configuration
- **CPU PCIe Lanes:** 128 PCIe 5.0 lanes
- **GPU Interface:** PCIe Gen 5.0 x16 per card
- **Total Bandwidth:** High-bandwidth interconnect support for dual-GPU configurations

### Software/Compute Features
- AVX-512 CPU instructions for AI acceleration
- 5th Gen Tensor Cores for AI/ML workloads
- Universal MIG for GPU virtualization
- ECC memory on both system RAM and GPU VRAM

### Use Cases
This hardware configuration is optimized for:
- Large Language Model (LLM) inference and training
- Multi-billion-parameter model deployment
- High-throughput batch processing
- AI research and development
- vLLM engine testing and benchmarking

---

## Notes

- **TDP Total:** CPU (350W) + 2x GPU (600W each) = ~1550W maximum power draw
- **Cooling Requirements:** Adequate cooling required for 600W GPUs
- **Power Supply Recommendation:** 1600W+ PSU with proper PCIe 5.0 power connectors
