# RISC-V + 边缘 LLM 推理加速：研究方向全景

> 最后更新：2026-05-29

---

## 目录

- [1. 为什么这个方向值得做](#1-为什么这个方向值得做)
- [2. 竞争版图](#2-竞争版图)
- [3. 关键论文地图](#3-关键论文地图)
- [4. 六大创新方向](#4-六大创新方向)
- [5. RISC-V V扩展 + 稀疏性深度分析](#5-risc-v-v扩展--稀疏性深度分析)
- [6. 推荐研究路径](#6-推荐研究路径)
- [7. 参考文献](#7-参考文献)

---

## 1. 为什么这个方向值得做

### 1.1 行业趋势

目前边缘 LLM 推理的格局是 **ARM（Apple Neural Engine / M 系列 + 高通 Hexagon）+ NVIDIA（Jetson 系列）双寡头**，RISC-V 生态几乎空白。但四个趋势正在交汇：

1. **RISC-V Vector 1.0 规范已冻结**，量产芯片正在铺开（玄铁 C910/C920 已集成 V 扩展，算能 SG2042/SG2380 等）
2. **地缘政治因素**推动中国大力投入 RISC-V 自主生态（"十四五"重点支持）
3. **LLM 小型化趋势**：1B-3B 参数模型已具备可用推理能力，MiniCPM、Qwen2.5、Phi-3 等证明小模型在边缘可行
4. **学术界空白**：目前还没有论文系统解决"RISC-V 上高效运行大模型"的全栈问题

### 1.2 核心瓶颈

边缘 LLM 推理的真正瓶颈**不是算力，是内存带宽**：

| 平台 | 内存带宽 | 7B 模型推理速度（典型） |
|------|---------|----------------------|
| Apple M3 Max | 400 GB/s | ~30 tok/s |
| NVIDIA Jetson Orin | 204 GB/s | ~20 tok/s |
| 树莓派 5 | ~17 GB/s | <1 tok/s |
| RISC-V (C910 单核) | ~10-20 GB/s | ❓未评测 |
| RISC-V (SG2042 64核) | ~100+ GB/s (理论) | ❓未评测 |

**稀疏性**和**量化**是突破这一瓶颈的两把钥匙。

---

## 2. 竞争版图

```
                        硬件定制程度
                           ↑
    ASIC/NPU          │  VitaLLM, NVLLM        ← 专用加速器
    (专用芯片)        │  TENET (三元LLM)
                      │
    SIMD + 扩展指令   │  IndexMAC,              ← ★ 核心创新空间
    (向量/张量)       │  Structured-Sparse MM,
                      │  RISC-V V 优化
                      │
    通用 CPU 优化     │  PowerInfer, DejaVu,    ← 已有大量工作
    (软件层)          │  LLM in a Flash,
                      │  AWQ, Scissorhands
                      │
                       └──────────────────────────→ 模型规模
                         TinyML    边缘 LLM    云端 LLM
                        (<100M)   (1B-7B)     (70B+)

    图例：
    绿色区域 → 已有大量工作，竞争激烈
    ★ 区域  → RISC-V 的核心机会窗口
    黄色区域 → 有探索但不成体系
```

**关键洞察**：RISC-V 目前在 TinyML/MCU 级别（绿色左下）已有不错积累（Daghero 2025, SpikeStream 2025），但在**边缘 LLM 级别（1B-7B）几乎完全是空白**。

---

## 3. 关键论文地图

### 3.1 稀疏性相关（已有论文）

| 论文 | 会议 | 核心思想 |
|------|------|---------|
| **DejaVu** | ICML 2023 | 上下文稀疏性：每层动态选神经元，仅激活~15%参数 |
| **PowerInfer** | SOSP 2024 | 热神经元稀疏：激活值呈现幂律分布，利用GPU局部性 |
| **PowerInfer2** | OSDI 2025 | PowerInfer的智能手机版：异构计算+离线权重聚类 |
| **Scissorhands** | ICLR 2023 | KV Cache重要性具有时间持久性，可大量剪枝 |

### 3.2 量化/内存相关（已有论文）

| 论文 | 会议 | 核心思想 |
|------|------|---------|
| **AWQ** | ICML 2024 | 激活感知权重量化：关注显著权重通道 |
| **LLM in a Flash** | ACL 2024 | 闪存+DRAM混合推理：分页式权重加载 |
| **FlexGen** | OSDI 2023 | Offloading引擎：GPU-CPU-磁盘三级存储调度 |
| **MobileLLM** | NeurIPS 2024 | 亚十亿参数小模型的最优架构设计 |
| **FlashAttention** | NeurIPS 2022 | IO-Aware精确注意力：分块+online softmax |

### 3.3 RISC-V 相关（新发现）

| 论文 | 时间 | 核心思想 |
|------|------|---------|
| **IndexMAC** | 2023.11 | 自定义RISC-V向量指令加速结构化稀疏GEMM |
| **Optimizing Structured-Sparse MM in RISC-V** | 2025.01 | 同一团队：N:M稀疏在RISC-V V上的系统优化 |
| **Lightweight Kernels for Sparse DNN on MCU** | 2025.03 | MCU级RISC-V核上稀疏DNN的软件kernel+硬件扩展 |
| **SpikeStream** | 2025.04 | RISC-V集群上脉冲神经网络稀疏计算扩展 |

### 3.4 边缘 LLM 最新进展（2025-2026）

| 论文 | 时间 | 核心思想 |
|------|------|---------|
| **When NPUs Are Not Always Faster** | 2026.05 | 手机端LLM推理的阶段级分析：NPU并非万能 |
| **Cassandra** | 2026.05 | 边缘端推理LLM的Self-Speculative Decoding |
| **VitaLLM** | 2026.04 | 混合精度(三元+INT8)的紧凑型LLM加速器 |
| **Cloud to Edge Benchmark** | 2026.04 | 硬件加速单板机上的LLM推理基准测试 |
| **Collapse or Preserve** | 2026.03 | ⚠️ SNN稀疏计算在SIMD上全部无法超越密集卷积 |
| **Resting Neurons (SPON)** | 2025.12→ICML 2026 | 用自发神经元机制让LLM激活稀疏性更鲁棒 |
| **Inference Time Context Sparsity** | 2026.05 | 质疑LLM推理中上下文稀疏性的实际价值 |
| **Sparse-on-Dense** | 2026.04 | 在密集加速器上高效计算稀疏网络的新范式 |

---

## 4. 六大创新方向

### 🥇 方向一：RISC-V V 扩展的稀疏 GEMM 优化

**研究问题**：如何利用 RISC-V Vector Extension 的 mask/gather/scatter 指令高效执行稀疏矩阵乘法？

**Baseline**：IndexMAC (2023)、Titopoulos et al. (2025)

**创新空间**：
- 混合精度+稀疏联合设计（如 INT4 量化 + N:M 稀疏在 RISC-V V 上的 co-design）
- 动态稀疏模式选择：不同层用不同 (N,M) 组合，用轻量级 predictor 在线决策
- 稀疏感知的向量寄存器分配：不同 VLEN 下的最优分块策略
- 非结构化稀疏→向量友好的数据重组格式

**难度**：⭐⭐⭐  
**创新性**：⭐⭐⭐⭐  
**出活速度**：⭐⭐⭐⭐⭐  
**目标会议**：ASPLOS, MICRO, HPCA, OSDI

---

### 🥈 方向二：RISC-V 上的 Attention 算子优化

**研究问题**：FlashAttention 系列完全针对 CUDA GPU 优化，如何高效移植到 RISC-V V 扩展？

**Baseline**：FlashAttention (NeurIPS 2022)、FlashAttention-2/3

**创新空间**：
- FlashAttention for RISC-V V：用 V 扩展实现 tiling + online softmax
- KV Cache 的 RISC-V 友好压缩格式
- Sliding window / sparse attention 的向量化实现
- 针对 RISC-V 向量寄存器宽度的最优 tile size 选择

**难度**：⭐⭐⭐⭐  
**创新性**：⭐⭐⭐⭐  
**出活速度**：⭐⭐⭐  
**目标会议**：OSDI, ASPLOS, EuroSys

---

### 🥉 方向三：RISC-V 存储层次感知的 LLM 推理

**研究问题**：边缘 LLM 推理的内存带宽瓶颈如何在 RISC-V 的 cache/memory 架构下缓解？

**Baseline**：FlexGen (OSDI 2023)、LLM in a Flash (ACL 2024)

**创新空间**：
- RISC-V 上 LLM 的 memory-aware 算子调度策略
- 权重预取+计算流水线：利用 RISC-V scalar/vector 并行性
- RISC-V 内存模型下的 KV Cache 管理策略
- 异构内存（SRAM+DRAM+Flash）协同调度

**难度**：⭐⭐⭐⭐  
**创新性**：⭐⭐⭐⭐  
**出活速度**：⭐⭐⭐  
**目标会议**：ISCA, MICRO, OSDI

---

### 🏅 方向四：面向 LLM 推理的 RISC-V 定制指令扩展

**研究问题**：能否设计一组 RISC-V 定制指令，在不大幅改动核心微架构的前提下加速 LLM 核心算子？

**Baseline**：IndexMAC (一条指令)、NVIDIA Tensor Core、ARM SME/SSVE

**创新空间**：
- 面向 LLM 推理的一组 RISC-V 定制指令集：
  - 稀疏 GEMM 指令（含索引编码）
  - 向量化 Layernorm/RMSNorm 指令
  - 向量化 Softmax 指令
  - 向量化 SiLU/GELU 等激活函数指令
- RISC-V custom-0/1/2/3 opcode 空间的高效编码
- LLVM/GCC 编译器 intrinsics 支持

**难度**：⭐⭐⭐⭐⭐  
**创新性**：⭐⭐⭐⭐⭐  
**出活速度**：⭐⭐  
**目标会议**：ISCA, MICRO, HPCA

---

### 🏅 方向五：RISC-V 异构多核 LLM 推理架构

**研究问题**：RISC-V 的开源 ISA 天然支持异构核设计。如何设计一个针对 LLM 推理最优的 RISC-V 多核架构？

**Baseline**：PowerInfer2 (OSDI 2025)、ARM big.LITTLE

**创新空间**：
- 大小核协作推理：大核处理 Attention（延迟敏感），小核处理 FFN（吞吐敏感）
- Prefill vs Decode 分离：不同阶段分配不同核
- RISC-V Cluster 上的模型并行 + 流水线并行
- 核间通信开销与分块粒度的 trade-off 建模

**难度**：⭐⭐⭐⭐⭐  
**创新性**：⭐⭐⭐⭐⭐  
**出活速度**：⭐⭐  
**目标会议**：ISCA, ASPLOS

---

### 🏅 方向六：RISC-V 平台 LLM 推理基准评测与分析

**研究问题**：RISC-V 平台运行 LLM 的性能特征如何？与 ARM/x86 的差距在哪里？优化机会是什么？

**Baseline**：无（空白领域！）

**创新空间**：
- 在 RISC-V 开发板（SiFive、玄铁 C910/C920、SG2042 等）上系统性评测小模型
- 计算瓶颈 vs 带宽瓶颈的精确分布分析
- RISC-V vs ARM vs x86 的逐算子性能对比
- RISC-V 特有的性能瓶颈识别与优化建议
- 开源 benchmark suite

**难度**：⭐⭐  
**创新性**：⭐⭐⭐⭐  
**出活速度**：⭐⭐⭐⭐⭐  
**目标会议**：ISPASS, IISWC, MLSys, 或作为系统类论文的 motivation/evaluation 部分

---

## 5. RISC-V V 扩展 + 稀疏性深度分析

### 5.1 为什么 SIMD/向量架构天然不利于不规则稀疏

参考论文 **Collapse or Preserve** (Qin, 2026) 的核心发现：在 Apple M3 Max 上测试 5 种稀疏计算策略，**全部无法超越密集卷积**。原因：

| 问题 | 描述 |
|------|------|
| **不规则访存** | 非零元素位置随机，向量 load 仍需取连续数据，mask 只屏蔽计算不减少访存 |
| **控制流发散** | 各 lane 需要处理的数据量不同，lane 利用率极低 |
| **元数据开销** | CSR/CSC 格式的索引解码、指针追踪开销可能超过省下的计算 |
| **稀疏度阈值** | 需 70-90%+ 稀疏度才能见到正收益，实际模型稀疏度常远低于此 |

### 5.2 结构化稀疏的权衡

| 特性 | 非结构化稀疏 | 结构化稀疏 (N:M) |
|------|------------|-----------------|
| 访存连续性 | ❌ 差 | ✅ 好 |
| 元数据开销 | ❌ 大 | ✅ 无 |
| Lane 利用率 | ❌ 低 | ✅ 100%（对齐时） |
| 精度损失 | ✅ 小 | ❌ 较大，需 retrain |
| 稀疏度灵活性 | ✅ 任意 | ❌ 固定档位 (2:4, 4:8...) |
| VLEN 对齐要求 | ⚠️ 无关 | ❌ 必须对齐 |

### 5.3 RISC-V V 扩展的关键特性

```
VLEN (向量寄存器宽度) → 决定了单条指令处理多少个元素
SEW (Selected Element Width) → 每个元素的位宽 (8/16/32/64)
LMUL (Length Multiplier) → 组合多个寄存器形成逻辑向量

例如：VLEN=256bit, SEW=32bit → 每向量 8 个 FP32 元素
      VLEN=128bit, SEW=16bit → 每向量 8 个 FP16 元素

关键指令：
  vle.v    → 连续加载（可配合 mask）
  vlxei.v  → 索引加载（gather，慢但灵活）
  vfmacc.v → 向量乘加（FMA）
  vsetvli  → 动态设置向量长度
```

**设计挑战**：N:M 的 M 值需要和 VLEN/SEW 对齐，否则 lane 利用率下降。不同层的稀疏模式可能需要不同的 (N,M) 组合。

---

## 6. 推荐研究路径

### 短期（3-6 个月）：方向六 + 方向一

1. **Benchmark 驱动**：先在 RISC-V 硬件上跑 LLM，获取第一手性能数据
2. **稀疏 GEMM 实现**：基于 IndexMAC 的 baseline，实现并改进稀疏 GEMM kernel
3. **产出**：一篇 workshop/short paper 描述 benchmark 发现 + 初步优化

### 中期（6-12 个月）：方向一深入

1. 完整的稀疏 GEMM 框架（多种 N:M 模式 + 量化融合）
2. 端到端 LLM 推理 demo（整合 Attention + FFN + 稀疏化）
3. **产出**：一篇系统类顶会（ASPLOS/MICRO/OSDI）

### 长期（12-24 个月）：方向二/三/四/五任选

根据中期实验结果，选择最有潜力的方向深入：
- 如果瓶颈在 Attention → 方向二
- 如果瓶颈在 Memory → 方向三
- 如果有 FPGA/模拟器资源 → 方向四/五

### 建议的故事线

> **RISC-V 上稀疏感知的边缘 LLM 推理加速**
>
> 1. **Motivation**：边缘 LLM 推理的内存带宽瓶颈 + RISC-V 生态的空白
> 2. **Observation**：稀疏性是突破瓶颈的关键，但 RISC-V V 扩展面临非结构化稀疏的挑战
> 3. **Solution**：针对 RISC-V V 特点设计的结构化稀疏 GEMM + 动态模式选择 + 量化融合
> 4. **Evaluation**：在真实 RISC-V 硬件上验证，达到 YY× 加速比，YY% 能耗降低

---

## 7. 参考文献

### 已有论文（用户文件夹）

1. Song, Y. et al. "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." SOSP 2024.
2. Alizadeh, K. et al. "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory." ACL 2024.
3. Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
4. Lin, J. et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." ICML 2024.
5. Sheng, Y. et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." OSDI 2023.
6. Xue, Z. et al. "PowerInfer-2: Fast Large Language Model Inference on a Smartphone." OSDI 2025.
7. Liu, Z. et al. "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time." ICML 2023.
8. Liu, Z. et al. "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases." NeurIPS 2024.
9. Sheng, Y. et al. "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time." ICLR 2023.
10. Tempus: Versal AI Edge. arXiv 2026.
11. RISC-V Vector Extension Specification v1.0.

### RISC-V + 稀疏相关

12. Titopoulos, V. et al. "IndexMAC: A Custom RISC-V Vector Instruction to Accelerate Structured-Sparse Matrix Multiplications." arXiv, 2023.
13. Titopoulos, V. et al. "Optimizing Structured-Sparse Matrix Multiplication in RISC-V Vector Processors." arXiv, 2025.
14. Daghero, F. et al. "Lightweight Software Kernels and Hardware Extensions for Efficient Sparse Deep Neural Networks on Microcontrollers." arXiv, 2025.
15. Manoni, S. et al. "SpikeStream: Accelerating Spiking Neural Network Inference on RISC-V Clusters with Sparse Computation Extensions." arXiv, 2025.

### 边缘 LLM 最新进展

16. "When NPUs Are Not Always Faster: A Stage-Level Analysis of Mobile LLM Inference." arXiv, 2026.
17. Choi, S. et al. "Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding." arXiv, 2026.
18. Lin, Z. & Chang, T. "VitaLLM: A Versatile and Tiny Accelerator for Mixed-Precision LLM Inference on Edge Devices." arXiv, 2026.
19. "Cloud to Edge: Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers." arXiv, 2026.

### 稀疏性理论/方法

20. Qin, J. "Collapse or Preserve: Data-Dependent Temporal Aggregation for Spiking Neural Network Acceleration." arXiv, 2026. ⚠️ 关键警示：SIMD 不适合不规则稀疏
21. Xu, H. et al. "Resting Neurons, Active Insights: Robustifying Activation Sparsity in LLMs via Spontaneity." ICML 2026.
22. Joshi, S. et al. "Inference Time Context Sparsity: Illusion or Opportunity?" arXiv, 2026.
23. Yoon, H. et al. "Sparse-on-Dense: Area and Energy-Efficient Computing of Sparse Neural Networks on Dense Matrix Multiplication Accelerators." arXiv, 2026.

---

## 附录：术语速查

| 术语 | 全称 | 含义 |
|------|------|------|
| V扩展 | RISC-V Vector Extension (RVV) | RISC-V 的 SIMD 向量处理单元 |
| VLEN | Vector Length | 向量寄存器位宽（如 128/256/512/1024 bit） |
| SEW | Selected Element Width | 向量元素的位宽（8/16/32/64 bit） |
| LMUL | Length Multiplier | 将多个向量寄存器组合为一个逻辑向量 |
| N:M 稀疏 | N:M Structured Sparsity | 每组 M 个连续值中恰好保留 N 个非零 |
| GEMM | General Matrix Multiply | 通用矩阵乘法 = 神经网络计算的核心 |
| KV Cache | Key-Value Cache | Transformer 自注意力中缓存的 Key 和 Value 矩阵 |
| FFN | Feed-Forward Network | Transformer 中的前馈网络层 |
| CSR/CSC | Compressed Sparse Row/Column | 稀疏矩阵的压缩存储格式 |
