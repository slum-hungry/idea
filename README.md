# 📚 写论文 — 文献整理索引

> **研究方向：** RISC-V V 扩展的稀疏 GEMM 优化  
> **更新日期：** 2026-06-01  
> **目标会议：** ASPLOS / MICRO / HPCA / OSDI

---

## 📁 目录结构

```
写论文/
├── 📄 README.md                          ← 本文件
├── 📁 01-Baselines/                      ← 基线方法（核心必读）
├── 📁 02-MixedPrecision-Sparse/          ← 混合精度 + 稀疏联合设计
├── 📁 03-DynamicSparsePattern/           ← 动态稀疏模式选择
├── 📁 04-RegisterAlloc-Tiling/           ← 寄存器分配与分块策略
├── 📁 05-DataReorganize-Format/          ← 数据重组与向量友好格式
├── 📁 06-Related-LLM-Sparse/             ← LLM 稀疏化 & 推理加速
├── 📁 07-Surveys/                        ← 综述文章
└── 📁 zz-Previous-LLM-Inference/         ← 之前的 LLM 推理优化论文
```

---

## 🔴 01-Baselines — 基线方法（核心必读）

这些是 RISC-V 稀疏 GEMM 方向最直接相关的 baseline。

| # | 论文 | 出处 | 年份 | 核心贡献 |
|---|---|---|---|---|
| **1** | **IndexMAC: A Custom RISC-V Vector Instruction to Accelerate Structured-Sparse Matrix Multiplications** — `2311.07241_IndexMAC_DATE2024.pdf` | **DATE 2024** (CCF-B) | 2024 | 提出 vindexmac 指令的前身，用 RISC-V 自定义向量指令加速结构化稀疏 MM。**你方向的直接 baseline** |
| **2** | **Optimizing Structured-Sparse Matrix Multiplication in RISC-V Vector Processors** — `2501.10189v1.pdf` | **IEEE TC** (CCF-A) | 2025 | IndexMAC 的扩展版：完整设计空间探索 + vindexmac 指令 + gem5 评估。**最核心的 baseline** |
| **3** | **Accelerating Sparse Deep Neural Networks** — `2104.08378_Mishra_AcceleratingSparseDNN.pdf` | arXiv (NVIDIA) | 2021 | N:M 结构化稀疏的奠基性工作，定义了 2:4 稀疏的 GPU 加速范式 |
| **4** | **Optimization of SpGEMM with RISC-V Vector Instructions** — `2303.02471_LeFevre_SpGEMM_RISC-V.pdf` | **HPDC 2023** (CCF-B) | 2023 | 在 RISC-V 长向量架构上优化 SpGEMM，直接使用现有 V 扩展指令，无自定义指令 |

---

## 🟠 02-MixedPrecision-Sparse — 混合精度 + 稀疏联合设计

对应创新点：**INT4 量化 + N:M 稀疏在 RISC-V V 上的 co-design**

| # | 论文 | 出处 | 年份 | 核心贡献 |
|---|---|---|---|---|
| **5** | **Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity** — `2603.05168_Sparse-BitNet.pdf` | arXiv 2026 | 2026 | 1.58-bit 量化 + N:M 半结构化稀疏联合设计，探索量化与稀疏的协同效应 |
| **6** | **Vectorized FlashAttention with Low-cost Exponential Computation in RISC-V Vector Processors** — `2510.06834_FlashAttention_RISC-V.pdf` | arXiv 2025 | 2025 | 同作者组（Titopoulos 等）的 RISC-V 向量化 FlashAttention，展示了他们在 V 扩展上的新工作。**可用于参考混合精度下注意力计算的优化** |
| **7** | ⭐ **待添加：Quark RISC-V Vector Processor** | ISCAS 2023 | 2023 | 整数 RISC-V 向量处理器，支持 sub-byte 量化 DNN 推理。直接相关！arXiv: 2302.xxxxx |

---

## 🟡 03-DynamicSparsePattern — 动态稀疏模式选择

对应创新点：**不同层用不同 (N,M) 组合，轻量级 predictor 在线决策**

| # | 论文 | 出处 | 年份 | 核心贡献 |
|---|---|---|---|---|
| **8** | **Dynamic Sparse Training with Structured Sparsity** — `2305.05116_DynamicSparseTraining_Structured.pdf` | ICLR 2024 | 2024 | 动态稀疏训练 + 结构化稀疏，训练过程中动态调整稀疏模式。**直接启发你创新点的论文** |
| **9** | **Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch** — `2102.04010_LearningNM_StructuredSparse.pdf` | **ICLR 2022** | 2022 | 从零训练 N:M 结构化稀疏网络。提出了 SR-STE（稀疏化直通估计器） |
| **10** | **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot** — `2301.00774_SparseGPT.pdf` | **ICML 2024** (CCF-A) | 2024 | 一次性剪枝 LLM 至 50%-60% 稀疏，用 Hessian 逆矩阵逐层重建。**方法论的启发** |
| **11** | **Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time** — `07_DejaVu_ICML2023.pdf` | **ICML 2023** (CCF-A) | 2023 | 上下文稀疏推理：每个输入 token 仅激活模型的一小部分 | 
| **12** | **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU** — `01_PowerInfer_SOSP2024.pdf` | **SOSP 2024** (CCF-A) | 2024 | 利用 LLM 中热/冷神经元的访问模式差异，在消费级 GPU 上加速推理 |

---

## 🟢 04-RegisterAlloc-Tiling — 寄存器分配与分块策略

对应创新点：**稀疏感知的向量寄存器分配：不同 VLEN 下的最优分块策略**

| # | 论文 | 出处 | 年份 | 核心贡献 |
|---|---|---|---|---|
| **13** | **Ara: A 1 GHz+ Scalable and Energy-Efficient RISC-V Vector Processor with Multi-Precision Floating Point Support in 22 nm FD-SOI** — `1906.00478_Ara_RISC-V_Vector.pdf` | **IEEE TVLSI 2020** (CCF-B) | 2020 | 最著名的开源 RISC-V 向量处理器实现，支持多精度浮点。**了解 V 扩展硬件约束的必读** |
| **14** | **RISC-V V 扩展规范 v1.0** — `11_RISC-V_Vector_Spec_v1.0.pdf` | RISC-V International | 2021 | 官方规范文档，定义 mask/gather/scatter/vrgather 等关键指令 |

---

## 🔵 05-DataReorganize-Format — 数据重组与向量友好格式

对应创新点：**非结构化稀疏 → 向量友好的数据重组格式**

| # | 论文 | 出处 | 年份 | 核心贡献 |
|---|---|---|---|---|
| **15** | **SpChar: Characterizing the Sparse Puzzle via Decision Trees** — `2304.06944_SpChar.pdf` | **ISCA 2024** (CCF-A) | 2024 | 用决策树分析稀疏 kernel 的性能特征（缓存、内存延迟、分支预测）。**性能建模方法论** |

---

## 🟣 06-Related-LLM-Sparse — LLM 稀疏化相关

LLM 模型压缩/加速中可借鉴的方法。

| # | 论文 | 出处 | 年份 | 核心贡献 |
|---|---|---|---|---|
| **16** | **FlashAttention: Fast and Memory-Efficient Exact Attention** — `03_FlashAttention_NeurIPS2022.pdf` | **NeurIPS 2022** (CCF-A) | 2022 | IO-aware 算法设计范式的代表。**对设计稀疏 kernel 的访存优化有启发** |
| **17** | **AWQ: Activation-aware Weight Quantization** — `04_AWQ_ICML2024.pdf` | **ICML 2024** (CCF-A) | 2024 | 基于激活感知的权重量化。**量化+稀疏协同的思路参考** |
| **18** | **LLM in a Flash: Efficient Large Language Model Inference with Limited Memory** — `02_LLM_in_a_Flash_ACL2024.pdf` | ACL 2024 (CCF-A) | 2024 | 有限内存下的 LLM 推理。**内存管理策略参考** |
| **19** | **FlexGen: High-Throughput Generative Inference of Large Language Models** — `05_FlexGen_OSDI2023.pdf` | **OSDI 2023** (CCF-A) | 2023 | 通过离线规划最大化 LLM 推理吞吐。**系统调度策略借鉴** |

---

## 📝 已确认但未下载的论文（待补充）

| 论文 | 出处 | 为什么重要 |
|---|---|---|
| **VIA: A Smart Scratchpad for Vector Units** (Pavon et al.) | HPCA 2021 | 为向量单元添加 scratchpad 处理非结构化稀疏。HPCA 同类型工作 |
| **Quark: An Integer RISC-V Vector Processor for Sub-Byte Quantized DNN Inference** | ISCAS 2023 | 直接相关：RISC-V V 扩展 + sub-byte 量化推理 |
| **SELL-C-sigma / CSR5 稀疏格式** | SC / IPDPS | 经典稀疏矩阵存储格式，向量友好的格式设计 baseline |
| **SparTA: Deep-Learning Model Sparsity via Tensor-with-Tensor Abstraction** | OSDI 2023 | 编译器级别处理多种稀疏模式的框架 |
| **PIT: Optimization of Dynamic Sparse Deep Learning Models** | OSDI 2024 | 动态稀疏的编译器优化 |

---

## 🗺️ 阅读路线建议

### 路线 A：快速了解方向（第一周）
1. → 01-Baselines/IndexMAC (DATE 2024)
2. → 01-Baselines/TC 2025 (核心 baseline)  
3. → 05-DataReorganize/SpChar (ISCA 2024)
4. → 03-DynamicSparse/Dynamic Sparse Training (ICLR 2024)

### 路线 B：深入 sparse GEMM（第二周）
5. → 01-Baselines/LeFevre SpGEMM (HPDC 2023)
6. → 01-Baselines/Mishra N:M (Arxiv 2021)
7. → 04-RegisterAlloc/Ara (TVLSI 2020)
8. → 02-MixedPrecision/FlashAttention RISC-V (2025)

### 路线 C：创新点收敛（第三周起）
9. → 06-Related/SparseGPT (ICML 2024)
10. → 02-MixedPrecision/Sparse-BitNet (2026)
11. → 06-Related/Learning N:M (ICLR 2022)
12. → ← 回到 baseline，找出空白点 → 写论文

---

## 🎯 四个创新方向 ↔ 论文映射

| 创新方向 | 最相关论文 | 难度 | 创新性 | 出活速度 |
|---|---|---|---|---|
| **混合精度+稀疏 co-design** | Sparse-BitNet (2026), AWQ (2024), FlashAttention-RISC-V (2025) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **动态稀疏模式选择** | Dynamic Sparse Training (ICLR 2024), Learning N:M (ICLR 2022) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **稀疏感知寄存器分配** | Ara (TVLSI 2020), TC 2025, LeFevre (HPDC 2023) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **向量友好数据重组** | SpChar (ISCA 2024), SpGEMM RISC-V (HPDC 2023) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

> 📌 标注 ⭐ 的论文建议补全下载  
> 📌 所有文件名以 arXiv ID 开头，方便精确引用
