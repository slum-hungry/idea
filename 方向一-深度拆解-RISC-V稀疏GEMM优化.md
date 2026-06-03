# 方向一深度拆解：RISC-V V 扩展的稀疏 GEMM 优化

> 基于论文 *"Optimizing Structured-Sparse Matrix Multiplication in RISC-V Vector Processors"* (Titopoulos et al., IEEE TC 2025) 的 Baseline 分析 + 创新方向探讨
>
> 日期：2026-06-03

---

## 目录

- [一、你要打败的 Baseline：Titopoulos 团队的算法梯子](#一你要打败的-baselinetitopoulos-团队的算法梯子)
- [二、Baseline 留下的坑（你的机会）](#二baseline-留下的坑你的机会)
- [三、四个创新空间的可行性分析](#三四個创新空间的可行性分析)
- [四、建议的论文方案（最小可发表单元）](#四建议的论文方案最小可发表单元)
- [五、现在就可以做的第一件事](#五现在就可以做的第一件事)

---

## 一、你要打败的 Baseline：Titopoulos 团队的算法梯子

这篇论文一共给了 6 个算法，逐层进化。理解它们的演化逻辑，才能找到突破口。

### 第一层：稠密 baseline（Alg 1 / 2 / 3）

论文先不考虑稀疏，实现三种 Row-wise GEMM 的向量化方案：

| 算法 | A 放哪 | B 放哪 | 核心操作 |
|------|--------|--------|---------|
| **Alg 1** | VRF（向量） | VRF | vload A 行 → 逐元素 vmv 到 SRF → FMA |
| **Alg 2** | SRF（标量） | VRF | A 的非零元直接 load 到 SRF，消除 VRF↔SRF 搬运 |
| **Alg 3** | VRF | VRF | 用 `vrgather.vx` 拷贝 A[j] 到全部 lane，FMA 全用 3 向量操作数 |

**Alg 3 是最优的**，因为它避免了 Alg 1 的 VRF↔SRF 搬运，也避免了 Alg 2 的 SRF 端口压力。

### 第二层：结构化稀疏适配（Alg 1-S / 2-S / 3-S）

将 A 改为 N:M 结构化稀疏存储（只有非零元 + 块内列号）。核心新增：**列号恢复**。

```
存储的 col_idx ∈ [0, M-1]   ← 块内偏移
实际列号 = block_id × M + col_idx
         = floor(j / N) × M + col_idx
```

j = 非零元在当前行非零序列中的位置，N = 每块最大非零数。

以 3:4 稀疏为例，非零元 'e' 在非零序列中排第 4 位（j=4），block_id = floor(4/3) = 1，块偏移 = 1×4 = 4，加上 col_idx=1，实际列号 = 5。

### 第三层：循环展开（Alg 4）

在 Alg 3-S 基础上**外循环×2 + 内循环×2 + 交错执行**：

```
外循环展开：同时处理 A 的行 i 和 i+1 → 同时产出 C 的两行
内循环展开：同时处理 A 的非零元 j 和 j+1

交错执行：
  组1: load s0 (col_i,j) → load s1 (col_i+1,j) → 算 block_id → 
       vload B[s0] → vload B[s1] → vrgather → FMA
  组2: 此处不是紧接组1的 FMA，而是交错插入组2的 load 操作，隐藏 load 延迟
```

寄存器用量：Alg 3-S 用 1 SR + 4 VR，Alg 4 用 2 SR + 8 VR（通过寄存器复用，没有 4× 膨胀）。

**展开到多少才够？** 论文 Fig.6 显示展开度 ×8 附近到达峰值。超过 ×16 就开始 spill 到内存，性能崩塌。

### 第四层：vindexmac 革命（Alg 5）

前三层的问题在哪？**A 有多少个非零元，就要从内存 load 多少次 B 的行**。每次 `vload` 都要过 cache 层级，延迟巨大。

vindexmac 的核心洞察：**结构化稀疏的"围栏"效应**。

```
非结构化稀疏：col_idx ∈ [0, N-1] 无界 
              → B 的某行可能被任何 A 非零元引用，无法预装
结构化稀疏：  col_idx ∈ [0, M-1] 有界 
              → A 的一个 block 只引用 B 的前 M 行！
```

所以可以：

1. **预装 B 的一个 tile（L×VL）到 VRF**
2. 用 `vindexmac` 指令**直接从 VRF 间接读 B 行**，不再走内存
3. 处理完一个 tile 的非零元后，换下一个 tile

```
vindexmac.vx vd, vs2, rs
语义：vd[i] += vs2[0] × VRF[ rs[4:0] ][i]

rs 的内容不当数值，当 VRF 地址！
唯一硬件代价：读端口地址总线上的 5-bit 2→1 MUX（4 µm² vs VRF 本身 56,527 µm²）
```

**效果**：纯标准 RVV 指令优化之后，再加 vindexmac 还能再快 **25-33%**。

### 第五层：多 tile + 多核（Alg 6 + 二维分块 Fig.5）

当一行非零元跨越的列数 > L 时，需要换 tile：

```
m = (M/N) × (VL/L) 个 tile

大矩阵：B 按 VL 列宽度切成垂直段（segments），每段独立执行 Alg 6
多核并行：每个核心负责一个垂直段，C 无写冲突
```

但论文发现 **8 核以上收益饱和**——原因不是计算，而是**内存带宽限制**。

---

## 二、Baseline 留下的坑（你的机会）

| Baseline 做了什么 | Baseline 没做什么 |
|---|---|
| ✅ 结构化稀疏在 RVV 上的 GEMM | ❌ **混合精度**：全是 FP32，没碰 INT8/INT4 |
| ✅ 一条自定义指令 vindexmac | ❌ **动态稀疏选择**：所有层同一个 N:M，不能逐层自适应 |
| ✅ CNN (ResNet) 上的评估 | ❌ **LLM (Transformer)** 上的评估 |
| ✅ gem5 模拟器 | ❌ 真实 RISC-V 芯片上跑 |
| ✅ 单核 + 多核 | ❌ 稀疏+量化 **co-design** |

**你路线图里列的四个创新空间，正好堵这五个缺口。**

---

## 三、四个创新空间的可行性分析

### 创新点 ①：混合精度 + 稀疏 co-design

**问题**：Baseline 全是 FP32 稀疏 GEMM。边缘设备上真正有意义的场景是 **INT4 量化 + N:M 稀疏** 同时使用。

**技术挑战（为什么不是简单叠加）**：

```
量化需要 dequant scale（每 group 一个 float），
稀疏 N:M 存储改变了权重排布，
两者叠加后的内存布局完全不同于单独的量化或稀疏。
```

**具体可以做什么**：

- 设计一种 **sparse-quantized 联合存储格式**：把 INT4 权重 + 2:4 稀疏编码 + dequant scale 打包进向量友好的 layout
- 实现 `int4_sparse_fma` kernel：用 RVV 的整数 vector 指令 + 位操作提取低精度权重，一条指令算 2× VLEN/4 个乘加
- 与纯 FP32 稀疏、纯 INT4 量化分别对比，展示 1+1>2 的带宽节省

**论文故事**：

> "现有工作要么做稀疏，要么做量化，我们在 RISC-V V 上首次做了二者的 joint optimization，在相同精度下带宽减少 YY%，延迟降低 XX%。"

---

### 创新点 ②：动态稀疏模式选择

**问题**：Baseline 所有 CNN 层用同一个 2:4。LLM 里不同层的稀疏度容忍度完全不同——Attention 的 Q/K/V 投影对稀疏极其敏感，FFN 的 up/gate 投影可以剪得很激进。

**具体可以做什么**：

```
Layer   │  推荐 (N,M)  │  原因
────────┼───────────────┼──────────
Q/K/V   │  4:4 (不剪)   │  注意力质量对稀疏极其敏感
O proj  │  2:4          │  可适度剪枝
FFN up  │  1:4 或 1:8   │  最冗余，大量剪
FFN gate│  1:4 或 1:8   │  同上
FFN down│  2:4          │  略保守
```

- 离线阶段：per-layer sensitivity analysis，确定每层的 (N,M)
- 在线阶段：用 **查找表**（一行 2 bit 编码 N:M 模式）驱动 GEMM kernel 选择对应路径
- 不需要 runtime predictor（这是加分项，说明你懂硬件——纯查表零开销）

**论文故事**：

> "我们观察到 LLM 各层对稀疏的容忍度差异可达 3×，提出 per-layer 自适应 N:M 选择，在精度几乎无损的前提下比 uniform 稀疏再多削减 30% 访存。"

---

### 创新点 ③：稀疏感知的 VRF 分配

**问题**：Baseline 假设 VLEN=512。实际 RISC-V 芯片的 VLEN 多种多样（128→1024）。VLEN 不同，(N,M) 选择与 tile size L 的最优组合也不同。

**具体可以做什么**：

- 建立一个分析模型：输入 VLEN、SEW、N、M，输出最优的 L（B tile 行数）和展开度
- **关键发现**：当 VLEN < M×SEW 时，一条向量指令装不下一个完整 N:M block 的 B 行，需要跨寄存器拼接
- 给出 **不同 VLEN 下的 N:M 推荐表**（这对 RISC-V 生态有普适价值）

```
VLEN=128, SEW=16: 一条指令 8 元素 → 对 2:4 对齐良好
VLEN=128, SEW=8:  一条指令 16 元素 → 对 1:4 也够用
VLEN=256, SEW=32: 一条指令 8 元素 → 和 VLEN=128/SEW=16 一样
VLEN=512, SEW=32: 一条指令 16 元素 → 对齐 1:16, 2:8, 4:4 均友好
```

---

### 创新点 ④：非结构化稀疏 → 向量友好的重组

**问题**：实际训练的稀疏模型是非结构化的。虽然可以 retrain 成结构化，但最务实的离线路径是**重组**。

**已有方法**：

- 行/列重排 + zero padding [9]
- 矩阵分裂 [10]

**你可以做的改进**：

- 设计一种 **RISC-V V 感知的稀疏重组算法**：不仅考虑 N:M 对齐，还考虑块内非零元的列号分布，使得 vindexmac 在 tile 内命中率最大化
- 核心指标：重组后每 tile 的 B 行复用率（一行 B 被 A 的多个非零元引用的次数）

---

## 四、建议的论文方案（最小可发表单元）

### 标题（草案）

> **SparseGEMM-RV: Mixed-Precision Structured-Sparse GEMM with Dynamic Pattern Selection on RISC-V Vector Processors**

### 贡献结构

| # | 贡献 | 会议价值 |
|---|------|---------|
| 1 | 首次在 RISC-V V 上实现 **量化+稀疏 co-design** 的 GEMM | 新问题 |
| 2 | **Per-layer 自适应 N:M 选择**，LLM 精度几乎无损失 | 新方法 |
| 3 | Sparse-quantized 联合存储格式 + RVV kernel 实现 | 系统工程 |
| 4 | gem5 + 真实 RISC-V 硬件上的系统评估（vs Titopoulos baseline） | 填补空白 |

### 实验设置

- **平台**：gem5 (对标 Titopoulos) + 如有条件，玄铁 C910/C920 开发板
- **模型**：LLaMA-3.2-1B / Qwen2.5-1.5B（边缘 LLM 级别）
- **对比**：Titopoulos 的 FP32 sparse GEMM、稠密 INT4 GEMM、你的 co-design
- **指标**：延迟、带宽利用率、能耗、per-layer 稀疏度分布

### 目标会议梯度

| 投稿轮次 | 目标 | 策略 |
|---------|------|------|
| 第一轮 | **MLSys / ISPASS** | 侧重系统创新 + benchmark |
| 第二轮 | **ASPLOS / MICRO** | 侧重架构贡献 + 硬件设计 |
| 第三轮 | **HPCA / ISCA** | 如果加了硬件扩展（如 vindexmac 的 LLM 定制版） |

---

## 五、现在就可以做的第一件事

1. **搭 gem5 RISC-V 向量处理器仿真环境**（论文用的配置在 Titopoulos et al. 的实验设置中有详细描述）
2. **复现 Alg 3-S 的 baseline**：不需要 vindexmac，纯标准 RVV 指令实现结构化稀疏 GEMM
3. **拿到第一条性能数据**（哪怕只是一个矩阵乘法 kernel 的延迟）——这就够你写 motivation 了
4. **选一个创新点主攻**：建议从 ①（混合精度+稀疏 co-design）or ②（动态 N:M 选择）起步，这两个出活最快

---

## 附录：相关术语速查

| 术语 | 全称 | 含义 |
|------|------|------|
| VRF | Vector Register File | 向量寄存器文件（32×512bit） |
| SRF | Scalar Register File | 标量寄存器文件 |
| VLEN | Vector Length | 向量寄存器位宽（128/256/512/1024 bit） |
| SEW | Selected Element Width | 向量元素位宽（8/16/32/64 bit） |
| N:M 稀疏 | N:M Structured Sparsity | 每 M 个连续元素中最多 N 个非零 |
| vrgather.vx | — | 按标量索引从向量寄存器中收集元素 |
| vindexmac | — | 自定义指令：间接读 VRF + 乘加 |
| FMA | Fused Multiply-Add | 融合乘加 |
| CSR/CSC | Compressed Sparse Row/Column | 稀疏矩阵压缩存储格式 |
| GEMM | General Matrix Multiply | 通用矩阵乘法 |
