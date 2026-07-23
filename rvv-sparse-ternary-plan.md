# RISC-V RVV 稀疏三元矩阵乘：从零到一 完整实施计划

> 目标：在 RISC-V 模拟器上实现 Sparse-BitNet 风格的 N:M 稀疏 + 三元量化矩阵乘 kernel，系统评估不同 N:M 模式在不同 VLEN 下的性能。

---

## 总览：你要交付什么

```
输入: 权重 W ∈ {-1, 0, +1}^{M×K}（三元 + N:M 稀疏）
       激活 X ∈ int8^{K×N}
输出: Y = γ · (W_q ⊙ M) · X̃   （≈ FP32/BF16 精度）

benchmark 产出:
  - 不同 N:M 模式 (2:4, 4:8, 6:8, 8:16...) 的 cycle 数
  - 不同 VLEN (128/256/512) 下的吞吐量 (GOPS)
  - 与 dense baseline、标量 baseline 的加速比
  - cache miss / 访存带宽分析
```

---

## 阶段零：环境搭建（预计 1-2 天）

### 0.1 安装 RISC-V GNU 工具链（带 RVV intrinsic 支持）

```bash
# 推荐用预编译的，不自己编译（省时间）
# 方法一（推荐）：从 sipeed 下载预编译工具链
# https://github.com/sipeed/riscv-gnu-toolchain/releases

# 方法二：从 emmtrix 下载 rvv-intrinsic 支持的版本
# wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases
# 选择 riscv64-elf- 或 riscv64-linux- 前缀的

# Windows (你的环境) 可以用 WSL，或者用 Docker
# 推荐 WSL2 + Ubuntu 22.04
```

### 0.2 安装 Spike 模拟器

```bash
# 安装依赖
sudo apt-get install device-tree-compiler libboost-all-dev

# 克隆并编译
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=/opt/riscv --enable-commitlog
make -j$(nproc)
sudo make install

# 测试
spike --isa=rv64gcv_zicntr_zihpm pk hello
# pk 是 proxy kernel，用于跑 bare-metal 程序
```

### 0.3 编译 proxy kernel (pk)

```bash
git clone https://github.com/riscv-software-src/riscv-pk.git
cd riscv-pk
mkdir build && cd build
../configure --prefix=/opt/riscv --host=riscv64-unknown-elf
make -j$(nproc)
sudo make install
```

### 0.4 设置环境变量

```bash
# 加到 ~/.bashrc
export RISCV=/opt/riscv
export PATH=$RISCV/bin:$PATH
export RISCV_GCC=$(which riscv64-unknown-elf-gcc)
export SPIKE=$(which spike)
```

### 0.5 验证环境

```bash
# 确认编译器有 RVV 支持
riscv64-unknown-elf-gcc -march=rv64gcv -O2 --print-multi-lib

# 写一个 hello RVV 测试
cat > test_rvv.c << 'EOF'
#include <riscv_vector.h>
int main() {
    size_t vl;
    vuint8m1_t a, b, c;
    vl = __riscv_vsetvl_e8m1(128);
    c = __riscv_vadd_vv_u8m1(a, b, vl);
    return 0;
}
EOF

riscv64-unknown-elf-gcc -march=rv64gcv -O2 test_rvv.c -o test_rvv
spike --isa=rv64gcv_zicntr pk test_rvv
echo "如果没报错，环境 OK"
```

---

## 阶段一：知识准备（边做边学，2-3 天）

### 1.1 必须掌握的 RVV 概念

| 概念 | 说明 | 对应你的需求 |
|---|---|---|
| `vsetvl` | 设置向量长度 | 每行/每块 K 维度不同时需要 |
| `vle8.v` / `vse8.v` | 向量 load/store | 加载激活值、写出结果 |
| `vadd.vv` / `vsub.vv` | 向量加减 | 三元权重用加减代替乘法 |
| `vmerge.vxm` | mask 选择 | 稀疏模式——跳过零值 |
| `vrgather.vv` | 向量索引 gather | 稀疏权重按非零索引 gather 激活 |
| `vmacc.vv` | 向量乘累加 | 核心运算 |
| `vredsum.vs` | 向量归约求和 | 内积归约 |

### 1.2 阅读材料（按优先级）

1. **RISC-V Vector Extension Spec v1.0**（重点看 2-5 章）：https://github.com/riscv/riscv-v-spec
2. **RVV Intrinsic 文档**：`riscv-non-isa/rvv-intrinsic-doc`（github）
3. **Spike 源码中的 vector 执行**：`riscv-isa-sim/riscv/insns/vadd_vv.h` 等
4. **你的 Sparse-BitNet 论文**：重点看 Section 2.2（Sparse-BitLinear 前向公式）

---

## 阶段二：实现 Dense Ternary MatMul Baseline（核心第一步，3-5 天）

### 2.1 数据结构设计

```c
// 三元权重编码：每个元素 2 bit
// 00 = 0, 01 = +1, 10 = -1  (留一个编码扩展)
typedef struct {
    uint8_t *data;       // packed 2-bit 编码
    float    gamma;       // scaling factor γ
    uint32_t rows;        // M (output dim)
    uint32_t cols;        // K (input dim)  
} TernaryWeight;

// 激活值：int8
typedef struct {
    int8_t  *data;
    float    scale;       // absmax scale
    uint32_t rows;        // K
    uint32_t cols;        // N
} QuantizedActivation;

// 输出：float32
typedef struct {
    float   *data;
    uint32_t rows;        // M
    uint32_t cols;        // N
} OutputMatrix;
```

### 2.2 Dense Ternary MatMul：标量版（验证逻辑）

先写标量版确保计算逻辑正确，这步很关键，后续做正确性 gold reference。

```c
// 标量版：y = γ * W_q * x
void dense_ternary_matmul_scalar(
    const TernaryWeight *W,
    const QuantizedActivation *X,
    OutputMatrix *Y)
{
    float s = W->gamma * X->scale / 127.0f;
    
    for (int m = 0; m < W->rows; m++) {
        for (int n = 0; n < X->cols; n++) {
            float sum = 0;
            for (int k = 0; k < W->cols; k++) {
                int8_t w = ternary_decode(W->data, m * W->cols + k); // -1/0/+1
                if (w != 0) {
                    sum += w * X->data[k * X->cols + n];
                }
            }
            Y->data[m * Y->cols + n] = s * sum;
        }
    }
}
```

### 2.3 Dense Ternary MatMul：RVV 向量化版

```
核心思路：
  将 K 维度向量化 —— 一条向量指令处理 VLEN/SEW 个 K 元素
  外层循环 M×N，内层循环 K 分块

算法伪代码：
  for m in 0..M:
    for n_block in 0..N step VLEN/SEW:
      vsum = [0, 0, ..., 0]   // 向量累加器
      for k_block in 0..K step VLEN/SEW:
        vl = min(VLEN/SEW, K - k_block)
        v_w = vload_ternary(W[m, k_block:k_block+vl])  // 解码三元值为 int8 向量
        v_x = vload(X[k_block:k_block+vl, n_block])     // 加载激活列
        v_mask = (v_w != 0)                              // 生成 mask
        // 用 mask 做条件乘累加：只计算非零权重位置
        vsum = vmacc_masked(vsum, v_w, v_x, v_mask, vl)
      vstore(Y[m, n_block:n_block+vl], vsum)
```

### 2.4 关键 RVV Intrinsic 写法

```c
#include <riscv_vector.h>

// 三元解码函数：从 packed 2-bit 中提取 int8 值
vint8m1_t ternary_decode_packed(const uint8_t *packed, size_t vl) {
    // 用 vrgather 或 lut 查表方式
    // 备选：预解压为 int8 数组再 vload
    // 简化：先存为 int8 数组方便开发
    return __riscv_vle8_v_i8m1((const int8_t*)unpacked, vl);
}

// 核心内积循环
void dense_ternary_matmul_rvv_row(
    const int8_t *w_row,      // 一行权重 [-1, 0, +1], 长度 K
    const int8_t *x,          // 激活矩阵, K×N
    float       *y_row,       // 输出行, 长度 N
    uint32_t     K, uint32_t  N,
    float        scale)
{
    for (uint32_t n = 0; n < N; ) {
        size_t vl = __riscv_vsetvl_e8m1(N - n);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        
        for (uint32_t k = 0; k < K; ) {
            size_t vlk = __riscv_vsetvl_e8m1(K - k);
            
            // 加载权重和对应激活
            vint8m1_t vw = __riscv_vle8_v_i8m1(&w_row[k], vlk);
            
            // 生成非零 mask
            vbool8_t mask = __riscv_vmsne_vx_i8m1_b8(vw, 0, vlk);
            
            // 对 N 维度的每一列做乘累加
            // （这里需要展开 N 维度，或使用 outer-product 策略）
            // ... 具体展开见 2.5 节
            
            k += vlk;
        }
        
        // 写入结果
        __riscv_vse32_v_f32m1(&y_row[n], vsum, vl);
        n += vl;
    }
}
```

### 2.5 优化：Outer Product 策略（更适合 RVV）

上面的 inner-product 策略每次只处理一行权重，需要反复加载激活矩阵。对于 LLM 推理场景（M=4096, K=4096, N=1 或 N=seq_len），更推荐：

```
策略选择：
  - 推理 bs=1 (N小) → inner-product（一行权重 × 少量激活列）
  - 推理 bs>1 (N大) → outer-product（向量外积，复用寄存器）
  - 训练场景 (N大)  → 分块 outer-product

对 benchmark：两种都做，对比分析
```

---

## 阶段三：实现 N:M 稀疏 Ternary MatMul（5-8 天）

### 3.1 N:M 稀疏数据布局

```
原始权重展平：[-1, 0, +1, 0, 0, -1, +1, 0, ...]
         分块: [  -1, 0,+1, 0 | 0,-1,+1, 0 | ... ]
                ├── M=4 ──┤   ├── M=4 ──┤

压缩存储方案（选 6:8, M=8, N=6）：
  - values:    [-1, -1, +1, +1, -1, +1]   // 每8个中保留6个非零
  - indices:   [ 0,   5,  1,  3,  2,  6]   // 各非零在块内的相对位置
  - bitmask:   0b11011010                   // 1=保留, 0=剪掉 (LSB first)
```

### 3.2 RVV 稀疏 MatMul 核心算法

```
方案A：mask register 方案（推荐，简洁）
  1. 用 N:M 规则计算 bitmask → 转为 vbool mask
  2. 加载整块 8 个权重（含零）→ vle8
  3. 加载对应 8 个激活列 → vle8
  4. 用 mask 做 vmacc: vsum += mask ? (w * x) : 0
  5. RVV 的 mask 操作是零开销的（数据路径上不执行被 mask 的 lane）
  6. 优点：代码简单，RVV 天然支持
  7. 关键确认：Spike 的 mask 操作计算被 mask 掉的 cycle 数

方案B：gather 方案（可能更高效）
  1. 只存储非零权重值 + 对应的 K 索引
  2. 用 vrgather 按索引 gather 激活值
  3. 对所有非零值做 vmac
  4. 优点：只加载/计算非零值，无浪费
  5. 缺点：vrgather 可能比 masked vmacc 慢（取决于微架构）
  
结论：两种都实现，benchmark 对比。这是论文里有价值的实验。
```

### 3.3 N:M Mask 生成函数

```c
// 输入：float 或 int8 master weights (M 个元素)
// 输出：vbool mask，标记保留的 N 个元素
// 用时：O(M log M) - 但 M 很小 (4/8/16)，实际开销极小

vbool8_t nm_generate_mask_per_block(
    const int8_t *weights_block,  // M 个元素
    uint32_t M, uint32_t N)
{
    // 1. 计算绝对值
    // 2. 找 Top-N 绝对值（可以用选择网络或排序网络）
    // 3. 生成 mask
    // 对三元权重 {-1,0,+1}：非零权重的绝对值都相等(=1)
    //    所以 N:M 等价于从 M 个中随机选 N 个非零
    //    如果非零数量 < N，保留全部非零
    // 简化：用 threshold
}
```

**重要洞察**：对于三元权重 {-1,0,+1}，因为非零值绝对值全为 1，N:M magnitude pruning 退化成了"保留任意 N 个非零，如果非零数 < N 则全保留"。这意味着 mask 生成极其简单，不需要排序。

这是一个**可以在论文里强调的优势**：三元量化使 N:M mask 生成从 O(M log M) 排序降为 O(M) 计数 + 选择。

### 3.4 稀疏 MatMul RVV 实现大纲

```c
// 核心函数签名
void sparse_ternary_matmul_nm_rvv(
    const TernaryWeight *W,         // 三元权重（已应用 N:M 稀疏）
    const uint32_t      *nm_mask,   // 预计算的 N:M mask (packed bits)
    const QuantizedActivation *X,
    OutputMatrix *Y,
    uint32_t N, uint32_t M          // N:M 参数
);

// 实现结构：
//   for m in 0..W->rows:
//     for k_block in 0..W->cols step M:
//       // 1. 取当前块的 mask bits
//       // 2. 取当前块的 M 个权重
//       // 3. 取对应当前块的 M 行激活
//       // 4. 用 mask 做 masked vmacc
//     // 5. 乘 scaling factor 写入结果
```

---

## 阶段四：Benchmark 框架（3-5 天）

### 4.1 实验矩阵

```
独立变量：
  模型规模:    M×K×N = {512×512×512, 4096×4096×4096, 4096×14336×4096}
              （模拟 LLM 的 FFN 层典型尺寸）
  N:M 模式:   {2:4, 3:4, 4:8, 6:8, 8:16, 12:16, ...}
  VLEN:       {128, 256, 512}  (Spike 用 --varch 参数控制)
  SEW:        {8, 16, 32}      (int8 vs fp16 vs fp32)

因变量（测什么）:
  - 总 cycle 数（Spike -l 输出）
  - 指令计数（按类型分：vector load/store/alu/mask）
  - 有效 GOPS（根据 cycle 数和 FLOP 推算）
  - 加速比 = cycle_dense / cycle_sparse
  - 理论利用率 = 有效操作数 / VLEN
```

### 4.2 Spike 性能统计

```bash
# Spike commit log 模式：输出每条提交指令的详细信息
spike --isa=rv64gcv_zicntr \
      --varch=vlen:256,elen:64 \
      -l --log-commits \
      pk ./sparse_ternary_bench

# 用脚本解析 log，统计：
#  - vector load 次数
#  - vector alu 次数  
#  - mask 操作次数
#  - 总 cycle 数

# 备选：Spike 的 --histogram 看 PC 分布
```

### 4.3 RISC-V 性能计数器

```c
// 使用 RISC-V 硬件性能计数器 (mcycle, minstret)
static inline uint64_t read_mcycle() {
    uint64_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

static inline uint64_t read_minstret() {
    uint64_t insts;
    asm volatile("csrr %0, minstret" : "=r"(insts));
    return insts;
}
```

---

## 阶段五：结果分析与论文撰写（5-7 天）

### 5.1 需要产出的图表

1. **主图：加速比 vs N:M 稀疏度**
   - X轴 = sparsity ratio (0%, 25%, 50%, 75%)
   - Y轴 = 加速比 (vs dense ternary matmul)
   - 多条线 = 不同 VLEN (128/256/512)
   - 预期：VLEN 越大加速越明显

2. **主图：GOPS vs 矩阵规模**
   - X轴 = K 维度大小
   - Y轴 = 有效 GOPS
   - 多条线 = mask 方案 vs gather 方案

3. **分析图：RVV 利用率**
   - 饼图/柱状图：vector 指令中各类型占比
   - load/store 占比 vs compute 占比
   - 识别瓶颈

4. **对比表：vs NVIDIA Sparse Tensor Core**
   - 引用 Sparse-BitNet 论文的 GPU 数据
   - 理论峰值 vs 实测利用率
   - 架构差异分析

### 5.2 论文叙事线

```
1. Sparse-BitNet 证明了 1.58-bit 量化 + N:M 稀疏在算法层面有效
2. 但现有工作依赖 NVIDIA Sparse Tensor Core（固定 2:4）
3. RISC-V RVV 提供了机会：灵活的向量长度 + mask 机制支持任意 N:M
4. 我们实现了 RVV 原生的稀疏三元 MatMul，系统探索 N:M 设计空间
5. 发现：
   a. 三元权重使 mask 生成开销降为 O(M)（vs O(M log M)）
   b. masked vmacc 比 vrgather 更高效（对 int8/小 M）
   c. VLEN>=256 时 6:8 稀疏即可逼近理论峰值
   d. RISC-V 的灵活性在某些 N:M 模式下甚至超过 GPU 固定 2:4
6. 为 RISC-V 生态中稀疏量化 LLM 推理提供了首个系统研究
```

---

## 完整时间线

| 阶段 | 内容 | 预计 |
|---|---|---|
| 0 | 环境搭建 | 1-2 天 |
| 1 | 知识准备 (边学边做) | 2-3 天 |
| 2 | Dense Ternary MatMul | 3-5 天 |
| 3 | N:M Sparse Kernel | 5-8 天 |
| 4 | Benchmark 框架 | 3-5 天 |
| 5 | 结果分析 + 论文 | 5-7 天 |
| **合计** | | **19-30 天** |

---

## 可能踩的坑 & 解决方案

| 问题 | 解决方案 |
|---|---|
| Spike 太慢，跑大模型不行 | 用小尺寸矩阵做 micro benchmark；关注相对加速比而非绝对性能 |
| RVV intrinsic API 变化 | 锁定一个工具链版本，记录版本号 |
| 编译时 -march 不对 | 仔细确认 `-march=rv64gcv_zicntr` flag |
| Mask 操作实际性能 | Spike 是功能模拟器，cycle 不反映真实微架构；需要 gem5 做精确建模（后续扩展） |
| packed 2-bit 三元解码开销大 | 初期用 int8 存储，优化阶段再做 packing |

---

## 如果卡住了从哪里求助

1. RISC-V Vector SIG mailing list: sig-vector@lists.riscv.org
2. Spike 源码 issue：github.com/riscv-software-src/riscv-isa-sim
3. RISC-V 中文社区微信群 / 知乎相关话题
4. 相关论文的 repo：github.com/AAzdi/Sparse-BitNet
