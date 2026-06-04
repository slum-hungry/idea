# RISC-V 稀疏 GEMM 研究工作清单 & 操作手册

> 目标：在 gem5 上复现 Titopoulos (IEEE TC 2025) 的 Alg 3-S，并扩展到 INT4 混合精度 co-design
>
> 日期：2026-06-04

---

## 总览：四阶段路线图

```
阶段 0 ── 环境搭建（1-2 天）
阶段 1 ── 复现 Baseline：FP32 Alg 3-S（3-5 天）
阶段 2 ── INT8/INT4 混合精度 co-design（5-7 天）
阶段 3 ── 自定义 vindexmac 指令 + 论文实验（10-14 天）
```

---

# 阶段 0：环境搭建

## 0.1 WSL2 安装（Windows 上做）

### 操作步骤

```powershell
# === 在 Windows PowerShell（管理员）中执行 ===

# 1. 启用 WSL
wsl --install -d Ubuntu-24.04
# 重启电脑

# 2. 首次进入 Ubuntu，创建用户名密码（记住就行）
```

### 验证

```bash
# 在 WSL2 Ubuntu 终端中
wsl.exe --version    # 应显示 WSL 版本 2.x
uname -a             # 应显示 Linux ... x86_64
```

---

## 0.2 安装系统依赖

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    build-essential \
    git \
    m4 \
    scons \
    zlib1g \
    zlib1g-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libprotoc-dev \
    libgoogle-perftools-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libboost-all-dev \
    pkg-config \
    device-tree-compiler \
    libpng-dev \
    libhdf5-dev

# 验证
gcc --version       # 应 >= 10
python3 --version   # 应 >= 3.8
scons --version     # 应 >= 3.0
```

---

## 0.3 编译 gem5

```bash
# 1. 创建工作目录
mkdir -p ~/riscv-sparse
cd ~/riscv-sparse

# 2. 克隆 gem5
git clone https://github.com/gem5/gem5.git
cd gem5

# 3. 切到稳定版
git checkout v24.1.0.0

# 4. 编译 RISC-V 版本（30-60 分钟）
scons build/RISCV/gem5.opt -j$(nproc)
```

### 验证

```bash
./build/RISCV/gem5.opt --version
# 应输出 gem5 版本信息
```

---

## 0.4 安装 RISC-V 交叉编译工具链

```bash
cd ~/riscv-sparse

# 方法 A: 用 apt（简单快速，适合第一轮验证）
sudo apt install -y gcc-riscv64-unknown-elf

# 方法 B: 用 prebuilt 工具链（推荐，RV V 支持更好）
# 下载地址（选最新 nightly）：
# https://github.com/riscv-collab/riscv-gnu-toolchain/releases

# 例如：
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.04.12/riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz
tar xzf riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz

# 加到 PATH（写入 ~/.bashrc）
echo 'export RISCV_HOME=$HOME/riscv-sparse/riscv' >> ~/.bashrc
echo 'export PATH=$RISCV_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 验证

```bash
riscv64-unknown-elf-gcc --version     # 应 >= 13.0
riscv64-unknown-elf-gcc -march=rv64gcv -E -dM - < /dev/null | grep riscv_vector
# 应输出: #define __riscv_vector 1
```

---

## 0.5 环境冒烟测试

### 0.5.1 创建测试文件

```bash
mkdir -p ~/riscv-sparse/kernels
cd ~/riscv-sparse/kernels
```

**`smoke_test.c`**：

```c
#include <stdint.h>

#define N 16
float A[N] __attribute__((aligned(64)));
float B[N] __attribute__((aligned(64)));
float C[N] __attribute__((aligned(64)));

void _start() {
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
        C[i] = 0;
    }

    // RVV 向量加法: C = A + B
    asm volatile(
        "li     t0, %[len]              \n\t"
        "vsetvli zero, t0, e32, m1      \n\t"
        "vle32.v v0, (%[a])             \n\t"
        "vle32.v v1, (%[b])             \n\t"
        "vfadd.vv v2, v0, v1            \n\t"
        "vse32.v v2, (%[c])             \n\t"
        :
        : [len] "i"(N), [a] "r"(A), [b] "r"(B), [c] "r"(C)
        : "t0", "memory"
    );

    // exit(0)
    asm volatile("li a7, 93; li a0, 0; ecall");
}
```

### 0.5.2 编译 + 跑

```bash
# 编译
riscv64-unknown-elf-gcc -march=rv64gcv -O2 -static -nostdlib smoke_test.c -o smoke_test

# 创建 gem5 运行脚本
cat > run_smoke.py << 'PYEOF'
import m5
from m5.objects import *

system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '2GHz'
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('512MB')]

system.cpu = MinorCPU()
system.cpu.isa = RiscvISA()

system.membus = SystemXBar()
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.createInterruptController()

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.workload = SEWorkload.init_compatible('smoke_test')
process = Process()
process.cmd = ['smoke_test']
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()
print("=== Simulation started ===")
exit_event = m5.simulate()
print(f"=== Exited: {exit_event.getCause()} @ tick {m5.curTick()} ===")
PYEOF

# 运行
~/riscv-sparse/gem5/build/RISCV/gem5.opt run_smoke.py
```

### ✅ 阶段 0 完成标准

- gem5 编译成功
- RISC-V 工具链可用（gcc 能识别 `-march=rv64gcv`）
- smoke_test 在 gem5 上跑通，输出 `Exited: exit @ tick ...`

---

# 阶段 1：复现 Baseline — FP32 Alg 3-S

## 1.1 理解输入数据格式

```
稀疏矩阵 A (2:4 结构化稀疏)：
  每 4 个连续权重 → 保留最多 2 个非零元
  存储格式（per row）：
    - values[]: 非零值的数组
    - col_idx[]: 块内偏移（0~3），每个非零元一个

  例子：A 的一行是 [0.5, 0, 1.2, 0, 0, 2.1, 0, 3.4, ...]
    block 0: [0.5, 0, 1.2, 0] → nonzeros: {0.5(col=0), 1.2(col=2)}
    block 1: [0, 2.1, 0, 3.4] → nonzeros: {2.1(col=1), 3.4(col=3)}
    
  values[]:  [0.5, 1.2, 2.1, 3.4, ...]
  col_idx[]: [0,   2,   1,   3,   ...]

B 矩阵 (dense)：
  正常行主序存储: B[row][col] = B[row * N + col]

C 矩阵 (dense)：
  正常行主序存储: C[i][j] = Σ_k A[i,k] * B[k,j]
```

## 1.2 创建 Alg 3-S kernel

### 1.2.1 稠密 GEMM baseline（用于对比）

**`dense_gemm.c`** — 标准稠密 GEMM，用 RVV intrinsics：

```c
#include <riscv_vector.h>
#include <stdint.h>

#define M 128
#define K 256
#define N 128

float A[M][K] __attribute__((aligned(64)));
float B[K][N] __attribute__((aligned(64)));
float C[M][N] __attribute__((aligned(64)));

void _start() {
    // ===== 初始化 =====
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i][j] = (float)((i * K + j) % 100) / 100.0f;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = (float)((i * N + j) % 100) / 100.0f;

    // ===== Dense GEMM (Gustavson 方法) =====
    for (int i = 0; i < M; i++) {
        // 清零 C[i]
        size_t vl;
        for (int n = 0; n < N; n += vl) {
            vl = __riscv_vsetvl_e32m2(N - n);
            vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            __riscv_vse32(&C[i][n], zero, vl);
        }

        // C[i] += Σ_k A[i,k] * B[k,:]
        for (int k = 0; k < K; k++) {
            float a_val = A[i][k];
            for (int n = 0; n < N; n += vl) {
                vl = __riscv_vsetvl_e32m2(N - n);
                vfloat32m2_t vb = __riscv_vle32_v_f32m2(&B[k][n], vl);
                vfloat32m2_t vc = __riscv_vle32_v_f32m2(&C[i][n], vl);
                vc = __riscv_vfmacc_vf_f32m2(vc, a_val, vb, vl);
                __riscv_vse32(&C[i][n], vc, vl);
            }
        }
    }

    asm volatile("li a7, 93; li a0, 0; ecall");
}
```

### 1.2.2 结构化稀疏 GEMM — Alg 3-S

**`alg3s_gemm.c`** — 2:4 结构化稀疏版本：

```c
#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>

#define M      128
#define K      256
#define N      128
#define NNZ_PER_ROW  (K * 2 / 4)   // 2:4 → 128 个非零元/行

// === 稀疏存储 ===
float   A_vals[M][NNZ_PER_ROW] __attribute__((aligned(64)));
uint8_t A_cols[M][NNZ_PER_ROW] __attribute__((aligned(64)));
float   B[K][N] __attribute__((aligned(64)));
float   C[M][N] __attribute__((aligned(64)));

void _start() {
    // ===== 生成测试数据 =====
    for (int i = 0; i < M; i++) {
        for (int b = 0; b < K/4; b++) {
            // block 0: 保留 col 0,2
            // block 1: 保留 col 1,3
            // ... 交替，确保 50% 稀疏
            int nz0 = (b % 2 == 0) ? 0 : 1;
            int nz1 = (b % 2 == 0) ? 2 : 3;
            
            A_vals[i][b*2 + 0] = (float)((i * K + b*4 + nz0) % 100) / 100.0f;
            A_cols[i][b*2 + 0] = nz0;
            A_vals[i][b*2 + 1] = (float)((i * K + b*4 + nz1) % 100) / 100.0f;
            A_cols[i][b*2 + 1] = nz1;
        }
    }

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = (float)((i * N + j) % 100) / 100.0f;

    memset(C, 0, sizeof(C));

    // ===== Alg 3-S: 结构化稀疏 GEMM =====
    for (int i = 0; i < M; i++) {
        // 清零 C[i]
        size_t vl;
        for (int n = 0; n < N; n += vl) {
            vl = __riscv_vsetvl_e32m2(N - n);
            vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            __riscv_vse32(&C[i][n], zero, vl);
        }

        // 遍历非零元
        for (int j = 0; j < NNZ_PER_ROW; j++) {
            float a_val = A_vals[i][j];
            int block_id = j / 2;               // 每个 2:4 block 有 2 个非零元
            int col_in_block = A_cols[i][j];     // 块内偏移 0~3
            int b_row = block_id * 4 + col_in_block;  // 实际 B 行号

            for (int n = 0; n < N; n += vl) {
                vl = __riscv_vsetvl_e32m2(N - n);
                vfloat32m2_t vb = __riscv_vle32_v_f32m2(&B[b_row][n], vl);
                vfloat32m2_t vc = __riscv_vle32_v_f32m2(&C[i][n], vl);
                vc = __riscv_vfmacc_vf_f32m2(vc, a_val, vb, vl);
                __riscv_vse32(&C[i][n], vc, vl);
            }
        }
    }

    asm volatile("li a7, 93; li a0, 0; ecall");
}
```

### 1.2.3 编译脚本

**`Makefile`**：

```makefile
RISCV_GCC = riscv64-unknown-elf-gcc
CFLAGS    = -march=rv64gcv -O3 -static -nostdlib
INCLUDE   = -I$(RISCV_HOME)/riscv64-unknown-elf/include

.PHONY: all clean

all:  dense_gemm  alg3s_gemm

dense_gemm: dense_gemm.c
	$(RISCV_GCC) $(CFLAGS) $(INCLUDE) $< -o $@

alg3s_gemm: alg3s_gemm.c
	$(RISCV_GCC) $(CFLAGS) $(INCLUDE) $< -o $@

# 大矩阵版本（做成参数化）
alg3s_gemm_1024: alg3s_gemm.c
	$(RISCV_GCC) $(CFLAGS) $(INCLUDE) -DM=1024 -DK=1024 -DN=1024 $< -o $@

clean:
	rm -f dense_gemm alg3s_gemm alg3s_gemm_1024
```

```bash
make all
```

## 1.3 创建 gem5 运行脚本（正式版，带统计）

**`run_gemm.py`**：

```python
import m5
from m5.objects import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('binary', help='Path to the RISC-V binary')
parser.add_argument('--cpu', default='minor', choices=['minor', 'o3'])
args = parser.parse_args()

system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '2GHz'
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('512MB')]

# === CPU 配置 ===
if args.cpu == 'o3':
    system.cpu = O3CPU()
    # 对标论文的配置（近似）
    system.cpu.fetchWidth = 4
    system.cpu.decodeWidth = 4
    system.cpu.renameWidth = 4
    system.cpu.issueWidth = 4
    system.cpu.wbWidth = 4
    system.cpu.commitWidth = 4
else:
    system.cpu = MinorCPU()

system.cpu.isa = RiscvISA()

# === Cache 层次 ===
system.cpu.icache = Cache(
    size='32kB', assoc=4,
    tag_latency=1, data_latency=1, response_latency=1,
    mshrs=4, tgts_per_mshr=8
)
system.cpu.dcache = Cache(
    size='32kB', assoc=4,
    tag_latency=1, data_latency=1, response_latency=1,
    mshrs=4, tgts_per_mshr=8
)

system.l2cache = Cache(
    size='256kB', assoc=8,
    tag_latency=10, data_latency=10, response_latency=10,
    mshrs=20, tgts_per_mshr=12
)

system.membus = SystemXBar()
system.l2bus = L2XBar()

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.l2cache.mem_side = system.membus.cpu_side_ports

system.cpu.createInterruptController()

# === 内存 ===
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# === Workload ===
system.workload = SEWorkload.init_compatible(args.binary)
process = Process()
process.cmd = [args.binary]
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()

print("=" * 60)
print(f"Simulation: {args.binary}, CPU: {args.cpu}")
print("=" * 60)

exit_event = m5.simulate()

print(f"\nExit cause: {exit_event.getCause()}")
print(f"Simulated ticks: {m5.curTick()}")
print(f"Simulated time: {m5.curTick() / 2e9 * 1000:.3f} ms (@ 2GHz)")

# Dump stats
m5.stats.dump()
```

### 1.3.1 运行 + 收集数据

```bash
GEM5=~/riscv-sparse/gem5/build/RISCV/gem5.opt

# 跑 dense baseline
$GEM5 run_gemm.py dense_gemm --cpu=minor 2>&1 | tee results/dense_minor.log

# 跑 Alg 3-S
$GEM5 run_gemm.py alg3s_gemm --cpu=minor 2>&1 | tee results/alg3s_minor.log
```

### 1.3.2 提取关键指标脚本

**`extract_stats.py`**：

```python
import sys, re

def extract(logfile):
    with open(logfile) as f:
        text = f.read()
    
    ticks = re.search(r'Simulated ticks:\s+(\d+)', text)
    time_ms = re.search(r'Simulated time:\s+([\d.]+)\s+ms', text)
    ipc = re.search(r'system\.cpu\.ipc\s+([\d.]+)', text)
    cpi = re.search(r'system\.cpu\.cpi\s+([\d.]+)', text)

    print(f"File: {logfile}")
    print(f"  Ticks:       {ticks.group(1) if ticks else 'N/A'}")
    print(f"  Time (ms):   {time_ms.group(1) if time_ms else 'N/A'}")
    print(f"  IPC:         {ipc.group(1) if ipc else 'N/A'}")
    print(f"  CPI:         {cpi.group(1) if cpi else 'N/A'}")
    print()

if __name__ == '__main__':
    for f in sys.argv[1:]:
        extract(f)
```

```bash
python3 extract_stats.py results/*.log
```

### ✅ 阶段 1 完成标准

- Dense GEMM 跑通，拿到延迟数据
- Alg 3-S 跑通，延迟 < Dense GEMM（因为标准 dense GEMM 做了 K 次 vload B 行，Alg 3-S 只做了 K/2 次）
- 加速比约 ~1.5-1.8×（接近论文趋势但不要求精确一致——论文有解耦向量处理器，gem5 标准版没有）

---

# 阶段 2：INT8 混合精度 co-design

## 2.1 INT8 sparse GEMM kernel

### 2.1.1 INT8 Alg 3-S

**`alg3s_gemm_int8.c`**：

```c
#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>

#define M      128
#define K      256
#define N      128
#define NNZ_PER_ROW  (K * 2 / 4)

// === INT8 稀疏存储（A 权重）===
int8_t   A_vals[M][NNZ_PER_ROW] __attribute__((aligned(64)));
uint8_t  A_cols[M][NNZ_PER_ROW] __attribute__((aligned(64)));

// === INT8 B 矩阵 ===
int8_t B[K][N] __attribute__((aligned(64)));

// === FP32 输出（dequant 后）===
float C[M][N] __attribute__((aligned(64)));

// Dequant scale（per-row, FP32）
float scale_A[M];

void _start() {
    // ===== 初始化 =====
    for (int i = 0; i < M; i++) {
        scale_A[i] = 0.01f;  // 假设
        for (int b = 0; b < K/4; b++) {
            int nz0 = (b % 2 == 0) ? 0 : 1;
            int nz1 = (b % 2 == 0) ? 2 : 3;
            A_vals[i][b*2+0] = (int8_t)((b*4 + nz0) % 127);
            A_cols[i][b*2+0] = nz0;
            A_vals[i][b*2+1] = (int8_t)((b*4 + nz1) % 127);
            A_cols[i][b*2+1] = nz1;
        }
    }

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = (int8_t)(((i * N + j) % 200) - 100);  // [-100, 99]

    memset(C, 0, sizeof(C));

    // ===== INT8 稀疏 GEMM =====
    for (int i = 0; i < M; i++) {
        // INT32 累加器（中间精度）
        int32_t acc[N];
        memset(acc, 0, sizeof(acc));

        // 遍历非零元：INT8 × INT8 → INT32 累加
        for (int j = 0; j < NNZ_PER_ROW; j++) {
            int8_t a_val = A_vals[i][j];
            int block_id = j / 2;
            int b_row = block_id * 4 + A_cols[i][j];

            size_t vl;
            for (int n = 0; n < N; n += vl) {
                vl = __riscv_vsetvl_e8m1(N - n);  // SEW=8
                vint8m1_t vb = __riscv_vle8_v_i8m1(&B[b_row][n], vl);

                // 用 vwmacc: INT8×INT8 → INT16, 累加到 INT32
                // RISC-V V 的 vwmacc: vd[i] = vd[i] + vs1[i] * vs2[i],
                // 其中 vd 是 2*SEW 宽度
                vint16m2_t vb_w = __riscv_vwadd_vx_i16m2(vb, 0, vl);  // widen B
                // 构造 A 的广播向量
                vint16m2_t va_w = __riscv_vmv_v_x_i16m2((int16_t)a_val, vl);

                // vwmacc: acc_int32 += a_val * b_val
                vint32m4_t vacc = __riscv_vle32_v_i32m4(&acc[n], vl);
                vacc = __riscv_vwmacc_vv_i32m4(vacc, va_w, vb_w, vl);
                __riscv_vse32(&acc[n], vacc, vl);
            }
        }

        // Dequant: C[i][n] = acc[n] * scale_A[i]
        float s = scale_A[i];
        size_t vl;
        for (int n = 0; n < N; n += vl) {
            vl = __riscv_vsetvl_e32m2(N - n);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&acc[n], vl);
            vfloat32m2_t vf = __riscv_vfcvt_f_x_v_f32m2(vacc, vl);
            vf = __riscv_vfmul_vf_f32m2(vf, s, vl);
            __riscv_vse32(&C[i][n], vf, vl);
        }
    }

    asm volatile("li a7, 93; li a0, 0; ecall");
}
```

> ⚠️ 上面的 vwmacc 路径可能需要根据工具链实际 intrinsic 名称调整。RISC-V V intrinsics API 在 GCC 14 和 LLVM 17 之间有差异。如果编译不过，可以先退回到标量 INT32 累加（用 `for` 循环），确保正确性，再逐步向量化。

## 2.2 实验矩阵

### 2.2.1 正确性验证

```c
// 在 kernel 末尾加验证代码
// 对比 C 和 reference（CPU 上算的黄金值）
// 允许 1e-3 相对误差（INT8 量化有精度损失）
```

### 2.2.2 四个对比点

```
配置                    延迟 (ms)  带宽 (MB)  注释
────────────────────────────────────────────────────
FP32 dense GEMM           T1         B1       稠密 baseline
FP32 2:4 sparse GEMM      T2         B2       Alg 3-S
INT8 dense GEMM           T3         B3       纯量化（无稀疏）
INT8 2:4 sparse GEMM      T4         B4       你的 co-design ← 预期最小
```

### 2.2.3 跑实验脚本

```bash
# run_all.sh
GEM5=~/riscv-sparse/gem5/build/RISCV/gem5.opt

mkdir -p results

echo "=== FP32 Dense ==="
$GEM5 run_gemm.py dense_gemm         2>&1 | tee results/dense.log

echo "=== FP32 Sparse ==="
$GEM5 run_gemm.py alg3s_gemm         2>&1 | tee results/alg3s.log

echo "=== INT8 Dense ==="
$GEM5 run_gemm.py dense_gemm_int8    2>&1 | tee results/dense_int8.log

echo "=== INT8 Sparse (co-design) ==="
$GEM5 run_gemm.py alg3s_gemm_int8    2>&1 | tee results/alg3s_int8.log

echo "=== Done ==="
python3 extract_stats.py results/*.log
```

### ✅ 阶段 2 完成标准

- 四个配置全部跑通
- INT8 sparse 延迟 < FP32 sparse 延迟（带宽减半 → 理论上延迟降低 30-50%）
- INT8 sparse 延迟 < INT8 dense 延迟（稀疏跳过零 → 减少计算量）
- 有完整的数据表格

---

# 阶段 3：自定义 vindexmac + 论文实验

## 3.1 修改 gem5 添加 INT8 vindexmac 指令

### 3.1.1 修改译码器

```bash
# 找到 RISC-V 译码文件
cd ~/riscv-sparse/gem5
# 文件路径: src/arch/riscv/decoder.isa
```

添加 `vindexmac.vx` 指令（FP32 或 INT8 版本）：

```
# 在 decoder.isa 的 vector 指令区添加
0x96: vindexmac_vx({{
    // 语义: vd[i] += vs2[0] * VRF[rs1[4:0]][i]
    // 需要修改 gem5 的 VRF 访问逻辑
}});
```

> 这部分需要深入 gem5 源码。论文的 vindexmac 是针对 FP32 的，你需要扩展出 INT8/INT16 变体。工作量不小（估计 3-5 天）。

### 3.1.2 备选方案：用 intrinsics 模拟

如果不改 gem5 源码，可以在 C 层面用 vrgather + vfmacc 模拟，虽然性能不如硬件指令，但可以证明"如果有了 INT8 vindexmac，性能会怎样"：

```
预估加速比 = 当前 vload B 行次数 / 使用 vindexmac 后的 VRF 访问次数
           = (每个非零元都 vload B 行) / (预装 B tile 后只在 VRF 内访问)
           ≈ 2-4× (取决于 tile 命中率)
```

## 3.2 CNN/LLM 层级实验

### 3.2.1 ResNet-18 结构

```
Layer     Input        Weight       Sparsity Pattern
─────────────────────────────────────────────────────
conv1     3×224×224    64×3×7×7     2:4
conv2_1   64×56×56     64×64×3×3    2:4
conv2_2   64×56×56     64×64×3×3    2:4
... (共 ~18 层卷积) ...
```

### 3.2.2 LLM 层级实验（LLaMA-3.2-1B）

```
Layer     Pattern   Reason
──────────────────────────────────────────
Q/K/V     4:4      注意力对稀疏极敏感
O proj    2:4      可适度裁剪
FFN up    1:4      最冗余
FFN gate  1:4      最冗余
FFN down  2:4      略保守
```

---

# 附录 A：文件组织结构

```
~/riscv-sparse/
├── gem5/                    # gem5 源码
├── riscv/                   # RISC-V 工具链
├── kernels/                 # 所有 kernel 源码
│   ├── smoke_test.c
│   ├── dense_gemm.c
│   ├── alg3s_gemm.c
│   ├── dense_gemm_int8.c
│   ├── alg3s_gemm_int8.c
│   ├── alg5_vindexmac.c     # (阶段 3)
│   └── Makefile
├── scripts/                 # 运行脚本
│   ├── run_smoke.py
│   ├── run_gemm.py
│   ├── run_all.sh
│   └── extract_stats.py
├── results/                 # 实验数据
│   ├── dense.log
│   ├── alg3s.log
│   └── ...
└── notes/                   # 笔记
    └── gem5_config_notes.md
```

# 附录 B：常用命令速查

```bash
# === gem5 ===
# 编译
scons build/RISCV/gem5.opt -j$(nproc)

# 运行 SE mode
./build/RISCV/gem5.opt <script.py>

# 查看 gem5 stats
less m5out/stats.txt

# === RISC-V GCC ===
# 编译 bare-metal
riscv64-unknown-elf-gcc -march=rv64gcv -O3 -static -nostdlib file.c -o file

# 生成汇编（调试用）
riscv64-unknown-elf-gcc -march=rv64gcv -O3 -S file.c -o file.s

# objdump 看反汇编
riscv64-unknown-elf-objdump -d file | less

# === 数据分析 ===
# 提取 ticks
grep 'Simulated ticks' results/*.log
grep 'system.cpu.ipc' m5out/stats.txt
grep 'system.cpu.cpi' m5out/stats.txt
```

# 附录 C：预计时间线

```
Week 1: 阶段 0-1
  Day 1-2: WSL2 + 依赖 + gem5 编译 + 工具链
  Day 3-4: dense_gemm + alg3s_gemm 实现 + 调试
  Day 5:   收集 baseline 数据

Week 2: 阶段 2
  Day 1-2: INT8 dense_gemm 实现
  Day 3-4: INT8 alg3s_gemm 实现（co-design）
  Day 5:   精度验证 + 收集对比数据

Week 3-4: 阶段 3（论文撰写同步）
  Day 1-3:   gem5 源码修改（vindexmac INT8）
  Day 4-5:   INT8 vindexmac 实验
  Day 6-7:   动态稀疏选择（per-layer N:M）实验
  Day 8-10:  整篇 CNN/LLM evaluation
  Day 11-14: 论文撰写 + 打磨
```
