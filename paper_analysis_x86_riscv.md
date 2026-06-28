# 📄 Practical and Efficient x86-64 Emulation on RISC-V

---

## 第一层：元信息 · 定位

| 维度 | 内容 |
|------|------|
| 标题 | Practical and Efficient x86-64 Emulation on RISC-V |
| 作者 | **共同一作**：Xiongchuan Tan（清华 RIOS Lab）、Yang Liu（中科院软件所）；Sebastien Chevalier（独立）、Yangyu Chen（重庆大学）、Xiaoyi Liu（清华）、Haohuan Fu（清华 RIOS Lab） |
| 发表状态 | **EUROSYS '26**（EuroSys 2026, Edinburgh, UK）— CCF-B / 系统领域顶会 |
| 年份 | 2026 |
| 代码 | 已上游化到 Box64 项目：`github.com/ptitSeb/box64` |
| Artifact | Zenodo DOI: `10.5281/zenodo.18611830` |

---

## 第二层：动机 · 问题定义

**要解决什么问题？**
> 在 RISC-V 平台上高效运行未经修改的 x86-64 应用程序（办公软件、桌面游戏等）。

**为什么重要？**
- RISC-V 生态最大的瓶颈不是硬件，而是「软件鸿沟」——大量 x86 应用没有 RISC-V 原生版本
- 二进制翻译是填补这个鸿沟最现实的路径
- 但现有方案（QEMU）性能损失太大，无法满足日常使用需求

**现有方法差在哪？**
- QEMU-TCG 性能低下（通用 IR 中间表示带来大量开销）
- 针对 Arm 的翻译器（Rosetta 2、FEX）无法直接移植到 RISC-V
- RISC-V 缺乏 Apple Silicon 那种硬件 TSO 模式，也没有 Arm RCpc 指令辅助

**核心难点：**
1. **ISA 鸿沟极大**：x86 和 RISC-V 的架构差异比 x86-Arm 更大（80-bit x87 FP、FLAGS 寄存器、非对齐原子操作、SIMD 碎片化）
2. **内存一致性模型差异**：x86 的 TSO vs RISC-V 的 WMM，不加 fence 多线程会出 data race
3. **RISC-V 硬件碎片化**：不同核心有不同扩展，翻译器必须同时保证兼容性和性能

---

## 第三层：贡献 · 创新点

论文声称的 contribution（原文引用+翻译）：

1. ✅ **首个实用 x86→RISC-V DBT 系统**：「extend Box64 to build the first DBT system for running real-world, complex, and unmodified x86-64 applications on RISC-V」
2. ✅ **高性能 JIT 引擎**：平均性能达原生 67%，QEMU 的 3.3 倍
3. ✅ **统一翻译框架**：覆盖 RISC-V 碎片化扩展，确保广泛兼容性
4. ✅ **实用 TSO 模拟方法**：基于真实应用行为的 selective memory barrier，比 state-of-the-art 低得多

**核心 insight：**
> 「不用追求理论上的完备 TSO 正确性——绝大多数真实多线程应用不需要全部的内存屏障。利用编译器元数据（MSVC volatile）+ 启发式按需插 fence，就能在实用层面做到正确且高性能。」

**本质区别 vs 前人：**
- vs QEMU：直接用 guest-host 一对一翻译，不用通用 IR，轻量得多
- vs Rosetta 2：没有硬件 TSO 辅助，纯软件——但用 pragmatism 弥补
- vs Risotto/CrossMapping：它们追求指令级 fence 最优，本文另辟蹊径——从应用层 + 编译器标注入手，且更专注于实际游戏/桌面软件验证

---

## 第四层：方法 · 技术细节

### 整体架构

```
x86 程序 → 内置 ELF Loader/Linker → JIT 引擎 → 翻译后的 RISC-V 代码
                ↕（跨架构调用桥接）
         RISC-V 原生动态库（glibc, OpenGL, Vulkan...）
```

- **四遍轻量 JIT 编译器**：不用 IR，直接逐条解码 x86 指令 → 生成 RISC-V 代码
- **静态寄存器映射**：利用 RISC-V 的 32 个寄存器静态映射 x86 的 16 个 GPR + FLAGS + RIP
- **高级模拟策略**：优先做库级互操作（wrapper native libs），而非系统调用级翻译

### 四大关键技术

**① 间接分支优化（CALL/RET 配对）**
- 核心 trick：把 guest 的 CALL/RET 翻译成 host 的 JAL/JALR
- 维护软件 shadow stack 存储 host 返回地址
- 匹配时用硬件 RAS 预测 → 极快；不匹配时 fallback 到慢路径
- 灵感来源：MAMBO-X64 [d'Antras et al. 2016]

**② 惰性 FLAGS 评估 + Macro-Op Fusion**
- 关键观察：大多数 FLAGS 更新是「死的」——后续指令不消费
- 后向数据流分析，仅在 FLAGS 被实际消费时才计算
- Macro-op fusion：检测 CMP + Jcc 这种 x86 pair，直接融合成一句 RISC-V 条件分支（如 `BGEU`），完全绕过 FLAGS 存储
- 将 10+ 条指令压缩到 1-2 条

**③ 统一 RISC-V 扩展适配**
- 基线：RV64G（保证任何 RISC-V 处理器都能跑）
- 动态检测可用扩展（Zba, Zbb, V 等），按需启用加速路径
- FP/SIMD 寄存器同步：x86 的 XMM 寄存器在 RISC-V 上需要映射到三处——内存（模拟状态）、F 寄存器（标量）、V 寄存器（向量），边界处插入 transfer 指令

**④ 实用 TSO 模拟（核心创新）**
- **5 级策略**：Level 0（无 fence / 默认）→ Level 4（Verbatim = QEMU 式）
- **Level 1 Pragmatic**：仅在包含 store 的 basic block 中插入 3 种 fence（entry: `fence r,rw` / pre-commit: `fence w,w` / post-commit: `fence w,w`）
- **编译器标注利用**：MSVC 的 `/volatileMetadata` 提供精确的 volatile 内存访问位置 → 仅在标注处插 fence
- **内置兼容性配置**：201 条已录入，自动检测 UnityPlayer.dll/libjvm.so 等已知多线程库并升级 TSO 级别

---

## 第五层：实验 · 验证

### 实验平台

| 设备 | ISA | 核心数 | 用途 |
|------|-----|--------|------|
| SOPHGO SG2044 | rv64gcvb_zvl128 | 64 核 | 主要评估 |
| Xiangshan Nanhu | rv64gcb | 1 核 | 对比验证 |

### 单线程性能（NBench）

| 指标 | SG2044 | Nanhu |
|------|--------|-------|
| 整体 vs 原生 | **67%**（几何平均） | — |
| vs QEMU-TCG | **3.3×** | — |
| 内存类 workload | 接近原生，QEMU 的 3× | — |
| 整数类 workload | QEMU 的 2× | — |
| FP（有 V 扩展） | 与整数类相当 | — |
| FP（无 V 扩展） | — | ⚠️ 比原生慢 **16×**（SSE → 标量 fallback） |

### 多线程性能 & TSO 开销（PARSEC 3.0, 8 线程）

| TSO 级别 | 相对性能 |
|----------|---------|
| Level 0 (None) | 最快（基线） |
| Level 1 (Pragmatic) | Level 4 的 **1.5×** |
| Level 4 (Verbatim) | 最慢 |

> 关键数据：Level 1 比 Verbatim 快 50%，远超 CrossMapping 文献报告的 7.3% 提升。

### 应用程序兼容性（桌面游戏，SG2044, 1080p）

| 游戏 | 平台 | 平均 FPS | 备注 |
|------|------|----------|------|
| The Witcher 3 | Win 64-bit | 18 | AAA 3D，可玩 |
| FlatOut 2 | Win 32-bit | **60** | 老 3D，满帧 |
| Besiege | Linux 64-bit | 14 | CPU-bounded |
| Hades | Win 64-bit | **52** | 2D 但 CPU 重 |
| Celeste | Linux 64-bit | **59** | .NET JIT → 验证了 SMC 处理 |
| Embodiment of Scarlet Devil | Win 32-bit | **59** | 经典 2D |

### 代码生成开销

- NBench：JIT 引擎占用 **<0.01%** 执行时间
- SciMark + LuaJIT：JIT 引擎占用约 **1%**

### 动态库加速

- bzip2 用原生 libbz2 → **1.38×** 提升
- ⚠️ NBench 用原生 libm → 反而只有模拟版的 **0.8×**（RISC-V 数学库实现质量 + wrapping 开销）

---

## 第六层：批判性分析

### 作者承认的局限性
- TSO Level 1 仍比 Level 0 有明显性能损失 → 需要更精细的 per-region 控制
- TSO 不提供理论正确性保证 → 需要手动 profiling + 大量测试验证
- 对新应用「开箱即用」的兼容性仍然是未知数

### 我认为的问题

**⚠️ 中等关注：**
- **TSO 正确性是「经验验证」而非「证明」的**。论文在游戏上跑了一小时没崩 → 但这不代表所有 corner case 都覆盖了。对于生产环境（服务器、数据库等对正确性要求高的），这种 best-effort 方法风险不可忽视
- **单线程评估用 NBench，过于微观**。NBench 是 kernel-level benchmark，不代表完整应用行为。好在游戏兼容性部分部分弥补了这个不足
- **FPU 精度是有意牺牲的**。80-bit x87 直接用 64-bit double 模拟，论文说「对绝大多数应用无影响」，但没有量化「绝大多数」是多少。对于数值敏感的 HPC/科学计算场景，这可能不成立

**✅ 做得好的：**
- **工程深度令人印象深刻**：从 FLAGS 的 OF bit 重定位到 FILD-FISTP idiom 检测，到非对齐 MMIO 的两阶段 signal-and-recompile——每个细节都解决得很实在
- **思路务实**：不追求理论完美，追求「能用、够快」，这在系统领域是正确的工程哲学
- **游戏作为 benchmark 选得好**：闭源、多线程、GPU 密集、使用各种反调试/DRM trick → 是二进制翻译器的终极压力测试
- **与 QEMU 的对比使用同一 JIT 引擎 + 不同 fence 策略**（Verbatim 模式），排除了不同系统的混淆变量

**🔴 OpenArena 数据**：x86 模拟版跑出了原生 RISC-V 版的 67%，和 NBench 的整体数据一致，这是一个很好的交叉验证

---

## 第七层：启示 · 连接

### 对你的启发
- 如果你在做**文本相似度模型**，这篇论文和你领域不同，但方法论有共通之处：
  - 「经验正确 > 理论完美」的工程哲学
  - 多层次优化策略（惰性评估 + macro-op fusion 类比于模型中的算子融合）
  - 用真实应用（而非合成 benchmark）做最终验证

### Follow-up 方向
1. **更精细的 per-region TSO 控制**：作者承认全局 Level 1 仍然开销大，这显然是下一步
2. **RVV 1.0 全面适配**：当前 V 扩展支持有限（向量化 x86 SIMD 还不够成熟）
3. **与其他 RISC-V 芯片（如 SiFive P870, Ventana Veyron）** 的性能对比

### 值得顺藤摸瓜的引用
1. **[CrossMapping 2024]** Gao et al., USENIX ATC — 指令级最优 fence 选择，用 DFA 推导
2. **[Risotto 2022]** Gouicem et al., ASPLOS — QEMU 的 fence 问题分析与改进
3. **[MAMBO-X64 2016]** d'Antras et al. — CALL/RET 配对的始祖方案
4. **[AtoMig 2023]** Beck et al. — 编译器层面从 TSO 迁移到 WMM
5. **[Tiaozhuan 2024]** Li et al. — 间接分支优化的最新进展

### 一句话 takeaway
> **「理论上的完美正确性」不如「对 99% 真实应用够用且快 3 倍」——这是系统研究的工程智慧，而这篇论文用 207 条内置配置 + 8 款桌面游戏的实际跑通证明了这一点。**
