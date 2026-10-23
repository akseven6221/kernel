# 实验操作指南

[返回路线](README.md)

## 阅读一个版本的顺序

打开本课 README，把“本课要解决什么”和“正确性约束”各用一句话复述出来。然后打开 `kernel.cu`，搜索 `[教程]`，只读标记处及其上下文。最后打开 `from_vN.diff`，检查修改是否确实围绕本课主题。

两种编号要分清：`tutorial/v08` 是第 08 课，对应原实验 `v12`；`--versions v12` 使用原实验目录。基准脚本不支持直接把 `tutorial/v08` 作为版本名，包含斜杠会破坏它的临时动态库路径。

## 复测未修改的原实验

每课提供实际父版和本版一起运行的命令。例如第 16 课原 v27：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v27 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v27 --mode profile
```

benchmark 使用固定的 8 个 shapes，并先执行每版 40 项 FP32 参考检查。双方使用同一个 GPU、输入和 stream，20-node CUDA Graph，9 轮交替计时；编译、context 创建、分配和 graph capture 不计入 kernel latency。

默认 profile 为 B1/S4096/H8/D128 non-causal，带官方 NCU 对照。短序列和 causal 可指定：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode profile --seq 512
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode profile --seq 8192 --causal
```

`--seq` 用于 profile/probe 目标，不会把 benchmark 的固定形状集改成单一 shape。扩展形状用 `--mode sweep`，它测试 S512/1024/2048/4096/8192、H2/8/32、两种 causal 模式共 30 个组合。

## 修改教学副本后测试

为实验创建一个新目录，避免改动已经与历史数据绑定的原版本。以下以第 03 课为例；目录已存在会停止，不会覆盖旧实验：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
root = Path('kernel/fa4')
dest = root / 'study_v03'
dest.mkdir()
shutil.copyfile(root / 'tutorial/v03/kernel.cu', dest / 'kernel.cu')
PY
```

在 `kernel/fa4/study_v03/kernel.cu` 修改一个点，先写下预期影响的指标和可能破坏的边界，再运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v2,study_v03 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions study_v03 --mode profile
.venv/bin/modal run kernel/fa4/run_modal.py --versions study_v03 --mode sanitize --sanitize-seq 511
```

如果改变 slot 复用、TMEM 别名或 dispatch，补测长尾块和独立参考：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions study_v03 --mode sanitize --sanitize-seq 8191
.venv/bin/modal run kernel/fa4/run_modal.py --versions study_v03 --mode stress
```

这些命令会实际创建 Modal B200 任务。运行结果自动保存到新的 `kernel/fa4/runs/<时间戳>_.../`；每次先确认退出码、源码 SHA 和 correctness，再读 timing。教程本次整理仅使用原先已保存的结果。

## 本地编译教学源码

已有支持本代码的 CUDA 工具链时，可将单个教学版本编译为动态库；目标为 B200 的 `sm_100a`：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a --shared -Xcompiler=-fPIC \
  kernel/fa4/tutorial/v23/kernel.cu -lcuda -o /tmp/fa4_tutorial_v23.so
```

导出的 C ABI 是 `fa4_create`、`fa4_launch`、`fa4_destroy`，参数使用方式见 [benchmark.py 的 Kernel 类](../benchmark.py)。只编译不等于验证输出；实际运行需要匹配 GPU、dtype、形状和输入布局。

## 怎样把 profile 变成下一步实验

| 观察 | 可以提出的假设 | 应检查的反证 |
| --- | --- | --- |
| global-store excessive sectors 很多 | 输出线程映射不合并 | epilogue 占比是否足够大，转置成本是否抵消收益 |
| local loads/stores 与 spill 多 | 活跃状态或预算不合适 | 零 spill 是否仍慢；增加预算是否使启动资源非法 |
| barrier polling 多、TC active 低 | producer-consumer 排程有空隙 | 等待是否保护必要依赖，删除后能否严格证明正确 |
| SM active cycles 分布很不均 | causal 任务大小导致尾部拖延 | remap 是否完整覆盖 batch/head/query，是否仅某个 shape 有效 |
| shared excessive wavefronts 多 | staging pitch 导致 bank conflict | 元素宽度和访问模式是否算对，端到端变化是否超过噪声 |
| 小 grid 与大 grid 排名反转 | 需要不同 tile/角色结构 | sweep 是否覆盖不同 H/B，是否只对固定八个 case 拟合 |

每次保存一页记录：父版本、修改点、为什么这么改、需保护的依赖、运行路径、正确性结果、配对样本、profile 变化、采用或回退理由。改变多个变量时直接标作结构/组合实验，再用消融分析因果。

## 教程的数据与重建

每课 `evidence.json` 保存父子版本的完整八个 case，包含九轮样本、误差、时钟记录和源 run 的结果文件哈希。NCU 通过链接复用原文件，不复制数百 MB 报告。

```bash
.venv/bin/python kernel/fa4/tutorial/build.py --check
```

该命令验证：原版 SHA 符合归档；教学代码去掉注释后完全一致；配对数据属于同一 run；每条数据有九轮样本且误差通过；生成文件和本地链接一致。它不执行 CUDA 测试。

若维护课程，在 [lessons.py](lessons.py) 修改人工讲解，然后运行 `build.py` 重建各课、diff、证据表与 manifest。重建会覆盖生成的教学副本，因此自己的代码实验应放在 `study_*` 目录。
