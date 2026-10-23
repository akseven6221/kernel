# 第二轮：从 v27 继续逼近官方 FA4

推荐入口是 [v51/kernel.cu](v51/kernel.cu)。本轮新增 v27–v51 共 25 个版本，23 个版本运行通过；v34 启动失败、v35 超时，均保留原始记录。
仍为 B200、FP16 dense self-attention forward、D=128、contiguous BSHD。官方固定为 flash-attn-4 4.0.0b35；双方均完整计算 attention，CUDA Graph 20 nodes、9 轮交替计时。

## 同一 B200 的最终结果

| B,S,H,D | causal | 上轮 v24 us | v51 us | 官方 us | v24→v51 | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1,512,8,128 | False | 16.946 | 12.038 | 10.731 | 1.408× | 89.14 |
| 1,1024,8,128 | False | 27.915 | 19.413 | 16.304 | 1.438× | 83.99 |
| 1,4096,8,128 | False | 94.867 | 73.442 | 53.210 | 1.292× | 72.45 |
| 1,8192,8,128 | False | 362.376 | 276.085 | 203.664 | 1.313× | 73.77 |
| 1,4096,8,128 | True | 95.037 | 66.978 | 50.741 | 1.419× | 75.76 |
| 1,8192,8,128 | True | 184.813 | 142.614 | 105.736 | 1.296× | 74.14 |
| 2,4096,16,128 | False | 382.861 | 298.672 | 217.224 | 1.282× | 72.73 |
| 2,4096,16,128 | True | 197.408 | 171.994 | 121.982 | 1.148× | 70.92 |

以上 v24 与 v51 来自本轮同机交替计时，不能用不同 run 的绝对时间直接相减。未达到官方速度；小任务更接近，大 grid 仍有明显差距。

## 为什么有收益

1. v27 把 QK0(j)、PV1(j−1)、QK1(j)、PV0(j) 交错，减少等待另一个 stage 的空隙。NCU TC active 从 v24 约 47% 提升到约 57%。
2. v31 利用同 warp MMA 的 A-read→D-write 顺序，去掉 P/S alias 的额外 completion roundtrip；SFULL 仍覆盖此前全部 MMA，保护 O correction。六项 sanitizer 与 96 项额外输入检查已保存。
3. v42 的 packed FMA 与四条 packed sum chains 在 non-causal 小幅有效。v47 max3 单项消融基本持平，收益不能仅归于 max3。v44 保留 causal 标量路径。
4. v43 的单 Q-stage 适合 CTA 少的情况；额外 30-shape sweep 揭示明显的 wave 边界。v50 改进 epilogue shared pitch，NCU excessive shared wavefronts 从 v43 的 131072 降至 0；延迟收益较小。v51 根据实际 SM 数与 CTA 数选择 v44/v50。

## 每个版本的取舍

| Version | Parent | 修改 | 取舍 |
| --- | --- | --- | --- |
| [v27](v27/README.md) | v24 | 交错 QK/PV 调度 | 保留，长序列首个主要收益 |
| [v28](v28/README.md) | v24 | 216/64 register 预算 | 回退 |
| [v29](v29/README.md) | v27 | 分离 packed scale/EX2/sum/convert 阶段 | 回退 |
| [v30](v30/README.md) | v27 | 循环边界数组改标量 | 回退，compiler spill 反而增加 |
| [v31](v31/README.md) | v27 | 同 warp MMA A-read→D-write 顺序，移除 PVDONE roundtrip | 保留 |
| [v32](v32/README.md) | v27 | P 分四段写回 TMEM | 回退 |
| [v33](v33/README.md) | v31 | 12.5% polynomial exponent | 回退 |
| [v34](v34/README.md) | v31 | 320 threads / uniform 200 regs | 启动失败：701 |
| [v35](v35/README.md) | v31 | 320 threads / 保留动态预算 | 首个输入超时：exit -9 |
| [v36](v36/README.md) | v31 | 两个自主 QK/softmax/PV 工作组 | 回退，零 spill 也没有加速 |
| [v37](v37/README.md) | v31 | 单次 4D TMA，保持 blocked D 布局 | 性能接近；保留布局实验 |
| [v38](v38/README.md) | v31 | 统一先减后乘，移除指数参数分支 | 回退 |
| [v39](v39/README.md) | v31 | M128/N64，单 Q，256 TMEM columns | 回退，资源上限改善但实际 TC 活跃度下降 |
| [v40](v40/README.md) | v31 | 两个软件 WG 交替使用 SFU | 回退 |
| [v41](v41/README.md) | v31 | PFULL 之后再求 rowsum | 回退 |
| [v42](v42/README.md) | v31 | max3 + packed FMA + 四条 packed sum chains | 保留 non-causal 路径 |
| [v43](v43/README.md) | v39 | M128/N128 单 Q / 单 KV pair | 保留小 grid 路径 |
| [v44](v44/README.md) | v42 | 按 causal 专门化 softmax | 作为最终大 grid 实现 |
| [v45](v45/README.md) | v44 | 232/40 register 预算 | 无全面稳定收益 |
| [v46](v46/README.md) | v44 | 240/24 register 预算 | 回退 |
| [v47](v47/README.md) | v31 | max3 单项消融 | 基本持平，v42 的收益不能单独归给 max3 |
| [v48](v48/README.md) | v43 | 单 Q 216/32 register 预算 | 回退 |
| [v49](v49/README.md) | v43 | 单 Q 224/24 register 预算 | 回退 |
| [v50](v50/README.md) | v43 | epilogue 129-half pitch | 小 grid 实现；性能差异较小，以 bank conflict 指标佐证 |
| [v51](v51/README.md) | v44+v50 | 依据 CTA/SM 数量选择两个自写 kernel | 推荐入口 |

## 被排除的方向与失败记录

增加寄存器、去掉源码数组、拆分 P stores、序列化 SFU、延后 rowsum、低比例 polynomial 都没有稳定改善。v36 消除 spill 后仍慢，说明 spill 数量不是唯一瓶颈。
v39 的理论资源允许两个 CTA，但实测 achieved active warps 仅约 6、TC active 约 33%；因此不能用理论 occupancy 直接预测收益。v43 的不同 grid 表现进一步说明调度/wave 边界很重要。
v34 按全 SM 的寄存器总数估算忽略 subpartition 约束，启动返回 701。v35 不完整 metadata warpgroup 的动态分配在首个输入上未完成，90 秒复核退出 -9；具体硬件停滞原因未被进一步证实。
v37 首轮 interleaved D 布局破坏 swizzle atom 内布局，已保留失败快照并修正为 4D TMA + 原 blocked 布局。v42 首轮 helper 名称生成错误也保留编译失败记录。

## 验证、选择规则与范围

最终统一 benchmark 包括本轮 23 个有效版本及 v1/v6/v24 三个基线：1040 项独立 FP32 正确性检查全部通过，8 shapes ×26=208 个性能结果，各有 9 个原始计时样本。
v51 另通过 120 项多 seed / S=1、17、127、129、511、513、1025、2048 / 多 B,H / scale=0.1、1、8 的独立 FP32 检查；包含自动选择的两条路径。30-shape sweep 同时验证输出与官方的差值。
v51 在 S=511（small）和 S=8191（large，含尾 tile）上，各完成 causal/non-causal × memcheck/synccheck/racecheck，共 12 项，全部退出 0。
v51 的选择规则：small CTA 数 ceil(S/128)×B×H ≤ SM 数时用单 Q-stage；causal 且 S≥2048 时允许 ≤2×SM 数；其余用双 Q-stage。仅依赖 shape/mode/device，不读取数据或复用历史输出。规则来自 30-shape sweep，其他 shape 的性能仍需实测。
正确性容差仍为 atol=rtol=0.003。官方 b35 对恒定 Q/K=30000 的压力案例仍有非有限输出；这些压力输入不用于性能排名。常规性能输入双方均通过对照。

## 本轮停止点

已经测试调度、硬件依赖、线程角色、寄存器预算、不同 N tile、TMA 布局、softmax 指令组织、近似指数、epilogue 与 shape dispatch。最后的预算/消融实验没有找到全面稳定的进一步收益；停止在 v51，并保留更快单项 case 的原始结果。
这不是全局最优证明。剩余差距涉及 Tensor Core/softmax 调度与编译器生成指令；多 CTA cluster、不同 TMEM load layout 或更大 N tile 需要新的实现与验证。本轮没有声称实现这些未测路径。

## 原始数据与复现

[最终同机复测](https://modal.com/apps/akseven6221/main/ap-SeNzAMu7MyFb1rjvdqAWje) · [源码与原始结果](runs/r2_final_all/) · [30-shape sweep](runs/r2_v51_dispatch_sweep/results.json) · [120 项 stress](runs/r2_v51_stress/results.json)
[small sanitizer](runs/r2_v51_sanitize_small/) · [large sanitizer](runs/r2_v51_sanitize_large/) · [全部历史索引](MEASUREMENTS.md)

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v51 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode sweep
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode stress
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode profile --seq 512
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode profile --seq 8192 --causal
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode sanitize --sanitize-seq 8191
.venv/bin/python kernel/fa4/finalize_round2.py
.venv/bin/python kernel/fa4/summarize.py
```

MMA 顺序依据：[NVIDIA PTX ISA，Pipelined tcgen05 Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-pipelined-instructions)。官方实现参考快照仍在 reference/，来源和 BSD 许可保留。
