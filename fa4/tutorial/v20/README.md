# 第 20 课：小任务支线：更高理论 occupancy 为什么仍慢

[路线总览](../README.md) · [上一课](../v19/README.md) · [下一课](../v21/README.md)

对应原实验 **v39**；真实父版本：[v31（第 17 课）](../v17/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

回到 v31 分叉。双 Q-stage 使用较多 SMEM/TMEM，小 grid 的 CTA 数可能不足；尝试用更小 CTA 并行补偿。

## 改法与原理

改为单 Q-stage、M128/N64、256 threads、256 TMEM columns，双 KV buffers。实际 N=64 由 qk descriptor、load_kv、循环步长和 tensor map 决定；原文件遗留 constexpr N=128 未驱动这些位置，不能仅看常量声明。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v31 → v39](from_v31.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:34](kernel.cu#L34)：K/V 每次只搬 64 rows；和 Q 的 128 rows 区分。
- [kernel.cu:44](kernel.cu#L44)：实际 QK N=64 编码在 descriptor。
- [kernel.cu:251](kernel.cu#L251)：host tensor map 也必须与 tile 一致。

## 正确性与同步必须守住什么

N 变化会同时影响 score 数、P packing、PV reduction 长度、V descriptor stride、TMA box 和 causal/tail。不能只改一个 constexpr 就认为完成 tile 变换。

## 如何读 profile 和性能结果

理论资源允许两个 CTA，但 NCU achieved active warps 约 5.96、TC active 约 33.1%，长序列显著变慢。此课是诊断过渡，不是提速成功。下一课保留轻量结构、恢复 N128。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v31 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 14.794 | 10.731 | 15.248 / 1.031× |
| 8192 | False | 643.013 | 203.664 | 278.870 / 0.434× |
| 4096 | True | 87.627 | 50.741 | 74.683 / 0.852× |
| 8192 | True | 312.677 | 105.736 | 142.707 / 0.456× |

[原版实验说明](../../v39/README.md) · [NCU 诊断](../../v39/profile/ncu_details.txt) · [NCU CSV](../../v39/profile/ncu_raw.csv) · [编译与 spill](../../v39/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

N128→N64 时，访问相同 key 长度需要多少次循环？为什么资源更少却不一定更快？

<details>
<summary>参考思路</summary>

大致翻倍；更多同步、softmax 更新和 issue 固定开销，可能超过额外并发带来的收益。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v31,v39 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v39 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
