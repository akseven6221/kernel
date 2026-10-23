# 第 06 课：预取支线：双 K/V buffer

[路线总览](../README.md) · [上一课](../v05/README.md) · [下一课](../v07/README.md)

对应原实验 **v6**；真实父版本：[v5（第 05 课）](../v05/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

串行基线每轮先等 TMA，再算 QK/softmax/PV；可以尝试在算 tile j 时加载 j+1。

## 改法与原理

增加第二组 K/V slot，以 j%2 选择 slot，以 (j/2)&1 追踪同一 slot 的 barrier phase。先填第一个 slot，进入循环后预取下一块。此课是 v5 的支线；下一课也从 v5 出发。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v5 → v6](from_v5.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:74](kernel.cu#L74)：多出的两个 TILE_BYTES 对应第二组 K/V。
- [kernel.cu:91](kernel.cu#L91)：在本轮计算开始前发起下一轮 TMA。

## 正确性与同步必须守住什么

旧 PV 完成后才能覆盖其 V；旧 QK 完成后才能覆盖 K。轮次 j 的 parity 与 slot 的重用次数相关，不能对所有 slot 使用同一个随意翻转的 phase。

## 如何读 profile 和性能结果

v6 的短序列和长序列都比 v5 快，但双 buffer 让数据 SMEM 从 96 KiB 增到 160 KiB。后续双 Q-stage 的资源与角色分工需重新设计，不能直接把 v6 的改动当成 v7 的父实现。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v5 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 14.642 | 10.640 | 15.669 / 1.070× |
| 8192 | False | 580.058 | 202.405 | 680.171 / 1.173× |
| 4096 | True | 130.267 | 53.230 | 148.920 / 1.143× |
| 8192 | True | 379.704 | 104.394 | 445.970 / 1.175× |

[原版实验说明](../../v6/README.md) · [NCU 诊断](../../v6/profile/ncu_details.txt) · [NCU CSV](../../v6/profile/ncu_raw.csv) · [编译与 spill](../../v6/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

列出 j=0..5 的 slot 和 phase，并标出每个 slot 何时可重新写入。

<details>
<summary>参考思路</summary>

(slot,phase) 为 (0,0),(1,0),(0,1),(1,1),(0,0),(1,0)。可重用时间由前次消费者完成决定。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v5,v6 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v6 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
