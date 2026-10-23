# 第 15 课：拆开 K/V 生命周期：尽早释放 K

[路线总览](../README.md) · [上一课](../v14/README.md) · [下一课](../v16/README.md)

对应原实验 **v24**；真实父版本：[v23（第 14 课）](../v14/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

QK 只读 K，PV 才读 V；把它们绑在 KVFULL/KVEMPTY 上，意味着 QK 等不需要的 V，K 复用也等不需要的 PV。

## 改法与原理

将 full/empty 分别拆成 KFULL、VFULL、KEMPTY、VEMPTY。最后一个 QK 后提交 KEMPTY，最后一个 PV 后提交 VEMPTY；loader 可先搬下一轮 K，再等 V slot。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v23 → v24](from_v23.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:81](kernel.cu#L81)：分别列出四种 K/V ownership barrier。
- [kernel.cu:124](kernel.cu#L124)：loader 不再必须把 K 和 V 一起等待。
- [kernel.cu:147](kernel.cu#L147)：K 生命周期在两个 QK 使用结束后关闭。

## 正确性与同步必须守住什么

empty 是旧消费者完成，不是新 producer 已经开始。异步 MMA 发出后不能立即普通 arrive 宣称 K/V 可覆盖，要让 completion 释放。

## 如何读 profile 和性能结果

SMEM 实际仍是双 K slots、双 V slots。对照 v23 的 wait/commit 位置和配对 latency：收益来自解除多余依赖，不是增加容量。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v23 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 16.784 | 10.640 | 16.885 / 1.006× |
| 8192 | False | 360.952 | 202.405 | 369.352 / 1.023× |
| 4096 | True | 94.755 | 53.230 | 96.573 / 1.019× |
| 8192 | True | 181.642 | 104.394 | 185.139 / 1.019× |

[原版实验说明](../../v24/README.md) · [NCU 诊断](../../v24/profile/ncu_details.txt) · [NCU CSV](../../v24/profile/ncu_raw.csv) · [编译与 spill](../../v24/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

为什么 KEMPTY 不能放在第一个 Q-stage 的 QK 后？

<details>
<summary>参考思路</summary>

两个 Q stages 共享同一个 K tile；第二个 QK 还会读它，必须等最后一个实际消费者。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v23,v24 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
