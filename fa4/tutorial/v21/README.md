# 第 21 课：小 grid 专用：恢复 N128，保留单 Q-stage

[路线总览](../README.md) · [上一课](../v20/README.md) · [下一课](../v22/README.md)

对应原实验 **v43**；真实父版本：[v39（第 20 课）](../v20/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

N64 的循环与同步次数太多；可以恢复较大 key tile，同时用单 KV pair 控制容量。

## 改法与原理

从 v39 恢复 N128，使用一个 Q tile、一个 K tile、一个 V tile；数据 SMEM 96 KiB 加 metadata，256 TMEM columns。softmax/metadata budget 调为 192/56。此课是 tile、buffer 与配额的联合设计，收益不能只归给 N。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v39 → v43](from_v39.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:15](kernel.cu#L15)：容量来自一个 Q 和一个 K/V pair。
- [kernel.cu:148](kernel.cu#L148)：单 Q-stage 的角色预算。
- [kernel.cu:260](kernel.cu#L260)：每 CTA 只负责 128 query rows。

## 正确性与同步必须守住什么

单 buffer 也需分别保护 K 和 V 的生命周期；K 可在 QK 后先覆盖，V 必须等 PV 完成。只有一个 Q-stage，不要照搬双 stage 的下标含义。

## 如何读 profile 和性能结果

短序列明显快于 v31/v44，长序列 non-causal 却慢很多。30-shape sweep 表明，应按 CTA 数与设备规模选择，而非简单断言“S 越小用它越好”。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v39 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 12.134 | 10.731 | 14.794 / 1.219× |
| 8192 | False | 479.982 | 203.664 | 643.013 / 1.340× |
| 4096 | True | 67.990 | 50.741 | 87.627 / 1.289× |
| 8192 | True | 238.501 | 105.736 | 312.677 / 1.311× |

[原版实验说明](../../v43/README.md) · [NCU 诊断](../../v43/profile/ncu_details.txt) · [NCU CSV](../../v43/profile/ncu_raw.csv) · [编译与 spill](../../v43/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

B=1,H=8 时，S512 和 S8192 的单 Q-stage CTA 数各是多少？

<details>
<summary>参考思路</summary>

ceil(S/128)*B*H，分别为 32 和 512。固定启动开销、填满 SM 的能力和多 wave 稳态成本的重要性不同。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v39,v43 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v43 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
