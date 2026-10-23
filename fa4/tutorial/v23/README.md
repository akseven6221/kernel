# 第 23 课：汇合：按 CTA/SM 数选择完整 kernel

[路线总览](../README.md) · [上一课](../v22/README.md)

对应原实验 **v51**；真实父版本：[v44（第 19 课）](../v19/README.md)、[v50（第 22 课）](../v22/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

大 grid 双 Q-stage 更擅长稳态吞吐，小 grid 单 Q-stage 更轻。单一版本无法在全部 shape 上取胜。

## 改法与原理

完整组合 v44 与 v50。create 查询设备 SM 数；launch 计算 T=ceil(S/128)*B*H。T≤SM，或 causal 且 S≥2048 且 T≤2*SM 时用 small，其余用 large。每次仅 launch 一个完整 attention kernel。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v44 → v51](from_v44.diff) · [v50 → v51](from_v50.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:15](kernel.cu#L15)：大 grid 组件来自原 v44。
- [kernel.cu:320](kernel.cu#L320)：小 grid 组件来自原 v50。
- [kernel.cu:574](kernel.cu#L574)：分别构造 context，设备属性查询放在 launch 之外。
- [kernel.cu:588](kernel.cu#L588)：本课真正新增的优化点：按工作量与设备规模选择。

## 正确性与同步必须守住什么

tensor map 创建和设备查询不进入 kernel 计时。dispatch 只依赖 shape/mode/device；CUDA Graph 捕获选定路径后，shape 改变需要重新准备对应 context/graph。

## 如何读 profile 和性能结果

规则来自 30-shape sweep，v51 再经过统一 benchmark、120 项压力测试和两条路径共 12 项 sanitizer。推荐是覆盖范围内的折中，不是每个 case 的绝对最快版本；S8192 约达到官方吞吐 74%。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v44 μs / 父版÷本版 | v50 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | False | 12.038 | 10.731 | 14.725 / 1.223× | 12.034 / 1.000× |
| 8192 | False | 276.085 | 203.664 | 275.558 / 0.998× | 484.781 / 1.756× |
| 4096 | True | 66.978 | 50.741 | 74.706 / 1.115× | 67.099 / 1.002× |
| 8192 | True | 142.614 | 105.736 | 141.886 / 0.995× | 241.070 / 1.690× |

[原版实验说明](../../v51/README.md) · [NCU 诊断](../../v51/profile/ncu_details.txt) · [NCU CSV](../../v51/profile/ncu_raw.csv) · [编译与 spill](../../v51/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

在 148 SM 的 B200 上，B1/H8/S4096 non-causal 与 causal 各走哪条路径？B1/H8/S8192 呢？

<details>
<summary>参考思路</summary>

S4096 的 T=256：non-causal 走 large，causal 满足 T≤296 走 small。S8192 的 T=512，两者都走 large。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v44,v50,v51 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
