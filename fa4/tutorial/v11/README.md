# 第 11 课：max reduction：四条独立依赖链

[路线总览](../README.md) · [上一课](../v10/README.md) · [下一课](../v12/README.md)

对应原实验 **v15**；真实父版本：[v14（第 10 课）](../v10/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

一条 max 链让下一次比较等待上一次结果；展开循环本身不自动消除这个依赖。

## 改法与原理

把 128 个 scores 分配给 maxima[c%4]，最后两层合并四个结果。本课只改 max，rowsum 的求和顺序保持前版。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v14 → v15](from_v14.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:168](kernel.cu#L168)：四个 accumulator 分散依赖。
- [kernel.cu:172](kernel.cu#L172)：最后使用短树合并。

## 正确性与同步必须守住什么

所有初始化值必须为 -infinity；掩码 score 参与 reduction 时仍不能成为有效概率。

## 如何读 profile 和性能结果

与 v14 做配对比较。v10 的八链 max+sum 同时增加多个状态后回退，所以这里收缩为较小的单因素实验。ILP 收益必须结合寄存器和 spill 判断。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v14 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 18.526 | 10.640 | 19.144 / 1.033× |
| 8192 | False | 384.298 | 202.405 | 407.381 / 1.060× |
| 4096 | True | 102.267 | 53.230 | 106.656 / 1.043× |
| 8192 | True | 308.850 | 104.394 | 323.158 / 1.046× |

[原版实验说明](../../v15/README.md) · [NCU 诊断](../../v15/profile/ncu_details.txt) · [NCU CSV](../../v15/profile/ncu_raw.csv) · [编译与 spill](../../v15/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

估算一条 128 次串行 max 与四条各 32 次 max 的最长依赖深度。为什么不直接增加到 128 条链？

<details>
<summary>参考思路</summary>

四链大致是 32 次加两层合并；更多链会增加 live state 和合并成本，编译器/发射资源也有限。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v14,v15 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v15 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
