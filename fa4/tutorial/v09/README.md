# 第 09 课：mask fast path：完整 tile 不逐元素检查

[路线总览](../README.md) · [上一课](../v08/README.md) · [下一课](../v10/README.md)

对应原实验 **v13**；真实父版本：[v12（第 08 课）](../v08/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

大部分非尾块和 causal 严格过去区域都无需 mask，旧代码仍对每个 score 做 bounds/select。

## 改法与原理

先做 tile/当前行级判断：key tile 最后一个位置小于 seq，且 causal 时不超过 query，则跳过 128 次逐元素 mask。对 diagonal 和 tail 保留原慢路径。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v12 → v13](from_v12.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:162](kernel.cu#L162)：一次判断决定是否执行逐元素 mask。

## 正确性与同步必须守住什么

fast path 必须同时满足 key bounds 和 causal 条件。TMA 越界填充值不等于 softmax 的 -infinity mask：零 score 仍会贡献概率。

## 如何读 profile 和性能结果

这是最终统一数据中较清楚的一步长序列收益。优先验证非 causal、causal、257/511 尾块，不能只测 4096 的整齐尺寸。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v12 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 18.222 | 10.640 | 20.069 / 1.101× |
| 8192 | False | 386.845 | 202.405 | 446.259 / 1.154× |
| 4096 | True | 101.699 | 53.230 | 132.882 / 1.307× |
| 8192 | True | 307.256 | 104.394 | 400.469 / 1.303× |

[原版实验说明](../../v13/README.md) · [NCU 诊断](../../v13/profile/ncu_details.txt) · [NCU CSV](../../v13/profile/ncu_raw.csv) · [编译与 spill](../../v13/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

S=257、query=256 时，哪些 key tile 可跳过 mask，最后一个 tile 为什么不行？

<details>
<summary>参考思路</summary>

j=0 和 j=1 的全部 128 keys 有效；j=2 只有 key=256 有效，其余必须设为 -infinity。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v12,v13 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v13 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
