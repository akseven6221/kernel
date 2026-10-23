# 第 13 课：rowsum reduction：四条 pair-sum 链

[路线总览](../README.md) · [上一课](../v12/README.md) · [下一课](../v14/README.md)

对应原实验 **v21**；真实父版本：[v16（第 12 课）](../v12/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

maximum 已有 ILP，psum 仍把 64 个 (a+b) 逐个串行累加。

## 改法与原理

改为 sums[c%4]+=a+b，最后合并四项。maximum、exponent 和 buffer 调度保持 v16，独立测 rowsum。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v16 → v21](from_v16.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:178](kernel.cu#L178)：只改变 rowsum 的依赖结构。
- [kernel.cu:199](kernel.cu#L199)：在更新 online rowsum 前合并四条链。

## 正确性与同步必须守住什么

加法重新排序改变 FP32 rounding，重新检查独立参考误差；不要仅凭数学结合律认定浮点输出完全相同。

## 如何读 profile 和性能结果

最终统一复测比 v16 的长序列更快。对照 v10 说明：同一个 ILP 想法放在不同资源预算和角色结构里，效果会不同。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v16 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 17.814 | 10.640 | 18.536 / 1.041× |
| 8192 | False | 365.176 | 202.405 | 383.254 / 1.050× |
| 4096 | True | 96.373 | 53.230 | 102.440 / 1.063× |
| 8192 | True | 290.459 | 104.394 | 310.190 / 1.068× |

[原版实验说明](../../v21/README.md) · [NCU 诊断](../../v21/profile/ncu_details.txt) · [NCU CSV](../../v21/profile/ncu_raw.csv) · [编译与 spill](../../v21/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

为什么此处四条链只有四个标量 accumulator，而后面的 packed 版本需要八个？

<details>
<summary>参考思路</summary>

这里先标量 a+b 再累加；packed 版本分别累计 a 和 b 两个分量，一条 packed 链有两个 FP32 accumulator。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v16,v21 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v21 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
