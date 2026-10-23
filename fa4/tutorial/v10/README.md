# 第 10 课：FMA 与数值稳定性：减少指令不保证更快

[路线总览](../README.md) · [上一课](../v09/README.md) · [下一课](../v11/README.md)

对应原实验 **v14**；真实父版本：[v13（第 09 课）](../v09/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

每个 score 都计算 (score-reference)*scale，可尝试把 reference 缩放提到循环外。

## 改法与原理

预计算 scaled_ref=-reference*scale，普通范围用 fmaf(score,scale,scaled_ref)。当 abs(reference)>65536 时保留先减后乘，使当前最大 score 的指数参数可靠地为零。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v13 → v14](from_v13.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:176](kernel.cu#L176)：把公共计算提出循环，同时引入公共舍入项。
- [kernel.cu:181](kernel.cu#L181)：数值稳定性分支与 FMA fast path 配套。

## 正确性与同步必须守住什么

scaled_ref 自身已舍入；FMA 不保证在 score==reference 时得到精确零。巨大的公共偏移会令中间 FP16 P 溢出，即使数学 softmax 不变。

## 如何读 profile 和性能结果

最终统一复测中 v14 比 v13 慢，包括 S8192 non-causal；早期试跑曾见收益，不能用它覆盖最终结果。保留此课因为它是后续真实祖先，并展示指令数、编译器调度与数值误差的区别。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v13 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 19.144 | 10.640 | 18.222 / 0.952× |
| 8192 | False | 407.381 | 202.405 | 386.845 / 0.950× |
| 4096 | True | 106.656 | 53.230 | 101.699 / 0.954× |
| 8192 | True | 323.158 | 104.394 | 307.256 / 0.951× |

[原版实验说明](../../v14/README.md) · [NCU 诊断](../../v14/profile/ncu_details.txt) · [NCU CSV](../../v14/profile/ncu_raw.csv) · [编译与 spill](../../v14/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

比较 fma(x,s,round(-x*s)) 与 (x-x)*s，为什么前者可能非零？

<details>
<summary>参考思路</summary>

第一种使用已舍入的公共项再融合乘加，不能消掉公共项的舍入误差；第二种先精确相减为零。检查中间 P，不能只分析最终归一化公式。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v13,v14 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v14 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
