# 第 02 课：指数指令：exp2f → ex2.approx

[路线总览](../README.md) · [上一课](../v01/README.md) · [下一课](../v03/README.md)

对应原实验 **v2**；真实父版本：[v1（第 01 课）](../v01/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

每行每 tile 有 128 个 exponent；通用 math 路径的额外指令会被重复很多次。

## 改法与原理

只替换指数 helper 及其调用。scale_log2=log2(e)/sqrt(128)，因此 exp(score/sqrt(D)) 可以用 exp2(score*scale_log2) 表示。PTX 版本是近似且带 ftz 的数值路径，不能称作精确等价。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v1 → v2](from_v1.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:64](kernel.cu#L64)：本课唯一核心 helper：近似 base-2 exponent。
- [kernel.cu:101](kernel.cu#L101)：reference rescale 也调用同一个 helper。

## 正确性与同步必须守住什么

alpha 和 P 的 exponent 都要用同一缩放定义；保留 mask 的 -infinity。容差验证包括尾块、大 logits 和恒定大 logits。

## 如何读 profile 和性能结果

对照原 v1 的 SASS 中 exponent 的范围处理，再看 v2 的 MUFU.EX2 和配对 latency。下表用最终修正后源码的统一复测，而非早期试跑数字。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v1 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 30.411 | 10.640 | 31.840 / 1.047× |
| 8192 | False | 760.104 | 202.405 | 829.216 / 1.091× |
| 4096 | True | 190.650 | 53.230 | 216.688 / 1.137× |
| 8192 | True | 525.330 | 104.394 | 612.062 / 1.165× |

[原版实验说明](../../v2/README.md) · [NCU 诊断](../../v2/profile/ncu_details.txt) · [NCU CSV](../../v2/profile/ncu_raw.csv) · [编译与 spill](../../v2/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

为什么这里比较 allclose，而不是要求 FP16 输出 bitwise 一样？

<details>
<summary>参考思路</summary>

指数近似及后续 FP16 P 的舍入会改变结果；是否可接受由独立 FP32 参考、有限性和既定容差共同判断。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v1,v2 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v2 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
