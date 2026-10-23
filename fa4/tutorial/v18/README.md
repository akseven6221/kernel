# 第 18 课：packed FP32：同时减少指令与依赖

[路线总览](../README.md) · [上一课](../v17/README.md) · [下一课](../v19/README.md)

对应原实验 **v42**；真实父版本：[v31（第 17 课）](../v17/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

标量 FMA/求和仍占 softmax 指令预算；直接打包但保留单一依赖链，又可能只是把瓶颈换个形式。

## 改法与原理

argument2 用 f32x2 FMA，add_sum2 用 f32x2 add，分成四条 packed chains，也就是八个标量 accumulator；配合 max3 风格归约。max3 helper 源码是两条 max.f32，不应仅凭函数名认定为一条原生 PTX 指令。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v31 → v42](from_v31.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:86](kernel.cu#L86)：两个 FP32 exponent 参数的 packed FMA。
- [kernel.cu:92](kernel.cu#L92)：独立 packed sum chains 与减少指令同时考虑。
- [kernel.cu:98](kernel.cu#L98)：阅读实际 helper，不按名字推断机器指令数量。

## 正确性与同步必须守住什么

f32x2 表示两个 FP32 运算，和将 P 转成 half2 是不同步骤。大 reference 稳定分支保留；psum 合并八项且每个分量只计一次。

## 如何读 profile 和性能结果

这是组合实验。v47 只改 max3 的消融接近持平，因此不能把 v42 的收益全部归给 max3。最终表有 case 差异；causal/non-causal 的取舍在下一课处理。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v31 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 14.693 | 10.731 | 15.248 / 1.038× |
| 8192 | False | 274.299 | 203.664 | 278.870 / 1.017× |
| 4096 | True | 75.264 | 50.741 | 74.683 / 0.992× |
| 8192 | True | 142.440 | 105.736 | 142.707 / 1.002× |

[原版实验说明](../../v42/README.md) · [NCU 诊断](../../v42/profile/ncu_details.txt) · [NCU CSV](../../v42/profile/ncu_raw.csv) · [编译与 spill](../../v42/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

为什么把 64 次 packed add 全放进同一个 accumulator 可能仍然慢？

<details>
<summary>参考思路</summary>

每次 add 仍依赖前一次结果；四条链给调度器更多独立工作，同时需要更多寄存器，应结合 spill 看效果。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v31,v42 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v42 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
