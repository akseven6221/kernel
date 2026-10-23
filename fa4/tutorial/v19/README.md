# 第 19 课：按模式专门化：causal 与 non-causal 分开选择

[路线总览](../README.md) · [上一课](../v18/README.md) · [下一课](../v20/README.md)

对应原实验 **v44**；真实父版本：[v42（第 18 课）](../v18/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

不同模式的循环长度、mask 和调度压力不同，一套 softmax 指令组织不一定兼顾所有 case。

## 改法与原理

template<bool CAUSAL> 中用 if constexpr：non-causal 用 v42 packed 路径，causal 用 v31 标量 FMA 和四条 sum chains。选择在编译期完成，不增加每元素运行时模式判断。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v42 → v44](from_v42.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:203](kernel.cu#L203)：同一个模板生成两种 softmax 指令组织。
- [kernel.cu:236](kernel.cu#L236)：causal 只合并四条有效标量链。

## 正确性与同步必须守住什么

分支改变计算组织，attention 定义、mask、scale 和稳定分支保持一致。最终结果重新验证，不能按组件历史数字推算组合结果。

## 如何读 profile 和性能结果

早期配对运行支持这个方向；最终统一表中 v42/v44 差异较小，不能写成每个 causal shape 都固定更快。v44 是 v51 大 grid 组件。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v42 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 14.725 | 10.731 | 14.693 / 0.998× |
| 8192 | False | 275.558 | 203.664 | 274.299 / 0.995× |
| 4096 | True | 74.706 | 50.741 | 75.264 / 1.007× |
| 8192 | True | 141.886 | 105.736 | 142.440 / 1.004× |

[原版实验说明](../../v44/README.md) · [NCU 诊断](../../v44/profile/ncu_details.txt) · [NCU CSV](../../v44/profile/ncu_raw.csv) · [编译与 spill](../../v44/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

if constexpr 与读 tensor 内容后选路径有什么区别？

<details>
<summary>参考思路</summary>

前者按调用模式生成固定 specialization；本实现不读输入内容来选择算法，也不依赖是否处于 benchmark。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v42,v44 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v44 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
