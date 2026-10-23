# 第 16 课：交错 QK/PV：让另一个 stage 填补等待

[路线总览](../README.md) · [上一课](../v15/README.md) · [下一课](../v17/README.md)

对应原实验 **v27**；真实父版本：[v24（第 15 课）](../v15/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

先发两个 QK、再集中等两个 softmax，仍可能留下 Tensor Core 空隙。双 Q-stage 只是提供了机会，还需选择合适 issue 顺序。

## 改法与原理

改成 QK0(j) → PV1(j-1) → QK1(j) → PV0(j)。stage1 的 PV 延迟到下一轮；最后独立 drain 一次 PV1。只画稳态不够，还要画首轮和末轮。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v24 → v27](from_v24.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:135](kernel.cu#L135)：本课核心是 issue 顺序，不是更多 tile。
- [kernel.cu:147](kernel.cu#L147)：延后一轮的 PV1 读取前一 slot。
- [kernel.cu:164](kernel.cu#L164)：独立 drain 保护最后一个 stage1 输出。

## 正确性与同步必须守住什么

V(j) 必须保留到延后的 PV1(j) 完成。causal 两个 stage 的循环长度不同，stage0 的 if 不能误删 stage1 的工作；FINAL 在 drain 后。

## 如何读 profile 和性能结果

NCU TC active 从 v24 约 46.9% 到 v27 约 56.8%；第二轮同机长序列 latency 有明显下降。NCU 的 active 指标不是官方吞吐比例，最终排名用 CUDA Graph 数据。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v24 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 15.566 | 10.731 | 16.946 / 1.089× |
| 8192 | False | 291.579 | 203.664 | 362.376 / 1.243× |
| 4096 | True | 77.360 | 50.741 | 95.037 / 1.229× |
| 8192 | True | 147.347 | 105.736 | 184.813 / 1.254× |

[原版实验说明](../../v27/README.md) · [NCU 诊断](../../v27/profile/ncu_details.txt) · [NCU CSV](../../v27/profile/ncu_raw.csv) · [编译与 spill](../../v27/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

写出 j=0、j=1 和 loop 结束后的 issue 顺序，标出 V0 最早什么时候能复用。

<details>
<summary>参考思路</summary>

j=0 无 PV1(-1)；j=1 才执行 PV1(0)，其 completion 后 V0 可重用；最后追加 PV1(last)。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v27 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v27 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
