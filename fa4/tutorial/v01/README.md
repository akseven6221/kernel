# 第 01 课：跑通串行 attention：先理解数据流

[路线总览](../README.md) · [下一课](../v02/README.md)

对应原实验 **v1**；真实父版本：无，串行基线。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

先建立可解释的低性能基线。一块 CTA 处理 128 个 query，每次读 128 个 key/value；搬运、QK、softmax、PV 顺序执行，等待期间缺少另一份工作填空。

## 改法与原理

Q/K/V 经 TMA 进入 swizzled SMEM；QK 把 FP32 scores 写入 TMEM；每线程取一行，在寄存器中做 online softmax，再把两个 FP16 P 打包成一个 32-bit word 写回 TMEM；PV 累加 FP32 O，最后归一化写出。原版 v1 使用 exp2f，v2 才直接用近似 PTX 指令。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · 本课没有父版本 diff

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:37](kernel.cu#L37)：先读 QK/PV helper：两次 GEMM 的输入位置不同。
- [kernel.cu:82](kernel.cu#L82)：每个 query row 持有跨 KV tiles 的 online 状态。
- [kernel.cu:109](kernel.cu#L109)：分母与 O 必须使用相同的 reference 更新。
- [kernel.cu:140](kernel.cu#L140)：Host 只负责 tensor map 和 launch，主要学习 kernel 内的数据流。

## 正确性与同步必须守住什么

P 覆盖 S 前必须读完 scores；QK 覆盖下一轮 S 前必须结束前次 PV 对 P 的读取。TMEM 分配 256 columns，S/P 位于前 128 列，O 位于后 128 列。

## 如何读 profile 和性能结果

NCU 和 SASS 暴露通用指数、非合并输出及逐元素除法开销。先认识这些可独立处理的问题，再改流水线；v1 已经使用 Tensor Core，低性能不等于没有使用矩阵硬件。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | 
| ---: | --- | ---: | ---: |
| 512 | False | 31.840 | 10.640 |
| 8192 | False | 829.216 | 202.405 |
| 4096 | True | 216.688 | 53.230 |
| 8192 | True | 612.062 | 104.394 |

[原版实验说明](../../v1/README.md) · [NCU 诊断](../../v1/profile/ncu_details.txt) · [NCU CSV](../../v1/profile/ncu_raw.csv) · [编译与 spill](../../v1/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

手画一个 CTA 的第 0、1 个 KV tile：rowmax、rowsum、O 各保存什么？为什么不能分别 softmax 每个 tile 后直接相加？

<details>
<summary>参考思路</summary>

每个 tile 的分母不同。必须统一 reference，并用 alpha 同时重缩放旧 rowsum 和旧 O；最后仅除一次全局 rowsum。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v1 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v1 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
