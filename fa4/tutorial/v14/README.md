# 第 14 课：causal 负载均衡：先安排较重的 query tiles

[路线总览](../README.md) · [上一课](../v13/README.md) · [下一课](../v15/README.md)

对应原实验 **v23**；真实父版本：[v21（第 13 课）](../v13/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

causal 前部 query 只处理少量 keys，后部 query 处理更多。相同 tile 大小不代表相同 CTA 工作量；尾部剩下少数重 CTA 会拖长结束时间。

## 改法与原理

从 v21 出发只加入 query-major reverse remap：把 linear CTA 坐标映射为逆序 query tile，batch/head 在 query 层内交错。non-causal 映射保持原状。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v21 → v23](from_v21.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:92](kernel.cu#L92)：先线性化 grid，再映射 query-major 顺序。
- [kernel.cu:96](kernel.cu#L96)：causal 重任务优先；输出仍写真实 query 位置。

## 正确性与同步必须守住什么

remap 只重新排列任务，必须双射覆盖每个 (batch,head,query tile)，不能交换输出内容或减少 causal 工作。

## 如何读 profile 和性能结果

独立 v22 实验中，NCU SM active cycles 的 max/mean 不均衡从约 +41% 降到约 +8%；v23 重新测组合，而非把两个历史 speedup 相乘。CUDA 不承诺严格按 blockIdx 顺序执行，这里是实测有效的映射。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v21 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 16.885 | 10.640 | 17.814 / 1.055× |
| 8192 | False | 369.352 | 202.405 | 365.176 / 0.989× |
| 4096 | True | 96.573 | 53.230 | 96.373 / 0.998× |
| 8192 | True | 185.139 | 104.394 | 290.459 / 1.569× |

[原版实验说明](../../v23/README.md) · [NCU 诊断](../../v23/profile/ncu_details.txt) · [NCU CSV](../../v23/profile/ncu_raw.csv) · [编译与 spill](../../v23/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

gridDim.x=3、B*H=2 时，枚举 linear=0..5 对应的 query tile 和 bh。

<details>
<summary>参考思路</summary>

依次为 (2,0),(2,1),(1,0),(1,1),(0,0)。相同重 query 的两个 heads 提前进入映射。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v21,v23 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v23 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
