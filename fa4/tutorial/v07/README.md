# 第 07 课：架构过渡：双 Q-stage 与四个 warpgroup

[路线总览](../README.md) · [上一课](../v06/README.md) · [下一课](../v08/README.md)

对应原实验 **v7**；真实父版本：[v5（第 05 课）](../v05/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

单 query tile 的 softmax 和同 tile 的 QK/PV 有数据依赖；仅增加 KV 预取无法让两者自由重叠。

## 改法与原理

一个 CTA 同时拥有两个 128-row Q tiles。WG0/WG1 分别做 softmax，WG2 做 correction/epilogue，WG3 内分 MMA 与 TMA issuer；一次 K/V 加载供两个 Q stages 使用。512 threads、512 TMEM columns，显式 mbarrier 交接 ownership。这是一次结构重写，不是单条指令优化。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v5 → v7](from_v5.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:81](kernel.cu#L81)：把 barrier 表与 ownership 图一起看。
- [kernel.cu:112](kernel.cu#L112)：metadata WG：独立 TMA 与 MMA issuer。
- [kernel.cu:155](kernel.cu#L155)：两个 software owners 独立消费各自 SFULL。
- [kernel.cu:174](kernel.cu#L174)：跨 WG correction mailbox 是下一课要减少的往返。

## 正确性与同步必须守住什么

S0/S1 位于 TMEM [0,128)/[128,256)，O0/O1 位于 [256,384)/[384,512)；P 分别覆盖各 S 的高 64 columns。明确 PFULL、PVDONE、CFULL、CACK 各保护谁。

## 如何读 profile 和性能结果

长序列下降明显，短序列反而慢于 v6。四角色建立并行机会，也增加 mailbox、资源和启动开销；这为下一课合并角色提供可观察的起点。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v5 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 23.245 | 10.640 | 15.669 / 0.674× |
| 8192 | False | 451.902 | 202.405 | 680.171 / 1.505× |
| 4096 | True | 133.307 | 53.230 | 148.920 / 1.117× |
| 8192 | True | 400.251 | 104.394 | 445.970 / 1.114× |

[原版实验说明](../../v7/README.md) · [NCU 诊断](../../v7/profile/ncu_details.txt) · [NCU CSV](../../v7/profile/ncu_raw.csv) · [编译与 spill](../../v7/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

如果 stage0 softmax 已结束，为什么不能立刻释放整块 TMEM？

<details>
<summary>参考思路</summary>

stage1 或 correction owner 可能仍读写其片段。释放必须等待全部使用者完成，而不是某个线程退出局部循环。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v5,v7 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v7 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
