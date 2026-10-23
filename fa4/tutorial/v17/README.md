# 第 17 课：消除多余等待：先证明 MMA 的读写顺序

[路线总览](../README.md) · [上一课](../v16/README.md) · [下一课](../v18/README.md)

对应原实验 **v31**；真实父版本：[v27（第 16 课）](../v16/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

PV 把 P 作为 A 读取，下一 QK 把别名 S 作为 D 写入；旧实现用 PVDONE commit/wait 强制完成后再提交。

## 改法与原理

同一 issuer warp、同一 cta_group 的这对 MMA 有 A-read→后续 D-write 顺序保证，所以去掉这条额外往返。SFULL 仍跟踪此前 MMA，使软件 correction 等到上一 PV 完成。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v27 → v31](from_v27.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:142](kernel.cu#L142)：此处不再等待前次 PV 的额外 PVDONE。
- [kernel.cu:156](kernel.cu#L156)：跨线程 P producer-consumer 交接仍然存在。
- [kernel.cu:178](kernel.cu#L178)：SFULL 也约束此前 O 的 PV 更新。

## 正确性与同步必须守住什么

只删除已证明冗余的依赖。PFULL、SFULL、K/V empty、FINAL、EPIDONE 和跨线程 fence 保留。不能推广为“同线程全部异步指令自动顺序完成”。

## 如何读 profile 和性能结果

本课结合 NVIDIA PTX pipelined tcgen05 规则与原 v31 的 96 项压力检查、六项 sanitizer 结果；参考链接见基础篇。sanitizer 是补充证据，不能替代对具体指令依赖的论证。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v27 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 15.248 | 10.731 | 15.566 / 1.021× |
| 8192 | False | 278.870 | 203.664 | 291.579 / 1.046× |
| 4096 | True | 74.683 | 50.741 | 77.360 / 1.036× |
| 8192 | True | 142.707 | 105.736 | 147.347 / 1.033× |

[原版实验说明](../../v31/README.md) · [NCU 诊断](../../v31/profile/ncu_details.txt) · [NCU CSV](../../v31/profile/ncu_raw.csv) · [编译与 spill](../../v31/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

为什么可以去掉 PVDONE，却不能去掉 softmax producer 的 PFULL？

<details>
<summary>参考思路</summary>

前者是同 warp MMA 的特定读写配对；后者跨 software owner 与 issuer，P 的 TMEM store 必须发布完成后才能被 PV 消费。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v27,v31 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v31 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
