# 第 22 课：修正 epilogue bank conflict：130 half → 129 half

[路线总览](../README.md) · [上一课](../v21/README.md) · [下一课](../v23/README.md)

对应原实验 **v50**；真实父版本：[v43（第 21 课）](../v21/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

v43 输出 staging 是 FP16，读取 half2 时相邻 lanes 的 c 相差 2。pitch=130 half 使 bank 步长为 2，产生两路冲突。

## 改法与原理

只把 staging 的写/读 pitch 全部由 130 改为 129。对固定 row，lane 每前进一次，地址跨 2*129 个 half，也就是 129 个 32-bit words，模 32 的步长为 1。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v43 → v50](from_v43.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:221](kernel.cu#L221)：写入 padding 与读取地址保持一致。
- [kernel.cu:229](kernel.cu#L229)：按 half 的字节数推导 bank 映射。

## 正确性与同步必须守住什么

写入和读出必须使用一致 pitch，分配区也必须容纳最大下标。不要把第 3 课 FP32 staging 的 bank 推导直接套到 FP16。

## 如何读 profile 和性能结果

NCU excessive shared wavefronts 从 v43 的 131072 降为 0；短序列 latency 只小幅下降，部分长序列还略慢。消除指标中的浪费不意味着端到端必有大提升。

下表均来自同一次 [r2_final_all](../../runs/r2_final_all/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v43 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 12.034 | 10.731 | 12.134 / 1.008× |
| 8192 | False | 484.781 | 203.664 | 479.982 / 0.990× |
| 4096 | True | 67.099 | 50.741 | 67.990 / 1.013× |
| 8192 | True | 241.070 | 105.736 | 238.501 / 0.989× |

[原版实验说明](../../v50/README.md) · [NCU 诊断](../../v50/profile/ncu_details.txt) · [NCU CSV](../../v50/profile/ncu_raw.csv) · [编译与 spill](../../v50/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

假设 pitch=128 half，同样 c 每次加 2，bank 步长是多少？

<details>
<summary>参考思路</summary>

128 个 32-bit words，模 32 为 0；这类访问会集中到同一个 bank，需要考虑实际指令和广播规则。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v43,v50 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v50 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
