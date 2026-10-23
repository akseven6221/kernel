# 第 03 课：合并写回：用 SMEM 转换线程到输出的映射

[路线总览](../README.md) · [上一课](../v02/README.md) · [下一课](../v04/README.md)

对应原实验 **v3**；真实父版本：[v2（第 02 课）](../v02/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

一个线程负责一整行适合 softmax，但同一时刻相邻 lanes 写不同 query 行，global 地址跨越 H×D。算得顺手的布局不一定写得高效。

## 改法与原理

结束主循环后复用已死亡的 Q/K/V SMEM；先写 staging[c*129+row]，同步，再按连续 head_dim 读取两个元素，用 half2 合并写 global。129 是 FP32 staging 的 pitch。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v2 → v3](from_v2.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:130](kernel.cu#L130)：SMEM 作为线程间交换缓冲，而不是额外 global transpose。
- [kernel.cu:135](kernel.cu#L135)：输出 ownership 从每线程一行改为连续 half2。

## 正确性与同步必须守住什么

只有最后一次 PV 完成后才能复用 SMEM；转置写入与读取之间保留 CTA barrier；最后一个 query tile 仍按 seq 屏蔽输出。

## 如何读 profile 和性能结果

基线输出有 4194304 global-store sectors，其中 3932160 excessive sectors。检查 v3 的 sector 指标和短序列 latency，解释为何固定 epilogue 开销在小任务中更显眼。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v2 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 19.250 | 10.640 | 30.411 / 1.580× |
| 8192 | False | 693.446 | 202.405 | 760.104 / 1.096× |
| 4096 | True | 165.568 | 53.230 | 190.650 / 1.151× |
| 8192 | True | 480.941 | 104.394 | 525.330 / 1.092× |

[原版实验说明](../../v3/README.md) · [NCU 诊断](../../v3/profile/ncu_details.txt) · [NCU CSV](../../v3/profile/ncu_raw.csv) · [编译与 spill](../../v3/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

固定 c，比较旧输出地址与新 half2 地址：lane 增加 1 时，地址各增加多少？

<details>
<summary>参考思路</summary>

旧地址通常跨 H*128 个 half；新映射同一行内相邻 lanes 各写两个 half，形成连续地址。跨行处需结合循环的 i/64 计算。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v2,v3 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v3 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
