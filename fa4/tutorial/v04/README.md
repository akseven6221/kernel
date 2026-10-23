# 第 04 课：减少 O 重缩放：延迟更新 reference

[路线总览](../README.md) · [上一课](../v03/README.md) · [下一课](../v05/README.md)

对应原实验 **v4**；真实父版本：[v3（第 03 课）](../v03/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

标准 online softmax 每当最大值上升就重缩放 O；O 有 128 个 FP32 元素，TMEM 读出、乘 alpha、写回有固定成本。

## 改法与原理

candidate 与 reference 的 base-2 距离超过 8 才更换 reference，否则 alpha=1；一个 warp 内所有行都无需 rescale 时，跳过该 warp 的 O 读写。此时 rowmax 更准确的名字是 reference，它可能小于实际最大 score。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v3 → v4](from_v3.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:99](kernel.cu#L99)：reference 不再每轮追随最大值。
- [kernel.cu:116](kernel.cu#L116)：warp 内任何一行需 correction 才执行集体 TMEM 读写。

## 正确性与同步必须守住什么

rowsum 和 O 必须同时以同一个 reference 表示。延迟更新期间最大 exponent 输入不超过 8，P≤256；更大变化及时 rescale。首轮显式初始化，避免 -infinity 运算污染状态。

## 如何读 profile 和性能结果

读主循环中的 __any_sync 和 TMEM correction；同机结果只有较小收益，说明此项不能独自解决串行等待。这里是减少工作量，不是改变 tile。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v3 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 19.152 | 10.640 | 19.250 / 1.005× |
| 8192 | False | 688.288 | 202.405 | 693.446 / 1.007× |
| 4096 | True | 159.114 | 53.230 | 165.568 / 1.041× |
| 8192 | True | 462.549 | 104.394 | 480.941 / 1.040× |

[原版实验说明](../../v4/README.md) · [NCU 诊断](../../v4/profile/ncu_details.txt) · [NCU CSV](../../v4/profile/ncu_raw.csv) · [编译与 spill](../../v4/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

alpha=1 时为什么仍要更新 rowsum？为什么不能简单让所有 alpha 永远为 1？

<details>
<summary>参考思路</summary>

新 tile 仍贡献 psum。永不更新 reference 会让指数范围失控，FP16 P 可能溢出；阈值限制了这个范围。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v3,v4 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v4 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
