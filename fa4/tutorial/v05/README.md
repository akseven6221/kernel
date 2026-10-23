# 第 05 课：归一化：一行一次 reciprocal

[路线总览](../README.md) · [上一课](../v04/README.md) · [下一课](../v06/README.md)

对应原实验 **v5**；真实父版本：[v4（第 04 课）](../v04/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

epilogue 里 128 个 output 元素除以同一个 rowsum，重复精确除法不必要。

## 改法与原理

每行先算一次 rcp.approx.ftz.f32(rowsum)，随后 o[c]*inverse。主循环、布局、reference 规则均继承前版。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v4 → v5](from_v4.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:131](kernel.cu#L131)：每行只做一次 reciprocal。
- [kernel.cu:135](kernel.cu#L135)：把结果广播给本行全部输出元素。

## 正确性与同步必须守住什么

这里仍采用近似数学；有效 query 行的 rowsum 为正。不要把越界行的中间数值当作有效输出。

## 如何读 profile 和性能结果

查看 PTX 的 div.rn 和 SASS 的 reciprocal/辅助调用是否减少，再比较短序列。源代码的 / 不一定只对应一条便宜机器指令。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v4 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 15.669 | 10.640 | 19.152 / 1.222× |
| 8192 | False | 680.171 | 202.405 | 688.288 / 1.012× |
| 4096 | True | 148.920 | 53.230 | 159.114 / 1.068× |
| 8192 | True | 445.970 | 104.394 | 462.549 / 1.037× |

[原版实验说明](../../v5/README.md) · [NCU 诊断](../../v5/profile/ncu_details.txt) · [NCU CSV](../../v5/profile/ncu_raw.csv) · [编译与 spill](../../v5/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

为什么这项优化更可能改变短序列的相对速度？

<details>
<summary>参考思路</summary>

归一化成本按 query 行数增长，长序列主循环还按 key 长度增长；固定 epilogue 在短序列占比更大。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v4,v5 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v5 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
