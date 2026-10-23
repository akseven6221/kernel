# 第 08 课：合并角色：softmax owner 自己修正 O

[路线总览](../README.md) · [上一课](../v07/README.md) · [下一课](../v09/README.md)

对应原实验 **v12**；真实父版本：[v7（第 07 课）](../v07/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

conditional rescale 让 correction 的实际工作变少，但每轮仍为独立 WG 交换 alpha 和 ack。NCU polling samples 提示通信可能比工作本身更贵。

## 改法与原理

把 O correction 和 epilogue 放回各自 softmax WG，CTA 从 512 降到 384 threads；PFULL 同时表示 P 与必要 O correction 完成。初始 register budget 和 staging 随角色重构一起调整，不能把性能变化归因于线程数一个变量。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v7 → v12](from_v7.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:86](kernel.cu#L86)：角色合并后使用 384-thread CTA 与新的 entry register 配额。
- [kernel.cu:187](kernel.cu#L187)：O correction 与 P stores 现在由同一个 owner 发布。
- [kernel.cu:203](kernel.cu#L203)：保护 causal 不同退出时刻的共享内存复用。

## 正确性与同步必须守住什么

causal 下 stage0 可能先结束；它复用 Q/K SMEM 作输出前，stage1 必须真正结束。不能用等待未来 parity 代替未来 epoch 的完成事件。

## 如何读 profile 和性能结果

non-causal 小幅改善，causal 并非所有 case 都改善。第一次实现暴露 future parity 错误，修正过程是本课重点：用一次性的 FINAL 等待全体 PV drain。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v7 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 20.069 | 10.640 | 23.245 / 1.158× |
| 8192 | False | 446.259 | 202.405 | 451.902 / 1.013× |
| 4096 | True | 132.882 | 53.230 | 133.307 / 1.003× |
| 8192 | True | 400.469 | 104.394 | 400.251 / 0.999× |

[原版实验说明](../../v12/README.md) · [NCU 诊断](../../v12/profile/ncu_details.txt) · [NCU CSV](../../v12/profile/ncu_raw.csv) · [编译与 spill](../../v12/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

画出 stage0 先退出、stage1 仍在读 Q1 的时间线，解释错误的提前 epilogue 会覆盖什么。

<details>
<summary>参考思路</summary>

stage0 staging 会复用仍被 stage1 QK 读取的共享内存。FINAL 在所有 PV 提交之后 commit，两个 owners 等其完成后再复用。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v7,v12 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v12 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
