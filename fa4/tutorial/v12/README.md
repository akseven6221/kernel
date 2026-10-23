# 第 12 课：寄存器预算：为 software owner 分配资源

[路线总览](../README.md) · [上一课](../v11/README.md) · [下一课](../v13/README.md)

对应原实验 **v16**；真实父版本：[v15（第 11 课）](../v11/README.md)。课号表示阅读顺序，父版本表示代码来源。

## 本课要解决什么

softmax 持有整行 scores、packed P 与 reduction 状态；metadata WG 的需求较小。静态统一配额与动态角色配额需共同考虑。

## 改法与原理

entry __maxnreg__ 从 160 到 168，两个 softmax WGs 从 200 到 224；metadata 仍为 48。角色预算 256*224+128*48=63488，entry pool 为 384*168=64512。

## 对照代码怎么读

[完整中文导读源码](kernel.cu) · [v15 → v16](from_v15.diff)

先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。

- [kernel.cu:86](kernel.cu#L86)：入口池与软件角色的动态配额是两层概念。
- [kernel.cu:156](kernel.cu#L156)：增加 softmax 配额前需由 metadata 角色释放预算。

## 正确性与同步必须守住什么

总数检查只是必要的资源核算，还需满足 warpgroup、分配粒度和 subpartition 约束。反例 v34/v35 分别启动失败/停滞。

## 如何读 profile 和性能结果

读 build.txt 的 spill 与 NCU local load/store，再看 latency。最终结果接近 v15，不把资源预算改善夸大为明显速度提升；后续 ILP 在这套预算上继续实验。

下表均来自同一次 [final_all_versions](../../runs/final_all_versions/results.json)，B=1、H=8、D=128、FP16。
父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。

| S | causal | 本版 μs | 官方 μs | v15 μs / 父版÷本版 |
| ---: | --- | ---: | ---: | ---: |
| 512 | False | 18.536 | 10.640 | 18.526 / 0.999× |
| 8192 | False | 383.254 | 202.405 | 384.298 / 1.003× |
| 4096 | True | 102.440 | 53.230 | 102.267 / 0.998× |
| 8192 | True | 310.190 | 104.394 | 308.850 / 0.996× |

[原版实验说明](../../v16/README.md) · [NCU 诊断](../../v16/profile/ncu_details.txt) · [NCU CSV](../../v16/profile/ncu_raw.csv) · [编译与 spill](../../v16/profile/build.txt)

这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。

## 动手检查

为什么 NCU 的 launch registers=168 不代表 softmax 永远只能用 168 个寄存器？

<details>
<summary>参考思路</summary>

它反映入口编译配置；setmaxnreg 可在角色间重新分配。要结合指令、角色预算、SASS 和 spill 看实际情况。

</details>

## 复现本课原实验

仓库根目录，已配置 Modal 的环境中运行：

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v15,v16 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v16 --mode profile
```

修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。
