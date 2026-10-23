# 支线与反例：其余 28 个实验

[返回路线](README.md)

主课保留 v51 的真实祖先链、预取支线和小 grid 的过渡设计。下面覆盖没有独立设课的 28 个实验。其原目录仍有源码、正确性结果和 profile；v34/v35 没有有效性能结果，保存的是失败证据。

## 流水线机制有成本

| 原版本 | 尝试 | 应带走的判断 |
| --- | --- | --- |
| [v8](../v8/README.md) | 50% polynomial exponent | 换用不同算术单元可能增加其他指令和寄存器压力，实测回退 |
| [v9](../v9/README.md) | warp arrivals | 减少 arrival 工作没有形成稳定收益，不代表所有等待都可删除 |
| [v11](../v11/README.md) | P split completion | 提前发布部分 P 同时增加交接复杂度；本实现没有稳定收益 |
| [v18](../v18/README.md) | 减为单 KV slot | 降低资源占用也减少预取空间；双 Q-stage 主线上变慢 |
| [v25](../v25/README.md) | 5-slot K/V ring | 更多 slot 不保证更好 overlap；需看实际 consumer 进度 |
| [v26](../v26/README.md) | 四 WG、split P、mixed exponent、TMA output 整合 | 适合与教程机制对照，但不是最终最快路径 |
| [v40](../v40/README.md) | 两个 software WGs 交替使用 SFU | 限制争用也可能损失并行，最终变慢 |
| [v41](../v41/README.md) | PFULL 后再求 rowsum | 提前发布与延长 live range 相互竞争，结果回退 |

建议在第 07–08、15–17 课后阅读这些分支。先画出每个 barrier 保护的边，再决定是否值得引入新的 mailbox 或阶段。

## 指令组织与 live range

| 原版本 | 尝试 | 应带走的判断 |
| --- | --- | --- |
| [v10](../v10/README.md) | 同时改为八链 max/sum | 同时增加两类状态后变慢；第 11/13 课分别缩小范围重试 |
| [v17](../v17/README.md) | 单条 x128 score load | 早期取舍未采用，但最终统一复测的部分 case 比 v16 快；不在 v51 祖先链，不应称作全面失败 |
| [v19](../v19/README.md) | packed FMA/add | 单链 packed reduction 不能代表全部 packed 设计；最终部分 case 有收益，第 18 课改为四条 packed chains |
| [v20](../v20/README.md) | pair-first max3 | 没有稳定提升；函数/指令名称不足以预测性能 |
| [v29](../v29/README.md) | 拆开 scale/EX2/sum/convert 阶段 | 某些最终 case 略快，但没有全面稳定收益；延长中间结果存活时间有代价 |
| [v30](../v30/README.md) | lengths 数组改两个标量 | 源码更“简单”却增加 compiler spill；看编译产物，不凭代码行数判断 |
| [v32](../v32/README.md) | 分四段写 P | 缩短部分 live range 与更多指令/调度开销之间需要实测，整体回退 |
| [v33](../v33/README.md) | 降到 12.5% polynomial exponent | 缩小混合比例仍没有改善；不能把 v8 失败只归因于比例过高 |
| [v38](../v38/README.md) | 统一先减后乘 | 简化数值分支却增加普通路径算术成本，未取代带 guard 的 FMA |
| [v47](../v47/README.md) | 只加入 max3 的消融 | 基本持平；v42 组合收益不能只归于 max3 |

这里特别保留“早期决定”和“最终复测”之间的差异。最终表现有小幅反转时，应读配对样本，不能为了画出漂亮学习曲线删掉反例。

## 寄存器与线程角色

| 原版本 | 尝试 | 应带走的判断 |
| --- | --- | --- |
| [v28](../v28/README.md) | v24 的 216/64 配额 | 少给 software、多给 metadata 没有带来改善 |
| [v34](../v34/README.md) | 320 threads、统一 200 registers | 启动返回 701；全 SM 总数预算不能替代 subpartition 与分配约束 |
| [v35](../v35/README.md) | 320 threads、保留动态预算 | 首输入停滞，90 秒复核超时；具体硬件原因未证实，不能当成功方案 |
| [v36](../v36/README.md) | 两个自主 QK/softmax/PV WGs | 零 compiler spill 仍比主线慢，说明 spill 不是唯一瓶颈 |
| [v45](../v45/README.md) | v44 的 232/40 配额 | 最终部分 case 更快，但不是所有 shape；保留数据，不宣称 v51 每项都最佳 |
| [v46](../v46/README.md) | v44 的 240/24 配额 | 继续增加 software 预算未继续提速 |
| [v48](../v48/README.md) | v43 的 216/32 配额 | 小 grid 路径也不能只靠增加寄存器预算改善 |
| [v49](../v49/README.md) | v43 的 224/24 配额 | 相比 v43 回退，最终保留原来的 192/56 |

## 两个单独值得看的实验

**[v22：causal remap](../v22/README.md)** 从 v16 单独验证 query-major reverse。第 14 课对应的 v23 将同一思路应用到 v21，父子 diff 更便于保持主线。v22 的独立实验有助于判断调度变化的因果。

**[v37：4D TMA](../v37/README.md)** 尝试把两次 3D TMA 合成一次 4D TMA。首版改变 D 布局，破坏了 MMA 依赖的 swizzle atom 内映射；修正后保持 blocked D 布局，性能接近 v31。学习重点是 TMA tensor map、shared 布局和 MMA descriptor 必须一致，而不是“少一次发射一定更快”。

所有历史运行和失败索引见 [MEASUREMENTS.md](../MEASUREMENTS.md)。主课第 20 课已经详细解释 v39，因此本页不重复计算为额外实验。
