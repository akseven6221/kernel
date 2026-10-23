# v51：依据 CTA 数量和 SM 数量选择路径

从基础学习本版：[23 课学习路线](../tutorial/README.md)；只读最终组合：[第 23 课](../tutorial/v23/README.md)。

组合完整 v44（双 Q-stage）与 v50（单 Q-stage），均为本目录 CUDA/PTX 实现。
创建 context 时读取实际 SM 数量；launch 只根据 shape、causal 和 SM 数量选择一个 kernel：

- 单 Q-stage 的 CTA 数 ceil(S/128)×B×H ≤ SM 数，选 v50。
- causal 且 S≥2048，CTA 数 ≤2×SM 数，也选 v50。
- 其余选 v44。

策略来自额外 30-shape sweep（S=512/1024/2048/4096/8192、H=2/8/32、双 causal 模式），
并非针对固定的 8 个基准 case 写查找表。两条路径都完整计算相同 attention，选择不读取 tensor 内容、
不缓存输出，也不调用官方实现。CUDA Graph 捕获的是选定的单次 kernel launch。
v50 只比 sweep 中的 v43 多了 epilogue pitch 修正；整合后再次进行 30-shape sweep 与独立 stress/sanitizer。

补充证据：[30-shape sweep](../runs/r2_v51_dispatch_sweep/results.json)、
[120 项 FP32 压力检查](../runs/r2_v51_stress/results.json)、
[small sanitizer](../runs/r2_v51_sanitize_small/)、[large sanitizer](../runs/r2_v51_sanitize_large/)。
除下方默认 4096 non-causal profile 外，另保存 [512 small-path NCU](profile_small/profile.ncu-rep)
和 [8192 causal large-path NCU](profile_causal/profile.ncu-rep)，均附源码、编译输出及文本/CSV。



## 第二轮最终统一复测

父版本：v44+v50；取舍：推荐入口。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 12.038 | 10.731 | 89.14 |
| 1,1024,8,128 | False | 19.413 | 16.304 | 83.99 |
| 1,4096,8,128 | False | 73.442 | 53.210 | 72.45 |
| 1,8192,8,128 | False | 276.085 | 203.664 | 73.77 |
| 1,4096,8,128 | True | 66.978 | 50.741 | 75.76 |
| 1,8192,8,128 | True | 142.614 | 105.736 | 74.14 |
| 2,4096,16,128 | False | 298.672 | 217.224 | 72.73 |
| 2,4096,16,128 | True | 171.994 | 121.982 | 70.92 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "84.064000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "57.360414",
  "sm__warps_active.avg.per_cycle_active": "9.968734",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
