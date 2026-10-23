# v37：单次 4D TMA 加载整个 tile

父版本 v31。TensorMap 使用 Dlow/H/BS/Dhigh 和非单调 global strides，
一次 TMA 搬运两个 D=64 半块，保留原先受 MMA swizzle-atom 支持的 blocked 布局与 descriptor。
目标是减半 TMA 指令和地址准备开销。

首轮尝试 interleaved D 布局不满足 swizzled MMA atom 内的连续行规则，正确性失败
（max_abs≈0.914）；源码/结果保留在 runs/r2_v37_tma4d。当前版改为保持合法 atom 布局。



## 第二轮最终统一复测

父版本：v31；取舍：性能接近；保留布局实验。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.203 | 10.731 | 70.59 |
| 1,1024,8,128 | False | 23.378 | 16.304 | 69.74 |
| 1,4096,8,128 | False | 73.846 | 53.210 | 72.05 |
| 1,8192,8,128 | False | 280.230 | 203.664 | 72.68 |
| 1,4096,8,128 | True | 75.450 | 50.741 | 67.25 |
| 1,8192,8,128 | True | 143.336 | 105.736 | 73.77 |
| 2,4096,16,128 | False | 305.395 | 217.224 | 71.13 |
| 2,4096,16,128 | True | 162.966 | 121.982 | 74.85 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "86.976000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "57.261112",
  "sm__warps_active.avg.per_cycle_active": "9.958171",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
