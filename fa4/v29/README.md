# v29：拆分 softmax 指令阶段

父版本 v27。将逐 pair 的 FMA→EX2→sum→convert 改为四个连续阶段，
用 packed FP32 FMA 原地生成指数参数，提供更多独立 EX2 指令，提前归约后再转换 FP16。
保留极大 reference 的先减再乘分支，避免指数公共偏移导致溢出。
目的是改善 SFU 调度与寄存器生存期；对比 ptxas spill 和 NCU。



## 第二轮最终统一复测

父版本：v27；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.304 | 10.731 | 70.12 |
| 1,1024,8,128 | False | 23.853 | 16.304 | 68.35 |
| 1,4096,8,128 | False | 76.637 | 53.210 | 69.43 |
| 1,8192,8,128 | False | 289.933 | 203.664 | 70.25 |
| 1,4096,8,128 | True | 78.150 | 50.741 | 64.93 |
| 1,8192,8,128 | True | 147.690 | 105.736 | 71.59 |
| 2,4096,16,128 | False | 305.730 | 217.224 | 71.05 |
| 2,4096,16,128 | True | 168.173 | 121.982 | 72.53 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "87.520000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "52.131901",
  "sm__warps_active.avg.per_cycle_active": "9.942475",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
