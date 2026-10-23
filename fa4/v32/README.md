# v32：分片写出 P，缩短 packed 值的生存期

父版本 v27，独立 softmax 实验。每次处理 32 个 scores，写出 16 个 packed words，
四次 TMEM store 代替保持 64 个 packed words 后一次写出。
目的是减少寄存器压力，让 P store 与后面的 exponent 计算重叠；PFULL 仍等全部 store 完成。
仍保留 v27 的显式 PVDONE 同步，便于独立测量分片策略。



## 第二轮最终统一复测

父版本：v27；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 19.552 | 10.731 | 54.89 |
| 1,1024,8,128 | False | 32.296 | 16.304 | 50.48 |
| 1,4096,8,128 | False | 109.664 | 53.210 | 48.52 |
| 1,8192,8,128 | False | 425.810 | 203.664 | 47.83 |
| 1,4096,8,128 | True | 111.643 | 50.741 | 45.45 |
| 1,8192,8,128 | True | 216.109 | 105.736 | 48.93 |
| 2,4096,16,128 | False | 442.091 | 217.224 | 49.14 |
| 2,4096,16,128 | True | 232.034 | 121.982 | 52.57 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "120.640000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "36.243796",
  "sm__warps_active.avg.per_cycle_active": "9.960360",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
