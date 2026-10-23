# v50：单 Q-stage epilogue 的无冲突 pitch

父版本 v43，只把共享内存转置缓冲的 pitch 从 130 half 改为 129 half。
原读取各 lane 的 c 相差 2，130-half pitch 让 bank index 步长为 2，产生 2-way conflict；
129-half pitch 的 bank index 步长为 1。输出仍是相同 coalesced half2 global stores。
重点观察短序列的固定 epilogue 成本和 NCU shared excessive wavefronts。



## 第二轮最终统一复测

父版本：v43；取舍：小 grid 实现；性能差异较小，以 bank conflict 指标佐证。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 12.034 | 10.731 | 89.18 |
| 1,1024,8,128 | False | 19.419 | 16.304 | 83.96 |
| 1,4096,8,128 | False | 123.595 | 53.210 | 43.05 |
| 1,8192,8,128 | False | 484.781 | 203.664 | 42.01 |
| 1,4096,8,128 | True | 67.099 | 50.741 | 75.62 |
| 1,8192,8,128 | True | 241.070 | 105.736 | 43.86 |
| 2,4096,16,128 | False | 451.131 | 217.224 | 48.15 |
| 2,4096,16,128 | True | 275.701 | 121.982 | 44.24 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "136.128000",
  "launch__registers_per_thread": "128",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "31.534486",
  "sm__warps_active.avg.per_cycle_active": "5.998255",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "0",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "0",
  "derived__memory_l1_wavefronts_shared_excessive": "0"
}
```
