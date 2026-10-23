# v49：单 Q-stage 的 224/24 register 预算

父版本 v43，仍为 256 threads / 97 KiB / 256 TMEM columns，调整软件与 metadata 的寄存器分配。
软件需持有 128 scores 与 packed P；希望减少 spill，并保留短序列/小 causal grid 的优势。



## 第二轮最终统一复测

父版本：v43；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 12.722 | 10.731 | 84.35 |
| 1,1024,8,128 | False | 20.528 | 16.304 | 79.42 |
| 1,4096,8,128 | False | 133.682 | 53.210 | 39.80 |
| 1,8192,8,128 | False | 518.147 | 203.664 | 39.31 |
| 1,4096,8,128 | True | 71.731 | 50.741 | 70.74 |
| 1,8192,8,128 | True | 259.400 | 105.736 | 40.76 |
| 2,4096,16,128 | False | 488.181 | 217.224 | 44.50 |
| 2,4096,16,128 | True | 292.658 | 121.982 | 41.68 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "146.144000",
  "launch__registers_per_thread": "128",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "36.718293",
  "sm__warps_active.avg.per_cycle_active": "5.993607",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "264704",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "137472",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
