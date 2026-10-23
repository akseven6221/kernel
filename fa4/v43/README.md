# v43：N=128 的单 Q-stage / 单 KV pair

父版本 v39。N64 的迭代次数和同步开销太大；恢复 N128，但只保留一组 K/V，
维持 97 KiB SMEM 和 256 TMEM columns 的资源目标。K 在 QK 后先释放，V 在 PV 后释放。
软件/metadata 预算改为 192/56，适应 128-score live range。



## 第二轮最终统一复测

父版本：v39；取舍：保留小 grid 路径。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 12.134 | 10.731 | 88.44 |
| 1,1024,8,128 | False | 19.621 | 16.304 | 83.10 |
| 1,4096,8,128 | False | 122.875 | 53.210 | 43.30 |
| 1,8192,8,128 | False | 479.982 | 203.664 | 42.43 |
| 1,4096,8,128 | True | 67.990 | 50.741 | 74.63 |
| 1,8192,8,128 | True | 238.501 | 105.736 | 44.33 |
| 2,4096,16,128 | False | 451.736 | 217.224 | 48.09 |
| 2,4096,16,128 | True | 261.626 | 121.982 | 46.62 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "135.328000",
  "launch__registers_per_thread": "128",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "31.476987",
  "sm__warps_active.avg.per_cycle_active": "5.996452",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "0",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "0",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
