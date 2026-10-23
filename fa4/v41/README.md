# v41：把 rowsum 移出 P-ready 的关键路径

父版本 v31。FP32 exponent 保留在 scores 中，先转换/写 P、完成必要 O correction 并发布 PFULL，
随后才进行 row-sum reduction 与 alpha 更新。PV 不依赖 rowsum，这部分工作可以与下一段硬件计算重叠。
代价是 FP32 exponents 的更长生存期，必须检查是否增加 spill。



## 第二轮最终统一复测

父版本：v31；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.715 | 10.731 | 68.29 |
| 1,1024,8,128 | False | 24.514 | 16.304 | 66.51 |
| 1,4096,8,128 | False | 78.342 | 53.210 | 67.92 |
| 1,8192,8,128 | False | 296.099 | 203.664 | 68.78 |
| 1,4096,8,128 | True | 81.310 | 50.741 | 62.40 |
| 1,8192,8,128 | True | 153.754 | 105.736 | 68.77 |
| 2,4096,16,128 | False | 317.354 | 217.224 | 68.45 |
| 2,4096,16,128 | True | 173.200 | 121.982 | 70.43 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "90.944000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "53.278875",
  "sm__warps_active.avg.per_cycle_active": "9.963059",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
