# v45：软件/metadata 预算 232/40

父版本 v44，只调整动态 register 分配。完整 384-thread CTA，每 subpartition 的预算仍在 16384 内。
增加软件寄存器、减少 metadata 配额，测试相反于 v28 的预算方向；以 spill 和配对实测决定。



## 第二轮最终统一复测

父版本：v44；取舍：无全面稳定收益。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 14.768 | 10.731 | 72.67 |
| 1,1024,8,128 | False | 22.790 | 16.304 | 71.54 |
| 1,4096,8,128 | False | 71.806 | 53.210 | 74.10 |
| 1,8192,8,128 | False | 272.040 | 203.664 | 74.87 |
| 1,4096,8,128 | True | 74.982 | 50.741 | 67.67 |
| 1,8192,8,128 | True | 141.179 | 105.736 | 74.89 |
| 2,4096,16,128 | False | 291.536 | 217.224 | 74.51 |
| 2,4096,16,128 | True | 166.586 | 121.982 | 73.23 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "84.448000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "59.429769",
  "sm__warps_active.avg.per_cycle_active": "9.957931",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "36736",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "23040",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
