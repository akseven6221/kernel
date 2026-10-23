# v31：利用 MMA 读 P / 写 S 的硬件顺序

父版本 v27。删除 MMA issuer 对 PVDONE 的 commit/wait roundtrip。
同一 warp、同一 cta_group 的 MMA A/metadata reads → 后续 MMA D writes 保证顺序，
因此 PV 读取 P 后可以立即提交重用该 TMEM 的下一个 QK。
旧 PTX 8.7 文档未明确列出这一对，最新 ISA 明确列出；官方实现也直接连续提交。
依据：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-pipelined-instructions

SFULL commit 仍追踪该线程此前全部 MMA，因此软件 correction 观察 SFULL 后，上一 PV 已完成。
K/V empty、PFULL、SFULL、FINAL、EPIDONE 的跨线程同步均保留。
本版必须同时通过独立正确性与 Compute Sanitizer，再判断速度。



## 第二轮最终统一复测

父版本：v27；取舍：保留。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.248 | 10.731 | 70.38 |
| 1,1024,8,128 | False | 23.520 | 16.304 | 69.32 |
| 1,4096,8,128 | False | 73.971 | 53.210 | 71.93 |
| 1,8192,8,128 | False | 278.870 | 203.664 | 73.03 |
| 1,4096,8,128 | True | 74.683 | 50.741 | 67.94 |
| 1,8192,8,128 | True | 142.707 | 105.736 | 74.09 |
| 2,4096,16,128 | False | 295.880 | 217.224 | 73.42 |
| 2,4096,16,128 | True | 162.202 | 121.982 | 75.20 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "85.792000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "57.143804",
  "sm__warps_active.avg.per_cycle_active": "9.966518",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
