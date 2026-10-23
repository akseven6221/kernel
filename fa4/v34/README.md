# v34：去掉两个空闲 metadata warps

父版本 v31。MMA/TMA 实际仅使用 WG2 的 warp0/warp1；从 384 threads 降到 320，
删除 setmaxnreg，统一编译预算设为 200（320×200=64000）。
尝试减少编译器跨角色约束及 local spill。所有 barrier 的参与者仍对应实际 software owners。

实测：launch error 701，保留失败版本。仅按总 registers 估算不足：320 threads 在四个 SM
subpartitions 上分配不均，部分 subpartition 有 3 warps，200 regs/thread 超过其容量。
原始编译、失败日志与 v35 的初始设计快照见 runs/r2_v34_v35_threads。
