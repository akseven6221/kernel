# BF16 CuTe softmax：重点优化 4096 列

[softmax.cu](softmax.cu) 是手写 CUDA C++ kernel，使用 CUTLASS 内的 CuTe Tensor 与 Copy_Atom。
默认目标为连续的 BF16 `[4096,4096]`，逐行计算：

```text
m = max(x)
y[i] = exp(x[i] - m) / sum(exp(x - m))
```

BF16 输入/输出，max、exp、sum、除法都在 FP32 中完成。
先减最大值，避免大正数使 exp 溢出。不是调用完整 CUTLASS 算子，也没有手写 PTX。

## 怎么写出这个 kernel

Softmax 每个元素只需读一次、写一次，而且同一行的所有元素共享 max 和 sum。
因此先选“一行一个 CTA”，再把每行输入保留在寄存器中：加载、求 max、计算 exp、
求 sum、归一化、写回都在同一个 kernel 里，避免额外的中间矩阵与 kernel launch。

4096 列试验了 64、128、256 线程三个配置。
以 128 线程为例，每线程持有 `4096/128=32` 个 FP32 值，每次加载 8 个 BF16，
循环四次就加载完。相邻线程处理相邻的 16B，访问连续且合并。

```cpp
auto global = cute::make_tensor(
    cute::make_gmem_ptr(input + row * 4096 + column),
    cute::make_layout(cute::Int<8>{}));
auto packed = cute::make_tensor<cutlass::bfloat16_t>(
    cute::make_layout(cute::Int<8>{}));
cute::copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>,
                          cutlass::bfloat16_t>{}, global, packed);
```

- `make_gmem_ptr` 标记数据在 global memory。
- 第一种 `make_tensor` 是指针加布局的 view，不分配内存。
- 第二种 `make_tensor<T>` 创建静态大小的 owning tensor，编译器将小数组放进寄存器。
- 8 个 BF16 恰好 128 bits，因此选择 `UniversalCopy<uint128_t>`。
  这是完整的向量 copy atom，不是八次独立的标量搬运。
- 加载后转为 FP32，保存在 `values` tensor 中；写回时再转 BF16。

归约分三层：每线程先处理自己的寄存器值；warp 内用 shuffle；warp 的 lane 0
将结果写进 shared，再由第一个 warp 合并。这里 shuffle/CTA 同步使用 CUDA intrinsics，
CuTe 负责数据组织与搬运，没有把同步藏在一个不透明的完整算子里。

max 与 sum 使用两组独立 shared 槽位。这样快 warp 开始 sum 归约时，
不会覆盖慢 warp 尚未读取的 max 结果；只需每次归约的两次 CTA 同步。

4096 列的专用模板 `FullTile=true` 让列数成为编译时常量，消除 padding/tail 判断。
这也是为什么不只写一个完全动态、所有尺寸都走同一条路径的 kernel。

## 使用方式与范围

```cpp
#include "softmax.cu"

// input/output: 独立的连续 CUDA BF16 buffer，类型为 cutlass::bfloat16_t*。
// stream: 调用方提供的 CUDA stream；函数不分配 buffer，不同步 stream。
cudaError_t status = cute_softmax::launch(input, output, 4096, 4096, stream);
```

[softmax_binding.cu](softmax_binding.cu) 提供给 Python benchmark 用的轻量 C ABI，
用 ctypes 调用，不需要编译 PyTorch C++ 扩展。

重点验证 BF16；kernel 内部用模板组织 dtype，但未对其他 dtype 作性能承诺。
要求二维连续输入/输出，输出 dtype 与输入相同；不支持两个 buffer 重叠。
`rows=0` 可直接返回；负 rows 或非正 cols 返回 `cudaErrorInvalidValue`。

非 2 次幂与未对齐指针有带边界检查的标量路径。
短行（不超过 2048 列）采用 warp/subwarp 处理，多行共用一个 CTA；
长行用 CTA 归约。超过 32768 列有三遍读输入的 streaming fallback，限制寄存器用量。
这些是通用性支持，性能重点仍为 BF16 4096 列。

支持有限值、部分 `-inf` mask；全 `-inf`、含 `+inf` 或 NaN 的行输出 NaN，
与所比较的 PyTorch softmax 一致，不将全 mask 行改成全 0。
使用 `__expf` 与 FP32 归约，测试允许 BF16 舍入附近的小误差，不要求位级一致。

## B200 验证与 benchmark

2026-10-07（北京时间），NVIDIA B200，CUDA 12.8.1 / nvcc 12.8.93，
PyTorch 2.8.0+cu128。BF16 `4096×4096`，逐行 softmax。

| 工作集 | CuTe 128 线程 | PyTorch eager CUDA softmax | 加速比 |
| --- | ---: | ---: | ---: |
| 重复同一 buffer | 10.820 µs | 48.753 µs | 4.51× |
| 轮换 8 对 buffer，512 MiB | 13.364 µs | 51.103 µs | 3.82× |

这里的 PyTorch 基线为内置 CUDA `torch.softmax`，没有使用 `torch.compile` 或 Triton。
结果只说明此 BF16 4096² 场景、上述版本与计时方法下的比较，不代表胜过所有 softmax 实现。

配置对比，默认选择在两种工作集下均最快的 **128 线程**：

| 每行线程数 | 重复 buffer | 轮换 buffer | 每线程寄存器 | shared | spill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 11.041 µs | 13.882 µs | 80 | 16 B | 0 |
| 128 | 10.820 µs | 13.364 µs | 48 | 32 B | 0 |
| 256 | 11.515 µs | 14.044 µs | 32 | 64 B | 0 |

生成 PTX 确认 128 线程专用路径每线程有四条 `ld.global.nc.v2.u64` 与四条
`st.global.v2.u64`，每条 16B，符合四次加载/四次写回 32 个 BF16 的设计。
32 个输入转为 FP32 后留在寄存器中；不重复读取输入，也没有全局中间矩阵。

正确性与检查：19 个 GPU FP64 参考用例全部通过，4096² 测试中最大绝对误差
为 `7.62939453125e-06`（与 FP64 结果舍入成 BF16 后比较）。
检查阈值 `rtol=0.008,atol=2e-7`，每行输出和误差小于 0.008。
三种线程配置、尾部列、未对齐指针、超长行均通过独立 CPU double 参考；
B200 的 memcheck、synccheck 为 0 errors，racecheck 为 0 hazards / 0 errors / 0 warnings。

[完整结果、样本与日志](results_softmax_bf16_b200_4096x4096.json)，
[生成的 PTX](softmax.sm100a.ptx)，
[Modal 运行记录](https://modal.com/apps/akseven6221/main/ap-l7HbME1WeoDjB3ltvDfiT9)。
结果中的三份源码 SHA256 已与本地文件核对一致。


[run_softmax_modal.py](run_softmax_modal.py) 使用 CUDA 12.8.1、CUTLASS 4.2.1 固定 commit，
`nvcc -std=c++17 -O3 -DNDEBUG -lineinfo -arch=sm_100a`，对比
PyTorch 2.8.0+cu128 的 `torch.softmax(x,dim=-1,out=output)`，双方预分配输出。

CUDA Graph 每次包含 100 次 softmax，以 CUDA events 计时。
预热后测量 9 轮，轮换测试顺序，报告每次 softmax 的中位耗时。
Python、ctypes 与分配开销不计入这个 GPU 时间。

同时测两种工作集：

- 重复同一输入/输出 buffer，可能命中 B200 L2。
- 轮换 8 对输入/输出 buffer，4096² 时共 512 MiB，避免把完全缓存驻留的时间当显存吞吐量。

有效带宽按 `2B input + 2B output` 计算，每次 64 MiB。这是算法有效流量，
不代表测量到的物理 HBM 事务量，尤其不能把缓存工作集的数字当成 HBM 带宽。

正确性对照来自实际 BF16 输入的 GPU FP64 softmax，再舍入成 BF16，
也检查输出非负、每行和接近 1。覆盖随机值、大 logits、常数行、非整齐列数、
未对齐指针、partial/all mask、NaN/inf。独立的 [softmax_check.cu](softmax_check.cu)
使用 CPU double 参考并供 compute-sanitizer 检查。

```bash
.venv/bin/modal run kernel/cutlass_kernel/run_softmax_modal.py --sanitize
```

官方接口参考：[CuTe copy atom](https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/atom/copy_atom.hpp)、
[UniversalCopy](https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/copy.hpp)、
[Tensor 的 owning/view 构造](https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/tensor_impl.hpp)。
