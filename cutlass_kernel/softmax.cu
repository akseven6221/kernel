#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cstdint>

// 连续 [rows, cols]，沿最后一维 softmax，输入/输出同类型，FP32 归约。
// CuTe 管理寄存器 Tensor 与 128-bit global copy；归约/调度由本 kernel 手写。
// 无手写 PTX，无 Tensor Core：softmax 的主要工作是带宽、exp 和归约。
namespace cute_softmax {

constexpr int kMaxRegisterColumns = 32768;

template <int Width, bool IsMax>
__device__ __forceinline__ float warp_reduce(float value) {
  const unsigned mask = __activemask();
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset /= 2) {
    const float other = __shfl_xor_sync(mask, value, offset, Width);
    if constexpr (IsMax) value = fmaxf(value, other);
    else value += other;
  }
  return value;
}

template <int Threads, bool IsMax>
__device__ __forceinline__ float row_reduce(float value, float* shared) {
  if constexpr (Threads <= 32) {
    return warp_reduce<Threads, IsMax>(value);
  } else {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    value = warp_reduce<32, IsMax>(value);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
      float partial = lane < Threads / 32 ? shared[lane] : (IsMax ? -CUDART_INF_F : 0.f);
      partial = warp_reduce<32, IsMax>(partial);
      if (lane == 0) shared[0] = partial;
    }
    __syncthreads();
    return shared[0];
  }
}

// Capacity 是本行列数向上取整后的 2 次幂；有效列数仍由 cols 决定。
// 向量路径：每线程每次搬 16B；行/指针不对齐时由 host 选择 Vec=1。
// 多个短行放同一 CTA；长行使用一个 CTA，只有此时 row_reduce 才需要 shared。
template <class T, int Capacity, int ThreadsPerRow, int RowsPerBlock, int Vec, bool FullTile = false>
__global__ __launch_bounds__(ThreadsPerRow * RowsPerBlock)
void softmax_register(const T* __restrict__ input, T* __restrict__ output,
                      int rows, int columns) {
  const int cols = FullTile ? Capacity : columns;
  constexpr int Chunks = (Capacity + ThreadsPerRow * Vec - 1) / (ThreadsPerRow * Vec);
  const int row = blockIdx.x * RowsPerBlock + threadIdx.x / ThreadsPerRow;
  const int lane = threadIdx.x % ThreadsPerRow;
  if (row >= rows) return;  // 短行：整个子 warp 一起退出；长行：整个 CTA 一起退出。
  constexpr int PartialCount = ThreadsPerRow > 32 ? ThreadsPerRow / 32 : 1;
  // max/sum 用不同槽位，避免快 warp 覆盖慢 warp 尚未读取的 max 结果。
  __shared__ float partial[2 * PartialCount];

  // owning Tensor 的静态形状让编译器把值保留在寄存器中。
  auto values = cute::make_tensor<float>(cute::make_layout(cute::Int<Chunks * Vec>{}));
  float maximum = -CUDART_INF_F;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    const int column = (chunk * ThreadsPerRow + lane) * Vec;
    auto packed = cute::make_tensor<T>(cute::make_layout(cute::Int<Vec>{}));
    if (column + Vec <= cols) {
      auto global = cute::make_tensor(cute::make_gmem_ptr(input + int64_t(row) * cols + column),
                                      cute::make_layout(cute::Int<Vec>{}));
      if constexpr (Vec > 1) {
        // 选择 128-bit Atom，而不依赖编译器猜测哪些标量访问可以合并。
        cute::copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, T>{}, global, packed);
      } else {
        cute::copy(global, packed);
      }
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        const float value = float(packed(v));
        values(chunk * Vec + v) = value;
        maximum = fmaxf(maximum, value);
      }
    } else {
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        const float value = column + v < cols ? float(input[int64_t(row) * cols + column + v])
                                               : -CUDART_INF_F;
        values(chunk * Vec + v) = value;
        maximum = fmaxf(maximum, value);
      }
    }
  }
  maximum = row_reduce<ThreadsPerRow, true>(maximum, partial);
  float sum = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      const int column = (chunk * ThreadsPerRow + lane) * Vec + v;
      // 无效列 exp 必须为 0；否则全 -inf 行的 padding 也会引入 NaN。
      const float exponent = column < cols ? __expf(values(chunk * Vec + v) - maximum) : 0.f;
      values(chunk * Vec + v) = exponent;
      sum += exponent;
    }
  }
  sum = row_reduce<ThreadsPerRow, false>(sum, partial + PartialCount);
  const float reciprocal = __fdividef(1.f, sum);
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    const int column = (chunk * ThreadsPerRow + lane) * Vec;
    auto packed = cute::make_tensor<T>(cute::make_layout(cute::Int<Vec>{}));
#pragma unroll
    for (int v = 0; v < Vec; ++v) packed(v) = T(values(chunk * Vec + v) * reciprocal);
    if (column + Vec <= cols) {
      auto global = cute::make_tensor(cute::make_gmem_ptr(output + int64_t(row) * cols + column),
                                      cute::make_layout(cute::Int<Vec>{}));
      if constexpr (Vec > 1)
        cute::copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, T>{}, packed, global);
      else cute::copy(packed, global);
    } else {
#pragma unroll
      for (int v = 0; v < Vec; ++v)
        if (column + v < cols) output[int64_t(row) * cols + column + v] = packed(v);
    }
  }
}

// 超长行的有界寄存器 fallback：三次读输入，避免寄存器数组随 cols 增长而 spill。
// 此路径优先保证通用性；高性能主路径覆盖 cols <= 32768。
template <class T>
__global__ void softmax_streaming(const T* input, T* output, int cols) {
  constexpr int Threads = 256;
  __shared__ float partial[2 * (Threads / 32)];
  const int64_t base = int64_t(blockIdx.x) * cols;
  float maximum = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += Threads) maximum = fmaxf(maximum, float(input[base + col]));
  maximum = row_reduce<Threads, true>(maximum, partial);
  float sum = 0;
  for (int col = threadIdx.x; col < cols; col += Threads) sum += __expf(float(input[base + col]) - maximum);
  sum = row_reduce<Threads, false>(sum, partial + Threads / 32);
  const float reciprocal = __fdividef(1.f, sum);
  for (int col = threadIdx.x; col < cols; col += Threads)
    output[base + col] = T(__expf(float(input[base + col]) - maximum) * reciprocal);
}

template <class T, int Capacity, int Vec>
void dispatch_capacity(const T* input, T* output, int rows, int cols,
                       cudaStream_t stream, int threads_override) {
  if constexpr (Capacity <= 2048) {
    constexpr int Threads = Capacity < 32 ? Capacity : 32;
    constexpr int RowCount = 128 / Threads;
    softmax_register<T, Capacity, Threads, RowCount, Vec>
        <<< (rows + RowCount - 1) / RowCount, Threads * RowCount, 0, stream >>>(input, output, rows, cols);
  } else {
    if constexpr (Capacity <= 8192) {
      softmax_register<T, Capacity, 128, 1, Vec><<<rows, 128, 0, stream>>>(input, output, rows, cols);
    } else {
      softmax_register<T, Capacity, 256, 1, Vec><<<rows, 256, 0, stream>>>(input, output, rows, cols);
    }
  }
}

template <class T, int Vec>
cudaError_t dispatch(const T* input, T* output, int rows, int cols,
                     cudaStream_t stream, int threads_override) {
#define CUTE_SOFTMAX_CASE(C) case C: dispatch_capacity<T, C, Vec>(input, output, rows, cols, stream, threads_override); break
  int capacity = 1;
  while (capacity < cols) capacity *= 2;
  switch (capacity) {
    CUTE_SOFTMAX_CASE(1); CUTE_SOFTMAX_CASE(2); CUTE_SOFTMAX_CASE(4); CUTE_SOFTMAX_CASE(8);
    CUTE_SOFTMAX_CASE(16); CUTE_SOFTMAX_CASE(32); CUTE_SOFTMAX_CASE(64); CUTE_SOFTMAX_CASE(128);
    CUTE_SOFTMAX_CASE(256); CUTE_SOFTMAX_CASE(512); CUTE_SOFTMAX_CASE(1024); CUTE_SOFTMAX_CASE(2048);
    CUTE_SOFTMAX_CASE(4096); CUTE_SOFTMAX_CASE(8192); CUTE_SOFTMAX_CASE(16384); CUTE_SOFTMAX_CASE(32768);
    default: return cudaErrorInvalidValue;
  }
#undef CUTE_SOFTMAX_CASE
  return cudaGetLastError();
}

// 不分配内存，不同步 stream；支持任意正 cols、非 2 次幂及未对齐指针。
// input/output 应为独立且足够大的 CUDA buffer（主路径使用 restrict）。
template <class T>
cudaError_t launch(const T* input, T* output, int rows, int cols,
                   cudaStream_t stream = nullptr, int threads_override = 0) {
  if (rows < 0 || cols <= 0 || (threads_override != 0 && threads_override != 64 && threads_override != 128 && threads_override != 256))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!input || !output || input == output) return cudaErrorInvalidValue;
  if (cols > kMaxRegisterColumns) {
    softmax_streaming<T><<<rows, 256, 0, stream>>>(input, output, cols);
    return cudaGetLastError();
  }
  constexpr int Vec = 16 / sizeof(T);
  const bool aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(output) % 16 == 0) && cols % Vec == 0 && cols >= 32 * Vec;
  if (cols == 4096 && aligned) {
    // 用户指定尺寸的专用路径：编译时确定完整 4096 列，移除 padding/尾部判断。
    if (threads_override == 256)
      softmax_register<T, 4096, 256, 1, Vec, true><<<rows, 256, 0, stream>>>(input, output, rows, cols);
    else if (threads_override == 64)
      softmax_register<T, 4096, 64, 1, Vec, true><<<rows, 64, 0, stream>>>(input, output, rows, cols);
    else
      softmax_register<T, 4096, 128, 1, Vec, true><<<rows, 128, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
  }
  return aligned ? dispatch<T, Vec>(input, output, rows, cols, stream, threads_override)
                 : dispatch<T, 1>(input, output, rows, cols, stream, threads_override);
}

}  // namespace cute_softmax
