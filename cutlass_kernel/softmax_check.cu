// 独立 CUDA 检查程序，供 compute-sanitizer 使用，不启动 PyTorch。
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
extern "C" int softmax_bf16(const void*, void*, int, int, void*, int);
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void test(int rows, int cols, int threads, bool unaligned) {
  const size_t count = size_t(rows) * cols;
  std::vector<__nv_bfloat16> x(count), y(count);
  for (size_t i = 0; i < count; ++i) x[i] = __float2bfloat16(float(int(i % 83) - 41) / 5.f);
  __nv_bfloat16 *a_base, *b_base;
  check(cudaMalloc(&a_base, (count + 1) * sizeof(*a_base)));
  check(cudaMalloc(&b_base, (count + 1) * sizeof(*b_base)));
  auto* a = a_base + int(unaligned);
  auto* b = b_base + int(unaligned);
  check(cudaMemcpy(a, x.data(), count * 2, cudaMemcpyHostToDevice));
  check(cudaMemset(b, 0xff, count * 2));
  check(cudaError_t(softmax_bf16(a, b, rows, cols, nullptr, threads)));
  check(cudaMemcpy(y.data(), b, count * 2, cudaMemcpyDeviceToHost));
  for (int row = 0; row < rows; ++row) {
    double maximum = -INFINITY;
    for (int col = 0; col < cols; ++col) maximum = fmax(maximum, __bfloat162float(x[size_t(row) * cols + col]));
    double sum = 0;
    for (int col = 0; col < cols; ++col) sum += exp(__bfloat162float(x[size_t(row) * cols + col]) - maximum);
    for (int col = 0; col < cols; ++col) {
      const size_t index = size_t(row) * cols + col;
      const float expected = __bfloat162float(__float2bfloat16(float(exp(__bfloat162float(x[index]) - maximum) / sum)));
      const float actual = __bfloat162float(y[index]);
      if (!std::isfinite(actual) || fabs(actual - expected) > 2e-7 + 0.008 * fabs(expected))
        throw std::runtime_error("softmax CPU reference mismatch");
    }
  }
  check(cudaFree(a_base)); check(cudaFree(b_base));
  std::printf("PASS rows=%d cols=%d threads=%d unaligned=%d\n", rows, cols, threads, unaligned);
}
int main() {
  try {
    for (int threads : {64, 128, 256}) test(13, 4096, threads, false);
    test(7, 17, 0, false); test(7, 4095, 0, false);
    test(7, 4096, 0, true); test(2, 32769, 0, false);
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
