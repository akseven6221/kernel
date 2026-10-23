// 轻量 C ABI，只给 benchmark/ctypes 用；不依赖 PyTorch C++ 扩展。
#include "softmax.cu"
extern "C" int softmax_bf16(const void* input, void* output, int rows, int cols,
                            void* stream, int threads) {
  return int(cute_softmax::launch(
      static_cast<const cutlass::bfloat16_t*>(input),
      static_cast<cutlass::bfloat16_t*>(output), rows, cols,
      static_cast<cudaStream_t>(stream), threads));
}
