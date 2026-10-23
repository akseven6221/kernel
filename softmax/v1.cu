#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void launch(uint32_t *input, uint32_t *output) {
    
}

torch::Tensor softmax(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(y.is_cuda(), "Input tensor must be on CUDA device");


}