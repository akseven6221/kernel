// Copyright (c) 2026, Modular Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// CUDA C++ port of the uploaded Mojo Blackwell BF16 GEMM example.
//
// Semantics preserved from the Mojo program:
//   A: M x K, BF16, row-major
//   B: N x K, BF16, row-major (mathematically this GEMM uses B^T)
//   C: M x N, BF16, row-major, with FP32 accumulation
//
// Hardware path:
//   TMA -> shared memory -> tcgen05.mma -> Tensor Memory -> BF16 output.
//
// The Mojo epilogue uses stmatrix + shared-memory staging + TMA store.  This
// standalone version writes the TMEM epilogue directly to global memory.  This
// removes the C TMA descriptor and keeps the file self-contained, while the
// numerics and the TMA/UMMA mainloop remain Blackwell-specific.
//
// Build on a machine with CUDA 12.8+ and a datacenter Blackwell GPU:
//   nvcc -std=c++17 -O3 -arch=sm_100a 黑石矩阵乘法.cu -o blackwell_matmul -lcuda
//
// Run:
//   ./blackwell_matmul                  # correctness test, M=N=K=4096
//   ./blackwell_matmul --benchmark      # benchmark path, 100 warmup + 100 runs
//   ./blackwell_matmul 4096 4096 4096   # explicit dimensions
//   ./blackwell_matmul --benchmark 4096 4096 4096
//
// Shape restrictions mirror the original kernel:
//   correctness mode: M % 256 == 0, N % 128 == 0, K % 64 == 0
//   benchmark mode:   M % 256 == 0, N % 256 == 0, K % 64 == 0

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 6;              // 4 epilogue + 1 TMA + 1 MMA
constexpr int THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M = 128;              // per-CTA output rows
constexpr int BLOCK_K = 64;
constexpr int CTA_GROUP = 2;              // 2-SM UMMA
constexpr int MMA_M = BLOCK_M * CTA_GROUP;
constexpr int MMA_K = 16;                 // 32 bytes of BF16 along K

__host__ __device__ constexpr int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

void check_cu(CUresult status, const char* what) {
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    if (cuGetErrorString(status, &message) != CUDA_SUCCESS || message == nullptr) {
      message = "unknown CUDA driver error";
    }
    std::fprintf(stderr, "CUDA driver error at %s: %s\n", what, message);
    std::exit(EXIT_FAILURE);
  }
}

__device__ __forceinline__ uint32_t elect_one_sync() {
  uint32_t predicate = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%p;\n\t"
      "elect.sync _|%%p, %1;\n\t"
      "@%%p mov.u32 %0, 1;\n\t"
      "}"
      : "+r"(predicate)
      : "r"(0xFFFFFFFFu));
  return predicate;
}

__device__ __forceinline__ void mbarrier_init(uint32_t address, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(address), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(uint32_t address, uint32_t phase) {
  constexpr uint32_t suspend_ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "WAIT_LOOP:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@p bra.uni WAIT_DONE;\n\t"
      "bra.uni WAIT_LOOP;\n\t"
      "WAIT_DONE:\n\t"
      "}"
      ::"r"(address), "r"(phase), "r"(suspend_ticks));
}

__device__ __forceinline__ void tma_3d_gmem_to_smem(
    uint32_t destination,
    const void* tensor_map,
    int32_t x,
    int32_t y,
    int32_t z,
    uint32_t barrier_address) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
      "[%0], [%1, {%2, %3, %4}], [%5];"
      ::"r"(destination), "l"(tensor_map), "r"(x), "r"(y), "r"(z),
        "r"(barrier_address), "n"(CTA_GROUP)
      : "memory");
}

__device__ __forceinline__ void tcgen05_mma_bf16(
    uint32_t tensor_memory_address,
    uint64_t a_descriptor,
    uint64_t b_descriptor,
    uint32_t instruction_descriptor,
    uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}"
      ::"r"(tensor_memory_address), "l"(a_descriptor), "l"(b_descriptor),
        "r"(instruction_descriptor), "r"(enable_input_d), "n"(CTA_GROUP));
}

__device__ __forceinline__ constexpr uint64_t descriptor_encode(uint64_t value) {
  return (value & 0x3FFFFULL) >> 4;
}

// Shared-memory descriptor for a K-major, 128-byte-swizzled BF16 tile.
__device__ __forceinline__ uint64_t make_shared_descriptor(uint32_t address) {
  constexpr uint64_t stride_byte_offset = 8 * 128;
  return descriptor_encode(address) |
         (descriptor_encode(stride_byte_offset) << 32) |
         (1ULL << 46) |  // leading-byte-offset mode
         (2ULL << 61);   // 128-byte swizzle
}

template <int OUTPUT_N, int NUM_STAGES>
__global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(THREADS_PER_BLOCK)
void blackwell_bf16_gemm_kernel(
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    __nv_bfloat16* c,
    int m,
    int n,
    int k) {
  static_assert(BLOCK_K == 64, "This port uses the Mojo kernel's 64-wide K tile.");
  static_assert(OUTPUT_N == 128 || OUTPUT_N == 256,
                "The translated configurations use MMA_N=128 or 256.");

  const int thread_id = threadIdx.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;

  // gridDim.x counts 128-row CTAs, so each cluster covers 256 output rows.
  const int output_m = blockIdx.x * BLOCK_M;
  const int output_n = blockIdx.y * OUTPUT_N;

  uint32_t cluster_rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cluster_rank));

  extern __shared__ __align__(1024) unsigned char dynamic_smem[];
  const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(dynamic_smem));

  constexpr uint32_t A_STAGE_BYTES = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
  constexpr uint32_t B_STAGE_BYTES =
      (OUTPUT_N / CTA_GROUP) * BLOCK_K * sizeof(__nv_bfloat16);
  constexpr uint32_t STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;

  __shared__ alignas(8) uint64_t barriers[NUM_STAGES * 2 + 1];
  __shared__ uint32_t tensor_memory_address_storage;

  const uint32_t tma_barrier_base =
      static_cast<uint32_t>(__cvta_generic_to_shared(barriers));
  const uint32_t mma_barrier_base = tma_barrier_base + NUM_STAGES * 8;
  const uint32_t mainloop_barrier = mma_barrier_base + NUM_STAGES * 8;

  // Tensor-memory allocation is a warp-collective tcgen05 operation.
  if (warp_id == 0) {
    const uint32_t address_storage =
        static_cast<uint32_t>(__cvta_generic_to_shared(&tensor_memory_address_storage));
    asm volatile(
        "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
        ::"r"(address_storage), "r"(OUTPUT_N), "n"(CTA_GROUP));
  }

  __syncthreads();

  if (warp_id == 0 && elect_one_sync()) {
    for (int stage = 0; stage < NUM_STAGES; ++stage) {
      // Both CTAs report their TMA transactions to the even/leader CTA.
      mbarrier_init(tma_barrier_base + stage * 8, CTA_GROUP);
      // The single tcgen05 commit is multicast to both CTAs.
      mbarrier_init(mma_barrier_base + stage * 8, 1);
    }
    mbarrier_init(mainloop_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }

  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

  const uint32_t tensor_memory_address = tensor_memory_address_storage;

  // Instruction descriptor: FP32 accumulator, BF16 A, BF16 B.
  constexpr uint32_t INSTRUCTION_DESCRIPTOR =
      (1u << 4) |                              // D type: FP32
      (1u << 7) |                              // A type: BF16
      (1u << 10) |                             // B type: BF16
      ((uint32_t(OUTPUT_N) >> 3) << 17) |      // MMA N
      ((uint32_t(MMA_M) >> 4) << 24);          // MMA M

  const int num_iterations = k / BLOCK_K;

  auto issue_tma_load = [&](int iteration, int& phase) {
    const int stage = iteration % NUM_STAGES;

    // The initial producer phase is one, matching the Mojo PipelineState.
    mbarrier_wait(mma_barrier_base + stage * 8, phase ^ 1);
    if (stage == NUM_STAGES - 1) {
      phase ^= 1;
    }

    // Both CTAs point their TMA completion at the leader CTA's mbarrier.
    const uint32_t leader_tma_barrier =
        (tma_barrier_base + stage * 8) & 0xFEFFFFFFu;
    const uint32_t a_smem = smem_base + stage * STAGE_BYTES;
    const uint32_t b_smem = a_smem + A_STAGE_BYTES;
    const int k_offset = iteration * BLOCK_K;

    tma_3d_gmem_to_smem(
        a_smem, &a_tensor_map, 0, output_m, k_offset / 64, leader_tma_barrier);
    tma_3d_gmem_to_smem(
        b_smem,
        &b_tensor_map,
        0,
        output_n + int(cluster_rank) * (OUTPUT_N / CTA_GROUP),
        k_offset / 64,
        leader_tma_barrier);

    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
        ::"r"(leader_tma_barrier), "r"(STAGE_BYTES)
        : "memory");
  };

  auto issue_mma = [&](int iteration, int& phase) {
    const int stage = iteration % NUM_STAGES;
    mbarrier_wait(tma_barrier_base + stage * 8, phase);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (stage == NUM_STAGES - 1) {
      phase ^= 1;
    }

    const uint32_t a_smem = smem_base + stage * STAGE_BYTES;
    const uint32_t b_smem = a_smem + A_STAGE_BYTES;

    // Each tcgen05 instruction consumes 16 BF16 values along K (32 bytes).
    // The first instruction of the first iteration initializes D; all later
    // instructions accumulate into the existing TMEM accumulator.
    tcgen05_mma_bf16(
        tensor_memory_address,
        make_shared_descriptor(a_smem),
        make_shared_descriptor(b_smem),
        INSTRUCTION_DESCRIPTOR,
        iteration == 0 ? 0u : 1u);

#pragma unroll
    for (int k_slice = 1; k_slice < BLOCK_K / MMA_K; ++k_slice) {
      const uint32_t byte_offset = k_slice * 32;
      tcgen05_mma_bf16(
          tensor_memory_address,
          make_shared_descriptor(a_smem + byte_offset),
          make_shared_descriptor(b_smem + byte_offset),
          INSTRUCTION_DESCRIPTOR,
          1u);
    }

    constexpr uint16_t cluster_mask = (1u << CTA_GROUP) - 1u;
    asm volatile(
        "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        ::"r"(mma_barrier_base + stage * 8), "h"(cluster_mask), "n"(CTA_GROUP)
        : "memory");
  };

  constexpr int TMA_WARP = 4;
  constexpr int MMA_WARP = 5;

  if (warp_id == TMA_WARP && elect_one_sync()) {
    int producer_phase = 0;
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
      issue_tma_load(iteration, producer_phase);
    }
  } else if (cluster_rank == 0 && warp_id == MMA_WARP && elect_one_sync()) {
    int consumer_phase = 0;
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
      issue_mma(iteration, consumer_phase);
    }

    constexpr uint16_t cluster_mask = (1u << CTA_GROUP) - 1u;
    asm volatile(
        "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        ::"r"(mainloop_barrier), "h"(cluster_mask), "n"(CTA_GROUP)
        : "memory");
  }

  __syncthreads();
  mbarrier_wait(mainloop_barrier, 0);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  // TMEM is organized as 128 32-bit lanes per CTA.  Four warps read all 128
  // rows; each thread receives eight consecutive FP32 columns per step.
  if (warp_id < 4) {
#pragma unroll 4
    for (int column = 0; column < OUTPUT_N; column += 8) {
      float values[8];
      const uint32_t tmem_row = cluster_rank * BLOCK_M + warp_id * 32;
      const uint32_t read_address =
          tensor_memory_address + (tmem_row << 16) + column;

      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
          "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
          : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
            "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7])
          : "r"(read_address));
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

      __nv_bfloat162 packed[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        packed[i] = __float22bfloat162_rn(make_float2(values[2 * i], values[2 * i + 1]));
      }

      const int row = output_m + warp_id * 32 + lane_id;
      const int col = output_n + column;
      *reinterpret_cast<int4*>(c + int64_t(row) * n + col) =
          *reinterpret_cast<const int4*>(packed);
    }
  }

  __syncthreads();

  if (warp_id == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::%0.sync.aligned;" ::"n"(CTA_GROUP));
    asm volatile(
        "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
        ::"r"(tensor_memory_address), "r"(OUTPUT_N), "n"(CTA_GROUP));
  }
}

void encode_operand_tensor_map(
    CUtensorMap* tensor_map,
    const __nv_bfloat16* pointer,
    uint64_t rows,
    uint64_t columns,
    uint32_t box_rows) {
  static_assert(BLOCK_K % 64 == 0);

  constexpr uint32_t rank = 3;
  const uint64_t global_dim[rank] = {64, rows, columns / 64};
  const uint64_t global_strides[rank - 1] = {
      columns * sizeof(__nv_bfloat16),
      64 * sizeof(__nv_bfloat16),
  };
  const uint32_t box_dim[rank] = {64, box_rows, BLOCK_K / 64};
  const uint32_t element_strides[rank] = {1, 1, 1};

  check_cu(
      cuTensorMapEncodeTiled(
          tensor_map,
          CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          rank,
          const_cast<__nv_bfloat16*>(pointer),
          global_dim,
          global_strides,
          box_dim,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
}

template <int OUTPUT_N, int NUM_STAGES>
void launch_blackwell_gemm(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k) {
  alignas(64) CUtensorMap a_tensor_map{};
  alignas(64) CUtensorMap b_tensor_map{};

  encode_operand_tensor_map(&a_tensor_map, a, uint64_t(m), uint64_t(k), BLOCK_M);
  encode_operand_tensor_map(
      &b_tensor_map, b, uint64_t(n), uint64_t(k), OUTPUT_N / CTA_GROUP);

  constexpr uint32_t SMEM_BYTES =
      (BLOCK_M + OUTPUT_N / CTA_GROUP) * BLOCK_K * NUM_STAGES *
      uint32_t(sizeof(__nv_bfloat16));

  auto* kernel = blackwell_bf16_gemm_kernel<OUTPUT_N, NUM_STAGES>;
  check_cuda(
      cudaFuncSetAttribute(
          reinterpret_cast<const void*>(kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          SMEM_BYTES),
      "cudaFuncSetAttribute");

  const dim3 grid(m / BLOCK_M, n / OUTPUT_N, 1);
  const dim3 block(THREADS_PER_BLOCK, 1, 1);
  kernel<<<grid, block, SMEM_BYTES>>>(a_tensor_map, b_tensor_map, c, m, n, k);
}

struct Options {
  bool benchmark = false;
  int m = 4096;
  int n = 4096;
  int k = 4096;
};

Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<int> dimensions;

  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--benchmark") {
      options.benchmark = true;
    } else if (argument == "--help" || argument == "-h") {
      std::printf(
          "Usage: %s [--benchmark] [M N K]\n"
          "Defaults: M=N=K=4096\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      char* end = nullptr;
      const long value = std::strtol(argument.c_str(), &end, 10);
      if (end == argument.c_str() || *end != '\0' || value <= 0 || value > INT32_MAX) {
        std::fprintf(stderr, "Invalid argument: %s\n", argument.c_str());
        std::exit(EXIT_FAILURE);
      }
      dimensions.push_back(int(value));
    }
  }

  if (!dimensions.empty() && dimensions.size() != 3) {
    std::fprintf(stderr, "Provide either zero dimensions or exactly M N K.\n");
    std::exit(EXIT_FAILURE);
  }
  if (dimensions.size() == 3) {
    options.m = dimensions[0];
    options.n = dimensions[1];
    options.k = dimensions[2];
  }
  return options;
}

void validate_shape(const Options& options) {
  const int output_n = options.benchmark ? 256 : 128;
  if (options.m % MMA_M != 0 || options.n % output_n != 0 || options.k % BLOCK_K != 0) {
    std::fprintf(
        stderr,
        "Unsupported shape %dx%dx%d for %s mode. Required: M%%%d=0, N%%%d=0, K%%%d=0.\n",
        options.m,
        options.n,
        options.k,
        options.benchmark ? "benchmark" : "correctness",
        MMA_M,
        output_n,
        BLOCK_K);
    std::exit(EXIT_FAILURE);
  }
}

template <int OUTPUT_N, int NUM_STAGES>
int run(const Options& options) {
  const int64_t a_elements = int64_t(options.m) * options.k;
  const int64_t b_elements = int64_t(options.n) * options.k;
  const int64_t c_elements = int64_t(options.m) * options.n;

  std::vector<__nv_bfloat16> a_host(a_elements);
  std::vector<__nv_bfloat16> b_host(b_elements);
  std::vector<__nv_bfloat16> c_host(c_elements);

  // Same initialization pattern as the Mojo test.
  for (int row = 0; row < options.m; ++row) {
    for (int col = 0; col < options.k; ++col) {
      a_host[int64_t(row) * options.k + col] = __float2bfloat16_rn(float(col));
    }
  }
  for (int row = 0; row < options.n; ++row) {
    for (int col = 0; col < options.k; ++col) {
      b_host[int64_t(row) * options.k + col] =
          __float2bfloat16_rn(row == col ? 1.0f : 0.0f);
    }
  }

  __nv_bfloat16* a_device = nullptr;
  __nv_bfloat16* b_device = nullptr;
  __nv_bfloat16* c_device = nullptr;
  check_cuda(cudaMalloc(&a_device, a_elements * sizeof(__nv_bfloat16)), "cudaMalloc A");
  check_cuda(cudaMalloc(&b_device, b_elements * sizeof(__nv_bfloat16)), "cudaMalloc B");
  check_cuda(cudaMalloc(&c_device, c_elements * sizeof(__nv_bfloat16)), "cudaMalloc C");
  check_cuda(cudaMemset(c_device, 0, c_elements * sizeof(__nv_bfloat16)), "cudaMemset C");
  check_cuda(
      cudaMemcpy(a_device, a_host.data(), a_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
      "copy A to device");
  check_cuda(
      cudaMemcpy(b_device, b_host.data(), b_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
      "copy B to device");

  if (!options.benchmark) {
    launch_blackwell_gemm<OUTPUT_N, NUM_STAGES>(
        a_device, b_device, c_device, options.m, options.n, options.k);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");
    check_cuda(
        cudaMemcpy(c_host.data(), c_device, c_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
        "copy C to host");

    // B is an identity matrix.  Each output therefore selects exactly one BF16
    // value from A, making an exact host-side check possible without cuBLAS.
    double max_absolute_error = 0.0;
    int64_t first_bad_index = -1;
    float first_expected = 0.0f;
    float first_actual = 0.0f;
    for (int row = 0; row < options.m; ++row) {
      for (int col = 0; col < options.n; ++col) {
        const float expected =
            col < options.k ? __bfloat162float(a_host[int64_t(row) * options.k + col]) : 0.0f;
        const float actual = __bfloat162float(c_host[int64_t(row) * options.n + col]);
        const double error = std::fabs(double(actual) - double(expected));
        max_absolute_error = std::max(max_absolute_error, error);
        if (actual != expected && first_bad_index < 0) {
          first_bad_index = int64_t(row) * options.n + col;
          first_expected = expected;
          first_actual = actual;
        }
      }
    }

    if (first_bad_index >= 0) {
      std::fprintf(
          stderr,
          "TEST FAILED at flat index %lld: expected %.9g, got %.9g\n",
          static_cast<long long>(first_bad_index),
          first_expected,
          first_actual);
      return EXIT_FAILURE;
    }

    std::printf("%dx%dx%d  max_abs_error=%.3g\n", options.m, options.n, options.k, max_absolute_error);
    std::printf("\n=== TEST PASSED ===\n\n");
  } else {
    constexpr int NUM_WARMUP = 100;
    constexpr int NUM_RUNS = 100;

    for (int i = 0; i < NUM_WARMUP; ++i) {
      launch_blackwell_gemm<OUTPUT_N, NUM_STAGES>(
          a_device, b_device, c_device, options.m, options.n, options.k);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < NUM_RUNS; ++i) {
      launch_blackwell_gemm<OUTPUT_N, NUM_STAGES>(
          a_device, b_device, c_device, options.m, options.n, options.k);
    }
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "event synchronize");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double seconds = double(elapsed_ms) * 1.0e-3 / NUM_RUNS;
    const double tflops = 2.0 * options.m * options.n * options.k / seconds * 1.0e-12;
    std::printf("%d, %d, %d, %.6f ms, %.2f TFLOPS\n",
                options.m,
                options.n,
                options.k,
                elapsed_ms / NUM_RUNS,
                tflops);
  }

  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  validate_shape(options);

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  std::printf("Device: %s (compute capability %d.%d)\n",
              properties.name,
              properties.major,
              properties.minor);
  if (properties.major != 10) {
    std::fprintf(
        stderr,
        "This binary uses tcgen05 and must be compiled/run for datacenter Blackwell (sm_100a).\n");
    return EXIT_FAILURE;
  }

  // Match the two configurations in the Mojo main() function.
  if (options.benchmark) {
    std::printf("Benchmarking blackwell_matmul_tma_umma_kernel\n");
    std::printf("============================================\n");
    std::printf("M, N, K, time(ms), TFLOPS\n");
    return run<256, 7>(options);
  }
  return run<128, 8>(options);
}