#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// descriptor(address, rows) 把 shared 内存的起始地址和读取步长打包成 64 位整数，告诉 MMA 数据在哪里、怎样读取。
// instruction_descriptor 把 A/B 类型、结果类型和计算块的 M/N 尺寸打包成 32 位整数，告诉 MMA 用什么类型、算多大的一块。

namespace gemm_v1{

int BLOCK_M = 64;
int BLOCK_N = 256;
int BLOCK_K = 64;
int THREADS = 128;
int TMEM_COLUMNS = 512;
int A_BYTES = BLOCK_M * BLOCK_K * 2;
int B_BYTES = BLOCK_N * BLOCK_K * 2;
int SHARED_BYTES = A_BYTES + B_BYTES + 24;

__device__ __forceinline__ void wait_barrier(uint32_t address, uint32_t phase) {
  asm volatile(
      "{ .reg .pred ready; wait_loop: "
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1; "
      "@!ready bra wait_loop; }"
      :: "r"(address), "r"(phase) : "memory");
}


// 为啥这个函数参数这里CUtensorMap前面要加一个const
__device__ void load_tile(uint32_t destination, const CUtensorMap* tensor_map,
    int row, int reduction_tile, uint32_t barrier) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        "[%0], [%1, {0, %2, %3}], [%4]"
        :: "r"(destination), "l"(tensor_map), "r"(row),
        "r"(reduction_tile), "r"(barrier) : "memory"
    );
}

__device__ __forceinline__ uint64_t descriptor(uint32_t address, int rows) {
    return uint64_t(address >> 4) |
    (uint64_t(rows) << 16) |
    (uint64_t(8) << 32) |
    (uint64_t(1) << 46);
}

__global__ __launch_bounds__(THREADS) void gemm(const __grid_constant__ CUtensorMap a_map,
          const __grid_constant__ CUtensorMap b_map,
          __nv_bfloat16* output, int columns, int reduction_iterations) {
    // 声明一块大小由 kernel 启动时指定、起始地址按 128 字节对齐的 shared memory，并用字节数组访问它。
    // extern： 声明在此，他处定义
    // __align__(128)：数组起始地址要是 128 的整数倍
    extern __shared__ __align__(128) unsigned char shared[];
    uint32_t shared_base = uint32_t(__cvta_generic_to_shared(shared));
    uint32_t tma_barrier = shared_base + A_BYTES + B_BYTES;
    uint32_t mma_barrier = tma_barrier + 8;
    uint32_t allocation_address = mma_barrier + 8;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        asm volatile("mabrrier.init.shared::cta.b64 [%0], 1;" :: "r"(tma_barrier));
        asm volatile("mabrrier.init.shared::cta.b64 [%0], 1;" :: "r"(mma_barrier));
        asm volatile("fence.mabrrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(allocation_address), "r"(TMEM_COLUMNS) : "memory"
        );
    }
    uint32_t tmem = *reinterpret_cast<uint32_t*>(shared + A_BYTES + B_BYTES + 16);

    uint32_t phase = 0;

    __syncthreads();
    // N 以 8 列为单位，M 以 16 行为单位。 文档具体规定：N 省略最低 3 位，M 省略最低 4 位，因此分别相当于除以 8、除以 16。
    uint32_t instruction_descriptor = (1u << 4) | (1u << 7) | (1u << 10) | ((BLOCK_N / 8) << 17) | ((BLOCK_M / 16) << 24);

    for (int iteration = 0; iteration < 16; ++iteration) {
        // tma
        if (threadIdx.x == 0) {
            asm volatile("mabrrier.arrive.shared::cta.b64 [%0], %1;"
            :: "r"(tma_barrier), "r"(A_BYTES + B_BYTES) : "memory");
            load_tile(shared_base, &a_map, blockIdx.y * BLOCK_M, iteration * 8, tma_barrier);
            load_tile(shared_base + A_BYTES, &b_map, blockIdx.y * BLOCK_M, iteration * 8, tma_barrier);
        }

        wait_barrier(tma_barrier, phase);
        asm volatile("mabrrier.fence::after_thread_sync;" ::: "memory");
        // mma
        if (threadIdx.x == 0) {
            for (int step = 0; step < BLOCK_K / 16; ++step) {
                const uint64_t a_descriptor = descriptor(shared_base + step * BLOCK_M * 32, BLOCK_M);
                const uint64_t b_descriptor = descriptor(shared_base + A_BYTES + step * BLOCK_N * 32, BLOCK_N);
                const uint32_t accumulate = iteration != 0 || step != 0;
                asm volatile(
                    "{ .reg .pred accumulate; setp.ne.u32 accumulate, %4, 0; "
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate; }"
                    :: "r"(tmem), "l"(a_descriptor), "l"(b_descriptor),
                    "r"(instruction_descriptor), "r"(accumulate) : "memory");
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mma_barrier) : "memory"
            );
        }
        wait_barrier(mma_barrier, phase);
        phase ^= 1;
    }

    asm volatile("tcgen05.fence::after_thread_syn c;" ::: "memory");
    float fragment[128];
    uint32_t read_address = tmem + (uint32_t(warp * 32) << 16);
    asm volatile(
        ""
    );
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    __syncthreads();

    if (warp == 0) {
        // 释放tmem
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem), "r"(TMEM_COLUMNS) : "memory");
    }

    for (int column_tile = 0; column_tile < BLOCK_N / 8; ++column_tile) {
        for (int row_tile = 0; row_tile < 2; ++row_tile) {
            int row = blockIdx.y * BLOCK_M + warp * 16 + lane % 4 + row_tile * 8;
            int column = blockIdx.x * BLOCK_N + column_tile * 8 + (lane % 4) * 2;
            int fragment_index = column_tile * 4 + row_tile * 2;
            *reinterpret_cast<__nv_bfloat162*>(output + int64_t(row) * columns + column) = 
            __float22bfloat162_rn(make_float2(fragment[fragment_index], fragment[fragment_index + 1]));
        }
    }
}

CUresult encode_tensor_map(CUtensorMap* tensor_map, __nv_bfloat16* pointer,
    int row, int reduction, int tile_rows) {
    uint64_t dimensions[] = {uint64_t(reduction), uint64_t(row)};
    uint64_t strides[] = {uint64_t(reduction) * 2};
    uint32_t box[] = {uint32_t(BLOCK_K), uint32_t(tile_rows)};
    uint32_t element_strides[] = {};
    return cuTensorMapEncodeTiled(
    tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
      dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map,
            __nv_bfloat16* output, int rows, int columns, int reduction,
            cudaStream_t stream) {
  gemm<<<dim3(columns / BLOCK_N, rows / BLOCK_M), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, output, columns, reduction / BLOCK_K);
}
}