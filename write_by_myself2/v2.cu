#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>

namespace v2 {

// 一个 CTA 算 64行×256列的输出；每轮沿 K 搬入 64 个元素。
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int THREADS = 128;
constexpr int TMEM_COLUMNS = 512;
constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
// C 分成四个 64×64 BF16 tile，每个 tile 使用 128B swizzle。
constexpr int TMA_N = 64;
constexpr int C_BYTES = BLOCK_M * BLOCK_N * 2;
constexpr int C_OFFSET = A_BYTES + B_BYTES;
// shared：[A][B][C][TMA barrier 8B][MMA barrier 8B][TMEM 地址 4B][预留 4B]。
constexpr int SHARED_BYTES = C_OFFSET + C_BYTES + 24;

using Mma = cute::SM100_MMA_F16BF16_SS<cutlass::bfloat16_t, cutlass::bfloat16_t, float, BLOCK_M, BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>;

__device__ __forceinline__ uint64_t descriptor(uint32_t address) {
    // 用具名字段替代手写位移。这里仍沿用原版布局，没有重新排布 shared。
    cute::UMMA::SmemDescriptor desc{};
    desc.start_address_ = address >> 4;    // 地址以 16B 为单位。
    desc.stride_byte_offset_ = 1024 / 16;  // 8 行 x 128B = 1024B。
    desc.version_ = 1;
    desc.layout_type_ = uint8_t(cute::UMMA::LayoutType::SWIZZLE_128B);
    return desc;
}

__global__ __launch_bounds__(THREADS)
void gemm(const __grid_constant__ CUtensorMap a_map,
          const __grid_constant__ CUtensorMap b_map,
          const __grid_constant__ CUtensorMap c_map, int reduction_iterations) {
    extern __shared__ __align__(1024) unsigned char shared[];
    uint32_t shared_base = cute::cast_smem_ptr_to_uint(shared);
    // API 要 C++ shared 指针；shared_base 是给 MMA 描述符用的地址数值。
    auto* tma_barrier = reinterpret_cast<uint64_t*>(shared + C_OFFSET + C_BYTES);
    auto* mma_barrier = tma_barrier + 1;
    // 分配器输出一个 32 位 TMEM 地址；uint64_t* 加 1 就跨过 8B barrier。
    auto* allocation_address = reinterpret_cast<uint32_t*>(mma_barrier + 1);
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    cute::TMEM::Allocator1Sm allocator;

    if (threadIdx.x == 0) {
        cutlass::arch::ClusterBarrier::init(tma_barrier, 1);
        cutlass::arch::ClusterBarrier::init(mma_barrier, 1);
        cutlass::arch::fence_barrier_init();
    }
    if (warp == 0) {
        allocator.allocate(TMEM_COLUMNS, allocation_address);
    }

    __syncthreads();
    uint32_t tmem = *allocation_address;
    const uint64_t instruction_descriptor = cute::UMMA::make_runtime_instr_desc<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    BLOCK_M, BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();

    uint32_t phase = 0;
    for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
        if (threadIdx.x == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(tma_barrier, A_BYTES + B_BYTES);
            cute::SM90_TMA_LOAD_2D::copy(&a_map, tma_barrier, uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
             shared, iteration * BLOCK_K, blockIdx.y * BLOCK_M);
            cute::SM90_TMA_LOAD_2D::copy(&b_map, tma_barrier, uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
            shared + A_BYTES, iteration * BLOCK_K, BLOCK_N * blockIdx.x);
        }
        cutlass::arch::ClusterBarrier::wait(tma_barrier, phase);  // 两次 TMA 的字节都完成后，MMA 才能使用 A/B。
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (warp == 0) {
#pragma unroll
            for (int step = 0; step < BLOCK_K / 16; ++step) {
                // 一条 MMA 消耗 16 个 BF16，即沿 K 偏移 32B。
                uint64_t a_descriptor = descriptor(shared_base + step * 32);
                uint64_t b_descriptor = descriptor(shared_base + A_BYTES + step * 32);
                uint32_t accumulate = iteration != 0 || step != 0;
                Mma::fma(a_descriptor, b_descriptor, tmem, accumulate, instruction_descriptor);
            }
            cutlass::arch::umma_arrive(mma_barrier);
        }
        cutlass::arch::ClusterBarrier::wait(mma_barrier, phase);
        phase ^= 1;
    }

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    float fragment[128];
    uint32_t read_address = tmem + (uint32_t(warp * 32) << 16);
    using TmemLoad = cute::SM100_TMEM_LOAD_16dp256b32x;
    using TmemLoadTraits = cute::Copy_Traits<TmemLoad>;
    auto src_layout = cute::group<0, 2>(cute::upcast<32>(TmemLoadTraits::ValID{}));
    auto src = cute::make_tensor(cute::make_tmem_ptr<float>(read_address), src_layout);
    auto dst = cute::make_tensor(fragment + 0, cute::make_layout(cute::Int<128>{}));
    // Atom 是一条 copy 指令的封装；call 将 dst 展开成 128 个引用参数。
    // 仍生成原版的一条 16x256b.x32，不是 128 次标量读。
    cute::Copy_Atom<TmemLoad, float>{}.call(src, dst);
    cutlass::arch::fence_view_async_tmem_load();

    // reg → shared：沿用 cuda_kernel/v3.cu 的 fragment 顺序与 stmatrix 布局。
    // 一个 output_tile 为 64×64，一个 matrix 组覆盖其中的 16 列。
#pragma unroll
    for (int output_tile = 0; output_tile < BLOCK_N / TMA_N; ++output_tile) {
#pragma unroll
        for (int matrix = 0; matrix < TMA_N / 16; ++matrix) {
            uint32_t packed[4];
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int index = output_tile * (TMA_N / 16) * 8 + matrix * 8 + pair * 2;
                const __nv_bfloat162 value = __float22bfloat162_rn(
                    make_float2(fragment[index], fragment[index + 1]));
                const __nv_bfloat162_raw raw = value;
                packed[pair] = uint32_t(raw.x) | (uint32_t(raw.y) << 16);
            }
            // stmatrix 的 lane 地址布局与原来的直接 global 写回布局不同。
            const uint32_t offset = ((warp * 16 + (lane & 15)) * TMA_N +
                                     (lane >> 4) * 8 + matrix * 16) * 2;
            const uint32_t swizzled_offset = offset ^ ((offset >> 3) & 0x70);
            auto* destination = reinterpret_cast<cute::uint128_t*>(
                shared + C_OFFSET + output_tile * BLOCK_M * TMA_N * 2 + swizzled_offset);
            cute::SM90_U32x4_STSM_N::copy(
                packed[0], packed[1], packed[2], packed[3], *destination);
        }
    }

    // 每个写 shared 的线程都向 async proxy 发布，再由 CTA barrier 汇合。
    cute::tma_store_fence();
    __syncthreads();

    // shared → global：线程 0～3 各负责一个 64×64 tile。
    if (threadIdx.x < BLOCK_N / TMA_N) {
        const void* source = shared + C_OFFSET + threadIdx.x * BLOCK_M * TMA_N * 2;
        cute::SM90_TMA_STORE_2D::copy(
            &c_map, source, int(blockIdx.x * BLOCK_N + threadIdx.x * TMA_N),
            int(blockIdx.y * BLOCK_M));
        cute::tma_store_arrive();
        // CuTe 的 tma_store_wait<0>() 仅封装 wait_group.read（shared 读取完成）。
        // 与 v3 一样等待整个 store 完成，保留不带 .read 的指令。
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }

    // 上面的 CTA barrier 已确保所有线程完成 TMEM load。
    if (warp == 0) {
        allocator.release_allocation_lock();
        allocator.free(tmem, TMEM_COLUMNS);
    }
}

CUresult encode_tensor_map(CUtensorMap* tensor_map, __nv_bfloat16* pointer,
                           int rows, int reduction, int tile_rows) {
    // 2D 坐标顺序为 {K坐标,行坐标}，维度 0 是内存连续的 K 方向。
    const uint64_t dimensions[] = {uint64_t(reduction), uint64_t(rows)};  // 整个矩阵各维元素数。
    const uint64_t strides[] = {uint64_t(reduction) * 2};                // 到下一行的字节数。
    // box[0]=K 方向覆盖 64 个元素，box[1]=覆盖 tile_rows 行；此处没有 3D 的 K 组拆分。
    // element_strides={1,1} 时，起点 (k,row) 覆盖 [k,k+63] × [row,row+tile_rows-1]。
    const uint32_t box[] = {uint32_t(BLOCK_K), uint32_t(tile_rows)};
    // 元素采样步长：K 每次 +1，行每次 +1，不跳元素；常规 GEMM 通常用 {1,1}。
    // 它与字节跨度 strides 不同；例如行采样步长 2 就会隔行取元素。
    const uint32_t element_strides[] = {1, 1};
    // 类型由 API 签名规定：全局尺寸/字节跨度用 uint64_t，局部 box/采样步长用 uint32_t。
    // 数组中的项表达数值，不是指针。
    return cuTensorMapEncodeTiled(
        tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
        dimensions, strides, box, element_strides,
        // shared 目的布局用 128B swizzle，必须匹配 descriptor() 的模式 2。
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

// 输出 map 的连续维是 N；box 必须匹配一个 64×64 shared tile。
CUresult encode_output(CUtensorMap* tensor_map, __nv_bfloat16* pointer,
                       int rows, int columns) {
    const uint64_t dimensions[] = {uint64_t(columns), uint64_t(rows)};
    const uint64_t strides[] = {uint64_t(columns) * 2};
    const uint32_t box[] = {uint32_t(TMA_N), uint32_t(BLOCK_M)};
    const uint32_t element_strides[] = {1, 1};
    return cuTensorMapEncodeTiled(
        tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
        dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map, const CUtensorMap& c_map,
            int rows, int columns, int reduction,
            cudaStream_t stream) {
  // grid.x 切 N、grid.y 切 M；K/64 为求和轮数。本实现要求 M/N/K 整除对应块大小。
  gemm<<<dim3(columns / BLOCK_N, rows / BLOCK_M), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, c_map, reduction / BLOCK_K);
}

}
