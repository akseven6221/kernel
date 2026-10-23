#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

namespace v5 {

// 对应 cuda_kernel/v6.cu：双 CTA 流水线，两个 128×32 的输出缓冲交替使用。
constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 64;
constexpr int MMA_M = 256;
constexpr int THREADS = 192;
constexpr int TMEM_COLUMNS = 512;
// 与 PTX 基线共用编译选项：256 对应 6 个 A/B stage，128 对应 8 个。
#ifndef MOJO_V6_MMA_N
#define MOJO_V6_MMA_N 256
#endif
constexpr int MMA_N = MOJO_V6_MMA_N;
static_assert(MMA_N == 128 || MMA_N == 256);
constexpr int BLOCK_N = MMA_N / 2;
constexpr int TMA_N = 32;
constexpr int C_STAGE_BYTES = BLOCK_M * TMA_N * 2;
constexpr int C_BYTES = 2 * C_STAGE_BYTES;

constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr int AB_STAGES = (233472 - 1024 - C_BYTES - 12) / (A_BYTES + B_BYTES + 16);
constexpr int B_OFFSET = AB_STAGES * A_BYTES;
constexpr int C_OFFSET = B_OFFSET + AB_STAGES * B_BYTES;
constexpr int TMA_BARRIER = C_OFFSET + C_BYTES;
constexpr int MMA_BARRIER = TMA_BARRIER + AB_STAGES * 8;
constexpr int COMPUTE_BARRIER = MMA_BARRIER + AB_STAGES * 8;
constexpr int TMEM_ADDRESS = COMPUTE_BARRIER + 8;
constexpr int SHARED_BYTES = TMEM_ADDRESS + 4;

// 每绕完一圈翻转 phase；producer 从 phase=1 开始首次使用空槽。
template <int Stages>
struct Pipeline {
  int index = 0;
  uint32_t phase = 0;
  __device__ void step() {
    if (++index == Stages) {
      index = 0;
      phase ^= 1;
    }
  }
};

using Mma = cute::SM100_MMA_F16BF16_2x1SM_SS<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float, MMA_M, MMA_N,
    cute::UMMA::Major::K, cute::UMMA::Major::K>;

__device__ __forceinline__ uint64_t descriptor(uint32_t address) {
  cute::UMMA::SmemDescriptor desc{};
  desc.start_address_ = address >> 4;
  desc.stride_byte_offset_ = 1024 / 16;
  desc.version_ = 1;
  desc.layout_type_ = uint8_t(cute::UMMA::LayoutType::SWIZZLE_128B);
  return desc;
}

__device__ __forceinline__ void load_operand(
    void* destination, const CUtensorMap* tensor_map, int row,
    int reduction_tile, uint64_t* barrier, uint32_t rank) {
  // 两个 CTA 各加载自己的 A/B；mask 只包含自己，完成字节都记到 CTA 0。
  // 2SM 封装会清除 barrier 地址的 peer bit，代替原版的 mapa(..., 0)。
  cute::SM100_TMA_2SM_LOAD_MULTICAST_2D::copy(
      tensor_map, barrier, uint16_t(1u << rank),
      uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
      destination, reduction_tile * BLOCK_K, row);
}

__device__ __forceinline__ void load_fragment(
    uint32_t address, float (&fragment)[16]) {
  using Load = cute::SM100_TMEM_LOAD_16dp256b4x;
  using Traits = cute::Copy_Traits<Load>;
  auto layout = cute::group<0, 2>(cute::upcast<32>(Traits::ValID{}));
  auto src = cute::make_tensor(cute::make_tmem_ptr<float>(address), layout);
  auto dst = cute::make_tensor(fragment + 0, cute::make_layout(cute::Int<16>{}));
  cute::Copy_Atom<Load, float>{}.call(src, dst);
}

__device__ __forceinline__ void store_fragment(
    void* destination, const float (&fragment)[16], int lane) {
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) {
    uint32_t packed[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int index = tile * 8 + pair * 2;
      const __nv_bfloat162 value = __float22bfloat162_rn(
          make_float2(fragment[index], fragment[index + 1]));
      const __nv_bfloat162_raw raw = value;
      packed[pair] = uint32_t(raw.x) | (uint32_t(raw.y) << 16);
    }
    // 输出每行 32 个 BF16，采用 64B swizzle；A/B 仍然是 128B swizzle。
    const uint32_t offset = ((lane & 15) * TMA_N + (lane >> 4) * 8 + tile * 16) * 2;
    auto* address = static_cast<unsigned char*>(destination) +
        (offset ^ ((offset >> 3) & 0x30));
    cute::SM90_U32x4_STSM_N::copy(
        packed[0], packed[1], packed[2], packed[3],
        *reinterpret_cast<cute::uint128_t*>(address));
  }
}

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS)
void gemm(const __grid_constant__ CUtensorMap a_map,
          const __grid_constant__ CUtensorMap b_map,
          const __grid_constant__ CUtensorMap c_map, int reduction_iterations) {
  extern __shared__ __align__(1024) unsigned char shared[];
  const uint32_t base = cute::cast_smem_ptr_to_uint(shared);
  auto* tma_barriers = reinterpret_cast<uint64_t*>(shared + TMA_BARRIER);
  auto* mma_barriers = reinterpret_cast<uint64_t*>(shared + MMA_BARRIER);
  auto* compute_barrier = reinterpret_cast<uint64_t*>(shared + COMPUTE_BARRIER);
  auto* allocation_address = reinterpret_cast<uint32_t*>(shared + TMEM_ADDRESS);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const uint32_t rank = cute::block_rank_in_cluster();
  cute::TMEM::Allocator2Sm allocator;

  cute::cluster_sync();
  if (warp == 0) {
    allocator.allocate(TMEM_COLUMNS, allocation_address);
  }
  __syncthreads();
  if (warp == 0 && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&a_map);
    cute::prefetch_tma_descriptor(&b_map);
    cute::prefetch_tma_descriptor(&c_map);
#pragma unroll
    for (int stage = 0; stage < AB_STAGES; ++stage) {
      cutlass::arch::ClusterBarrier::init(tma_barriers + stage, 1);
      cutlass::arch::ClusterBarrier::init(mma_barriers + stage, 1);
    }
    cutlass::arch::ClusterBarrier::init(compute_barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  cute::cluster_sync();
  const uint32_t tmem = *allocation_address;
  const uint64_t instruction = cute::UMMA::make_runtime_instr_desc<
      cutlass::bfloat16_t, cutlass::bfloat16_t, float, MMA_M, MMA_N,
      cute::UMMA::Major::K, cute::UMMA::Major::K>();

  if (warp == 4) {
    // 两个 CTA 各有一个 producer：等待本地 MMA-empty，再加载本地 A/B。
    if (cute::elect_one_sync()) {
      Pipeline<AB_STAGES> producer{0, 1};
      for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
        cutlass::arch::ClusterBarrier::wait(
            mma_barriers + producer.index, producer.phase);
        if (rank == 0) {
          cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
              tma_barriers + producer.index, 2 * (A_BYTES + B_BYTES));
        }
        load_operand(shared + producer.index * A_BYTES, &a_map,
                     blockIdx.x * BLOCK_M, iteration,
                     tma_barriers + producer.index, rank);
        load_operand(shared + B_OFFSET + producer.index * B_BYTES, &b_map,
                     blockIdx.y * MMA_N + rank * BLOCK_N, iteration,
                     tma_barriers + producer.index, rank);
        producer.step();
      }
    }
  } else if (warp == 5 && rank == 0) {
    // 仅 leader 的 warp 5 发 MMA；完成通知释放两个 CTA 对应的槽位。
    Pipeline<AB_STAGES> consumer;
    for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
      cutlass::arch::ClusterBarrier::wait(
          tma_barriers + consumer.index, consumer.phase);
      // CUTLASS 4.2.1 无此 fence 的对应封装，沿用 v2/v3 的处理。
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int step = 0; step < BLOCK_K / 16; ++step) {
        const uint64_t a_descriptor = descriptor(
            base + consumer.index * A_BYTES + step * 32);
        const uint64_t b_descriptor = descriptor(
            base + B_OFFSET + consumer.index * B_BYTES + step * 32);
        const uint32_t accumulate = iteration != 0 || step != 0;
        // fma/umma_arrive 内部 elect，必须由整个 warp 调用，不能再包单 lane 条件。
        Mma::fma(a_descriptor, b_descriptor, tmem, accumulate, instruction);
      }
      cutlass::arch::umma_arrive_multicast_2x1SM(
          mma_barriers + consumer.index, uint16_t(3));
      consumer.step();
    }
    // 最终计算完成通知 multicast 给双方，让各自的输出 warp 开始读 TMEM。
    cutlass::arch::umma_arrive_multicast_2x1SM(compute_barrier, uint16_t(3));
  } else if (warp < 4) {
    cutlass::arch::ClusterBarrier::wait(compute_barrier, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
    for (int stage = 0; stage < MMA_N / TMA_N; ++stage) {
      // 每次只读 32 列：每线程 upper/lower 各 16 个 FP32，降低寄存器占用。
      float upper[16], lower[16];
      const uint32_t address = tmem + stage * TMA_N + (uint32_t(warp * 32) << 16);
      load_fragment(address, upper);
      load_fragment(address + (16 << 16), lower);
      cutlass::arch::fence_view_async_tmem_load();
      auto* destination = shared + C_OFFSET + (stage % 2) * C_STAGE_BYTES;
      store_fragment(destination + warp * 32 * TMA_N * 2, upper, lane);
      store_fragment(destination + (warp * 32 + 16) * TMA_N * 2, lower, lane);
      cute::tma_store_fence();
      // 只有输出的 128 个线程参加；枚举重载对应原版 bar.sync 1,128。
      cutlass::arch::NamedBarrier::sync(
          128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      if (warp == 0 && lane == 0) {
        cute::SM90_TMA_STORE_2D::copy(
            &c_map, destination, int(blockIdx.y * MMA_N + stage * TMA_N),
            int(blockIdx.x * BLOCK_M));
        cute::tma_store_arrive();
        // 保持完整 store 等待语义；CuTe tma_store_wait<N>() 只有 .read 版本。
        if (stage == MMA_N / TMA_N - 1) {
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        } else {
          asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
        }
      }
      if (stage > 0 && stage < MMA_N / TMA_N - 1) {
        // wait_group 1 保证旧一组写出完成后，通知所有输出线程可复用旧缓冲。
        // stage 0 之后直接写另一个缓冲即可；最后一轮由 cluster 同步收尾。
        cutlass::arch::NamedBarrier::sync(
            128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      }
    }

  }
  // 包括 rank 1 空闲的 warp 5 在内，所有线程都必须参加收尾同步。
  cute::cluster_sync();
  if (warp == 0) {
    allocator.release_allocation_lock();
    allocator.free(tmem, TMEM_COLUMNS);
  }
  cute::cluster_sync();
}

CUresult encode_tensor_map(CUtensorMap* tensor_map, __nv_bfloat16* pointer,
                           int rows, int reduction, int tile_rows) {
  const uint64_t dimensions[] = {uint64_t(reduction), uint64_t(rows)};
  const uint64_t strides[] = {uint64_t(reduction) * 2};
  const uint32_t box[] = {uint32_t(BLOCK_K), uint32_t(tile_rows)};
  const uint32_t element_strides[] = {1, 1};
  return cuTensorMapEncodeTiled(
      tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
      dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

CUresult encode_output(CUtensorMap* tensor_map, __nv_bfloat16* pointer, int rows, int columns) {
  const uint64_t dimensions[] = {uint64_t(columns), uint64_t(rows)};
  const uint64_t strides[] = {uint64_t(columns) * 2};
  const uint32_t box[] = {TMA_N, BLOCK_M};
  const uint32_t element_strides[] = {1, 1};
  return cuTensorMapEncodeTiled(
      tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
      dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map, const CUtensorMap& c_map,
            int rows, int columns, int reduction,
            cudaStream_t stream) {
  gemm<<<dim3(rows / BLOCK_M, columns / MMA_N), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, c_map, reduction / BLOCK_K);
}

}
