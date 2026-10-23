#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>

// v1_new.cu 的逐指令移植：保留 tile、单缓冲、线程数、shared/TMEM 布局及写回。
// CUTLASS 的 CuTe arch 层封装同一组硬件指令；不是换成另一个 GEMM 算法。
// CUTLASS 4.2.1 / CUDA 12.8 随附 C++ 头文件没有 tcgen05.fence 封装，保留两个 fence。
// 新手推导、PTX/API 对照、B200 实测见同目录 README.md。
namespace mojo_v1 {  // 保留接口，复用原 benchmark，无需改调用方。

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int THREADS = 128;
constexpr int TMEM_COLUMNS = 512;
constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr int SHARED_BYTES = A_BYTES + B_BYTES + 24;

// SS = A/B 都从 shared 读；类型参数明确指定 BF16 输入、FP32 累加。
// A[M,K]、B[N,K] 的连续维度都是 K，所以两个 Major 都是 K。
using Mma = cute::SM100_MMA_F16BF16_SS<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    BLOCK_M, BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>;

__device__ __forceinline__ void wait_barrier(const uint64_t* barrier, uint32_t phase) {
  cutlass::arch::ClusterBarrier::wait(barrier, phase);
}

__device__ __forceinline__ void load_tile(
    void* destination, const CUtensorMap* tensor_map,
    int row, int reduction_offset, uint64_t* barrier) {
  // API 接收 C++ shared 指针，内部转换地址；坐标按 tensor map 的 {K,row} 顺序。
  // SM90 表示这条 TMA 指令始于 Hopper；Blackwell 也支持。
  cute::SM90_TMA_LOAD_2D::copy(
      tensor_map, barrier, uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
      destination, reduction_offset, row);
}

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
          __nv_bfloat16* output, int columns, int reduction_iterations) {
  extern __shared__ __align__(1024) unsigned char shared[];
  const uint32_t shared_base = cute::cast_smem_ptr_to_uint(shared);
  // barrier/allocator API 要 generic shared 指针，不能把 shared 地址数值强转为指针。
  auto* tma_barrier = reinterpret_cast<uint64_t*>(shared + A_BYTES + B_BYTES);
  auto* mma_barrier = tma_barrier + 1;
  auto* allocation_address = reinterpret_cast<uint32_t*>(mma_barrier + 1);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  cute::TMEM::Allocator1Sm allocator;

  if (threadIdx.x == 0) {
    cutlass::arch::ClusterBarrier::init(tma_barrier, 1);
    cutlass::arch::ClusterBarrier::init(mma_barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 0) {
    // 必须整个 warp 参与分配；不能改成 threadIdx.x == 0。
    allocator.allocate(TMEM_COLUMNS, allocation_address);
  }
  __syncthreads();

  const uint32_t tmem = *allocation_address;
  // 构造同样的 instruction descriptor；此 API 将它放在 uint64_t 的高 32 位，
  // 低 32 位留给 sparse metadata（本例是 dense，保持 0）。fma 内部取高 32 位。
  const uint64_t instruction_descriptor = cute::UMMA::make_runtime_instr_desc<
      cutlass::bfloat16_t, cutlass::bfloat16_t, float,
      BLOCK_M, BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
  uint32_t phase = 0;

  for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
    if (threadIdx.x == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          tma_barrier, A_BYTES + B_BYTES);
      load_tile(shared, &a_map, blockIdx.y * BLOCK_M, iteration * BLOCK_K, tma_barrier);
      load_tile(shared + A_BYTES, &b_map, blockIdx.x * BLOCK_N,
                iteration * BLOCK_K, tma_barrier);
    }
    wait_barrier(tma_barrier, phase);
    // 所用 CUTLASS 版本没有对应 API，保留同步排序指令，不能用普通 CTA barrier 替代。
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (warp == 0) {
      // CuTe 的 MMA/commit 内部使用 elect_one_sync，必须整个 warp 统一进入。
      // 仅被选出的一个线程实际发指令，仍是一轮 4 条 MMA + 1 条 commit。
      // 一条 MMA 消耗 K=16；原来的 K=64 一轮仍发 4 条，偏移仍是 32B。
#pragma unroll
      for (int step = 0; step < BLOCK_K / 16; ++step) {
        const uint64_t a_descriptor = descriptor(shared_base + step * 32);
        const uint64_t b_descriptor = descriptor(shared_base + A_BYTES + step * 32);
        const uint32_t accumulate = iteration != 0 || step != 0;
        Mma::fma(a_descriptor, b_descriptor, tmem, accumulate, instruction_descriptor);
      }
      cutlass::arch::umma_arrive(mma_barrier);
    }
    wait_barrier(mma_barrier, phase);
    phase ^= 1;
  }

  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  float fragment[128];
  const uint32_t read_address = tmem + (uint32_t(warp * 32) << 16);
  // Tensor = 指针 + 索引布局的视图，不是分配内存，也不额外复制数据。
  // TMEM load 的源布局由指令 traits 给出（单位是 bit），upcast<32> 转成 FP32 元素。
  // group 将二维布局包成一个嵌套的一维 mode，满足 Atom::call 的 rank-1 接口。
  using TmemLoad = cute::SM100_TMEM_LOAD_16dp256b32x;
  using TmemLoadTraits = cute::Copy_Traits<TmemLoad>;
  auto src_layout = cute::group<0, 2>(cute::upcast<32>(TmemLoadTraits::ValID{}));
  auto src = cute::make_tensor(cute::make_tmem_ptr<float>(read_address), src_layout);
  auto dst = cute::make_tensor(fragment + 0, cute::make_layout(cute::Int<128>{}));
  // Atom 是一条 copy 指令的封装；call 将 dst 展开成 128 个引用参数。
  // 仍生成原版的一条 16x256b.x32，不是 128 次标量读。
  cute::Copy_Atom<TmemLoad, float>{}.call(src, dst);
  cutlass::arch::fence_view_async_tmem_load();
  __syncthreads();

  if (warp == 0) {
    allocator.release_allocation_lock();
    allocator.free(tmem, TMEM_COLUMNS);
  }

  // 保留原版 fragment 到 C 的映射及一对 BF16 的向量写回。
#pragma unroll
  for (int column_tile = 0; column_tile < BLOCK_N / 8; ++column_tile) {
#pragma unroll
    for (int row_tile = 0; row_tile < 2; ++row_tile) {
      const int row = blockIdx.y * BLOCK_M + warp * 16 + lane / 4 + row_tile * 8;
      const int column = blockIdx.x * BLOCK_N + column_tile * 8 + (lane % 4) * 2;
      const int fragment_index = column_tile * 4 + row_tile * 2;
      *reinterpret_cast<__nv_bfloat162*>(output + int64_t(row) * columns + column) =
          __float22bfloat162_rn(make_float2(fragment[fragment_index], fragment[fragment_index + 1]));
    }
  }
}
// 
// CPU 端编码 TMA 元数据，不搬数据；GPU 的 load_tile 才按描述符执行搬运。
// reduction 就是 K（乘法中归约/求和的维度），不是输出列数 N。
// A 用 rows=M,tile_rows=64；B 用 rows=N,tile_rows=256，均按 [rows,K] 存储。
// API 查阅：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
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

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map,
            __nv_bfloat16* output, int rows, int columns, int reduction,
            cudaStream_t stream) {
  // grid.x 切 N、grid.y 切 M；K/64 为求和轮数。本实现要求 M/N/K 整除对应块大小。
  gemm<<<dim3(columns / BLOCK_N, rows / BLOCK_M), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, output, columns, reduction / BLOCK_K);
}

}
