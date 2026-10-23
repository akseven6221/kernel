#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// 本版本：2D TMA + 128B swizzle + tcgen05 MMA。
// A[M,K]、B[N,K] 按行主序存储，输出 C[M,N] = A * B^T；K 是求和维度。
// 指令/字段查阅：https://docs.nvidia.com/cuda/parallel-thread-execution/
// asm 约束查阅：https://docs.nvidia.com/cuda/inline-ptx-assembly/
namespace mojo_v1 {

// 一个 CTA 算 64行×256列的输出；每轮沿 K 搬入 64 个元素。
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int THREADS = 128;
constexpr int TMEM_COLUMNS = 512;
// BF16 每元素 2B：A tile 占 8192B，B tile 占 32768B。
constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
// shared：[A][B][TMA barrier 8B][MMA barrier 8B][TMEM 地址 4B][预留 4B]。
constexpr int SHARED_BYTES = A_BYTES + B_BYTES + 24;

__device__ __forceinline__ void wait_barrier(uint32_t address, uint32_t phase) {
  // [%0] 是 shared 中的 barrier 地址，%1 是要等待完成的轮次奇偶位（0/1）。
  // .pred ready 是布尔寄存器；try_wait 检查该轮是否完成，@!ready 表示“未完成则跳回”。
  // acquire.cta 提供 CTA 范围的获取语义；shared::cta.b64 指定 8B barrier 对象。
  // {} 限定 PTX 局部寄存器/标签作用域，重复内联时避免重名。
  // :: 后为输入，%0/%1 按顺序编号；“r”=32 位、“l”=64 位寄存器。
  // volatile / "memory" 约束编译器；实际等待和 GPU 内存同步由 PTX 指令完成。
  asm volatile(
      "{ .reg .pred ready; wait_loop: "
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1; "
      "@!ready bra wait_loop; }"
      :: "r"(address), "r"(phase) : "memory");
}

__device__ __forceinline__ void load_tile(
    uint32_t destination, const CUtensorMap* tensor_map,
    int row, int reduction_offset, uint32_t barrier) {
  // tensor_map 描述源矩阵地址、形状、行间距、box、swizzle；它本身不是矩阵数据。
  // [%0]：shared 目的起点；[%1,{%3,%2}]：源描述符 + 起始坐标 (K偏移,row)。
  // 搬多大由 box={BLOCK_K,tile_rows} 决定：tile_rows 行 × 64 个 K 元素。
  // [%4]：搬完向这个 barrier 报告完成字节数；async 发起后仍需 wait_barrier。
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3, %2}], [%4];"
      :: "r"(destination), "l"(tensor_map), "r"(row),
         "r"(reduction_offset), "r"(barrier) : "memory");
}

// shared 描述符：把“shared 中从哪读、按什么布局读”打包进 64 位。
// 此 K-major + 128B swizzle 布局：每行 128B，每 8 行跨度 1024B，编码为 1024/16=64。
// leading 字段此时不使用；1024B 对齐使 swizzle base offset 可保持 0。
// 移位和 | 不自动限宽：64 须装入 bits32–45，address>>4 须装入 bits0–14。
// 本例符合范围；换参数/布局时须重新检查，否则超宽值会串到相邻字段。
__device__ __forceinline__ uint64_t descriptor(uint32_t address) {
  return uint64_t(address >> 4) |       // bits0–14：16B 对齐的 shared 起点，以 16B 为单位。
         (uint64_t(64) << 32) |        // bits32–45：stride = 64 个 16B，即 1024B。
         (uint64_t(1) << 46) |         // bits46–48：规范要求固定为 001。
         (uint64_t(2) << 61);          // bits61–63：2=128B swizzle；先转 64 位再移位。
}

__global__ __launch_bounds__(THREADS)
void gemm(const __grid_constant__ CUtensorMap a_map,
          const __grid_constant__ CUtensorMap b_map,
          __nv_bfloat16* output, int columns, int reduction_iterations) {
  // tensor_map 按值作为内核参数传入；__grid_constant__ 让整个 grid 对同一参数对象取地址。
  // A/B 起点均在 1024B 边界，匹配 descriptor() 中 swizzle base offset=0 的设定。
  extern __shared__ __align__(1024) unsigned char shared[];
  // shared 是 C++ 通用指针；cvta 返回 shared 地址空间中的地址数值（size_t）。shared是指向共享内存的 generic 指针
  // size_t 装的是转换结果；这里转成 uint32_t，供 PTX 的 32 位 shared 地址操作数使用。(__cvta_generic_to_shared(shared)是和shared同一位置的 shared 专用地址数值)
  const uint32_t shared_base = uint32_t(__cvta_generic_to_shared(shared));
  // A/B 数据之后才放 TMA barrier，故其地址为 shared_base + A_BYTES + B_BYTES。
  // uint32_t 的 4B 是地址变量宽度；所指的 mbarrier.b64 对象占 8B，下一项应加 8。
  const uint32_t tma_barrier = shared_base + A_BYTES + B_BYTES;
  const uint32_t mma_barrier = tma_barrier + 8;
  const uint32_t allocation_address = mma_barrier + 8;  // 两个 8B barrier 后的 4B 返回值槽位。
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    // 初始化 8B barrier；1 是每轮预期 arrival 数，不是把这块内存写成整数 1。
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(tma_barrier) : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mma_barrier) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) {
    // 第一个 warp 集体分配 TMEM，指令把返回的 32 位 TMEM 地址写入 shared 槽位。
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(allocation_address), "r"(TMEM_COLUMNS) : "memory");
  }
  __syncthreads();

  // 读取 alloc 写入的 TMEM 地址；+16 是跳过两个 8B barrier。
  // allocation_address 是 PTX shared 地址数值，单纯强转不是地址空间转换。
  // 此处用原始 shared 指针加偏移；也可先 __cvta_shared_to_generic 转换后再构造指针。
  const uint32_t tmem = *reinterpret_cast<uint32_t*>(shared + A_BYTES + B_BYTES + 16);
  // 指令描述符：打包“算多大的矩阵、输入/累加用什么类型”，不包含 A/B 地址。
  constexpr uint32_t instruction_descriptor =
      (1u << 4) | (1u << 7) | (1u << 10) |  // bits4–5：FP32；bits7–9/10–12：A/B 为 BF16。
      ((BLOCK_N / 8) << 17) |               // bits17–22：N/8=32，表示输出 256 列。
      ((BLOCK_M / 16) << 24);               // bits24–28：M/16=4，表示输出 64 行。
  // /8、/16 是硬件形状编码单位；字段宽度为 6/5 位，本例 32/4 均装得下。
  uint32_t phase = 0;  // 初始化为第 0 轮，每轮完成后翻转奇偶位。

  for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
    if (threadIdx.x == 0) {
      // 一次 arrival，登记本轮预期完成 A_BYTES+B_BYTES=40960B，来自下面两次 TMA。
      asm volatile(
          "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
          :: "r"(tma_barrier), "r"(A_BYTES + B_BYTES) : "memory");
      // A：从第 blockIdx.y*64 行、K=iteration*64 开始，搬 64行×64元素到 shared 的 A 区。
      load_tile(shared_base, &a_map, blockIdx.y * BLOCK_M, iteration * BLOCK_K, tma_barrier);
      // B 按 [N,K] 存储：从第 blockIdx.x*256 行开始，搬 256行×64元素到 shared 的 B 区。
      // 同一 iteration*BLOCK_K 让 A/B 对齐同一段求和 K；本 2D 版偏移单位是元素。
      load_tile(shared_base + A_BYTES, &b_map, blockIdx.x * BLOCK_N,
                iteration * BLOCK_K, tma_barrier);
    }
    wait_barrier(tma_barrier, phase);  // 两次 TMA 的字节都完成后，MMA 才能使用 A/B。
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (threadIdx.x == 0) {
      // 一条 BF16 MMA 消耗 K=16，本轮 K=64 拆成 4 条；每片 K 偏移为 16×2=32B。
#pragma unroll
      for (int step = 0; step < BLOCK_K / 16; ++step) {
        const uint64_t a_descriptor = descriptor(shared_base + step * 32);
        const uint64_t b_descriptor = descriptor(shared_base + A_BYTES + step * 32);
        const uint32_t accumulate = iteration != 0 || step != 0;  // 首次不读取旧累加值，其余累加。
        asm volatile(
            "{ .reg .pred accumulate; setp.ne.u32 accumulate, %4, 0; "
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate; }"
            :: "r"(tmem), "l"(a_descriptor), "l"(b_descriptor),
               "r"(instruction_descriptor), "r"(accumulate) : "memory");
      }
      // 前面的异步 MMA 完成后，commit 向 mma_barrier 发出一次 arrival。
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
          :: "r"(mma_barrier) : "memory");
    }
    wait_barrier(mma_barrier, phase);  // MMA 读完本轮 A/B，下一轮才可覆盖 shared。
    phase ^= 1;                      // 0→1→0…；同轮 TMA/MMA barrier 等待相同 parity。
  }

  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  // TMEM 结果读到线程寄存器：每线程 128 个 FP32，共 128×128=64×256 个输出。
  float fragment[128];
  const uint32_t read_address = tmem + (uint32_t(warp * 32) << 16);
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x32.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
      : "=f"(fragment[0]), "=f"(fragment[1]), "=f"(fragment[2]), "=f"(fragment[3]),
        "=f"(fragment[4]), "=f"(fragment[5]), "=f"(fragment[6]), "=f"(fragment[7]),
        "=f"(fragment[8]), "=f"(fragment[9]), "=f"(fragment[10]), "=f"(fragment[11]),
        "=f"(fragment[12]), "=f"(fragment[13]), "=f"(fragment[14]), "=f"(fragment[15]),
        "=f"(fragment[16]), "=f"(fragment[17]), "=f"(fragment[18]), "=f"(fragment[19]),
        "=f"(fragment[20]), "=f"(fragment[21]), "=f"(fragment[22]), "=f"(fragment[23]),
        "=f"(fragment[24]), "=f"(fragment[25]), "=f"(fragment[26]), "=f"(fragment[27]),
        "=f"(fragment[28]), "=f"(fragment[29]), "=f"(fragment[30]), "=f"(fragment[31]),
        "=f"(fragment[32]), "=f"(fragment[33]), "=f"(fragment[34]), "=f"(fragment[35]),
        "=f"(fragment[36]), "=f"(fragment[37]), "=f"(fragment[38]), "=f"(fragment[39]),
        "=f"(fragment[40]), "=f"(fragment[41]), "=f"(fragment[42]), "=f"(fragment[43]),
        "=f"(fragment[44]), "=f"(fragment[45]), "=f"(fragment[46]), "=f"(fragment[47]),
        "=f"(fragment[48]), "=f"(fragment[49]), "=f"(fragment[50]), "=f"(fragment[51]),
        "=f"(fragment[52]), "=f"(fragment[53]), "=f"(fragment[54]), "=f"(fragment[55]),
        "=f"(fragment[56]), "=f"(fragment[57]), "=f"(fragment[58]), "=f"(fragment[59]),
        "=f"(fragment[60]), "=f"(fragment[61]), "=f"(fragment[62]), "=f"(fragment[63]),
        "=f"(fragment[64]), "=f"(fragment[65]), "=f"(fragment[66]), "=f"(fragment[67]),
        "=f"(fragment[68]), "=f"(fragment[69]), "=f"(fragment[70]), "=f"(fragment[71]),
        "=f"(fragment[72]), "=f"(fragment[73]), "=f"(fragment[74]), "=f"(fragment[75]),
        "=f"(fragment[76]), "=f"(fragment[77]), "=f"(fragment[78]), "=f"(fragment[79]),
        "=f"(fragment[80]), "=f"(fragment[81]), "=f"(fragment[82]), "=f"(fragment[83]),
        "=f"(fragment[84]), "=f"(fragment[85]), "=f"(fragment[86]), "=f"(fragment[87]),
        "=f"(fragment[88]), "=f"(fragment[89]), "=f"(fragment[90]), "=f"(fragment[91]),
        "=f"(fragment[92]), "=f"(fragment[93]), "=f"(fragment[94]), "=f"(fragment[95]),
        "=f"(fragment[96]), "=f"(fragment[97]), "=f"(fragment[98]), "=f"(fragment[99]),
        "=f"(fragment[100]), "=f"(fragment[101]), "=f"(fragment[102]), "=f"(fragment[103]),
        "=f"(fragment[104]), "=f"(fragment[105]), "=f"(fragment[106]), "=f"(fragment[107]),
        "=f"(fragment[108]), "=f"(fragment[109]), "=f"(fragment[110]), "=f"(fragment[111]),
        "=f"(fragment[112]), "=f"(fragment[113]), "=f"(fragment[114]), "=f"(fragment[115]),
        "=f"(fragment[116]), "=f"(fragment[117]), "=f"(fragment[118]), "=f"(fragment[119]),
        "=f"(fragment[120]), "=f"(fragment[121]), "=f"(fragment[122]), "=f"(fragment[123]),
        "=f"(fragment[124]), "=f"(fragment[125]), "=f"(fragment[126]), "=f"(fragment[127])
      : "r"(read_address));
  // 上面 “=f” 是 FP32 输出寄存器约束；等 TMEM 读完，才能使用 fragment / 释放 TMEM。
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  __syncthreads();

  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem), "r"(TMEM_COLUMNS) : "memory");
  }

  // 256 列分 32 组，每组 8 列；同组 4 个 lane 各写 2 列，合起来覆盖这 8 列。
  // 每 warp 写 16 行：lane/4 选 0..7，row_tile=0/1 选前/后 8 行；4 warp 覆盖 64 行。
#pragma unroll
  for (int column_tile = 0; column_tile < BLOCK_N / 8; ++column_tile) {
#pragma unroll
    for (int row_tile = 0; row_tile < 2; ++row_tile) {
      const int row = blockIdx.y * BLOCK_M + warp * 16 + lane / 4 + row_tile * 8;
      const int column = blockIdx.x * BLOCK_N + column_tile * 8 + (lane % 4) * 2;
      // 每线程每列组持有 2行×2列=4 个值；row_tile 每增加 1 就跳过一对值。
      // 如 column_tile=1,row_tile=1：索引=6，写 fragment[6]、[7] 到同一行的相邻两列。
      const int fragment_index = column_tile * 4 + row_tile * 2;
      // 两个 FP32 转为一对 BF16，按行主序写回相邻两列。
      *reinterpret_cast<__nv_bfloat162*>(output + int64_t(row) * columns + column) =
          __float22bfloat162_rn(make_float2(fragment[fragment_index], fragment[fragment_index + 1]));
    }
  }
}

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