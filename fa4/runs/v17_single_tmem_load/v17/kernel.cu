// FA4 data path baseline: TMA -> QK tcgen05 -> TMEM softmax -> PV tcgen05.
// v17: issue one x128 score load under the larger startup register budget.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
namespace fa4 {
constexpr int M=128, N=128, D=128, THREADS=384;
constexpr int TILE_BYTES=32768, SHARED_BYTES=6*TILE_BYTES+1168;
__device__ __forceinline__ void wait(uint32_t a, uint32_t phase) {
  asm volatile("{ .reg .pred p; L: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; @!p bra L; }" :: "r"(a), "r"(phase) : "memory");
}
__device__ __forceinline__ void init(uint32_t a) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(a):"memory");
}
__device__ __forceinline__ void expect(uint32_t a,int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(a),"r"(bytes):"memory");
}
__device__ __forceinline__ void commit(uint32_t a) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a):"memory");
}
__device__ __forceinline__ void fence() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void load(uint32_t dst,const CUtensorMap* map,int row,int head,uint32_t bar) {
#pragma unroll
  for(int d=0;d<2;d++)
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" :: "r"(dst+d*16384),"l"(map),"r"(d*64),"r"(head),"r"(row),"r"(bar):"memory");
}
__device__ __forceinline__ uint64_t desc(uint32_t a, bool mn=false) {
  return uint64_t(a>>4) | (uint64_t(mn?1024:0)<<16) | (uint64_t(64)<<32) | (uint64_t(1)<<46) | (uint64_t(2)<<61);
}
__device__ __forceinline__ void qk(uint32_t s,uint32_t q,uint32_t k) {
  constexpr uint32_t id=(1u<<4)|((128/8)<<17)|((128/16)<<24);
#pragma unroll
  for(int i=0;i<8;i++) {
    const int off=(i/4)*16384+(i%4)*32;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" :: "r"(s),"l"(desc(q+off)),"l"(desc(k+off)),"r"(id),"r"(i):"memory");
  }
}
__device__ __forceinline__ void pv(uint32_t o,uint32_t p,uint32_t v,bool accumulate) {
  constexpr uint32_t id=(1u<<4)|(1u<<16)|((128/8)<<17)|((128/16)<<24);
#pragma unroll
  for(int i=0;i<8;i++)
    asm volatile("{ .reg .pred a; setp.ne.u32 a, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, a; }" :: "r"(o),"r"(p+i*8),"l"(desc(v+i*2048,true)),"r"(id),"r"(int(accumulate||i>0)):"memory");
}
// Generated register-vector PTX wrappers follow.
__device__ __forceinline__ void ld128(uint32_t a,float (&x)[128]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]), "=f"(x[5]), "=f"(x[6]), "=f"(x[7]), "=f"(x[8]), "=f"(x[9]), "=f"(x[10]), "=f"(x[11]), "=f"(x[12]), "=f"(x[13]), "=f"(x[14]), "=f"(x[15]), "=f"(x[16]), "=f"(x[17]), "=f"(x[18]), "=f"(x[19]), "=f"(x[20]), "=f"(x[21]), "=f"(x[22]), "=f"(x[23]), "=f"(x[24]), "=f"(x[25]), "=f"(x[26]), "=f"(x[27]), "=f"(x[28]), "=f"(x[29]), "=f"(x[30]), "=f"(x[31]), "=f"(x[32]), "=f"(x[33]), "=f"(x[34]), "=f"(x[35]), "=f"(x[36]), "=f"(x[37]), "=f"(x[38]), "=f"(x[39]), "=f"(x[40]), "=f"(x[41]), "=f"(x[42]), "=f"(x[43]), "=f"(x[44]), "=f"(x[45]), "=f"(x[46]), "=f"(x[47]), "=f"(x[48]), "=f"(x[49]), "=f"(x[50]), "=f"(x[51]), "=f"(x[52]), "=f"(x[53]), "=f"(x[54]), "=f"(x[55]), "=f"(x[56]), "=f"(x[57]), "=f"(x[58]), "=f"(x[59]), "=f"(x[60]), "=f"(x[61]), "=f"(x[62]), "=f"(x[63]), "=f"(x[64]), "=f"(x[65]), "=f"(x[66]), "=f"(x[67]), "=f"(x[68]), "=f"(x[69]), "=f"(x[70]), "=f"(x[71]), "=f"(x[72]), "=f"(x[73]), "=f"(x[74]), "=f"(x[75]), "=f"(x[76]), "=f"(x[77]), "=f"(x[78]), "=f"(x[79]), "=f"(x[80]), "=f"(x[81]), "=f"(x[82]), "=f"(x[83]), "=f"(x[84]), "=f"(x[85]), "=f"(x[86]), "=f"(x[87]), "=f"(x[88]), "=f"(x[89]), "=f"(x[90]), "=f"(x[91]), "=f"(x[92]), "=f"(x[93]), "=f"(x[94]), "=f"(x[95]), "=f"(x[96]), "=f"(x[97]), "=f"(x[98]), "=f"(x[99]), "=f"(x[100]), "=f"(x[101]), "=f"(x[102]), "=f"(x[103]), "=f"(x[104]), "=f"(x[105]), "=f"(x[106]), "=f"(x[107]), "=f"(x[108]), "=f"(x[109]), "=f"(x[110]), "=f"(x[111]), "=f"(x[112]), "=f"(x[113]), "=f"(x[114]), "=f"(x[115]), "=f"(x[116]), "=f"(x[117]), "=f"(x[118]), "=f"(x[119]), "=f"(x[120]), "=f"(x[121]), "=f"(x[122]), "=f"(x[123]), "=f"(x[124]), "=f"(x[125]), "=f"(x[126]), "=f"(x[127]) : "r"(a) : "memory");
}
__device__ __forceinline__ void st128(uint32_t a, const float (&x)[128]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x128.b32 [%128], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127};"
    :: "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]), "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]), "f"(x[16]), "f"(x[17]), "f"(x[18]), "f"(x[19]), "f"(x[20]), "f"(x[21]), "f"(x[22]), "f"(x[23]), "f"(x[24]), "f"(x[25]), "f"(x[26]), "f"(x[27]), "f"(x[28]), "f"(x[29]), "f"(x[30]), "f"(x[31]), "f"(x[32]), "f"(x[33]), "f"(x[34]), "f"(x[35]), "f"(x[36]), "f"(x[37]), "f"(x[38]), "f"(x[39]), "f"(x[40]), "f"(x[41]), "f"(x[42]), "f"(x[43]), "f"(x[44]), "f"(x[45]), "f"(x[46]), "f"(x[47]), "f"(x[48]), "f"(x[49]), "f"(x[50]), "f"(x[51]), "f"(x[52]), "f"(x[53]), "f"(x[54]), "f"(x[55]), "f"(x[56]), "f"(x[57]), "f"(x[58]), "f"(x[59]), "f"(x[60]), "f"(x[61]), "f"(x[62]), "f"(x[63]), "f"(x[64]), "f"(x[65]), "f"(x[66]), "f"(x[67]), "f"(x[68]), "f"(x[69]), "f"(x[70]), "f"(x[71]), "f"(x[72]), "f"(x[73]), "f"(x[74]), "f"(x[75]), "f"(x[76]), "f"(x[77]), "f"(x[78]), "f"(x[79]), "f"(x[80]), "f"(x[81]), "f"(x[82]), "f"(x[83]), "f"(x[84]), "f"(x[85]), "f"(x[86]), "f"(x[87]), "f"(x[88]), "f"(x[89]), "f"(x[90]), "f"(x[91]), "f"(x[92]), "f"(x[93]), "f"(x[94]), "f"(x[95]), "f"(x[96]), "f"(x[97]), "f"(x[98]), "f"(x[99]), "f"(x[100]), "f"(x[101]), "f"(x[102]), "f"(x[103]), "f"(x[104]), "f"(x[105]), "f"(x[106]), "f"(x[107]), "f"(x[108]), "f"(x[109]), "f"(x[110]), "f"(x[111]), "f"(x[112]), "f"(x[113]), "f"(x[114]), "f"(x[115]), "f"(x[116]), "f"(x[117]), "f"(x[118]), "f"(x[119]), "f"(x[120]), "f"(x[121]), "f"(x[122]), "f"(x[123]), "f"(x[124]), "f"(x[125]), "f"(x[126]), "f"(x[127]), "r"(a) : "memory");
}
__device__ __forceinline__ void st64(uint32_t a, const uint32_t (&x)[64]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x64.b32 [%64], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63};"
    :: "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]), "r"(x[10]), "r"(x[11]), "r"(x[12]), "r"(x[13]), "r"(x[14]), "r"(x[15]), "r"(x[16]), "r"(x[17]), "r"(x[18]), "r"(x[19]), "r"(x[20]), "r"(x[21]), "r"(x[22]), "r"(x[23]), "r"(x[24]), "r"(x[25]), "r"(x[26]), "r"(x[27]), "r"(x[28]), "r"(x[29]), "r"(x[30]), "r"(x[31]), "r"(x[32]), "r"(x[33]), "r"(x[34]), "r"(x[35]), "r"(x[36]), "r"(x[37]), "r"(x[38]), "r"(x[39]), "r"(x[40]), "r"(x[41]), "r"(x[42]), "r"(x[43]), "r"(x[44]), "r"(x[45]), "r"(x[46]), "r"(x[47]), "r"(x[48]), "r"(x[49]), "r"(x[50]), "r"(x[51]), "r"(x[52]), "r"(x[53]), "r"(x[54]), "r"(x[55]), "r"(x[56]), "r"(x[57]), "r"(x[58]), "r"(x[59]), "r"(x[60]), "r"(x[61]), "r"(x[62]), "r"(x[63]), "r"(a) : "memory");
}
__device__ __forceinline__ float fast_exp2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ void ld32(uint32_t a, float (&x)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]), "=f"(x[5]), "=f"(x[6]), "=f"(x[7]), "=f"(x[8]), "=f"(x[9]), "=f"(x[10]), "=f"(x[11]), "=f"(x[12]), "=f"(x[13]), "=f"(x[14]), "=f"(x[15]), "=f"(x[16]), "=f"(x[17]), "=f"(x[18]), "=f"(x[19]), "=f"(x[20]), "=f"(x[21]), "=f"(x[22]), "=f"(x[23]), "=f"(x[24]), "=f"(x[25]), "=f"(x[26]), "=f"(x[27]), "=f"(x[28]), "=f"(x[29]), "=f"(x[30]), "=f"(x[31]) : "r"(a) : "memory");
}
__device__ __forceinline__ void st32(uint32_t a, const float (&x)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31};"
    :: "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]), "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]), "f"(x[16]), "f"(x[17]), "f"(x[18]), "f"(x[19]), "f"(x[20]), "f"(x[21]), "f"(x[22]), "f"(x[23]), "f"(x[24]), "f"(x[25]), "f"(x[26]), "f"(x[27]), "f"(x[28]), "f"(x[29]), "f"(x[30]), "f"(x[31]), "r"(a) : "memory");
}
__device__ __forceinline__ void init_count(uint32_t a,int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a),"r"(count):"memory"); }
__device__ __forceinline__ void arrive(uint32_t a) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a):"memory"); }
__device__ __forceinline__ void wg_sync() { asm volatile("bar.sync 2, 128;" ::: "memory"); }
// Barrier offsets protect distinct ownership transitions; all are 8-byte aligned.
constexpr int QFULL=6*TILE_BYTES, KVFULL=QFULL+8, KVEMPTY=QFULL+24;
constexpr int SFULL=QFULL+40, PFULL=QFULL+56, CFULL=QFULL+72;
constexpr int CACK=QFULL+88, PVDONE=QFULL+104, EPIDONE=QFULL+120;
constexpr int ALLOC=QFULL+128, FINAL=QFULL+136, SCALE=QFULL+144;
template<bool CAUSAL>
__global__ __maxnreg__(168) void attention(
    const __grid_constant__ CUtensorMap qm,const __grid_constant__ CUtensorMap km,
    const __grid_constant__ CUtensorMap vm,half* output,int seq,int heads) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const uint32_t base=uint32_t(__cvta_generic_to_shared(sm));
  const int tid=threadIdx.x, wg=tid/128, local=tid%128, warp=local/32, lane=tid%32;
  const int head=blockIdx.y%heads, batch=blockIdx.y/heads, qr=blockIdx.x*256;
  int lengths[2];
#pragma unroll
  for(int s=0;s<2;s++) lengths[s]=CAUSAL?min((qr+(s+1)*128+127)/128,(seq+127)/128):(seq+127)/128;
  if(tid==0) {
    init(base+QFULL);init(base+FINAL); init_count(base+EPIDONE,256);
#pragma unroll
    for(int s=0;s<2;s++) {
      init(base+KVFULL+s*8);init(base+KVEMPTY+s*8);init(base+SFULL+s*8);
      init_count(base+PFULL+s*8,128);init_count(base+CFULL+s*8,128);
      init_count(base+CACK+s*8,128);init(base+PVDONE+s*8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if(tid/32==8) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(base+ALLOC):"memory");
  __syncthreads();
  const uint32_t tm=*reinterpret_cast<uint32_t*>(sm+ALLOC);
  const uint32_t rowtm=uint32_t(warp*32)<<16;
  float* scales=reinterpret_cast<float*>(sm+SCALE);
  if(wg==2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;" ::: "memory");
    if(warp==1 && lane==0) {
      expect(base+QFULL,2*TILE_BYTES);
      load(base,&qm,batch*seq+qr,head,base+QFULL);
      load(base+TILE_BYTES,&qm,batch*seq+qr+128,head,base+QFULL);
      for(int j=0;j<lengths[1];j++) {
        const int slot=j%2;
        if(j>=2) wait(base+KVEMPTY+slot*8,((j-2)/2)&1);
        const uint32_t kb=base+2*TILE_BYTES+slot*2*TILE_BYTES;
        expect(base+KVFULL+slot*8,2*TILE_BYTES);
        load(kb,&km,batch*seq+j*128,head,base+KVFULL+slot*8);
        load(kb+TILE_BYTES,&vm,batch*seq+j*128,head,base+KVFULL+slot*8);
      }
    }
    if(warp==0) {
      if(lane==0) {
        wait(base+QFULL,0);fence();
        for(int j=0;j<lengths[1];j++) {
          const int slot=j%2;
          const uint32_t kb=base+2*TILE_BYTES+slot*2*TILE_BYTES;
          wait(base+KVFULL+slot*8,(j/2)&1);fence();
#pragma unroll
          for(int s=0;s<2;s++) if(j<lengths[s]) {
            // P reuses S storage: wait until previous PV has finished reading P.
            if(j>0) wait(base+PVDONE+s*8,(j-1)&1);
            fence();qk(tm+s*128,base+s*TILE_BYTES,kb);commit(base+SFULL+s*8);
          }
#pragma unroll
          for(int s=0;s<2;s++) if(j<lengths[s]) {
            wait(base+PFULL+s*8,j&1);fence();
            pv(tm+256+s*128,tm+s*128+64,kb+TILE_BYTES,j>0);
            commit(base+PVDONE+s*8);
          }
          commit(base+KVEMPTY+slot*8);
        }
        commit(base+FINAL);
      }
      // Correction has finished all TMEM reads before relinquishing allocation.
      wait(base+EPIDONE,0);
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm):"memory");
    }
  } else if(wg<2) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 224;" ::: "memory");
    const int stage=wg, query=qr+stage*128+local;
    float rowmax=-INFINITY, rowsum=0;
    for(int j=0;j<lengths[stage];j++) {
      wait(base+SFULL+stage*8,j&1);fence();
      float scores[128];ld128(tm+stage*128+rowtm,scores);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      if(j*128+127>=seq || (CAUSAL && j*128+127>query)) {
#pragma unroll
        for(int c=0;c<128;c++) {
          if(j*128+c>=seq || (CAUSAL && j*128+c>query)) scores[c]=-INFINITY;
        }
      }
      float maxima[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
#pragma unroll
      for(int c=0;c<128;c++) maxima[c%4]=fmaxf(maxima[c%4],scores[c]);
      float mx=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
      const float candidate=fmaxf(rowmax,mx);
      const float next=(j==0||(rowmax-candidate)*0.127517430824596f<-8.0f)?candidate:rowmax;
      const float alpha=j==0?0.0f:fast_exp2((rowmax-next)*0.127517430824596f);
      rowmax=next;

      const float scaled_ref=-next*0.127517430824596f;
      uint32_t packed[64];float psum=0;
#pragma unroll
      for(int c=0;c<64;c++) {
        float a=fast_exp2(fmaf(scores[2*c],0.127517430824596f,scaled_ref));
        float b=fast_exp2(fmaf(scores[2*c+1],0.127517430824596f,scaled_ref));
        psum+=a+b;
        const __half2_raw h=__floats2half2_rn(a,b);packed[c]=uint32_t(h.x)|(uint32_t(h.y)<<16);
      }
      rowsum=rowsum*alpha+psum;
      st64(tm+stage*128+64+rowtm,packed);
      asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      // SFULL implies the previous PV is complete. Fold the rare correction into this owner.
      if(j>0 && __any_sync(0xffffffff,alpha!=1.0f)) {
#pragma unroll
        for(int c=0;c<4;c++) {
          float o[32];ld32(tm+256+stage*128+c*32+rowtm,o);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
          for(int d=0;d<32;d++) o[d]*=alpha;
          st32(tm+256+stage*128+c*32+rowtm,o);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      }
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      arrive(base+PFULL+stage*8);
    }
    // A causal stage can end earlier; wait for the last stage-1 PV before reusing Q/K storage.
    wait(base+FINAL,0);fence();
    const float sum=rowsum;float inverse;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(sum));
    half* staging=reinterpret_cast<half*>(sm+stage*128*130*2);
#pragma unroll
    for(int c=0;c<4;c++) {
      float o[32];ld32(tm+256+stage*128+c*32+rowtm,o);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
      for(int d=0;d<32;d++) staging[(c*32+d)*130+local]=__float2half_rn(o[d]*inverse);
    }
    if(stage==0) asm volatile("bar.sync 2,128;" ::: "memory");
    else asm volatile("bar.sync 3,128;" ::: "memory");
    for(int i=local;i<128*64;i+=128) {
      int r=i/64,c=(i%64)*2;
      if(qr+stage*128+r<seq) {
        const __half2 value=__halves2half2(staging[c*130+r],staging[(c+1)*130+r]);
        *reinterpret_cast<__half2*>(output+((int64_t(batch)*seq+qr+stage*128+r)*heads+head)*128+c)=value;
      }
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    arrive(base+EPIDONE);
  }
}

struct Context { CUtensorMap q,k,v; int batch,seq,heads; };
CUresult encode(CUtensorMap* map,void* p,int batch,int seq,int heads) {
  uint64_t dims[]={128,uint64_t(heads),uint64_t(batch)*seq};
  uint64_t strides[]={256,uint64_t(heads)*256};
  uint32_t box[]={64,1,128}, es[]={1,1,1};
  return cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,3,p,dims,strides,box,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
}
extern "C" int fa4_create(void** result,void* q,void* k,void* v,int batch,int seq,int heads) {
  auto* ctx=new fa4::Context{}; ctx->batch=batch;ctx->seq=seq;ctx->heads=heads;
  CUresult r=fa4::encode(&ctx->q,q,batch,seq,heads);
  if(!r) r=fa4::encode(&ctx->k,k,batch,seq,heads);
  if(!r) r=fa4::encode(&ctx->v,v,batch,seq,heads);
  if(r) {delete ctx;return int(r);}
  cudaError_t e=cudaFuncSetAttribute(fa4::attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa4::SHARED_BYTES);
  if(!e) e=cudaFuncSetAttribute(fa4::attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa4::SHARED_BYTES);
  if(e) {delete ctx;return 10000+int(e);}
  *result=ctx; return 0;
}
extern "C" int fa4_launch(void* ptr,void* out,int causal,void* stream) {
  auto& c=*static_cast<fa4::Context*>(ptr); dim3 grid((c.seq+255)/256,c.batch*c.heads);
  if(causal) fa4::attention<true><<<grid,fa4::THREADS,fa4::SHARED_BYTES,(cudaStream_t)stream>>>(c.q,c.k,c.v,(half*)out,c.seq,c.heads);
  else fa4::attention<false><<<grid,fa4::THREADS,fa4::SHARED_BYTES,(cudaStream_t)stream>>>(c.q,c.k,c.v,(half*)out,c.seq,c.heads);
  return int(cudaGetLastError());
}
extern "C" void fa4_destroy(void* ptr) {delete static_cast<fa4::Context*>(ptr);}
