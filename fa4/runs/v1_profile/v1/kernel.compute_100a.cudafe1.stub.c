#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernel.fatbin.c"
static void __device_stub__ZN3fa49attentionILb0EEEv14CUtensorMap_stS1_S1_P6__halfii(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, half *, int, int);
static void __device_stub__ZN3fa49attentionILb1EEEv14CUtensorMap_stS1_S1_P6__halfii(const CUtensorMap&, const CUtensorMap&, const CUtensorMap&, half *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
static void __device_stub__ZN3fa49attentionILb0EEEv14CUtensorMap_stS1_S1_P6__halfii(const CUtensorMap&__par0, const CUtensorMap&__par1, const CUtensorMap&__par2, half *__par3, int __par4, int __par5){__cudaLaunchPrologue(6);__cudaSetupArg(__par0, 0UL);__cudaSetupArg(__par1, 128UL);__cudaSetupArg(__par2, 256UL);__cudaSetupArgSimple(__par3, 384UL);__cudaSetupArgSimple(__par4, 392UL);__cudaSetupArgSimple(__par5, 396UL);__cudaLaunch(((char *)((void ( *)(const CUtensorMap, const CUtensorMap, const CUtensorMap, half *, int, int))fa4::attention<(bool)0> )));}namespace fa4{

template<> __specialization_static void __wrapper__device_stub_attention<false>( const ::CUtensorMap &__cuda_0,const ::CUtensorMap &__cuda_1,const ::CUtensorMap &__cuda_2,::half *&__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__ZN3fa49attentionILb0EEEv14CUtensorMap_stS1_S1_P6__halfii( (::CUtensorMap &)__cuda_0,(::CUtensorMap &)__cuda_1,(::CUtensorMap &)__cuda_2,(::half *&)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}}
static void __device_stub__ZN3fa49attentionILb1EEEv14CUtensorMap_stS1_S1_P6__halfii(const CUtensorMap&__par0, const CUtensorMap&__par1, const CUtensorMap&__par2, half *__par3, int __par4, int __par5){__cudaLaunchPrologue(6);__cudaSetupArg(__par0, 0UL);__cudaSetupArg(__par1, 128UL);__cudaSetupArg(__par2, 256UL);__cudaSetupArgSimple(__par3, 384UL);__cudaSetupArgSimple(__par4, 392UL);__cudaSetupArgSimple(__par5, 396UL);__cudaLaunch(((char *)((void ( *)(const CUtensorMap, const CUtensorMap, const CUtensorMap, half *, int, int))fa4::attention<(bool)1> )));}namespace fa4{

template<> __specialization_static void __wrapper__device_stub_attention<true>( const ::CUtensorMap &__cuda_0,const ::CUtensorMap &__cuda_1,const ::CUtensorMap &__cuda_2,::half *&__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__ZN3fa49attentionILb1EEEv14CUtensorMap_stS1_S1_P6__halfii( (::CUtensorMap &)__cuda_0,(::CUtensorMap &)__cuda_1,(::CUtensorMap &)__cuda_2,(::half *&)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}}
static void __nv_cudaEntityRegisterCallback(void **__T94){__nv_dummy_param_ref(__T94);__nv_save_fatbinhandle_for_managed_rt(__T94);__cudaRegisterEntry(__T94, ((void ( *)(const CUtensorMap, const CUtensorMap, const CUtensorMap, half *, int, int))fa4::attention<(bool)1> ), _ZN3fa49attentionILb1EEEv14CUtensorMap_stS1_S1_P6__halfii, 128);__cudaRegisterEntry(__T94, ((void ( *)(const CUtensorMap, const CUtensorMap, const CUtensorMap, half *, int, int))fa4::attention<(bool)0> ), _ZN3fa49attentionILb0EEEv14CUtensorMap_stS1_S1_P6__halfii, 128);}
static void __sti____cudaRegisterAll(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}

#pragma GCC diagnostic pop
