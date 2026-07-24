// probe_expf.cu — characterize the device single-precision expf that XGBoost's
// binary:logistic / multiclass objectives use (common::Sigmoid = 1/(expf(min(-x,
// 88.7))+1+1e-16); softmax uses expf too). XGBoost builds with NO --use_fast_math,
// so this is the ACCURATE expf (Cody-Waite reduction + polynomial), not __expf.
//
// Run on the NVIDIA box that produced the reference fixtures:
//   nvcc -arch=native -O2 probe_expf.cu -o probe_expf && ./probe_expf
//
// Dumps expf_sweep.npy — device expf(x) bit patterns for x on a fixed, regenerable
// grid, so the host side can diff device-expf vs libm-expf (what cajeta's Math.exp
// uses) and decide whether there is any fidelity gap at all. The grid is
// x_i = X_LO + i*(X_HI-X_LO)/(N-1), float32, i=0..N-1 — regenerate it identically on
// the host. Also reports how device expf relates to cheap reproducible models.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N   (1 << 23)
#define X_LO (-90.0f)
#define X_HI ( 90.0f)

__device__ __forceinline__ uint32_t bits(float x){ uint32_t u; memcpy(&u,&x,4); return u; }
__device__ __forceinline__ float grid_x(int i){ return X_LO + (float)i * ((X_HI - X_LO) / (float)(N - 1)); }

__global__ void dump_expf(uint32_t* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= N) return;
    out[i] = bits(expf(grid_x(i)));           // the accurate device expf
}

// Compare device expf to reproducible-on-CPU candidates over the same grid.
// M1 __expf (fast SFU), M2 exp2f(x*log2e) (single-mult reduction + accurate exp2f),
// M3 ex2.approx.f32(x*log2e) (raw MUFU.EX2). mism[0..2].
__device__ __forceinline__ float ex2_approx(float x){ float r; asm("ex2.approx.f32 %0,%1;":"=f"(r):"f"(x)); return r; }
__global__ void models(uint64_t* mism){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= N) return;
    float x = grid_x(i); uint32_t e = bits(expf(x));
    const float log2e = 1.4426950408889634f;
    if (bits(__expf(x))              != e) atomicAdd((unsigned long long*)&mism[0],1ULL);
    if (bits(exp2f(x*log2e))         != e) atomicAdd((unsigned long long*)&mism[1],1ULL);
    if (bits(ex2_approx(x*log2e))    != e) atomicAdd((unsigned long long*)&mism[2],1ULL);
}

static void write_npy_i32(const char* path, const uint32_t* d, int64_t n){
    char h[128]; int hl = snprintf(h,sizeof(h),"{'descr': '<i4', 'fortran_order': False, 'shape': (%lld,), }",(long long)n);
    int tot = 10+hl, pad = (64-(tot%64))%64; FILE* f=fopen(path,"wb");
    const unsigned char magic[8]={0x93,'N','U','M','P','Y',1,0}; fwrite(magic,1,8,f);
    uint16_t len=(uint16_t)(hl+pad+1); fwrite(&len,2,1,f); fwrite(h,1,hl,f);
    for(int i=0;i<pad;i++) fputc(' ',f); fputc('\n',f); fwrite(d,4,(size_t)n,f); fclose(f);
}

int main(){
    uint32_t* d; cudaMalloc(&d,(size_t)N*4);
    dump_expf<<<(N+255)/256,256>>>(d); cudaDeviceSynchronize();
    uint32_t* hbuf=(uint32_t*)malloc((size_t)N*4); cudaMemcpy(hbuf,d,(size_t)N*4,cudaMemcpyDeviceToHost);

    uint64_t* dm; cudaMalloc(&dm,24); cudaMemset(dm,0,24);
    models<<<(N+255)/256,256>>>(dm); cudaDeviceSynchronize();
    uint64_t m[3]; cudaMemcpy(m,dm,24,cudaMemcpyDeviceToHost);
    printf("grid: x in [%.1f, %.1f], N=%d\n", X_LO, X_HI, N);
    printf("device expf vs models (mismatches / %d):\n", N);
    printf("  __expf (fast SFU)          = %llu%s\n",(unsigned long long)m[0], m[0]==0?"  <-- MATCH":"");
    printf("  exp2f(x*log2e)             = %llu%s\n",(unsigned long long)m[1], m[1]==0?"  <-- MATCH":"");
    printf("  ex2.approx.f32(x*log2e)    = %llu%s\n",(unsigned long long)m[2], m[2]==0?"  <-- MATCH":"");

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("GPU: %s  sm_%d%d  CUDART %d\n", p.name, p.major, p.minor, CUDART_VERSION);
    write_npy_i32("expf_sweep.npy", hbuf, N);
    printf("wrote expf_sweep.npy (%d entries, %.1f MB) — regenerate x on the host with the same grid.\n", N, N*4.0/1e6);
    return 0;
}
