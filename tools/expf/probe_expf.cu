// probe_expf.cu (v2) — device expf uses MUFU.EX2 (ex2.approx.f32) + argument
// reduction (njuffa; ~2.3 ulp). Reproduce it on CPU = accurate (x*log2e) then
// ex2.approx from a probed table. This probe (a) tests which reduction makes
// ex2.approx(reduce(x)) == expf(x) bit-for-bit, and (b) dumps the ex2.approx table.
//
//   nvcc -arch=sm_89 -O2 probe_expf.cu -o probe_expf && ./probe_expf
//
// Prints per-reduction mismatch vs device expf over x in [-90,90], and writes
// ex2_table.npy — ex2.approx.f32(f) for f = i/2^23, i=0..2^23-1 (f in [0,1)); with
// ex2.approx(y)=ldexp(ex2.approx(frac(y)), floor(y)) that reconstructs ex2 anywhere.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N   (1 << 23)
#define XLO (-90.0f)
#define XHI ( 90.0f)
// log2(e) split hi+lo (hi is the round-to-24-bit head; lo the tail)
#define L2E_HI 1.442695021629333496e+0f
#define L2E_LO 1.925962991460541939e-8f

__device__ __forceinline__ uint32_t bits(float x){ uint32_t u; memcpy(&u,&x,4); return u; }
__device__ __forceinline__ float gx(int i){ return fmaf((float)i, (XHI-XLO)/(float)(N-1), XLO); }
__device__ __forceinline__ float ex2a(float x){ float r; asm("ex2.approx.f32 %0,%1;":"=f"(r):"f"(x)); return r; }

// reduction candidates -> ex2.approx
__device__ __forceinline__ float R_single(float x){ return ex2a(x * 1.442695041f); }
__device__ __forceinline__ float R_fma2(float x){ return ex2a(fmaf(x, L2E_HI, x*L2E_LO)); }
__device__ __forceinline__ float R_fma2b(float x){ float t=fmaf(x,L2E_HI, 0.0f); return ex2a(fmaf(x,L2E_LO,t)); }

__global__ void models(uint64_t* m){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    float x=gx(i); uint32_t e=bits(expf(x));
    if(!isfinite(__int_as_float(e))) return;          // skip overflow/underflow tail
    if(bits(R_single(x))!=e) atomicAdd((unsigned long long*)&m[0],1ULL);
    if(bits(R_fma2(x))  !=e) atomicAdd((unsigned long long*)&m[1],1ULL);
    if(bits(R_fma2b(x)) !=e) atomicAdd((unsigned long long*)&m[2],1ULL);
    atomicAdd((unsigned long long*)&m[3],1ULL);       // finite count
}
__global__ void ex2tab(uint32_t* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    float f=(float)i / (float)N;                       // f in [0,1)
    out[i]=bits(ex2a(f));
}

static void wnpy(const char* p,const uint32_t* d,int64_t n){
    char h[128]; int hl=snprintf(h,sizeof(h),"{'descr': '<i4', 'fortran_order': False, 'shape': (%lld,), }",(long long)n);
    int pad=(64-((10+hl)%64))%64; FILE* f=fopen(p,"wb");
    const unsigned char mg[8]={0x93,'N','U','M','P','Y',1,0}; fwrite(mg,1,8,f);
    uint16_t L=(uint16_t)(hl+pad+1); fwrite(&L,2,1,f); fwrite(h,1,hl,f);
    for(int i=0;i<pad;i++) fputc(' ',f); fputc('\n',f); fwrite(d,4,(size_t)n,f); fclose(f);
}

int main(){
    uint64_t* dm; cudaMalloc(&dm,32); cudaMemset(dm,0,32);
    models<<<(N+255)/256,256>>>(dm); cudaDeviceSynchronize();
    uint64_t m[4]; cudaMemcpy(m,dm,32,cudaMemcpyDeviceToHost);
    printf("device expf vs ex2.approx(reduction), over %llu finite x:\n",(unsigned long long)m[3]);
    printf("  ex2(x*log2e single)        = %llu%s\n",(unsigned long long)m[0], m[0]==0?"  <-- MATCH":"");
    printf("  ex2(fma2 x*log2e hi+lo)     = %llu%s\n",(unsigned long long)m[1], m[1]==0?"  <-- MATCH":"");
    printf("  ex2(fma2b variant)          = %llu%s\n",(unsigned long long)m[2], m[2]==0?"  <-- MATCH":"");

    uint32_t* dt; cudaMalloc(&dt,(size_t)N*4); ex2tab<<<(N+255)/256,256>>>(dt); cudaDeviceSynchronize();
    uint32_t* h=(uint32_t*)malloc((size_t)N*4); cudaMemcpy(h,dt,(size_t)N*4,cudaMemcpyDeviceToHost);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0); printf("GPU: %s sm_%d%d\n",p.name,p.major,p.minor);
    wnpy("ex2_table.npy",h,N);
    printf("wrote ex2_table.npy (%d entries) — ex2.approx(i/2^23), i in [0,2^23).\n",N);
    return 0;
}
