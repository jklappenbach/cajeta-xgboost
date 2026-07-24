// probe_rcp.cu — characterize the SFU FP32 reciprocal (MUFU.RCP / rcp.approx.f32)
// bit-for-bit, so cajeta can reproduce device XGBoost's __fdividef split-gain
// scoring exactly. Run this ON THE NVIDIA BOX that produces the parity reference
// (same GPU family as the reference fixtures).
//
//   nvcc -arch=native -O2 probe_rcp.cu -o probe_rcp && ./probe_rcp
//
// It writes rcp_mantissa.npy — a [2^23] int32 array where entry i holds the raw
// 32-bit pattern of rcp.approx.f32(m) for m = uint_as_float(0x3F800000 + i), i.e.
// every float32 in [1.0, 2.0). That is the whole table: for any positive normal
// b = 2^E * m, MUFU.RCP(b) = ldexp(MUFU.RCP(m), -E), which the program VERIFIES
// (exponent factoring) before writing, and it also confirms
// __fdividef(a,b) == a * rcp.approx.f32(b) so cajeta can implement fdividef as a
// table lookup + one float32 multiply.
//
// Commit rcp_mantissa.npy under tools/fdividef/ and ping me; I bake it into the
// cajeta FastMath table (delta-compressed) and wire it into SplitFinder.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float rcp_approx(float x) {
    float r;
    asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float fdividef_dev(float a, float b) {
    return __fdividef(a, b);   // what XGBoost's Divide() emits on device
}

// Candidate CPU-reproducible models of __fdividef (all use MUFU.RCP + FMA, both
// exactly reproducible off-device). The probe reports which matches bit-for-bit.
__device__ __forceinline__ uint32_t bits(float x){ uint32_t u; memcpy(&u,&x,4); return u; }
__device__ __forceinline__ float M0(float a,float b){ return a*rcp_approx(b); }                // raw
__device__ __forceinline__ float M1(float a,float b){ float r=rcp_approx(b); float q=a*r;      // 1 Newton
    return fmaf(fmaf(-b,q,a), r, q); }
__device__ __forceinline__ float M2(float a,float b){ float r=rcp_approx(b); float q=a*r;      // 2 Newton
    q=fmaf(fmaf(-b,q,a),r,q); return fmaf(fmaf(-b,q,a),r,q); }
__device__ __forceinline__ float M3(float a,float b){ return __fdiv_rn(a,b); }                 // correct rnd
__device__ __forceinline__ float M4(float a,float b){ return a*__frcp_rn(b); }                 // rn recip

__global__ void fill_table(uint32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m; uint32_t mb = 0x3F800000u + (uint32_t)i; // m in [1,2)
    memcpy(&m, &mb, 4);
    float r = rcp_approx(m);
    uint32_t rb; memcpy(&rb, &r, 4);
    out[i] = rb;
}

// Over a pseudo-random sweep of positive-normal (a,b): count (a) exponent-factoring
// failures for rcp, and (b) per-model mismatches vs __fdividef. mism[0]=factoring,
// mism[1..5]=M0..M4.
__global__ void verify(const uint32_t* table, uint64_t* mism, int reps) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t s[6] = {0,0,0,0,0,0};
    uint32_t rng = 0x9E3779B9u * (uint32_t)(t + 1);
    for (int k = 0; k < reps; ++k) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        uint32_t exp = 1u + (rng % 254u), man = (rng >> 3) & 0x7FFFFFu;
        uint32_t bb = (exp << 23) | man; float b; memcpy(&b, &bb, 4);
        // (a) factoring: rcp(b) == ldexp(table[man], -(exp-127))
        float rb = rcp_approx(b);
        float tm; uint32_t tb = table[man]; memcpy(&tm, &tb, 4);
        float recon = ldexpf(tm, -((int)exp - 127));
        if (bits(rb) != bits(recon)) s[0]++;
        // (b) which model reproduces __fdividef?
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        uint32_t ab = ((1u + (rng % 254u)) << 23) | ((rng >> 3) & 0x7FFFFFu);
        float a; memcpy(&a, &ab, 4);
        uint32_t fd = bits(fdividef_dev(a, b));
        if (bits(M0(a,b)) != fd) s[1]++;
        if (bits(M1(a,b)) != fd) s[2]++;
        if (bits(M2(a,b)) != fd) s[3]++;
        if (bits(M3(a,b)) != fd) s[4]++;
        if (bits(M4(a,b)) != fd) s[5]++;
    }
    for (int j = 0; j < 6; ++j) atomicAdd((unsigned long long*)&mism[j], (unsigned long long)s[j]);
}

// Dump a handful of concrete (a,b) with every candidate's bits, for manual diff.
__global__ void examples(uint32_t* out) {
    int i = threadIdx.x; if (i >= 8) return;
    uint32_t ab[8] = {0x40400000,0x40a00000,0x3fc00000,0x41800000,0x3f000000,0x42f60000,0x3e800000,0x40e00000};
    uint32_t bb[8] = {0x40e00000,0x40100000,0x41600000,0x3fa00000,0x43160000,0x40000000,0x41c80000,0x3f400000};
    float a,b; memcpy(&a,&ab[i],4); memcpy(&b,&bb[i],4);
    uint32_t* o = out + i*8;
    o[0]=ab[i]; o[1]=bb[i]; o[2]=bits(fdividef_dev(a,b));
    o[3]=bits(M0(a,b)); o[4]=bits(M1(a,b)); o[5]=bits(M2(a,b)); o[6]=bits(M3(a,b)); o[7]=bits(M4(a,b));
}

static void write_npy_i32(const char* path, const uint32_t* data, int64_t n) {
    // minimal .npy v1.0 writer, dtype '<i4', shape (n,)
    char hdr[128];
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '<i4', 'fortran_order': False, 'shape': (%lld,), }", (long long)n);
    int total = 10 + hlen;
    int pad = (64 - (total % 64)) % 64;               // align to 64
    FILE* f = fopen(path, "wb");
    const unsigned char magic[8] = {0x93,'N','U','M','P','Y',1,0};
    fwrite(magic, 1, 8, f);
    uint16_t len = (uint16_t)(hlen + pad + 1);
    fwrite(&len, 2, 1, f);
    fwrite(hdr, 1, hlen, f);
    for (int i = 0; i < pad; ++i) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, 4, (size_t)n, f);
    fclose(f);
}

int main() {
    const int N = 1 << 23;
    uint32_t* d_out; cudaMalloc(&d_out, (size_t)N * 4);
    fill_table<<<(N + 255) / 256, 256>>>(d_out, N);
    cudaDeviceSynchronize();
    uint32_t* h = (uint32_t*)malloc((size_t)N * 4);
    cudaMemcpy(h, d_out, (size_t)N * 4, cudaMemcpyDeviceToHost);

    uint64_t* d_m; cudaMalloc(&d_m, 48); cudaMemset(d_m, 0, 48);
    verify<<<256, 256>>>(d_out, d_m, 4096);
    cudaDeviceSynchronize();
    uint64_t m[6]; cudaMemcpy(m, d_m, 48, cudaMemcpyDeviceToHost);
    uint64_t total = (uint64_t)256 * 256 * 4096;
    const char* names[6] = {"exponent-factoring",
        "M0 a*rcp", "M1 rcp+1 Newton", "M2 rcp+2 Newton", "M3 __fdiv_rn (correct)", "M4 a*__frcp_rn"};
    printf("over %llu samples:\n", (unsigned long long)total);
    for (int j = 0; j < 6; ++j)
        printf("  %-24s mismatches = %llu%s\n", names[j], (unsigned long long)m[j],
               (m[j]==0 ? "   <-- MATCH" : ""));
    printf("=> factoring %s; __fdividef model = the M* line reading MATCH above.\n",
           m[0]==0 ? "HOLDS (mantissa table sufficient)" : "FAILED (ping me)");

    // concrete examples for manual diff
    uint32_t* d_ex; cudaMalloc(&d_ex, 8*8*4); examples<<<1,8>>>(d_ex);
    cudaDeviceSynchronize();
    uint32_t ex[64]; cudaMemcpy(ex, d_ex, 8*8*4, cudaMemcpyDeviceToHost);
    printf("examples (hex bits): a b fdividef | M0 M1 M2 M3 M4\n");
    for (int i = 0; i < 8; ++i) {
        uint32_t* o = ex + i*8;
        printf("  %08x %08x %08x | %08x %08x %08x %08x %08x\n",
               o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7]);
    }

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s  sm_%d%d\n", p.name, p.major, p.minor);
    write_npy_i32("rcp_mantissa.npy", h, N);
    printf("wrote rcp_mantissa.npy (%d entries, %.1f MB)\n", N, N * 4.0 / 1e6);
    return 0;
}
