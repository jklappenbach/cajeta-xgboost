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

__global__ void fill_table(uint32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m; uint32_t mb = 0x3F800000u + (uint32_t)i; // m in [1,2)
    memcpy(&m, &mb, 4);
    float r = rcp_approx(m);
    uint32_t rb; memcpy(&rb, &r, 4);
    out[i] = rb;
}

// Verify (a) exponent factoring and (b) fdividef == a*rcp, over a pseudo-random
// sweep of positive normal inputs. Returns mismatch counts via out[0], out[1].
__global__ void verify(const uint32_t* table, uint64_t* mism, int reps) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t s0 = 0, s1 = 0;
    uint32_t rng = 0x9E3779B9u * (uint32_t)(t + 1);
    for (int k = 0; k < reps; ++k) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        // build a positive normal float: exponent in [1,254], random mantissa
        uint32_t exp = 1u + (rng % 254u);
        uint32_t man = (rng >> 3) & 0x7FFFFFu;
        uint32_t bb = (exp << 23) | man;
        float b; memcpy(&b, &bb, 4);
        // (a) factoring: rcp(b) == ldexp(table[man], -(exp-127))
        float rb = rcp_approx(b);
        uint32_t tb = table[man];             // rcp(mantissa in [1,2)) bits
        float tm; memcpy(&tm, &tb, 4);
        int E = (int)exp - 127;
        float recon = ldexpf(tm, -E);
        uint32_t rbb, rcb; memcpy(&rbb, &rb, 4); memcpy(&rcb, &recon, 4);
        if (rbb != rcb) s0++;
        // (b) fdividef(a,b) == a * rcp(b), a random too
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        uint32_t ab = ((1u + (rng % 254u)) << 23) | ((rng >> 3) & 0x7FFFFFu);
        float a; memcpy(&a, &ab, 4);
        float fd = fdividef_dev(a, b);
        float am = a * rb;
        uint32_t fdb, amb; memcpy(&fdb, &fd, 4); memcpy(&amb, &am, 4);
        if (fdb != amb) s1++;
    }
    atomicAdd((unsigned long long*)&mism[0], (unsigned long long)s0);
    atomicAdd((unsigned long long*)&mism[1], (unsigned long long)s1);
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

    uint64_t* d_m; cudaMalloc(&d_m, 16); cudaMemset(d_m, 0, 16);
    verify<<<256, 256>>>(d_out, d_m, 4096);
    cudaDeviceSynchronize();
    uint64_t m[2]; cudaMemcpy(m, d_m, 16, cudaMemcpyDeviceToHost);
    uint64_t total = (uint64_t)256 * 256 * 4096;
    printf("verify: exponent-factoring mismatches = %llu / %llu\n",
           (unsigned long long)m[0], (unsigned long long)total);
    printf("verify: fdividef==a*rcp mismatches     = %llu / %llu\n",
           (unsigned long long)m[1], (unsigned long long)total);
    if (m[0] != 0)
        printf("WARNING: factoring failed — the mantissa table is NOT sufficient; ping me.\n");
    if (m[1] != 0)
        printf("WARNING: __fdividef != a*rcp — fdividef needs a different model; ping me.\n");

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s  sm_%d%d\n", p.name, p.major, p.minor);
    write_npy_i32("rcp_mantissa.npy", h, N);
    printf("wrote rcp_mantissa.npy (%d entries, %.1f MB)\n", N, N * 4.0 / 1e6);
    return 0;
}
