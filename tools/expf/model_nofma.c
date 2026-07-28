// model_nofma.c — the U1.2.3 implementation strategy, validated before porting.
//
// model_libdevice.c proved the MODEL. This proves the MODEL IS REACHABLE FROM
// CAJETA, which has neither an FMA intrinsic nor rounding-mode control:
//
//   * fmaf(a,b,c)          -> emu_fmaf: exact double product + 2Sum residual +
//                             round-to-odd, so the float32 result is rounded ONCE.
//   * fma_rd(j,252,2^23+1) -> floor(j*252) in double, which is exact.
//
// Build deliberately without FMA and with contraction off, so nothing sneaks a
// hardware fma back in and flatters the result:
//   cc -O2 -ffp-contract=off -mno-fma model_nofma.c -o model_nofma -lm && ./model_nofma

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static const float L2E_HI = 1.442695021629333496e+0f;
static const float L2E_LO = 1.925962991460541939e-8f;

static inline float b2f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static inline uint32_t f2b(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static inline double b2d(uint64_t u) { double x; memcpy(&x, &u, 8); return x; }
static inline uint64_t d2b(double x) { uint64_t u; memcpy(&u, &x, 8); return u; }

// Exact float32 FMA with no hardware FMA.
//
// a*b is EXACT in double (24+24 = 48 <= 53 significand bits), so the only
// rounding to worry about is the add. Knuth 2Sum recovers that error exactly,
// giving the true sum as the unevaluated pair (s, err). Rounding s to float32
// directly would round twice (exact -> 53 bits -> 24 bits) and can land on the
// wrong side of a float32 midpoint. Round-to-odd on the double first makes the
// second rounding exact: a float32 midpoint needs 25 significand bits, so as a
// double its low bits are zero and it is never odd — an odd double can never be
// a tie, so round-to-nearest float32 cannot break the wrong way. 53 >= 2*24+2,
// which is the condition that makes this correct.
static float emu_fmaf(float a, float b, float c) {
    double p  = (double) a * (double) b;      // exact
    double cc = (double) c;
    double s  = p + cc;                       // the one rounding

    double bv  = s - p;                       // Knuth 2Sum, exact for any inputs
    double err = (p - (s - bv)) + (cc - bv);  // p + cc == s + err, exactly

    if (err != 0.0) {
        uint64_t sb = d2b(s);
        if ((sb & 1ull) == 0) {               // already odd -> already unambiguous
            int towardPos = err > 0.0;
            // Bit patterns grow away from zero, so the direction flips for s < 0.
            if (s >= 0.0) sb = towardPos ? sb + 1 : sb - 1;
            else          sb = towardPos ? sb - 1 : sb + 1;
            s = b2d(sb);
        }
    }
    return (float) s;
}

static uint32_t *load_npy_u32(const char *path, int64_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fprintf(stderr, "short read\n"); exit(1); }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
    if (fseek(f, 10 + hlen, SEEK_SET) != 0) { fprintf(stderr, "seek fail\n"); exit(1); }
    long here = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, here, SEEK_SET);
    int64_t n = (end - here) / 4;
    uint32_t *d = malloc((size_t) n * 4);
    if (fread(d, 4, (size_t) n, f) != (size_t) n) { fprintf(stderr, "short data\n"); exit(1); }
    fclose(f);
    *count = n;
    return d;
}

static uint32_t *EX2TAB;
static int64_t   EX2N;

static inline float ex2_approx(float y) {
    double yd = (double) y;
    double nf = floor(yd);
    double fr = yd - nf;
    int64_t idx = (int64_t) floor(fr * (double) EX2N);
    if (idx < 0) idx = 0;
    if (idx >= EX2N) idx = EX2N - 1;
    float base = b2f(EX2TAB[idx]);
    return (float) ldexp((double) base, (int) nf);
}

static float model_nofma(float a) {
    const float C = L2E_HI / 252.0f;

    float j = emu_fmaf(a, C, 0.5f);
    if (isnan(j)) j = 0.0f;
    if (j < 0.0f) j = 0.0f;
    if (j > 1.0f) j = 1.0f;

    // i = fma_rd(j, 252, 2^23+1). The exact value lands in [2^23+1, 2^23+253],
    // a range where every float32 IS an integer (ulp == 1), so rounding toward
    // -inf is plain floor -- and floor(2^23+1 + j*252) = 2^23+1 + floor(j*252).
    // j*252 needs at most 30 significand bits, so it is exact in double and the
    // floor is exact too. No rounding mode required.
    int32_t k = (int32_t) floor((double) j * 252.0);   // 0..252
    int32_t m = k + 1;                                 // i - 2^23, 1..253
    float   n = (float) (k - 126);                     // i - (2^23+127), -126..126
    float   s = b2f((uint32_t) m << 23);               // 2^n

    float f = emu_fmaf(a, L2E_HI, -n);
    f = emu_fmaf(a, L2E_LO, f);

    return ex2_approx(f) * s;
}

int main(void) {
    int64_t tn, sn;
    EX2TAB = load_npy_u32("ex2_table.npy", &tn);
    uint32_t *sweep = load_npy_u32("expf_sweep.npy", &sn);
    EX2N = tn;

    const float step = 180.0f / 8388607.0f;
    int64_t tot = 0, bad = 0, maxulp = 0, xg_tot = 0, xg_bad = 0, denorm_bad = 0;

    for (int64_t k = 0; k < sn; k++) {
        float x = fmaf((float) k, step, -90.0f);   // fixture index math, not the model
        float dev = b2f(sweep[k]);
        if (!isfinite(dev) || dev == 0.0f) continue;
        float got = model_nofma(x);
        tot++;
        int64_t d = (int64_t)(int32_t) f2b(got) - (int64_t)(int32_t) f2b(dev);
        int in_xg = (x <= 88.7f);
        if (in_xg) xg_tot++;
        if (d != 0) {
            bad++;
            if (in_xg) xg_bad++;
            if (fabsf(dev) < 1.17549435e-38f) denorm_bad++;
            int64_t ad = d < 0 ? -d : d;
            if (ad > maxulp) maxulp = ad;
        }
    }
    printf("finite points : %lld\n", (long long) tot);
    printf("mismatches    : %lld (%.6f%%)  maxULP=%lld\n",
           (long long) bad, 100.0 * (double) bad / (double) tot, (long long) maxulp);
    printf("  of those, denormal-result: %lld\n", (long long) denorm_bad);
    printf("XGBoost domain: %lld / %lld mismatch (%.6f%%)\n",
           (long long) xg_bad, (long long) xg_tot,
           100.0 * (double) xg_bad / (double) xg_tot);
    return 0;
}
