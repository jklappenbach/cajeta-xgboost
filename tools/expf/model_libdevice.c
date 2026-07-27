// model_libdevice.c — transcription of __nv_expf from the pinned toolchain's
// libdevice IR (captured 2026-07-27 on Phoenix), validated against expf_sweep.npy.
//
//   cc -O2 -mfma model_libdevice.c -o model_libdevice -lm && ./model_libdevice
//
// The IR (non-FTZ path):
//   C  = L2E_HI / 252.0f                       (compile-time fold, round-to-nearest)
//   j  = saturate(fmaf(a, C, 0.5f))            -> [0,1]
//   i  = fma_rd(j, 252.0f, 2^23 + 1.0f)        round-toward -inf
//   n  = i - (2^23 + 127.0f)                   exact; n in [-126,126]
//   f  = fmaf(a, L2E_LO, fmaf(a, L2E_HI, -n))  -> reduced arg in [0,1)
//   s  = bitcast<float>(bitcast<int>(i) << 23) = 2^n
//   r  = ex2.approx.ftz(f) * s
//
// So the SFU *is* the core (on a reduced argument), not a minimax polynomial.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>

#define N (1 << 23)

static const float L2E_HI = 1.442695021629333496e+0f;
static const float L2E_LO = 1.925962991460541939e-8f;

static uint32_t *load_npy_u32(const char *path, int64_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fprintf(stderr, "short read %s\n", path); exit(1); }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fprintf(stderr, "short read %s\n", path); exit(1); }
    if (fseek(f, 10 + hlen, SEEK_SET) != 0) { fprintf(stderr, "seek fail %s\n", path); exit(1); }
    long here = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, here, SEEK_SET);
    int64_t n = (end - here) / 4;
    uint32_t *d = malloc((size_t)n * 4);
    if (fread(d, 4, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short data %s\n", path); exit(1); }
    fclose(f);
    *count = n;
    return d;
}

static inline float b2f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static inline uint32_t f2b(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }

static uint32_t *EX2TAB;
static int64_t   EX2N;

// ex2.approx.ftz.f32, reconstructed from the captured uniform table
// ex2.approx(i/2^23) over [0,1), via ex2(y) = 2^floor(y) * ex2(frac(y)).
// The reduction normally lands y in [0,1), but when `saturate` clamps
// (|x| > 126/log2e ~ 87.34) y leaves the unit interval, so handle the general case.
// MODE 0 = truncate the fraction to 23 bits, 1 = round.
static int EX2_MODE = 0;

static inline float ex2_approx(float y) {
    double yd = (double)y;
    double nf = floor(yd);
    double fr = yd - nf;                       // [0,1), exact (Sterbenz / same binade)
    double t = fr * (double)EX2N;
    int64_t idx = EX2_MODE == 0 ? (int64_t)floor(t) : (int64_t)llrint(t);
    if (idx < 0) idx = 0;
    if (idx >= EX2N) idx = EX2N - 1;
    float base = b2f(EX2TAB[idx]);
    return (float)ldexp((double)base, (int)nf);
}

static float g_f, g_n;                         // last reduction, for diagnostics

static float model_expf(float a) {
    const float C = L2E_HI / 252.0f;          // folded at compile time, round-to-nearest

    float j = fmaf(a, C, 0.5f);
    if (isnan(j)) j = 0.0f;                    // saturate: NaN -> 0
    if (j < 0.0f) j = 0.0f;
    if (j > 1.0f) j = 1.0f;

    int rm = fegetround();
    fesetround(FE_DOWNWARD);
    float i = fmaf(j, 252.0f, 8388609.0f);     // 2^23 + 1
    fesetround(rm);

    float n = i - 8388735.0f;                  // 2^23 + 127

    float f = fmaf(a, L2E_HI, -n);
    f = fmaf(a, L2E_LO, f);

    float s = b2f(f2b(i) << 23);               // = 2^n
    g_f = f; g_n = n;

    return ex2_approx(f) * s;
}

int main(void) {
    int64_t tn, sn;
    EX2TAB = load_npy_u32("ex2_table.npy", &tn);
    uint32_t *sweep = load_npy_u32("expf_sweep.npy", &sn);
    EX2N = tn;
    printf("ex2_table: %lld entries   expf_sweep: %lld entries\n",
           (long long)tn, (long long)sn);

    const float step = 180.0f / 8388607.0f;

    for (EX2_MODE = 0; EX2_MODE <= 1; EX2_MODE++) {
        int64_t tot = 0, bad = 0, maxulp = 0;
        int64_t xg_tot = 0, xg_bad = 0;        // XGBoost's Sigmoid domain: x <= 88.7
        float first_bad_x = 0.0f; int have_first = 0;
        int64_t hist[11]; memset(hist, 0, sizeof hist);

        for (int64_t k = 0; k < sn && k < N; k++) {
            float x = fmaf((float)k, step, -90.0f);
            float dev = b2f(sweep[k]);
            if (!isfinite(dev) || dev == 0.0f) continue;   // skip overflow/underflow tail
            float got = model_expf(x);
            tot++;
            int64_t d = (int64_t)(int32_t)f2b(got) - (int64_t)(int32_t)f2b(dev);
            int in_xg = (x <= 88.7f);
            if (in_xg) xg_tot++;
            if (d != 0) {
                bad++;
                if (in_xg) xg_bad++;
                if (!have_first) { first_bad_x = x; have_first = 1; }
                int64_t ad = d < 0 ? -d : d;
                if (ad > maxulp) maxulp = ad;
            }
            int64_t c = d < -5 ? -5 : (d > 5 ? 5 : d);
            hist[c + 5]++;
        }
        printf("\n--- ex2 index mode: %s ---\n", EX2_MODE == 0 ? "truncate" : "round");
        printf("  finite points   : %lld\n", (long long)tot);
        printf("  mismatches      : %lld  (%.4f%%)   maxULP=%lld\n",
               (long long)bad, 100.0 * (double)bad / (double)tot, (long long)maxulp);
        printf("  XGBoost domain  : %lld / %lld mismatch (%.4f%%)\n",
               (long long)xg_bad, (long long)xg_tot,
               100.0 * (double)xg_bad / (double)xg_tot);
        if (have_first) printf("  first mismatch x: %.9g\n", first_bad_x);
        printf("  bit-error hist  :");
        for (int c = 0; c <= 10; c++) if (hist[c]) printf(" %+d:%lld", c - 5, (long long)hist[c]);
        printf("\n");

        if (EX2_MODE != 0) continue;

        // Where do the survivors live? Bucket by the reduced arg's low bit and by
        // whether the result is denormal / the saturate clamp fired.
        int64_t denorm = 0, clamped = 0, lowbit_set = 0, nbad = 0;
        int64_t byfrac[8]; memset(byfrac, 0, sizeof byfrac);
        printf("  samples (x, f, n, model, device):\n");
        int shown = 0;
        for (int64_t k = 0; k < sn && k < N; k++) {
            float x = fmaf((float)k, step, -90.0f);
            float dev = b2f(sweep[k]);
            if (!isfinite(dev) || dev == 0.0f) continue;
            float got = model_expf(x);
            if (f2b(got) == f2b(dev)) continue;
            nbad++;
            if (fabsf(dev) < 1.17549435e-38f) denorm++;
            if (g_f < 0.0f || g_f >= 1.0f) clamped++;
            if (f2b(g_f) & 1u) lowbit_set++;
            int bucket = (int)(g_f < 0 ? 0 : (g_f >= 1 ? 7 : g_f * 8));
            byfrac[bucket < 0 ? 0 : (bucket > 7 ? 7 : bucket)]++;
            if (shown < 6) {
                printf("    x=%-14.9g f=%-14.9g n=%-6.0f model=%-14.9g dev=%-14.9g (bits %08x vs %08x)\n",
                       x, g_f, g_n, got, dev, f2b(got), f2b(dev));
                shown++;
            }
        }
        printf("  of %lld mismatches: denormal-result=%lld  outside-[0,1)=%lld  f-lowbit-set=%lld\n",
               (long long)nbad, (long long)denorm, (long long)clamped, (long long)lowbit_set);
        printf("  by f octant     :");
        for (int c = 0; c < 8; c++) printf(" [%d/8]:%lld", c, (long long)byfrac[c]);
        printf("\n");
    }
    return 0;
}
