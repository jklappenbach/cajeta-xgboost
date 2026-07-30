// probe_device_grads.cu — capture REAL device grad/hess from the reference
// objective arithmetic, on the fixture margins, on NVIDIA hardware.
//
// Why this exists (gpu-numeric-fidelity 1.1.2): the committed fixtures'
// grad.npy for rounds >= 1 is a NUMPY reconstruction (tools/fixtures/
// generate.py::compute_grad_hess), not a device dump — so the multi-round
// grad parity tests never had device ground truth. This probe produces it:
// for EVERY margin row it computes grad/hess with a verbatim transcription
// of the pinned xgboost 3.1.2 device objective code, run as real CUDA on the
// real GPU (so expf IS libdevice expf, on the SFU).
//
// Output semantics: device_grad.npy / device_hess.npy are keyed by INPUT
// margin row — row r holds the grads the device computes FROM margins[r].
// The fixture mapping is grads[r] <- margins[r-1] for r >= 1 (round 0 comes
// from the base margin and is already covered by the exact analytic tests),
// so the consumer aligns rows; the probe itself does no round bookkeeping.
//
//   nvcc -arch=native -O2 probe_device_grads.cu -o probe_device_grads
//   ./probe_device_grads <fixture-dir> binary|multi <out-dir>
//
// Reads <fixture-dir>/margins.npy (float64 [R,n] or [R,n,K]) + y.npy
// (float64 [n]). Writes <out-dir>/device_grad.npy + device_hess.npy with the
// same shape as margins (float64 holding float32-exact values — the fixture
// convention). No fast-math flags: xgboost builds its objectives without
// them, which is precisely why device expf == the libdevice sequence the
// FastMath model reproduces.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Verbatim reference transcriptions. Provenance: xgboost 3.1.2 (the pinned
// xgboost-ref), cited per function. Any edit here invalidates the capture.
// ---------------------------------------------------------------------------

// src/common/math.h:28-34 — Sigmoid(float), including the 88.7 overflow clamp
// and the kEps in the denominator.
__device__ __forceinline__ float ref_sigmoid(float x) {
    float constexpr kEps = 1e-16;  // avoid 0 div
    x = fminf(-x, 88.7f);          // std::min on device floats == fminf
    float denom = expf(x) + 1.0f + kEps;
    float y = 1.0f / denom;
    return y;
}

// src/objective/regression_loss.h:63-72 — LogisticRegression first/second
// order gradients (weights = 1 in every fixture).
__global__ void k_binary(const double* margin, const double* y, int64_t n,
                         double* grad, double* hess) {
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p = ref_sigmoid((float) margin[i]);
    float g = p - (float) y[i];
    const float eps = 1e-16f;
    float h = fmaxf(p * (1.0f - p), eps);
    grad[i] = (double) g;
    hess[i] = (double) h;
}

// src/objective/multiclass_obj.cu:120-143 — the SoftmaxMultiClassObj
// GetGradient kernel body (weights = 1): wmax seeded with float-min, wsum
// accumulates the FLOAT-ROUNDED expf terms into a double, and p is
// recomputed per class ("duplicated to avoid creating a cache").
__global__ void k_multi(const double* margin, const double* y, int64_t n, int64_t K,
                        double* grad, double* hess) {
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double* point = margin + i * K;

    float wmax = 1.17549435e-38f;               // std::numeric_limits<float>::min()
    for (int64_t k = 0; k < K; ++k) {
        wmax = fmaxf((float) point[k], wmax);
    }
    double wsum = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        wsum += expf((float) point[k] - wmax);
    }
    long label = (long) y[i];
    for (int64_t k = 0; k < K; ++k) {
        float p = expf((float) point[k] - wmax) / static_cast<float>(wsum);
        constexpr float kEps = 1e-16f;
        float h = fmax(2.0f * p * (1.0f - p), kEps);
        p = label == (long) k ? p - 1.0f : p;
        grad[i * K + k] = (double) p;
        hess[i * K + k] = (double) h;
    }
}

// ---------------------------------------------------------------------------
// Minimal npy IO — float64 ('<f8'), C order, up to 3 dims. Matches the
// simplistic reader in tools/expf/model_libdevice.c.
// ---------------------------------------------------------------------------

static std::vector<double> rd_npy_f64(const std::string& path,
                                      std::vector<int64_t>& shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fprintf(stderr, "short %s\n", path.c_str()); exit(1); }
    uint16_t hlen = 0;
    if (fread(&hlen, 2, 1, f) != 1) { fprintf(stderr, "short %s\n", path.c_str()); exit(1); }
    std::vector<char> hdr(hlen + 1, 0);
    if (fread(hdr.data(), 1, hlen, f) != hlen) { fprintf(stderr, "short %s\n", path.c_str()); exit(1); }
    std::string h(hdr.data());
    if (h.find("'<f8'") == std::string::npos) {
        fprintf(stderr, "%s: expected float64 npy\n", path.c_str()); exit(1);
    }
    size_t sp = h.find("'shape': (");
    if (sp == std::string::npos) { fprintf(stderr, "%s: no shape\n", path.c_str()); exit(1); }
    sp += 10;
    shape.clear();
    while (sp < h.size() && h[sp] != ')') {
        while (sp < h.size() && (h[sp] == ' ' || h[sp] == ',')) sp++;
        if (h[sp] == ')') break;
        shape.push_back(strtoll(&h[sp], nullptr, 10));
        while (sp < h.size() && h[sp] != ',' && h[sp] != ')') sp++;
    }
    int64_t count = 1;
    for (int64_t d : shape) count *= d;
    std::vector<double> data((size_t) count);
    if (fread(data.data(), 8, (size_t) count, f) != (size_t) count) {
        fprintf(stderr, "%s: short data\n", path.c_str()); exit(1);
    }
    fclose(f);
    return data;
}

static void wr_npy_f64(const std::string& path, const double* d,
                       const std::vector<int64_t>& shape) {
    std::string dims;
    for (size_t i = 0; i < shape.size(); ++i) {
        dims += std::to_string(shape[i]);
        dims += (shape.size() == 1 || i + 1 < shape.size()) ? "," : "";
    }
    char h[192];
    int hl = snprintf(h, sizeof(h),
        "{'descr': '<f8', 'fortran_order': False, 'shape': (%s), }", dims.c_str());
    int pad = (64 - ((10 + hl) % 64)) % 64;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    const unsigned char mg[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    fwrite(mg, 1, 8, f);
    uint16_t L = (uint16_t)(hl + pad + 1);
    fwrite(&L, 2, 1, f); fwrite(h, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    int64_t count = 1;
    for (int64_t dd : shape) count *= dd;
    fwrite(d, 8, (size_t) count, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <fixture-dir> binary|multi <out-dir>\n", argv[0]);
        return 1;
    }
    std::string fdir = argv[1], mode = argv[2], odir = argv[3];

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0; cudaRuntimeGetVersion(&rt);
    printf("device: %s (sm_%d%d), CUDA runtime %d\n",
           prop.name, prop.major, prop.minor, rt);

    std::vector<int64_t> mshape, yshape;
    std::vector<double> margins = rd_npy_f64(fdir + "/margins.npy", mshape);
    std::vector<double> y       = rd_npy_f64(fdir + "/y.npy", yshape);
    int64_t R = mshape[0];
    int64_t n = yshape[0];
    int64_t K = 1;
    if (mode == "multi") {
        if (mshape.size() == 3)      K = mshape[2];
        else if (mshape.size() == 2) K = mshape[1] / n;   // flattened [R, n*K]
        if (K < 2) { fprintf(stderr, "bad K=%lld\n", (long long) K); return 1; }
    }
    printf("margins: R=%lld n=%lld K=%lld\n", (long long) R, (long long) n, (long long) K);

    double *dM, *dY, *dG, *dH;
    size_t row = (size_t)(n * K);
    CK(cudaMalloc(&dM, row * 8)); CK(cudaMalloc(&dY, (size_t) n * 8));
    CK(cudaMalloc(&dG, row * 8)); CK(cudaMalloc(&dH, row * 8));
    CK(cudaMemcpy(dY, y.data(), (size_t) n * 8, cudaMemcpyHostToDevice));

    std::vector<double> grad(margins.size()), hess(margins.size());
    for (int64_t r = 0; r < R; ++r) {
        CK(cudaMemcpy(dM, margins.data() + (size_t) r * row, row * 8, cudaMemcpyHostToDevice));
        int tpb = 256;
        int blocks = (int)((n + tpb - 1) / tpb);
        if (mode == "binary") k_binary<<<blocks, tpb>>>(dM, dY, n, dG, dH);
        else                  k_multi<<<blocks, tpb>>>(dM, dY, n, K, dG, dH);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(grad.data() + (size_t) r * row, dG, row * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hess.data() + (size_t) r * row, dH, row * 8, cudaMemcpyDeviceToHost));
    }

    wr_npy_f64(odir + "/device_grad.npy", grad.data(), mshape);
    wr_npy_f64(odir + "/device_hess.npy", hess.data(), mshape);
    printf("wrote %s/device_grad.npy + device_hess.npy\n", odir.c_str());
    return 0;
}
