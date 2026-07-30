// probe_evaluate_gain.cu — replay ONE node's split evaluation with a verbatim
// transcription of the reference GPU evaluate agent, on NVIDIA hardware, and
// dump every boundary's gain bits (gpu-numeric-fidelity 2.2.1).
//
// Why this exists: after modeling the device's three-level selection rule
// (mcw-blind 32-bin tile argmax -> gated tile winner -> cross-feature reduce),
// tiny_reg_mcw10 node 14 still diverges: the port computes a VALID gain 96.706
// on feature 1 while the reference records feature 4 at 96.521. Either the
// device computes a different number at that boundary (an op-order/narrowing
// residual the CPU model misses) or it discards the candidate somewhere no
// source reading has found. This probe answers which, with silicon.
//
// Inputs (a directory written by cajeta-xgboost's TreeBuilder dump hook —
// XGBOOST_NODE_DUMP=<dir> XGBOOST_NODE_DUMP_NID=<nid> during a test run):
//   hist_g.npy, hist_h.npy  int64 [nbins]   node histogram (quantised)
//   parent.npy              int64 [2]       parent sum (g, h)
//   cut_ptrs.npy            int64 [nf+1]    per-feature bin ranges
//   scales.npy              float64 [5]     to_float_g, to_float_h, lambda,
//                                           alpha, min_child_weight
//
//   nvcc -arch=native -O2 probe_evaluate_gain.cu -o probe_evaluate_gain
//   ./probe_evaluate_gain <dump-dir> <out-dir>
//
// Outputs:
//   <out-dir>/gain_bits.npy  float64 [nbins, 4] — per boundary, as exact
//     uint32 bit patterns stored in f64: [missLeft gain, missRight gain,
//     chosen gain, chosen missing_left flag]
//   <out-dir>/feature_best.npy float64 [nf, 6] — per feature: [gidx, gain
//     bits, missing_left, left_h_quantised, right_h_quantised, valid]
//   stdout: per-feature winners + the cross-feature fold result.
//
// Provenance (xgboost 3.1.2, the pinned xgboost-ref; any edit invalidates
// the capture):
//   EvaluateSplitAgent / Numerical      src/tree/gpu_hist/evaluate_splits.cu:42-148
//   LossChangeMissing                   src/tree/gpu_hist/evaluate_splits.cu:16-33
//   CalcSplitGain / CalcGainGivenWeight src/tree/split_evaluator.h:76-145
//   ThresholdL1                         src/tree/param.h:226-236
//   GradientQuantiser::ToFloatingPoint  src/tree/gpu_hist/quantiser.cuh:32-37
//   DeviceSplitCandidate::Update        src/tree/updater_gpu_common.cuh:69-84
//   SumCallbackOp                       src/tree/updater_gpu_common.cuh:139-150
//   GPUTrainingParam (float32 fields!)  src/tree/updater_gpu_common.cuh:15-28
//   cross-feature reduce operator+      src/tree/gpu_hist/evaluate_splits.cu:312-315
//
// The one deliberately non-verbatim piece: the reference folds per-feature
// candidates with cub::DeviceSegmentedReduce::Sum, whose reduction TREE ORDER
// is unspecified; ties between features resolve by that shape. This probe
// folds sequentially in feature order (keeps the earlier feature on ties) and
// prints every per-feature candidate so any cross-feature near-tie is visible
// in the output regardless of fold order.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Verbatim reference types (trimmed to what evaluation touches).
// ---------------------------------------------------------------------------

// include/xgboost/base.h:271-322 — element-wise int64 ops, zero-init.
struct GradientPairInt64 {
    int64_t g = 0, h = 0;
    __host__ __device__ GradientPairInt64() {}
    __host__ __device__ GradientPairInt64(int64_t g_, int64_t h_) : g(g_), h(h_) {}
    __host__ __device__ GradientPairInt64 operator+(const GradientPairInt64& o) const {
        return {g + o.g, h + o.h};
    }
    __host__ __device__ GradientPairInt64& operator+=(const GradientPairInt64& o) {
        g += o.g; h += o.h; return *this;
    }
    __host__ __device__ GradientPairInt64 operator-(const GradientPairInt64& o) const {
        return {g - o.g, h - o.h};
    }
};

// src/tree/updater_gpu_common.cuh:15-28 — the params are FLOAT32 on device.
struct GPUTrainingParam {
    float min_child_weight;
    float reg_lambda;
    float reg_alpha;
    // max_delta_step == 0, no monotone constraints in every parity fixture.
};

// src/tree/gpu_hist/quantiser.cuh:32-37 — int64 -> double, exact pow-2 scale.
struct Quantiser {
    double to_float_g, to_float_h;
    __host__ __device__ double2 ToFloatingPoint(const GradientPairInt64& p) const {
        return { (double) p.g * to_float_g, (double) p.h * to_float_h };
    }
};

// src/tree/updater_gpu_common.cuh:139-150.
template <typename T>
struct SumCallbackOp {
    T running_total;
    __device__ SumCallbackOp() : running_total() {}
    __device__ T operator()(T block_aggregate) {
        T old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

// src/tree/param.h:226-236 — T1 = double, T2 = float on the device path.
__device__ __forceinline__ double ThresholdL1(double w, float alpha) {
    if (w > +alpha) { return w - alpha; }
    if (w < -alpha) { return w + alpha; }
    return 0.0;
}

// src/common/math.h:21.
__device__ __forceinline__ double Sqr(double w) { return w * w; }

// src/tree/split_evaluator.h:123-145 — the default (unconstrained,
// max_delta_step==0) path: double numerator/denominator, both narrowed to
// float at the call, then __fdividef.
__device__ __forceinline__ float CalcGainGivenWeight(const GPUTrainingParam& p,
                                                     double2 stats) {
    if (stats.y <= 0) { return .0f; }
    return __fdividef((float) Sqr(ThresholdL1(stats.x, p.reg_alpha)),
                      (float) (stats.y + p.reg_lambda));
}

// src/tree/split_evaluator.h:76-94 — constraint == 0, gain = left + right.
__device__ __forceinline__ float CalcSplitGain(const GPUTrainingParam& p,
                                               double2 left, double2 right) {
    return CalcGainGivenWeight(p, left) + CalcGainGivenWeight(p, right);
}

// src/tree/gpu_hist/evaluate_splits.cu:16-33.
__device__ __forceinline__ float LossChangeMissing(
        const GradientPairInt64& scan, const GradientPairInt64& missing,
        const GradientPairInt64& parent_sum, const GPUTrainingParam& param,
        const Quantiser& quantiser, bool& missing_left_out,
        float* dbg_left_gain, float* dbg_right_gain) {
    const auto left_sum = scan + missing;
    float missing_left_gain = CalcSplitGain(
        param, quantiser.ToFloatingPoint(left_sum),
        quantiser.ToFloatingPoint(parent_sum - left_sum));
    float missing_right_gain = CalcSplitGain(
        param, quantiser.ToFloatingPoint(scan),
        quantiser.ToFloatingPoint(parent_sum - scan));
    *dbg_left_gain = missing_left_gain;
    *dbg_right_gain = missing_right_gain;
    missing_left_out = missing_left_gain > missing_right_gain;
    return missing_left_out ? missing_left_gain : missing_right_gain;
}

// src/tree/updater_gpu_common.cuh:54,69-84 — per-feature running best.
struct DeviceSplitCandidate {
    float loss_chg = -FLT_MAX;
    int   gidx = -1;
    bool  missing_left = true;
    GradientPairInt64 left_sum, right_sum;
    __device__ void Update(float loss_chg_in, bool missing_left_in, int gidx_in,
                           GradientPairInt64 left_sum_in, GradientPairInt64 right_sum_in,
                           const GPUTrainingParam& param, const Quantiser& quantiser) {
        if (loss_chg_in > loss_chg &&
            quantiser.ToFloatingPoint(left_sum_in).y >= param.min_child_weight &&
            quantiser.ToFloatingPoint(right_sum_in).y >= param.min_child_weight) {
            loss_chg = loss_chg_in;
            missing_left = missing_left_in;
            gidx = gidx_in;
            left_sum = left_sum_in;
            right_sum = right_sum_in;
        }
    }
};

// ---------------------------------------------------------------------------
// The agent kernel — one block of 32 threads per feature, mirroring
// EvaluateSplitAgent<32>::Numerical (evaluate_splits.cu:42-148).
// ---------------------------------------------------------------------------

constexpr int kBlockSize = 32;
constexpr float kNullGain = -std::numeric_limits<float>::infinity();

struct FeatureBest {
    float loss_chg;
    int   gidx;
    int   missing_left;
    long long left_g, left_h, right_g, right_h;
};

__global__ void k_evaluate(const int64_t* hist_g, const int64_t* hist_h,
                           const int64_t* cut_ptrs, int64_t nf,
                           GradientPairInt64 parent_sum,
                           GPUTrainingParam param, Quantiser quantiser,
                           FeatureBest* out_best,
                           // debug: per global bin [nbins]
                           float* dbg_gain_ml, float* dbg_gain_mr,
                           float* dbg_gain, unsigned char* dbg_missleft) {
    int fidx = blockIdx.x;
    if (fidx >= nf) return;
    int gidx_begin = (int) cut_ptrs[fidx];
    int gidx_end   = (int) cut_ptrs[fidx + 1];

    using BlockScanT = cub::BlockScan<GradientPairInt64, kBlockSize>;
    using ArgMaxT = cub::KeyValuePair<int, float>;
    using MaxReduceT = cub::WarpReduce<ArgMaxT>;
    using SumReduceT = cub::WarpReduce<GradientPairInt64>;
    __shared__ union {
        typename BlockScanT::TempStorage scan;
        typename MaxReduceT::TempStorage max_reduce;
        typename SumReduceT::TempStorage sum_reduce;
    } temp_storage;

    // ReduceFeature (evaluate_splits.cu:92-102) — int64 exact, order-free.
    GradientPairInt64 local_sum;
    for (int idx = gidx_begin + threadIdx.x; idx < gidx_end; idx += kBlockSize) {
        local_sum += GradientPairInt64{hist_g[idx], hist_h[idx]};
    }
    local_sum = SumReduceT(temp_storage.sum_reduce).Sum(local_sum);
    GradientPairInt64 feature_sum{
        __shfl_sync(0xffffffff, (long long) local_sum.g, 0),
        __shfl_sync(0xffffffff, (long long) local_sum.h, 0)};
    GradientPairInt64 missing = parent_sum - feature_sum;

    DeviceSplitCandidate best;
    SumCallbackOp<GradientPairInt64> prefix_op;

    // Numerical (evaluate_splits.cu:113-148).
    for (int scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += kBlockSize) {
        bool thread_active = (scan_begin + (int) threadIdx.x) < gidx_end;
        GradientPairInt64 bin = thread_active
            ? GradientPairInt64{hist_g[scan_begin + threadIdx.x],
                                hist_h[scan_begin + threadIdx.x]}
            : GradientPairInt64();
        BlockScanT(temp_storage.scan).ExclusiveScan(bin, bin, cub::Sum{}, prefix_op);

        bool missing_left = true;
        float g_ml = kNullGain, g_mr = kNullGain;
        float gain = thread_active
            ? LossChangeMissing(bin, missing, parent_sum, param, quantiser,
                                missing_left, &g_ml, &g_mr)
            : kNullGain;

        if (thread_active) {
            int p = scan_begin + threadIdx.x;
            dbg_gain_ml[p] = g_ml;
            dbg_gain_mr[p] = g_mr;
            dbg_gain[p] = gain;
            dbg_missleft[p] = missing_left ? 1 : 0;
        }

        auto best_arg = MaxReduceT(temp_storage.max_reduce)
                            .Reduce({(int) threadIdx.x, gain}, cub::ArgMax());
        auto best_thread = __shfl_sync(0xffffffff, best_arg.key, 0);

        if ((int) threadIdx.x == best_thread) {
            GradientPairInt64 left = missing_left ? bin + missing : bin;
            GradientPairInt64 right = parent_sum - left;
            best.Update(gain, missing_left, scan_begin + (int) threadIdx.x,
                        left, right, param, quantiser);
        }
        __syncwarp();
    }

    if (threadIdx.x == 0) {
        out_best[fidx] = {best.loss_chg, best.gidx, best.missing_left ? 1 : 0,
                          (long long) best.left_sum.g, (long long) best.left_sum.h,
                          (long long) best.right_sum.g, (long long) best.right_sum.h};
    }
}

// ---------------------------------------------------------------------------
// Minimal npy IO (matches probe_device_grads.cu's reader, plus '<i8').
// ---------------------------------------------------------------------------

static std::vector<char> rd_raw(const std::string& path, const char* descr,
                                std::vector<int64_t>& shape, size_t elem) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fprintf(stderr, "short %s\n", path.c_str()); exit(1); }
    uint16_t hlen = 0;
    if (fread(&hlen, 2, 1, f) != 1) { fprintf(stderr, "short %s\n", path.c_str()); exit(1); }
    std::vector<char> hdr(hlen + 1, 0);
    if (fread(hdr.data(), 1, hlen, f) != hlen) { fprintf(stderr, "short %s\n", path.c_str()); exit(1); }
    std::string h(hdr.data());
    if (h.find(descr) == std::string::npos) {
        fprintf(stderr, "%s: expected %s npy\n", path.c_str(), descr); exit(1);
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
    std::vector<char> data((size_t) count * elem);
    if (fread(data.data(), elem, (size_t) count, f) != (size_t) count) {
        fprintf(stderr, "%s: short data\n", path.c_str()); exit(1);
    }
    fclose(f);
    return data;
}

static std::vector<int64_t> rd_npy_i64(const std::string& p, std::vector<int64_t>& s) {
    auto raw = rd_raw(p, "'<i8'", s, 8);
    std::vector<int64_t> v(raw.size() / 8);
    memcpy(v.data(), raw.data(), raw.size());
    return v;
}

static std::vector<double> rd_npy_f64(const std::string& p, std::vector<int64_t>& s) {
    auto raw = rd_raw(p, "'<f8'", s, 8);
    std::vector<double> v(raw.size() / 8);
    memcpy(v.data(), raw.data(), raw.size());
    return v;
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

static uint32_t f32_bits(float v) { uint32_t b; memcpy(&b, &v, 4); return b; }

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <dump-dir> <out-dir>\n", argv[0]);
        return 1;
    }
    std::string ddir = argv[1], odir = argv[2];

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    std::vector<int64_t> s;
    auto hist_g   = rd_npy_i64(ddir + "/hist_g.npy", s);
    int64_t nbins = s[0];
    auto hist_h   = rd_npy_i64(ddir + "/hist_h.npy", s);
    auto parent   = rd_npy_i64(ddir + "/parent.npy", s);
    auto cut_ptrs = rd_npy_i64(ddir + "/cut_ptrs.npy", s);
    int64_t nf    = s[0] - 1;
    auto scales   = rd_npy_f64(ddir + "/scales.npy", s);

    Quantiser q{scales[0], scales[1]};
    // GPUTrainingParam narrows the config to float32 exactly as the device
    // does (updater_gpu_common.cuh:15-28) — this is deliberate, not lossy
    // transcription: reg params live as float on device.
    GPUTrainingParam param{(float) scales[4], (float) scales[2], (float) scales[3]};
    GradientPairInt64 parent_sum{parent[0], parent[1]};

    printf("nbins=%lld nf=%lld parent=(%lld,%lld) to_float=(%a,%a)\n",
           (long long) nbins, (long long) nf,
           (long long) parent[0], (long long) parent[1], scales[0], scales[1]);
    printf("param: lambda=%a alpha=%a min_child_weight=%a (as f32)\n",
           (double) param.reg_lambda, (double) param.reg_alpha,
           (double) param.min_child_weight);

    int64_t *d_hg, *d_hh, *d_cp;
    FeatureBest* d_best;
    float *d_ml, *d_mr, *d_gain;
    unsigned char* d_missleft;
    CK(cudaMalloc(&d_hg, nbins * 8)); CK(cudaMalloc(&d_hh, nbins * 8));
    CK(cudaMalloc(&d_cp, (nf + 1) * 8));
    CK(cudaMalloc(&d_best, nf * sizeof(FeatureBest)));
    CK(cudaMalloc(&d_ml, nbins * 4)); CK(cudaMalloc(&d_mr, nbins * 4));
    CK(cudaMalloc(&d_gain, nbins * 4)); CK(cudaMalloc(&d_missleft, nbins));
    CK(cudaMemcpy(d_hg, hist_g.data(), nbins * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_hh, hist_h.data(), nbins * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cp, cut_ptrs.data(), (nf + 1) * 8, cudaMemcpyHostToDevice));

    k_evaluate<<<(int) nf, kBlockSize>>>(d_hg, d_hh, d_cp, nf, parent_sum,
                                         param, q, d_best,
                                         d_ml, d_mr, d_gain, d_missleft);
    CK(cudaDeviceSynchronize());

    std::vector<FeatureBest> best(nf);
    std::vector<float> ml(nbins), mr(nbins), gain(nbins);
    std::vector<unsigned char> missleft(nbins);
    CK(cudaMemcpy(best.data(), d_best, nf * sizeof(FeatureBest), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ml.data(), d_ml, nbins * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(mr.data(), d_mr, nbins * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gain.data(), d_gain, nbins * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(missleft.data(), d_missleft, nbins, cudaMemcpyDeviceToHost));

    // Cross-feature fold, sequential in feature order (see the header note on
    // cub::DeviceSegmentedReduce tree order). Strict >: earlier feature wins ties.
    int win = 0;
    for (int f = 1; f < nf; ++f) {
        if (best[f].loss_chg > best[win].loss_chg) { win = f; }
    }
    for (int f = 0; f < nf; ++f) {
        printf("feature %d: gidx=%d gain=%.9g (bits 0x%08x) missing_left=%d "
               "left_h=%lld right_h=%lld\n",
               f, best[f].gidx, (double) best[f].loss_chg, f32_bits(best[f].loss_chg),
               best[f].missing_left, best[f].left_h, best[f].right_h);
    }
    printf("WINNER: feature %d gidx=%d gain bits 0x%08x\n",
           win, best[win].gidx, f32_bits(best[win].loss_chg));

    // gain_bits.npy [nbins, 4]: exact uint32 bit patterns held in f64.
    std::vector<double> gb((size_t) nbins * 4);
    for (int64_t p = 0; p < nbins; ++p) {
        gb[p * 4 + 0] = (double) f32_bits(ml[(size_t) p]);
        gb[p * 4 + 1] = (double) f32_bits(mr[(size_t) p]);
        gb[p * 4 + 2] = (double) f32_bits(gain[(size_t) p]);
        gb[p * 4 + 3] = (double) missleft[(size_t) p];
    }
    wr_npy_f64(odir + "/gain_bits.npy", gb.data(), {nbins, 4});

    std::vector<double> fb((size_t) nf * 6);
    for (int64_t f = 0; f < nf; ++f) {
        fb[f * 6 + 0] = (double) best[f].gidx;
        fb[f * 6 + 1] = (double) f32_bits(best[f].loss_chg);
        fb[f * 6 + 2] = (double) best[f].missing_left;
        fb[f * 6 + 3] = (double) best[f].left_h;
        fb[f * 6 + 4] = (double) best[f].right_h;
        fb[f * 6 + 5] = best[f].gidx >= 0 ? 1.0 : 0.0;
    }
    wr_npy_f64(odir + "/feature_best.npy", fb.data(), {nf, 6});
    printf("wrote %s/gain_bits.npy + feature_best.npy\n", odir.c_str());
    return 0;
}
