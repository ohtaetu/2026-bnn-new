// XNOR-popcount inference of the 784-512-10 BNN (Algorithm 2) on an NVIDIA GPU.
//
// One fused kernel goes from raw uint8 pixels to class predictions:
//   1. bit packing: each lane of a warp tests one pixel (pixel >= threshold) and
//      __ballot_sync packs the 32 results into one uint32 word,
//   2. hidden layer: one thread per hidden neuron keeps its 25 packed weight words in
//      registers and computes  z = n_in - 2 * sum popc(x ^ w)  (32 synapses per popc),
//   3. sign + re-packing: z >= T is again packed with __ballot_sync, so warp k produces
//      hidden word k directly,
//   4. output layer: lanes 0..9 of warp 0 do the 512-bit XNOR-popcount, apply the folded
//      BatchNorm (a * z + c) and a warp-shuffle argmax picks the class.
// Each block processes IMGS images per step so every weight register is reused IMGS times.
//
// Usage: bnn_xnor <model.bin> <testset.bin> [json_out]

#include <cuda_runtime.h>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                         __FILE__, __LINE__);                                         \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

constexpr int N_IN = 784;
constexpr int W_IN = (N_IN + 31) / 32;  // 25 words per image
constexpr int HIDDEN = 512;             // one thread per hidden neuron
constexpr int W_H = HIDDEN / 32;        // 16 words of hidden activations
constexpr int MAX_OUT = 32;
constexpr int IMGS = 4;                 // images per block step
constexpr unsigned FULL = 0xffffffffu;

struct Model {
    uint32_t n_in, n_hidden, n_out, w_in, w_h, pixel_threshold;
    std::vector<int32_t> t1;
    std::vector<uint32_t> w1, w2;
    std::vector<float> a2, c2;
};

struct TestSet {
    uint32_t n, dim;
    std::vector<uint8_t> pixels, labels, expected;
};

static void read_exact(FILE* f, void* dst, size_t bytes, const char* what) {
    if (std::fread(dst, 1, bytes, f) != bytes) {
        std::fprintf(stderr, "unexpected end of file while reading %s\n", what);
        std::exit(1);
    }
}

static FILE* open_checked(const char* path, const char* magic) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    char m[4];
    read_exact(f, m, 4, "magic");
    if (std::memcmp(m, magic, 4) != 0) {
        std::fprintf(stderr, "%s: bad magic, expected %.4s\n", path, magic);
        std::exit(1);
    }
    return f;
}

static Model load_model(const char* path) {
    FILE* f = open_checked(path, "BNN1");
    Model m;
    uint32_t hdr[6];
    read_exact(f, hdr, sizeof hdr, "model header");
    m.n_in = hdr[0]; m.n_hidden = hdr[1]; m.n_out = hdr[2];
    m.w_in = hdr[3]; m.w_h = hdr[4]; m.pixel_threshold = hdr[5];
    if (m.n_in != N_IN || m.n_hidden != HIDDEN || m.w_in != W_IN || m.w_h != W_H || m.n_out > MAX_OUT) {
        std::fprintf(stderr, "model shape %u-%u-%u not supported (kernel is built for %d-%d-<=%d)\n",
                     m.n_in, m.n_hidden, m.n_out, N_IN, HIDDEN, MAX_OUT);
        std::exit(1);
    }
    m.t1.resize(m.n_hidden);
    m.w1.resize(size_t(m.n_hidden) * m.w_in);
    m.w2.resize(size_t(m.n_out) * m.w_h);
    m.a2.resize(m.n_out);
    m.c2.resize(m.n_out);
    read_exact(f, m.t1.data(), m.t1.size() * 4, "t1");
    read_exact(f, m.w1.data(), m.w1.size() * 4, "w1");
    read_exact(f, m.w2.data(), m.w2.size() * 4, "w2");
    read_exact(f, m.a2.data(), m.a2.size() * 4, "a2");
    read_exact(f, m.c2.data(), m.c2.size() * 4, "c2");
    std::fclose(f);
    return m;
}

static TestSet load_testset(const char* path) {
    FILE* f = open_checked(path, "BNT1");
    TestSet t;
    uint32_t hdr[2];
    read_exact(f, hdr, sizeof hdr, "testset header");
    t.n = hdr[0]; t.dim = hdr[1];
    if (t.dim != N_IN) {
        std::fprintf(stderr, "testset has %u pixels per image, expected %d\n", t.dim, N_IN);
        std::exit(1);
    }
    t.pixels.resize(size_t(t.n) * t.dim);
    t.labels.resize(t.n);
    t.expected.resize(t.n);
    read_exact(f, t.pixels.data(), t.pixels.size(), "pixels");
    read_exact(f, t.labels.data(), t.n, "labels");
    read_exact(f, t.expected.data(), t.n, "expected predictions");
    std::fclose(f);
    return t;
}

// ---------------------------------------------------------------- CPU reference

static int popc32(uint32_t x) { return int(std::bitset<32>(x).count()); }

static void cpu_predict(const Model& m, const TestSet& t, std::vector<uint8_t>& pred) {
    pred.resize(t.n);
    uint32_t x[W_IN], h[W_H];
    for (uint32_t i = 0; i < t.n; ++i) {
        const uint8_t* img = &t.pixels[size_t(i) * N_IN];
        std::fill(x, x + W_IN, 0u);
        for (int p = 0; p < N_IN; ++p)
            if (img[p] >= m.pixel_threshold) x[p / 32] |= 1u << (p % 32);
        std::fill(h, h + W_H, 0u);
        for (int j = 0; j < HIDDEN; ++j) {
            int mism = 0;
            for (int k = 0; k < W_IN; ++k) mism += popc32(x[k] ^ m.w1[size_t(j) * W_IN + k]);
            if (N_IN - 2 * mism >= m.t1[j]) h[j / 32] |= 1u << (j % 32);
        }
        int best = 0;
        float best_score = -FLT_MAX;
        for (uint32_t o = 0; o < m.n_out; ++o) {
            int mism = 0;
            for (int k = 0; k < W_H; ++k) mism += popc32(h[k] ^ m.w2[o * W_H + k]);
            float score = m.a2[o] * float(HIDDEN - 2 * mism) + m.c2[o];
            if (score > best_score) { best_score = score; best = int(o); }
        }
        pred[i] = uint8_t(best);
    }
}

// ---------------------------------------------------------------- GPU kernel

__global__ void __launch_bounds__(HIDDEN)
bnn_xnor_fused(const uint8_t* __restrict__ pixels, int n_images, uint32_t pixel_threshold,
               const uint32_t* __restrict__ w1, const int32_t* __restrict__ t1,
               const uint32_t* __restrict__ w2, const float* __restrict__ a2,
               const float* __restrict__ c2, int n_out, uint8_t* __restrict__ pred) {
    __shared__ uint32_t xs[IMGS][W_IN];
    __shared__ uint32_t hs[IMGS][W_H];

    const int j = threadIdx.x;  // hidden neuron
    const int lane = j & 31;
    const int warp = j >> 5;

    uint32_t w[W_IN];  // this neuron's packed weight row, kept in registers
#pragma unroll
    for (int k = 0; k < W_IN; ++k) w[k] = w1[j * W_IN + k];
    const int thr = t1[j];

    uint32_t wo[W_H];  // output weights: lane o of warp 0 owns class o
    float ao = 0.f, co = 0.f;
    if (warp == 0 && lane < n_out) {
#pragma unroll
        for (int k = 0; k < W_H; ++k) wo[k] = w2[lane * W_H + k];
        ao = a2[lane];
        co = c2[lane];
    }

    for (int base = blockIdx.x * IMGS; base < n_images; base += gridDim.x * IMGS) {
        __syncthreads();  // previous step's output stage is done with hs

        // 1. pack pixels: warp w builds words w and w + 16 of every image in this step
        for (int word = warp; word < W_IN; word += HIDDEN / 32) {
            const int p = word * 32 + lane;
#pragma unroll
            for (int i = 0; i < IMGS; ++i) {
                const int img = base + i;
                const bool bit = img < n_images && p < N_IN &&
                                 pixels[size_t(img) * N_IN + p] >= pixel_threshold;
                const uint32_t packed = __ballot_sync(FULL, bit);
                if (lane == 0) xs[i][word] = packed;
            }
        }
        __syncthreads();

        // 2-3. hidden layer: XNOR-popcount, threshold, re-pack with ballot
#pragma unroll
        for (int i = 0; i < IMGS; ++i) {
            int mism = 0;
#pragma unroll
            for (int k = 0; k < W_IN; ++k) mism += __popc(xs[i][k] ^ w[k]);
            const uint32_t packed = __ballot_sync(FULL, N_IN - 2 * mism >= thr);
            if (lane == 0) hs[i][warp] = packed;
        }
        __syncthreads();

        // 4. output layer and argmax (warp 0)
        if (warp == 0) {
#pragma unroll
            for (int i = 0; i < IMGS; ++i) {
                float score = -FLT_MAX;
                if (lane < n_out) {
                    int mism = 0;
#pragma unroll
                    for (int k = 0; k < W_H; ++k) mism += __popc(hs[i][k] ^ wo[k]);
                    // explicit rounding (no FMA contraction) so scores match the CPU/NumPy path bit for bit
                    score = __fadd_rn(__fmul_rn(ao, float(HIDDEN - 2 * mism)), co);
                }
                int idx = lane;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const float s2 = __shfl_down_sync(FULL, score, off);
                    const int i2 = __shfl_down_sync(FULL, idx, off);
                    if (s2 > score || (s2 == score && i2 < idx)) { score = s2; idx = i2; }
                }
                if (lane == 0 && base + i < n_images) pred[base + i] = uint8_t(idx);
            }
        }
    }
}

// ---------------------------------------------------------------- driver

struct Timing { double us_per_image, ms_per_call; };

static Timing time_kernel(int n, int grid, int reps, const uint8_t* d_pix, const Model& m,
                          const uint32_t* d_w1, const int32_t* d_t1, const uint32_t* d_w2,
                          const float* d_a2, const float* d_c2, uint8_t* d_pred) {
    auto launch = [&] {
        bnn_xnor_fused<<<grid, HIDDEN>>>(d_pix, n, m.pixel_threshold, d_w1, d_t1, d_w2, d_a2, d_c2,
                                         int(m.n_out), d_pred);
    };
    for (int r = 0; r < 20; ++r) launch();
    CUDA_CHECK(cudaGetLastError());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    const double per_call = double(ms) / reps;
    return {1000.0 * per_call / n, per_call};
}

template <typename T>
static T* to_device(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.bin> <testset.bin> [json_out]\n", argv[0]);
        return 1;
    }
    const Model m = load_model(argv[1]);
    const TestSet t = load_testset(argv[2]);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, bnn_xnor_fused, HIDDEN, 0));
    const int n = int(t.n);
    const int grid = std::min((n + IMGS - 1) / IMGS, prop.multiProcessorCount * blocks_per_sm);

    uint8_t* d_pix = to_device(t.pixels);
    uint32_t* d_w1 = to_device(m.w1);
    int32_t* d_t1 = to_device(m.t1);
    uint32_t* d_w2 = to_device(m.w2);
    float* d_a2 = to_device(m.a2);
    float* d_c2 = to_device(m.c2);
    uint8_t* d_pred = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pred, n));

    // correctness: GPU vs CPU reference vs the Python integer path
    bnn_xnor_fused<<<grid, HIDDEN>>>(d_pix, n, m.pixel_threshold, d_w1, d_t1, d_w2, d_a2, d_c2,
                                     int(m.n_out), d_pred);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint8_t> gpu(n), cpu;
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_pred, n, cudaMemcpyDeviceToHost));

    const auto c0 = std::chrono::high_resolution_clock::now();
    cpu_predict(m, t, cpu);
    const double cpu_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - c0).count();

    int correct = 0, gpu_vs_cpu = 0, gpu_vs_python = 0;
    for (int i = 0; i < n; ++i) {
        correct += gpu[i] == t.labels[i];
        gpu_vs_cpu += gpu[i] != cpu[i];
        gpu_vs_python += gpu[i] != t.expected[i];
    }
    const double acc = 100.0 * correct / n;

    const Timing all = time_kernel(n, grid, 200, d_pix, m, d_w1, d_t1, d_w2, d_a2, d_c2, d_pred);
    const Timing one = time_kernel(1, 1, 2000, d_pix, m, d_w1, d_t1, d_w2, d_a2, d_c2, d_pred);

    std::printf("GPU            : %s (sm_%d%d, %d SMs), grid %d x %d threads\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount, grid, HIDDEN);
    std::printf("accuracy       : %.2f%% (%d/%d)\n", acc, correct, n);
    std::printf("mismatches     : GPU vs CPU reference %d, GPU vs Python integer path %d\n", gpu_vs_cpu,
                gpu_vs_python);
    std::printf("GPU, %5d imgs : %.4f ms per call, %.5f us/image (%.1f M images/s)\n", n, all.ms_per_call,
                all.us_per_image, 1.0 / all.us_per_image);
    std::printf("GPU, 1 image   : %.4f ms per call (launch latency included)\n", one.ms_per_call);
    std::printf("CPU reference  : %.4f ms/image (scalar, single thread, unoptimized)\n", cpu_ms / n);

    if (argc > 3) {
        FILE* f = std::fopen(argv[3], "w");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", argv[3]);
            return 1;
        }
        std::fprintf(f,
                     "{\n  \"gpu\": \"%s\",\n  \"accuracy\": %.4f,\n  \"mismatch_gpu_vs_cpu\": %d,\n"
                     "  \"mismatch_gpu_vs_python\": %d,\n  \"batch_all\": {\"images\": %d, \"ms_per_call\": %.6f, "
                     "\"us_per_image\": %.6f},\n  \"batch_1\": {\"ms_per_call\": %.6f},\n"
                     "  \"cpu_reference_ms_per_image\": %.6f\n}\n",
                     prop.name, acc, gpu_vs_cpu, gpu_vs_python, n, all.ms_per_call, all.us_per_image,
                     one.ms_per_call, cpu_ms / n);
        std::fclose(f);
    }

    cudaFree(d_pix); cudaFree(d_w1); cudaFree(d_t1); cudaFree(d_w2);
    cudaFree(d_a2); cudaFree(d_c2); cudaFree(d_pred);
    return (gpu_vs_cpu == 0 && gpu_vs_python == 0) ? 0 : 2;
}
