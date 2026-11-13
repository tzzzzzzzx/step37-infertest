// step37_model.cu — T1 装载编排：收货 → 入库
// 施工说明（BUILD-LOG §2）：
//   1. Manifest（收货登记处）→ mmap 三卷 → 专家切片表（主仓库 in-place 指针）
//   2. 常驻集单一大 arena H2D (单块 cudaMalloc + 逐张量 cudaMemcpy)
#include "step37_model.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "step37_dump.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "step37_attn.hpp"
#include "step37_aux.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace step37 {

namespace {

constexpr uint64_t kAlign = 256;

// IQ4_XS: 256 值 / 136 字节（4.25 bpw）
constexpr uint64_t kIq4XsBlockElems = 256;
constexpr uint64_t kIq4XsBlockBytes = 136;

uint64_t expert_slice_bytes(uint64_t in, uint64_t ff) {
    return in * ff / kIq4XsBlockElems * kIq4XsBlockBytes;
}

// mmap 只读映射
struct MappedFileLocal {
    uint8_t* base = nullptr;
    uint64_t bytes = 0;
    int fd = -1;

    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            fd = -1;
            return false;
        }
        bytes = (uint64_t)st.st_size;
        void* p = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd);
            fd = -1;
            return false;
        }
        base = (uint8_t*)p;
        return true;
    }
};

} // namespace

// ===== MappedFile（hpp 声明版）=====

bool MappedFile::open(const char* path) {
    fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        fd = -1;
        return false;
    }
    bytes = (uint64_t)st.st_size;
    void* p = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        fd = -1;
        return false;
    }
    base = (uint8_t*)p;
    return true;
}

// ===== Model: 析构 / 卸载 =====

Model::~Model() {
    unload();
}

void Model::unload() {
    if (d_arena_) {
        cudaFree(d_arena_);
        d_arena_ = nullptr;
    }
    for (auto& m : maps_) {
        if (m.base) ::munmap(m.base, m.bytes);
        if (m.fd >= 0) ::close(m.fd);
    }
    maps_.clear();
    experts_.clear();
    loaded_ = false;
}

// ===== Model::load 步骤 1-2：收货 + mmap =====

bool Model::load(const ModelOptions& opts, std::string& err) {
    unload();
    opts_ = opts;
    const auto t0 = std::chrono::steady_clock::now();

    // 1) 收货登记 + 分类
    manifest_ = scan_directory(opts.model_dir);
    classify(manifest_);

    // 2) mmap 三卷
    maps_.resize(manifest_.shards.size());
    for (size_t i = 0; i < manifest_.shards.size(); ++i) {
        if (!maps_[i].open(manifest_.shards[i].path.c_str())) {
            err = "mmap failed: " + manifest_.shards[i].path;
            return false;
        }
    }

    // 3) 常驻集 arena 布局（对齐 256B）
    arena_bytes_ = 0;
    d_offsets_.assign(manifest_.tensors.size(), UINT64_MAX);
    for (size_t i = 0; i < manifest_.tensors.size(); ++i) {
        if (manifest_.home[i] != Home::VRAM_RESIDENT) continue;
        const uint64_t b = manifest_.tensors[i].bytes();
        if (b == 0) {
            err = "unknown tensor type: " + manifest_.tensors[i].name;
            return false;
        }
        d_offsets_[i] = arena_bytes_;
        arena_bytes_ += (b + kAlign - 1) / kAlign * kAlign;
    }
    report_.resident_bytes = arena_bytes_;

    // 4) VRAM arena + 常驻集 H2D（单块 staging 全量打包一次上传）
    if (cudaMalloc(&d_arena_, arena_bytes_) != cudaSuccess) {
        err = "cudaMalloc arena failed";
        return false;
    }
    {
        std::vector<uint8_t> staging(arena_bytes_);
        for (size_t i = 0; i < manifest_.tensors.size(); ++i) {
            if (manifest_.home[i] != Home::VRAM_RESIDENT) continue;
            const TensorRef& t = manifest_.tensors[i];
            std::memcpy(staging.data() + d_offsets_[i],
                        maps_[t.shard].base + manifest_.shards[t.shard].data_start + t.offset, t.bytes());
        }
        if (cudaMemcpy(d_arena_, staging.data(), arena_bytes_,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "arena H2D failed";
            return false;
        }
    }

    // 5) 专家切片表：42 层 × 288 专家 × gate/up/down（mmap 页缓存 in-place 指针）
    {
        const uint64_t sb = expert_slice_bytes(4096, 1280); // 每专家每矩阵字节（IQ4_XS）
        std::vector<const uint8_t*> gbase(42, nullptr), ubase(42, nullptr), dbase(42, nullptr);
        std::vector<uint32_t> gshard(42), ushard(42), dshard(42);
        for (size_t i = 0; i < manifest_.tensors.size(); ++i) {
            if (manifest_.home[i] != Home::EXPERT_POOL) continue;
            const TensorRef& t = manifest_.tensors[i];
            // blk.N.ffn_gate_exps → 层 N → li = N-3
            const std::string& name = t.name;
            if (name.rfind("blk.", 0) != 0) continue;
            const int layer = std::atoi(name.c_str() + 4);
            if (layer < 3 || layer > 44) continue;
            const int li = layer - 3;
            const uint64_t data_off = manifest_.shards[t.shard].data_start + t.offset;
            if (name.find("ffn_gate_exps") != std::string::npos) {
                gbase[li] = maps_[t.shard].base + data_off;
            } else if (name.find("ffn_up_exps") != std::string::npos) {
                ubase[li] = maps_[t.shard].base + data_off;
            } else if (name.find("ffn_down_exps") != std::string::npos) {
                dbase[li] = maps_[t.shard].base + data_off;
            }
        }
        experts_.assign(42, {});
        for (int li = 0; li < 42; ++li) {
            experts_[li].assign(288, {});
            for (int e = 0; e < 288; ++e) {
                experts_[li][e] = {gbase[li] + (uint64_t)e * sb,
                                   ubase[li] + (uint64_t)e * sb,
                                   dbase[li] + (uint64_t)e * sb};
            }
        }
    }
    report_.expert_bytes = manifest_.expert_bytes;

    // 5.5) 常驻张量 device 地址表（name → {ptr,type}，前向时查）
    dev_map_.clear();
    dev_map_.reserve(manifest_.tensors.size());
    for (size_t i = 0; i < manifest_.tensors.size(); ++i) {
        if (manifest_.home[i] != Home::VRAM_RESIDENT) continue;
        dev_map_.push_back({manifest_.tensors[i].name,
                            {d_arena_ + d_offsets_[i], manifest_.tensors[i].type}});
    }

    // 5.6) KV cache（F32 [pos][8*128]，45 层；cap 由 STEP37_KV_CAP 定，默认 4096）
    {
        kv_cap_ = getenv("STEP37_KV_CAP") ? atoi(getenv("STEP37_KV_CAP")) : 4096;
        swa_cap_ = getenv("STEP37_SWA_CAP") ? atoi(getenv("STEP37_SWA_CAP")) : 4096;
        recompute_kv_layout();
        // q8_0：1088B/位置/层（32 块 × 34B）；SWA 层环形 cap 独立
        const uint64_t kv_bytes = (uint64_t)kv_base_bytes_[44] + (uint64_t)kv_lcap_[44] * 1088;
        if (cudaMalloc(&d_kcache_, kv_bytes) != cudaSuccess ||
            cudaMalloc(&d_vcache_, kv_bytes) != cudaSuccess) {
            err = "KV cache alloc failed";
            return false;
        }
        report_.vram_arena_bytes = arena_bytes_ + kv_bytes * 2;
    }
    // 5.7) M1.5-d: tail-N 层专家整体 VRAM 常驻（STEP37_GPU_TAIL，默认按剩余显存自适应到 11）
    {
        int want = getenv("STEP37_GPU_TAIL") ? atoi(getenv("STEP37_GPU_TAIL")) : 11;
        const uint64_t mat = 288ull * 2785280;          // 一个 _exps 张量
        const uint64_t per_layer = 3 * mat;
        while (want > 0) {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            if ((uint64_t)free_b > per_layer * (uint64_t)want + (1ull << 30)) break;  // 留 1GiB 余量
            --want;
        }
        gpu_tail_ = want;
        if (gpu_tail_ > 0) {
            if (cudaMalloc(&d_gpu_exp_, per_layer * (uint64_t)gpu_tail_) != cudaSuccess) {
                gpu_tail_ = 0;
                d_gpu_exp_ = nullptr;
            } else {
                for (int l = 0; l < gpu_tail_; ++l) {
                    const int li = 42 - gpu_tail_ + l;   // experts_ 索引（blk 3..44 → 0..41）
                    uint8_t* dst = d_gpu_exp_ + per_layer * (uint64_t)l;
                    cudaMemcpy(dst,               experts_[li][0].gate, mat, cudaMemcpyHostToDevice);
                    cudaMemcpy(dst + mat,         experts_[li][0].up,   mat, cudaMemcpyHostToDevice);
                    cudaMemcpy(dst + 2 * mat,     experts_[li][0].down, mat, cudaMemcpyHostToDevice);
                }
            }
        }
        fprintf(stderr, "GPU_TAIL: %d layers resident (%.2f GiB experts)\n",
                gpu_tail_, gpu_tail_ * (double)per_layer / 1073741824.0);
    }
    // 5.7) 工作缓冲 arena：先算尺寸再分配（两遍 seg）
    uint64_t ws_total = 0;
    for (int pass = 0; pass < 2; ++pass) {
        uint64_t off = 0;
        auto seg = [&](int n_f32) -> float* {
            off = (off + 255) / 256 * 256;
            float* p = pass == 1 ? (float*)(d_ws_ + off) : nullptr;
            off += (uint64_t)n_f32 * 4;
            return p;
        };
        ws_.x = seg(4096);
        ws_.xn = seg(4096);
        ws_.q = seg(12288);
        ws_.k = seg(1024);
        ws_.v = seg(1024);
        ws_.attn = seg(12288);
        ws_.gate = seg(96);
        ws_.o = seg(4096);
        ws_.ffn_x = seg(4096);
        ws_.h1 = seg(16 * 1280); // MoE 8 gate + 8 up blocks; dense 11264 fits
        ws_.h2 = seg(11264); // dense up; shared expert 2*1280
        ws_.ffn_y = seg(4096);
        ws_.logits = seg(128896);
        ws_.scores = seg(96 * 8192);
        ws_.router_logits = seg(288);
        ws_.exp_w = seg(8);
        ws_.shared_y = seg(4096);
        ws_.cos_full = seg(8192 * 32);
        ws_.sin_full = seg(8192 * 32);
        ws_.cos_swa = seg(8192 * 64);
        ws_.sin_swa = seg(8192 * 64);
        ws_.e_staging = (float*)seg(24 * 696320); // 24 IQ4_XS 切片（8 专家 × 3 矩阵）
        ws_.parts = seg(8 * 4096);
        ws_.gemv_scratch = seg(262144);           // native_mmvq 的 q8_1 scratch
        ws_.e_ids = (int*)seg(8);
        ws_total = off;
        if (pass == 0) {
            if (cudaMalloc(&d_ws_, ws_total) != cudaSuccess) {
                err = "ws arena alloc failed";
                return false;
            }
        }
    }
    report_.vram_arena_bytes = arena_bytes_ + ws_total;
    report_.tensors = manifest_.tensors.size();
    report_.load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    loaded_ = true;
    return true;
}


// ===== T2: generate =====

const DevTensor* Model::dev(const char* name) const {
    for (const auto& kv : dev_map_)
        if (kv.first == name) return &kv.second;
    return nullptr;
}



// llama3 rope scaling 表（factor 2.0 / low 1.0 / high 32 / orig 131072，HF config 语义）
void build_llama3_rope_table(int n_rot, double base, int max_pos, float* cos_tab, float* sin_tab) {
    const double factor = 2.0;
    const double low_freq_factor = 1.0;
    const double high_freq_factor = 32.0;
    const double orig = 131072.0;
    const int half = n_rot / 2;
    std::vector<float> ct((size_t)max_pos * half);
    std::vector<float> st((size_t)max_pos * half);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv_freq = pow(base, -2.0 * i / n_rot);
            const double wavelen = 2.0 * M_PI / inv_freq;
            const double low_wavelen = orig / low_freq_factor;
            const double high_wavelen = orig / high_freq_factor;
            double f = 1.0;
            if (wavelen >= low_wavelen) {
                f = 1.0 / factor;
            } else if (wavelen > high_wavelen) {
                const double smooth = (orig / wavelen - high_freq_factor) / (low_freq_factor - high_freq_factor);
                f = (1.0 - smooth) + smooth / factor;
            }
            const double ang = inv_freq * f * (double)p;
            ct[(size_t)p * half + i] = (float)cos(ang);
            st[(size_t)p * half + i] = (float)sin(ang);
        }
    }
    cudaMemcpy(cos_tab, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(sin_tab, st.data(), st.size() * 4, cudaMemcpyHostToDevice);
}

bool Model::generate(const std::vector<int32_t>& prompt_ids, int n_decode,
                     std::vector<int32_t>& out_tokens, double& decode_tps, std::string& err,
                     int resume_prompt, int base_pos) {
    if (!loaded_) { err = "not loaded"; return false; }
    {   // rope 表容量自适应（修雷：arena 表按 8192 定容，prefill_run 把 kv_cap 提到 50K+ 后
        // 旧代码仍按 kv_cap_ 写表 = 溢出写 arena；decode pos>8191 读越界）
        static uint32_t built_cap = 0;
        static float *dcf = nullptr, *dsf = nullptr, *dcs = nullptr, *dss = nullptr;
        if (built_cap != kv_cap_) {
            if (dcf) { cudaFree(dcf); cudaFree(dsf); cudaFree(dcs); cudaFree(dss); }
            const size_t nb_f = (size_t)kv_cap_ * 32 * 4, nb_s = (size_t)kv_cap_ * 64 * 4;
            cudaMalloc(&dcf, nb_f); cudaMalloc(&dsf, nb_f);
            cudaMalloc(&dcs, nb_s); cudaMalloc(&dss, nb_s);
            build_llama3_rope_table(64, 5e6, kv_cap_, dcf, dsf);
            build_llama3_rope_table(128, 10000.0, kv_cap_, dcs, dss);
            ws_.cos_full = dcf; ws_.sin_full = dsf; ws_.cos_swa = dcs; ws_.sin_swa = dss;
            built_cap = kv_cap_;
        }
        static float* df_scratch = nullptr;   // decode flash 段态 scratch（96 头 × 128 段 × 130）
        if (!df_scratch) {
            cudaMalloc(&df_scratch, (size_t)96 * 128 * 130 * 4);
            ws_.df_scratch = df_scratch;
        }
        // big_scores（96×kv_cap×4 = 262144 时 100MB）已砍：ws_.scores 只写不读——
        // 消费方是死路径（sm_89 恒走 attn_prefill_mma，scores 在 kernel 内部 smem）
    // tile per-head norm gammas to kernel shape ([n_rows*n], one gamma per row)
    if (!d_qn_gamma_) {
        cudaMalloc(&d_qn_gamma_, (size_t)45 * 96 * 128 * 4);
        cudaMalloc(&d_kn_gamma_, (size_t)45 * 8 * 128 * 4);
        std::vector<float> w128(128), tiled(96 * 128);
        for (int il = 0; il < 45; ++il) {
            const int nqh = (il % 4 == 0) ? 64 : 96;
            char gnm[64];
            snprintf(gnm, sizeof gnm, "blk.%d.attn_q_norm.weight", il);
            cudaMemcpy(w128.data(), dev(gnm)->ptr, 512, cudaMemcpyDeviceToHost);
            for (int h = 0; h < nqh; ++h) memcpy(tiled.data() + h * 128, w128.data(), 512);
            cudaMemcpy(d_qn_gamma_ + (size_t)il * 96 * 128, tiled.data(), nqh * 128 * 4, cudaMemcpyHostToDevice);
            snprintf(gnm, sizeof gnm, "blk.%d.attn_k_norm.weight", il);
            cudaMemcpy(w128.data(), dev(gnm)->ptr, 512, cudaMemcpyDeviceToHost);
            for (int h = 0; h < 8; ++h) memcpy(tiled.data() + h * 128, w128.data(), 512);
            cudaMemcpy(d_kn_gamma_ + (size_t)il * 8 * 128, tiled.data(), 8 * 128 * 4, cudaMemcpyHostToDevice);
        }
    }
    }
    const DevTensor* tok_embd = dev("token_embd.weight");
    const DevTensor* out_w = dev("output.weight");
    if (!tok_embd || !out_w) { err = "missing embed/output"; return false; }
    const int n_prompt = (int)prompt_ids.size();
    const auto t_start = std::chrono::steady_clock::now();
    int n_gen = 0;
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);
    // decode 侧 F32 GEMV（router）走 cuBLAS：maya 判例「router/mHC 强制 F32 SGEMM」。
    // 旧 f32_gemv 只有 288 线程串行扫 4096 → 实测 14.8 ms/token（本机最大单块）
    static cublasHandle_t cbh_dec = nullptr;
    if (!cbh_dec) { cublasCreate(&cbh_dec); }
    cublasSetStream(cbh_dec, st);

    // ---- M1.5 phase timing buckets (seconds, per generate() call) ----
    double t_gpu = 0, t_sync = 0, t_cpu = 0, t_tail = 0;
    double dbl_quant = 0, dbl_spin = 0;
    const auto tick = [] { return std::chrono::steady_clock::now(); };
    auto elapsed = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    // ---- CPU lane 工人池 v2：work-stealing（2026-11-11）----
    // 旧版静态 (专家, 四分块) 映射在池命中 71% 时只有 ~9/32 工人活跃，DRAM 带宽吃不满
    // （地板 2.8GB×(1-hit)÷98.9GB/s；实测 14.7ms vs 地板 9.6ms）。改为动态工位：
    // miss 专家的行区间切成 80/64 行小块进原子队列，32 工人全量抢活 → DRAM 打满。
    // 位等性：每行点积只依赖 act（只读共享）与该行权重，行间无归约 → 切法不影响结果；
    // quant_h 由最后完成 GU 的工人对全部 miss 专家各做一次（输入逐位同旧版 → hq 逐位同）。
    struct LaneCtx {
        std::atomic<uint64_t> gen{0};
        std::atomic<bool> stop{false};
        struct Item { const uint8_t* wg; const uint8_t* wu; const uint8_t* wd;
                      float* ff; float* po; const void* hq; int lo, hi; };
        Item ig[128], id[128];          // GU/DN 块 ≤8 专家×16
        int n_ig = 0, n_id = 0;
        std::atomic<int> next_ig{0}, next_id{0}, left_ig{0}, left_id{0}, phase{1}, done{0};
        const void* act = nullptr;
        float* ffq[8];                  // miss 专家的 ff 行基址（quant_h 用）
        void* hq_e[8];                  // 对应 hq 缓冲
        int n_exp = 0;
    };
    static strata::kernels::cpu::NativeFmt s_fmt;
    static bool s_fmt_ok = false;
    if (!s_fmt_ok) {
        std::string ferr;
        s_fmt_ok = strata::kernels::cpu::native_fmt(23, 23, 4096, 1280, s_fmt, ferr);
        if (!s_fmt_ok) { err = "native_fmt: " + ferr; return false; }
    }
    static LaneCtx lane;
    alignas(64) static uint8_t hq_shared[8 * strata::kernels::cpu::kNativeHBytes];
    static std::vector<float> ffbuf(8 * 1280);
    // partbuf 必须 pinned：池命中专家的 parts 靠 D2H 打进来，pageable 目标会走 driver
    // 中转（每 token 336 次小 D2H → 20ms 级停顿，prefill 的 pinned 判例同源）
    static float* partbuf_p = nullptr;
    if (!partbuf_p) cudaMallocHost((void**)&partbuf_p, 8 * 4096 * 4);
    struct PartBuf { float* p; float* data() const { return p; } };
    PartBuf partbuf{partbuf_p};
    static float* ysum = nullptr;  // pinned: the tail H2D reads it
    static bool pool_started = false;
    if (!pool_started) {
        pool_started = true;
        for (int w = 0; w < 32; ++w) {
            std::thread([w] {
                // CCD 亲和实验（负结果已回退）：CPU lane 是 DRAM 带宽墙（2.8GB/token ÷ 98.9
                // GB/s = 28.5ms 与实测精确吻合），绑核不能加带宽；同专家 4 工人聚同 CCD 反而
                // 把该 CCD 的 DRAM 通路挤成瓶颈——不绑，让调度器把工人散满全机。
                // v2：工位动态化——每工人从原子队列抢 (专家, 行块)，miss 少时 32 人全量摊
                // DRAM 读，贴 9.6ms 带宽地板。
                uint64_t seen = 0;
                for (;;) {
                    uint64_t g;
                    while ((g = lane.gen.load(std::memory_order_acquire)) == seen) {
                        if (lane.stop.load(std::memory_order_relaxed)) return;
                        __builtin_ia32_pause();
                    }
                    seen = g;
                    // ---- 相位 1：GU 行块（80 行/块）——按完成项计数（数工人会竞态：
                    //      抢空的工人先过点，最后项未完 quant 就开跑 → 读到未完成 ff）----
                    for (;;) {
                        const int i = lane.next_ig.fetch_add(1, std::memory_order_relaxed);
                        if (i >= lane.n_ig) break;
                        const LaneCtx::Item& it = lane.ig[i];
                        float* ffp[1] = { it.ff };
                        strata::kernels::cpu::native_gu_rows_split(s_fmt, it.wg, it.wu, &lane.act, 1, ffp,
                                                                   it.lo, it.hi);
                        if (lane.left_ig.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            // 最后一个完成项的工人：全部 miss 专家各 quant_h 一次
                            //（输入与旧版逐位同 → hq 逐位同）
                            for (int k = 0; k < lane.n_exp; ++k)
                                strata::kernels::cpu::native_quant_h(s_fmt, lane.ffq[k], lane.hq_e[k]);
                            lane.phase.store(2, std::memory_order_release);
                        }
                    }
                    while (lane.phase.load(std::memory_order_acquire) < 2) __builtin_ia32_pause();
                    // ---- 相位 2：DN 行块（64 行/块）----
                    for (;;) {
                        const int i = lane.next_id.fetch_add(1, std::memory_order_relaxed);
                        if (i >= lane.n_id) break;
                        const LaneCtx::Item& it = lane.id[i];
                        const void* hp[1] = { it.hq };
                        float* opp[1] = { it.po };
                        strata::kernels::cpu::native_down_rows_split(s_fmt, it.wd, hp, 1, opp,
                                                                     it.lo, it.hi);
                        if (lane.left_id.fetch_sub(1, std::memory_order_acq_rel) == 1)
                            lane.done.store(1, std::memory_order_release);
                    }
                }
            }).detach();
        }
    }
    // pinned staging for the per-layer host round-trips
    alignas(64) static uint8_t lane_act[strata::kernels::cpu::kNativeActBytes];  // q8_K activation
    static float* gu8 = nullptr, *uu8 = nullptr, *bh8 = nullptr;   // M2a resident-lane buffers [8][1280]
    static float* pin_x = nullptr;   // [4096] ffn_x
    static int*   pin_ids = nullptr; // [8]
    static float* pin_w = nullptr;   // [8]
    static float* pin_sh = nullptr;  // [4096] shared expert out
    static volatile int* h_door = nullptr;   // GPU->CPU doorbell (pinned, written by a graph node)
    static int* d_seq = nullptr;             // doorbell value source (device), uploaded per token
    static int* d_pos = nullptr;             // rope position array [96], uploaded once per token
    static cudaEvent_t ev_d2h = nullptr;     // ids/w/x 三 D2H 的完成门（T1 只等它们，不等 shared）
    // 预量化 q8_1 缓冲：同一个 x 服务多个权重矩阵（native_mmvq.hpp: "One quantization may
    // serve multiple weight matrices sharing the same input"）。旧 f32 wrapper 每次调用都重
    // 量化一遍 —— 同层 8 个专家 = 8 次冗余量化 + 每次 GEMV 变 2 次发射（实测 ~1100 次/token）
    static void *q8_xn = nullptr, *q8_attn = nullptr, *q8_ffnx = nullptr;
    static void *q8_h1 = nullptr, *q8_h1e = nullptr, *q8_h2 = nullptr;
    const size_t q8_row1280 = strata::kernels::native_q8_1_bytes(1280, 1);
    if (!q8_xn) {
        cudaMalloc(&q8_xn,   strata::kernels::native_q8_1_bytes(4096, 1));
        cudaMalloc(&q8_attn, strata::kernels::native_q8_1_bytes(12288, 1));
        cudaMalloc(&q8_ffnx, strata::kernels::native_q8_1_bytes(4096, 1));
        cudaMalloc(&q8_h1,   strata::kernels::native_q8_1_bytes(11264, 1));
        cudaMalloc(&q8_h1e,  q8_row1280 * 8);
        cudaMalloc(&q8_h2,   strata::kernels::native_q8_1_bytes(1280, 1));
        cudaMalloc(&ws_.gu_hi, 8 * 2560 * 4);   // gate/up 合并输出（每命中专家 2560）
    }
    if (!pin_x) {
        cudaMallocHost(&pin_x, 4096 * 4);
        cudaMallocHost(&pin_ids, 8 * 4);
        cudaMallocHost(&pin_w, 8 * 4);
        cudaMallocHost(&pin_sh, 4096 * 4);
        cudaMallocHost(&ysum, 4096 * 4);
        cudaMalloc(&gu8, 8 * 1280 * 4); cudaMalloc(&uu8, 8 * 1280 * 4); cudaMalloc(&bh8, 8 * 1280 * 4);
        cudaMallocHost((void**)&h_door, 4);
        cudaMalloc(&d_pos, 96 * 4);
        cudaMalloc(&d_seq, 4);
        cudaEventCreateWithFlags(&ev_d2h, cudaEventDisableTiming);
        *(int*)h_door = -1;
    }
    // ===== 影子专家池模拟（零风险 hit-rate 探针；maya LayerPool 判例的 host 侧复刻）=====
    // LFU 计数 + 每 32768 次路由事件整体减半（glm_fast_path.cu:4780-4785）+ recency 平票裁决
    struct PoolSim {
        int cap = 0, used = 0;
        std::vector<int32_t> slot_of;    // key -> slot, -1 缺席
        std::vector<int32_t> s_key;      // slot -> key
        std::vector<uint32_t> cnt;       // key -> LFU
        std::vector<uint64_t> tick;      // key -> 最近一次命中时刻
        uint64_t now = 0, hits = 0, miss = 0, evict = 0, ev_events = 0;
        void init(int c, int n_key) {
            cap = c; slot_of.assign(n_key, -1); s_key.assign(c, -1);
            cnt.assign(n_key, 0); tick.assign(n_key, 0);
        }
        void touch(int key) {
            ++now;
            if (slot_of[key] >= 0) { ++hits; ++cnt[key]; tick[key] = now; return; }
            ++miss;
            int slot;
            if (used < cap) slot = used++;
            else {   // 受害者：min cnt，平票取最旧 tick
                slot = 0; uint32_t bc = cnt[s_key[0]]; uint64_t bt = tick[s_key[0]];
                for (int i = 1; i < cap; ++i) {
                    const uint32_t c = cnt[s_key[i]]; const uint64_t t = tick[s_key[i]];
                    if (c < bc || (c == bc && t < bt)) { bc = c; bt = t; slot = i; }
                }
                slot_of[s_key[slot]] = -1; ++evict;
            }
            s_key[slot] = key; slot_of[key] = slot;
            ++cnt[key]; tick[key] = now;
            if ((ev_events += 1) >= 32768) { ev_events = 0; for (auto& c : cnt) c >>= 1; }
        }
        double rate() const { return (hits + miss) ? (double)hits / (double)(hits + miss) : 0.0; }
    };
    static PoolSim psim[5];
    static bool psim_init = false;
    static const int psim_caps[5] = {256, 512, 1024, 1536, 2048};
    if (!psim_init) {
        for (int i = 0; i < 5; ++i) psim[i].init(psim_caps[i], 42 * 288);
        psim_init = true;
    }
    // ===== 真专家池：profile 预热（maya --expert-profile 同构）=====
    // 槽布局 = gate|up|down 三段连续（3 × 2785280 B）；key = (il-3)*288 + expert
    constexpr size_t kSlotBytes = 3ull * 2785280;
    static uint8_t* d_pool = nullptr;
    static int pool_n = 0;
    static std::vector<int32_t> pool_slot_of;
    // 在线准入（LFU + recency 平票，maya 判例）：静态 profile 只有 53% 命中，
    // 影子池实测同槽数可达 74.9% —— 差在"不自适应"
    static std::vector<uint32_t> pool_cnt;
    static std::vector<uint64_t> pool_tick;
    static std::vector<int32_t> pool_slot_key;         // slot -> key, -1 空
    static std::vector<std::pair<int,int>> pool_pending;   // (key, slot) 待搬运
    static uint64_t pool_now = 0, pool_ev_events = 0;
    static int pool_used = 0, pool_admit_budget = 0, pool_admitted = 0;
    static uint8_t* pool_pin = nullptr;                    // pinned landing
    static bool pool_init_done = false;
    if (!pool_init_done) {
        pool_init_done = true;
        if (const char* pf = getenv("STEP37_POOL_PROFILE")) {
            if (FILE* f = fopen(pf, "r")) {
                std::vector<std::pair<unsigned long long, int>> v;
                unsigned long long k = 0, c = 0;
                while (fscanf(f, "%llu %llu", &k, &c) == 2 && k < 42ull * 288) v.push_back({c, (int)k});
                fclose(f);
                std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                int want = getenv("STEP37_POOL_SLOTS") ? atoi(getenv("STEP37_POOL_SLOTS")) : 1600;
                if ((int)v.size() > want) v.resize((size_t)want);
                if (!v.empty() && cudaMalloc(&d_pool, (size_t)v.size() * kSlotBytes) == cudaSuccess) {
                    pool_n = (int)v.size();
                    pool_slot_of.assign(42 * 288, -1);
                    uint8_t* pin = nullptr;
                    cudaMallocHost(&pin, kSlotBytes);
                    for (int i = 0; i < pool_n; ++i) {
                        const int key = v[(size_t)i].second;
                        const ExpertSlice& es = experts_[key / 288][key % 288];
                        memcpy(pin, es.gate, 2785280);
                        memcpy(pin + 2785280, es.up, 2785280);
                        memcpy(pin + 2 * 2785280, es.down, 2785280);
                        cudaMemcpy(d_pool + (size_t)i * kSlotBytes, pin, kSlotBytes, cudaMemcpyHostToDevice);
                        pool_slot_of[(size_t)key] = i;
                    }
                    cudaFreeHost(pin);
                    pool_cnt.assign(42 * 288, 0);
                    pool_tick.assign(42 * 288, 0);
                    pool_slot_key.assign((size_t)pool_n, -1);
                    for (int i = 0; i < pool_n; ++i) {
                        const int key = v[(size_t)i].second;
                        pool_slot_key[(size_t)i] = key;
                        pool_cnt[(size_t)key] = (uint32_t)std::min<unsigned long long>(v[(size_t)i].first, 1u << 30);
                    }
                    pool_used = pool_n;
                    cudaMallocHost((void**)&pool_pin, kSlotBytes);
                    fprintf(stderr, "POOL: %d slots (%.2f GB) warmed from profile\n",
                            pool_n, (double)pool_n * kSlotBytes / 1073741824.0);
                } else {
                    fprintf(stderr, "POOL: alloc failed, fallback to tail\n");
                }
            } else {
                fprintf(stderr, "POOL: cannot open %s\n", pf);
            }
        }
    }

            auto forward_one = [&](int tok, int pos) -> bool {
        iq4_xs_row((const uint8_t*)tok_embd->ptr, tok, ws_.x, st);
        {   // rope position array + doorbell sequence: one upload per token
            static int pos_h[96];
            for (int h = 0; h < 96; ++h) pos_h[h] = pos;
            static int seq_h;              // static: async copy source must outlive the call
            seq_h = pos + 1;
            cudaMemcpyAsync(d_pos, pos_h, 96 * 4, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(d_seq, &seq_h, 4, cudaMemcpyHostToDevice, st);
        }
        step37dump::f32f(pos, ws_.x, 4096, "tok_embd");
        for (int il = 0; il < 45; ++il) {
            if (getenv("STEP37_DBG")) fprintf(stderr, "DBG enter il=%d pos=%d\n", il, pos);
            // per-layer attention geometry (GGUF tensor shapes are ground truth):
            //   il%4==0 → 64 heads, full attention, rope n_rot=64 base 5M
            //   else    → 96 heads, SWA-512,           rope n_rot=128 base 10k
            const bool full_il = (il % 4 == 0);
            const int nqh = full_il ? 64 : 96;
            const int qb = nqh * 128;
            const int nrot = full_il ? 64 : 128;
            const float* cosT = full_il ? ws_.cos_full : ws_.cos_swa;
            const float* sinT = full_il ? ws_.sin_full : ws_.sin_swa;
            const int lo = full_il ? 0 : (pos > 511 ? pos - 511 : 0);
            char nm[64];
            snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", il);
            const DevTensor* tn1 = dev(nm);
            // per-layer CUDA graph: capture once (first token), replay after — kills ~26 launch
            // gaps per layer. All pointers are stable; pos flows through d_pos (device memory).
            static cudaGraphExec_t g_exec[45] = {};
            static cudaGraph_t g_graph[45] = {};
            static const bool no_graph = !getenv("STEP37_GRAPH");   // graphs parked: deterministic-wrong on replay (see notes); eager path verified gold
            static const int graph_max = getenv("STEP37_GRAPH_MAX") ? atoi(getenv("STEP37_GRAPH_MAX")) : 45;
            const bool graph_on = !no_graph && il < graph_max;
            const bool cap = graph_on && !g_exec[il];
            if (cap) {
                cudaError_t rc = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (rc != cudaSuccess) { fprintf(stderr, "DBG begin-capture il=%d rc=%s\n", il, cudaGetErrorString(rc)); }
            }
            step37dump::g_capturing = cap;
            strata::kernels::native_gr_rms_norm_weighted(ws_.x, (const float*)tn1->ptr, ws_.xn, 4096, 1, 1e-5, st);
            step37dump::f32f(pos, ws_.xn, 4096, "attn_norm-%d", il);
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", il);
            strata::kernels::native_quantize_q8_1(ws_.xn, q8_xn, 4096, 1, st);   // 一次量化服务 Q/K/V/gate
            strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_xn, ws_.q, 4096, qb, 1, st);
            step37dump::f32f(pos, ws_.q, qb, "Qcur-%d", il);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", il);
            strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_xn, ws_.k, 4096, 1024, 1, st);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", il);
            strata::kernels::native_q5_k_mmvq(dev(nm)->ptr, q8_xn, ws_.v, 4096, 1024, 1, st);
            step37dump::f32f(pos, ws_.v, 1024, "Vcur-%d", il);
            snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", il);
            strata::kernels::native_gr_rms_norm_weighted(ws_.q, d_qn_gamma_ + (size_t)il * 96 * 128, ws_.attn, 128, nqh, 1e-5, st);
            cudaMemcpyAsync(ws_.q, ws_.attn, qb * 4, cudaMemcpyDeviceToDevice, st);
            step37dump::f32f(pos, ws_.q, qb, "Qcur_normed-%d", il);
            snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", il);
            strata::kernels::native_gr_rms_norm_weighted(ws_.k, d_kn_gamma_ + (size_t)il * 8 * 128, ws_.attn, 128, 8, 1e-5, st);
            cudaMemcpyAsync(ws_.k, ws_.attn, 1024 * 4, cudaMemcpyDeviceToDevice, st);
            step37dump::f32f(pos, ws_.k, 1024, "Kcur_normed-%d", il);
            // rope（全注意力层 n_rot=64 base 5M；SWA 层 n_rot=128 base 10k）；pos 数组每 token 上传一次
            strata::kernels::rope_neox_apply(ws_.q, ws_.q, nqh, 128, nrot, cosT, sinT, d_pos, st);
            step37dump::f32f(pos, ws_.q, qb, "Qcur_pos-%d", il);
            strata::kernels::rope_neox_apply(ws_.k, ws_.k, 8, 128, nrot, cosT, sinT, d_pos, st);
            step37dump::f32f(pos, ws_.k, 1024, "Kcur_pos-%d", il);
            // KV append: k/v 写 [pos] 行（kernel 版：pos 设备读，graph 安全）
            kv_append(d_kcache_, d_vcache_, ws_.k, ws_.v, d_pos, kv_base_bytes_[il], kv_lcap_[il], st);
            {   // KV autopsy: dump the row just written (layer's k at row *d_pos)
                if (step37dump::dump_on() && step37dump::pos_ok(pos)) {
                    char pn[160];
                    int posv; cudaMemcpy(&posv, d_pos, 4, cudaMemcpyDeviceToHost);
                    snprintf(pn, sizeof pn, "%s/kvrow-%d.bin", getenv("STEP37_DUMP_DIR"), il);
                    uint8_t* row = (uint8_t*)d_kcache_ + kv_base_bytes_[il] + (size_t)(posv % kv_lcap_[il]) * 1088;
                    if (FILE* f = fopen(pn, "wb")) {
                        std::vector<char> buf(1088);
                        cudaMemcpy(buf.data(), row, 1088, cudaMemcpyDeviceToHost);
                        fwrite(buf.data(), 1, 1088, f); fclose(f);
                    }
                }
            }
            // attention (scores → softmax → weighted V)
            // P2.3: decode flash（pos 分段 online softmax + 二段规约）。旧三件在 50K ctx 下
            // 46.6 ms/token（网格 96 block ≈12% occupancy，延迟受限 7.4% 效率）
            attn_decode_flash(d_kcache_, d_vcache_, ws_.q, ws_.attn, d_pos, !full_il, nqh, 8,
                              0.08838834764831845f, kv_base_bytes_[il],
                              full_il ? (pos + 1) : std::min(pos + 1, 512), ws_.df_scratch, st, kv_lcap_[il]);
            step37dump::f32f(pos, ws_.attn, qb, "attn_out-%d", il);
            // attn_gate: sigmoid(g_proj(xn)) 逐头广播乘
            snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", il);
            strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_xn, ws_.gate, 4096, nqh, 1, st);
            step37dump::f32f(pos, ws_.gate, nqh, "attn_gate_raw-%d", il);
            attn_gate_mul(ws_.gate, ws_.attn, 128, nqh, st);
            step37dump::f32f(pos, ws_.attn, qb, "attn_gated-%d", il);
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", il);
            strata::kernels::native_quantize_q8_1(ws_.attn, q8_attn, qb, 1, st);
            strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_attn, ws_.o, qb, 4096, 1, st);
            step37dump::f32f(pos, ws_.o, 4096, "attn_proj-%d", il);
            strata::kernels::add_inplace(ws_.x, ws_.o, 4096, st);
            step37dump::f32f(pos, ws_.x, 4096, "ffn_inp-%d", il);
            // FFN（blk 0-2 dense，3-44 MoE）
            snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", il);
            strata::kernels::native_gr_rms_norm_weighted(ws_.x, (const float*)dev(nm)->ptr, ws_.ffn_x, 4096, 1, 1e-5, st);
            step37dump::f32f(pos, ws_.ffn_x, 4096, "ffn_norm-%d", il);
            if (il < 3) {
                snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", il);
                strata::kernels::native_quantize_q8_1(ws_.ffn_x, q8_ffnx, 4096, 1, st);
                strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_ffnx, ws_.h1, 4096, 11264, 1, st);
                snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", il);
                strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_ffnx, ws_.h2, 4096, 11264, 1, st);
                swiglu_rows_step37(ws_.h1, ws_.h2, ws_.h1, 11264, st);
                snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", il);
                strata::kernels::native_quantize_q8_1(ws_.h1, q8_h1, 11264, 1, st);
                strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_h1, ws_.ffn_y, 11264, 4096, 1, st);
                step37dump::f32f(pos, ws_.ffn_y, 4096, "ffn_out-%d", il);
                step37dump::g_capturing = false;
                if (cap) {
                    if (cudaStreamEndCapture(st, &g_graph[il]) != cudaSuccess ||
                        cudaGraphInstantiateWithFlags(&g_exec[il], g_graph[il], 0) != cudaSuccess) {
                        err = "graph capture/instantiate failed"; return false;
                    }
                }
                if (graph_on && g_exec[il]) cudaGraphLaunch(g_exec[il], st);
            } else {
                // 分段图 v2：attn 段到此为止（CPU 洞前收图回放），router 起全 eager——
                // 整层图的回放悖论（跨 CPU 洞的 D2H/H2D 时序）隔离实验
                if (cap) {
                    if (cudaStreamEndCapture(st, &g_graph[il]) != cudaSuccess ||
                        cudaGraphInstantiateWithFlags(&g_exec[il], g_graph[il], 0) != cudaSuccess) {
                        err = "attn-graph capture failed"; return false;
                    }
                }
                if (graph_on && g_exec[il]) cudaGraphLaunch(g_exec[il], st);
                const auto tg0 = tick();
                snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", il);
                if (getenv("STEP37_RTR_OLD")) {
                    f32_gemv((const float*)dev(nm)->ptr, ws_.ffn_x, ws_.router_logits, 4096, 288, st);
                } else {   // router: logits[288] = W[288][4096] · x[4096]。旧 f32_gemv 只有 288 线程
                    // 串行扫 4096 → 14.8 ms/token（本机最大单块）；cuBLAS Sgemv 走 split-K
                    const float one = 1.f, zero = 0.f;
                    cublasSgemv(cbh_dec, CUBLAS_OP_T, 4096, 288, &one,
                                (const float*)dev(nm)->ptr, 4096, ws_.ffn_x, 1, &zero,
                                ws_.router_logits, 1);
                }
                snprintf(nm, sizeof nm, "blk.%d.exp_probs_b.bias", il);
                strata::kernels::router_sigmoid(ws_.router_logits, (const float*)dev(nm)->ptr, 1, 288, 8, 3.0f, true, ws_.e_ids, ws_.exp_w, st);
                step37dump::f32f(pos, ws_.router_logits, 288, "ffn_moe_logits-%d", il);
                // router readback first — CPU can start while shared expert still runs on GPU
                cudaMemcpyAsync(pin_ids, ws_.e_ids, 32, cudaMemcpyDeviceToHost, st);
                cudaMemcpyAsync(pin_w, ws_.exp_w, 32, cudaMemcpyDeviceToHost, st);
                cudaMemcpyAsync(pin_x, ws_.ffn_x, 16384, cudaMemcpyDeviceToHost, st);
                cudaEventRecord(ev_d2h, st);   // CPU lane 的最小到货门：ids/w/x（shared 的 D2H 不在此门内）
                ring_host(h_door, d_seq, 64, il + 1, st);   // per-layer unique doorbell: (pos+1)*64 + il + 1
                // shared expert enqueued after the event (needs only ffn_x) — overlaps the CPU lane
                snprintf(nm, sizeof nm, "blk.%d.ffn_gate_shexp.weight", il);
                strata::kernels::native_quantize_q8_1(ws_.ffn_x, q8_ffnx, 4096, 1, st);
                strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_ffnx, ws_.h2, 4096, 1280, 1, st);
                snprintf(nm, sizeof nm, "blk.%d.ffn_up_shexp.weight", il);
                strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_ffnx, ws_.h2 + 1280, 4096, 1280, 1, st);
                swiglu_rows_step37(ws_.h2, ws_.h2 + 1280, ws_.h2, 1280, st);
                snprintf(nm, sizeof nm, "blk.%d.ffn_down_shexp.weight", il);
                strata::kernels::native_quantize_q8_1(ws_.h2, q8_h2, 1280, 1, st);
                strata::kernels::native_iq4_xs_mmvq(dev(nm)->ptr, q8_h2, ws_.shared_y, 1280, 4096, 1, st);
                cudaMemcpyAsync(pin_sh, ws_.shared_y, 16384, cudaMemcpyDeviceToHost, st);
                step37dump::g_capturing = false;
                if (cap && il < 3) {   // 整层图仅 dense 层（无 CPU 洞）；MoE 层已用 attn 分段图
                    cudaError_t rc = cudaStreamEndCapture(st, &g_graph[il]);
                    if (rc != cudaSuccess) fprintf(stderr, "DBG end-capture il=%d rc=%s\n", il, cudaGetErrorString(rc));
                    if (rc == cudaSuccess && cudaGraphInstantiateWithFlags(&g_exec[il], g_graph[il], 0) != cudaSuccess) {
                        fprintf(stderr, "DBG instantiate il=%d failed\n", il);
                    }
                }
                if (cap && il < 3) {   // dense 整层图首 token 补跑（MoE attn 图已在 else 开头回放）
                    cudaError_t rc = cudaGraphLaunch(g_exec[il], st);
                    if (rc != cudaSuccess) fprintf(stderr, "DBG graph-launch il=%d rc=%s\n", il, cudaGetErrorString(rc));
                    else {
                        rc = cudaStreamSynchronize(st);
                        if (rc != cudaSuccess) fprintf(stderr, "DBG graph-EXEC il=%d rc=%s\n", il, cudaGetErrorString(rc));
                    }
                }
                t_gpu += elapsed(tg0, tick());
                const auto ts0 = tick();
                // 最小到货门：只等 ids/w/x 三个 D2H（shared expert kernels 与 CPU lane 继续重叠，
                // pin_sh 由 T3 的 sync 收）。旧版整流 sync 每 token 白等 31×shared 段 ≈ 11ms
                cudaEventSynchronize(ev_d2h);
                t_sync += elapsed(ts0, tick());
                for (int e = 0; e < 8; ++e) {   // 影子池：只记账，不动数据
                    const int key = (il - 3) * 288 + pin_ids[e];
                    for (auto& ps : psim) ps.touch(key);
                }
                if (il >= 45 - gpu_tail_) {
                    // ---- resident fast path: experts already in VRAM; GEMVs hit them in place ----
                    const int l = il - (45 - gpu_tail_);
                    const uint64_t mat = 288ull * 2785280;
                    uint8_t* base = d_gpu_exp_ + 3 * mat * (uint64_t)l;
                    // M2a: 3 launches (gu8 / swiglu / dn8) instead of 24 GEMV launches.
                    // PARKED: moe_gu8 self-consistent (lstsq 1.4e-15) but greedy diverges from
                    // GEMV path — integration-layer diff unresolved; opt-in via STEP37_M2A=1
                    static const bool use_m2a = getenv("STEP37_M2A") != nullptr;
                    if (!use_m2a) {
                        strata::kernels::native_quantize_q8_1(ws_.ffn_x, q8_ffnx, 4096, 1, st);
                        for (int e = 0; e < 8; ++e) {
                            const uint64_t off = (uint64_t)pin_ids[e] * 2785280;
                            strata::kernels::native_iq4_xs_mmvq(base + off, q8_ffnx, ws_.gu_hi + (size_t)e * 2560, 4096, 2560, 1, st);
                        }
                        swiglu_gu8(ws_.gu_hi, ws_.h1, 1280, st);
                        strata::kernels::native_quantize_q8_1(ws_.h1, q8_h1e, 1280, 8, st);
                        for (int e = 0; e < 8; ++e) {
                            const uint64_t off = (uint64_t)pin_ids[e] * 2785280;
                            strata::kernels::native_iq4_xs_mmvq(base + 2 * mat + off,
                                (const uint8_t*)q8_h1e + (size_t)e * q8_row1280, ws_.parts + (size_t)e * 4096, 1280, 4096, 1, st);
                        }
                    } else {
                        moe_gu8(base, base + mat, ws_.ffn_x, (const int*)pin_ids, gu8, uu8, 2785280, 1280, st);
                        swiglu_batch(gu8, uu8, bh8, (size_t)8 * 1280, st);
                        moe_dn8(base + 2 * mat, bh8, (const int*)pin_ids, ws_.parts, 2785280, 1280, st);
                        static bool ab_done = false;
                        if (!ab_done && il == 34) {
                            ab_done = true;
                            // reference: native GEMV for expert pin_ids[0], row 0..3
                            // (offline comparison: numpy dequant of expert row vs gu8 dump)
                            std::vector<char> buf(8 * 1280 * 4);
                            cudaMemcpy(buf.data(), gu8, buf.size(), cudaMemcpyDeviceToHost);
                            char pn[160];
                            snprintf(pn, sizeof pn, "%s/gu8_L%d.bin", getenv("STEP37_DUMP_DIR") ? getenv("STEP37_DUMP_DIR") : ".", il);
                            FILE* f = fopen(pn, "wb"); fwrite(buf.data(), 1, buf.size(), f); fclose(f);
                            fprintf(stderr, "AB: gu8 L%d dumped, ids=[%d %d %d %d]\n", il,
                                    pin_ids[0], pin_ids[1], pin_ids[2], pin_ids[3]);
                        }
                    }
                    strata::kernels::moe_combine(ws_.parts, ws_.exp_w, ws_.shared_y, ws_.ffn_y, 4096, 8, st);
                } else if (pool_n > 0) {
                    // ---- 池：逐专家决定 on what device（命中=VRAM 槽，未命中=CPU lane）----
                    int hs[8]; int n_hit = 0;
                    for (int e = 0; e < 8; ++e) {
                        hs[e] = pool_slot_of[(size_t)((il - 3) * 288 + pin_ids[e])];
                        n_hit += hs[e] >= 0;
                    }
                    // 在线准入：未命中的 key 按 LFU 竞争槽位（预算内、且比受害者更热才换）
                    if (pool_cnt.empty()) { pool_cnt.assign(42 * 288, 0); pool_tick.assign(42 * 288, 0); }
                    for (int e = 0; e < 8; ++e) {
                        const int key = (il - 3) * 288 + pin_ids[e];
                        ++pool_now;
                        ++pool_cnt[(size_t)key]; pool_tick[(size_t)key] = pool_now;
                        // 热度阈值：只换入"已被路由 ≥2 次"的专家，避免为一闪而过的 key 付 PCIe 成本
                        const int kAdmitMinCnt = getenv("STEP37_POOL_MINCNT") ? atoi(getenv("STEP37_POOL_MINCNT")) : 2;
                        if (hs[e] >= 0 || pool_admit_budget <= 0) continue;
                        if (pool_cnt[(size_t)key] < (uint32_t)kAdmitMinCnt) continue;
                        int slot = -1;
                        if (pool_used < pool_n) slot = pool_used++;
                        else {
                            int best = -1; uint32_t bc = UINT32_MAX; uint64_t bt = ~0ull;
                            for (int s = 0; s < pool_n; ++s) {
                                const int k2 = pool_slot_key[(size_t)s];
                                if (k2 < 0) { best = s; break; }
                                const uint32_t c2 = pool_cnt[(size_t)k2]; const uint64_t t2 = pool_tick[(size_t)k2];
                                if (c2 < bc || (c2 == bc && t2 < bt)) { bc = c2; bt = t2; best = s; }
                            }
                            if (best < 0 || pool_cnt[(size_t)key] <= bc) continue;   // 不值得换
                            slot = best;
                            pool_slot_of[(size_t)pool_slot_key[(size_t)best]] = -1;
                        }
                        pool_slot_key[(size_t)slot] = key;
                        pool_slot_of[(size_t)key] = slot;
                        pool_pending.push_back({key, slot});
                        --pool_admit_budget;
                        ++pool_admitted;
                    }
                    static long long pool_hit = 0, pool_req = 0;
                    pool_hit += n_hit; pool_req += 8;
                    if (pool_req % (8 * 42 * 8) == 0)
                        fprintf(stderr, "POOL-HIT: %lld/%lld = %.1f%%\n", pool_hit, pool_req,
                                100.0 * (double)pool_hit / (double)pool_req);
                    if (n_hit > 0) {
                        strata::kernels::native_quantize_q8_1(ws_.ffn_x, q8_ffnx, 4096, 1, st);
                        // gate|up 槽内相邻 = 一个 [2560,4096] 矩阵：一次 mmvq（位等价，行独立），
                        // swiglu_gu8 一次 launch 拆两半 —— hit 专家 17 launch → 2。
                        // （此前"合并更慢"的结论是 __expf 契约污染 A/B 的假象，见定律 9/11）
                        for (int e = 0; e < 8; ++e) if (hs[e] >= 0) {
                            const uint8_t* b = d_pool + (size_t)hs[e] * kSlotBytes;
                            strata::kernels::native_iq4_xs_mmvq(b, q8_ffnx, ws_.gu_hi + (size_t)e * 2560, 4096, 2560, 1, st);
                        }
                        swiglu_gu8(ws_.gu_hi, ws_.h1, 1280, st);
                        strata::kernels::native_quantize_q8_1(ws_.h1, q8_h1e, 1280, 8, st);
                        for (int e = 0; e < 8; ++e) if (hs[e] >= 0) {
                            const uint8_t* b = d_pool + (size_t)hs[e] * kSlotBytes;
                            strata::kernels::native_iq4_xs_mmvq(b + 2 * 2785280,
                                (const uint8_t*)q8_h1e + (size_t)e * q8_row1280, ws_.parts + (size_t)e * 4096, 1280, 4096, 1, st);
                        }
                        for (int e = 0; e < 8; ++e) if (hs[e] >= 0)
                            cudaMemcpyAsync(partbuf.data() + (size_t)e * 4096, ws_.parts + (size_t)e * 4096,
                                            16384, cudaMemcpyDeviceToHost, st);
                    }
                    if (n_hit < 8) {   // CPU lane：miss 专家 → 动态工位（work-stealing v2）
                        const auto tc0 = tick();
                        const auto tq0 = tick();
                        strata::kernels::cpu::native_quant_act(s_fmt, pin_x, lane_act);
                        const double q_this = elapsed(tq0, tick());
                        dbl_quant += q_this;
                        lane.n_ig = 0; lane.n_id = 0; lane.n_exp = 0;
                        for (int e = 0; e < 8; ++e) {
                            if (hs[e] >= 0) continue;
                            const ExpertSlice& es = experts_[il - 3][pin_ids[e]];
                            float* ff_e = ffbuf.data() + (size_t)e * 1280;
                            float* po_e = partbuf.data() + (size_t)e * 4096;
                            void* hq_e = hq_shared + (size_t)lane.n_exp * strata::kernels::cpu::kNativeHBytes;
                            lane.ffq[lane.n_exp] = ff_e;
                            lane.hq_e[lane.n_exp] = hq_e;
                            ++lane.n_exp;
                            for (int c = 0; c < 16; ++c)   // GU：1280 行 → 16×80
                                lane.ig[lane.n_ig++] = { es.gate, es.up, nullptr, ff_e, nullptr,
                                                         nullptr, c * 80, c * 80 + 80 };
                            for (int c = 0; c < 16; ++c)   // DN：4096 行 → 16×256（修雷：原写 16×64
                                lane.id[lane.n_id++] = { nullptr, nullptr, es.down, nullptr, po_e,   // 只盖 1024 行，
                                                         hq_e, c * 256, c * 256 + 256 };              // 3/4 输出是旧垃圾）
                        }
                        lane.act = lane_act;
                        lane.next_ig.store(0, std::memory_order_relaxed);
                        lane.next_id.store(0, std::memory_order_relaxed);
                        lane.left_ig.store(lane.n_ig, std::memory_order_relaxed);   // 按完成项计数
                        lane.left_id.store(lane.n_id, std::memory_order_relaxed);
                        lane.phase.store(lane.n_ig ? 1 : 2, std::memory_order_relaxed);
                        lane.done.store(lane.n_id ? 0 : 1, std::memory_order_relaxed);
                        lane.gen.fetch_add(1, std::memory_order_release);
                        while (lane.done.load(std::memory_order_acquire) == 0) __builtin_ia32_pause();
                        dbl_spin += elapsed(tq0, tick()) - q_this;
                        t_cpu += elapsed(tc0, tick());
                        if (il == 3 && getenv("STEP37_WS_DBG")) {
                            char pn[300];
                            snprintf(pn, sizeof pn, "%s/wsdbg-partbuf-%d.bin",
                                     getenv("STEP37_DUMP_DIR") ? getenv("STEP37_DUMP_DIR") : "/tmp", il);
                            FILE* f = fopen(pn, "wb");
                            if (f) { fwrite(partbuf.data(), 4, 8 * 4096, f); fclose(f); }
                            fprintf(stderr, "WSDBG il=3 n_ig=%d n_id=%d n_exp=%d hits=%d\n",
                                    lane.n_ig, lane.n_id, lane.n_exp, n_hit);
                        }
                    }
                    const auto tt0 = tick();
                    for (int i = 0; i < 4096; ++i) ysum[i] = 0.f;
                    for (int e = 0; e < 8; ++e) {
                        const float* po = partbuf.data() + (size_t)e * 4096;
                        const float we = pin_w[e];
                        for (int i = 0; i < 4096; ++i) ysum[i] += we * po[i];
                    }
                    cudaStreamSynchronize(st);   // 等池命中专家的 D2H + shared expert
                    for (int i = 0; i < 4096; ++i) ysum[i] += pin_sh[i];
                    cudaMemcpyAsync(ws_.ffn_y, ysum, 16384, cudaMemcpyHostToDevice, st);
                    t_tail += elapsed(tt0, tick());
                } else {
                // ---- CPU lane（无池，全 8 专家）—— work-stealing 同款 ----
                const auto tc0 = tick();
                strata::kernels::cpu::native_quant_act(s_fmt, pin_x, lane_act);
                lane.n_ig = 0; lane.n_id = 0; lane.n_exp = 0;
                for (int e = 0; e < 8; ++e) {
                    const ExpertSlice& es = experts_[il - 3][pin_ids[e]];
                    float* ff_e = ffbuf.data() + (size_t)e * 1280;
                    void* hq_e = hq_shared + (size_t)lane.n_exp * strata::kernels::cpu::kNativeHBytes;
                    lane.ffq[lane.n_exp] = ff_e;
                    lane.hq_e[lane.n_exp] = hq_e;
                    ++lane.n_exp;
                    for (int c = 0; c < 16; ++c)
                        lane.ig[lane.n_ig++] = { es.gate, es.up, nullptr, ff_e, nullptr,
                                                 nullptr, c * 80, c * 80 + 80 };
                    for (int c = 0; c < 16; ++c)
                        lane.id[lane.n_id++] = { nullptr, nullptr, es.down, nullptr, partbuf.data() + (size_t)e * 4096,
                                                 hq_e, c * 256, c * 256 + 256 };
                }
                lane.act = lane_act;
                for (int i = 0; i < 4096; ++i) ysum[i] = 0.f;
                lane.next_ig.store(0, std::memory_order_relaxed);
                lane.next_id.store(0, std::memory_order_relaxed);
                lane.left_ig.store(lane.n_ig, std::memory_order_relaxed);   // 按完成项计数
                lane.left_id.store(lane.n_id, std::memory_order_relaxed);
                lane.phase.store(1, std::memory_order_relaxed);
                lane.done.store(0, std::memory_order_relaxed);
                lane.gen.fetch_add(1, std::memory_order_release);
                while (lane.done.load(std::memory_order_acquire) == 0) __builtin_ia32_pause();
                t_cpu += elapsed(tc0, tick());
                const auto tt0 = tick();
                for (int e = 0; e < 8; ++e) {
                    const float* po = partbuf.data() + (size_t)e * 4096;
                    const float we = pin_w[e];
                    for (int i = 0; i < 4096; ++i) ysum[i] += we * po[i];
                }
                { cudaError_t rc = cudaStreamSynchronize(st);
  if (rc != cudaSuccess) fprintf(stderr, "DBG tailsync rc=%s\n", cudaGetErrorString(rc)); }
                for (int i = 0; i < 4096; ++i) ysum[i] += pin_sh[i];
                { cudaError_t rc = cudaMemcpyAsync(ws_.ffn_y, ysum, 16384, cudaMemcpyHostToDevice, st);
  if (rc != cudaSuccess) fprintf(stderr, "DBG h2d rc=%s\n", cudaGetErrorString(rc)); }
                t_tail += elapsed(tt0, tick());
                step37dump::f32f(pos, ws_.shared_y, 4096, "ffn_shared_out-%d", il);
                // host-side wall dumps (router order) — parity wall only, env-gated
                if (const char* dd = getenv("STEP37_DUMP_DIR")) {
                    if (step37dump::pos_ok(pos)) {
                        char pn[160];
                        snprintf(pn, sizeof pn, "%s/ffn_moe_topk-%d.bin", dd, il);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(pin_ids, 4, 8, f); fclose(f); }
                        snprintf(pn, sizeof pn, "%s/ffn_moe_weights_scaled-%d.bin", dd, il);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(pin_w, 4, 8, f); fclose(f); }
                        snprintf(pn, sizeof pn, "%s/ffn_moe_swiglu-%d.bin", dd, il);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(ffbuf.data(), 4, 8 * 1280, f); fclose(f); }
                        snprintf(pn, sizeof pn, "%s/ffn_moe_down-%d.bin", dd, il);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(partbuf.data(), 4, 8 * 4096, f); fclose(f); }
                        snprintf(pn, sizeof pn, "%s/ffn_moe_out-%d.bin", dd, il);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(ysum, 4, 4096, f); fclose(f); }
                    }
                }
                }
            }
            strata::kernels::add_inplace(ws_.x, ws_.ffn_y, 4096, st);
            step37dump::f32f(pos, ws_.x, 4096, "l_out-%d", il);
        }
        // final norm + output head（Q6_K）→ logits
        {
            const DevTensor* fn = dev("output_norm.weight");
            strata::kernels::native_gr_rms_norm_weighted(ws_.x, (const float*)fn->ptr, ws_.xn, 4096, 1, 1e-5, st);
        step37dump::f32f(pos, ws_.xn, 4096, "result_norm");
            strata::kernels::native_quantize_q8_1(ws_.xn, q8_xn, 4096, 1, st);
            strata::kernels::native_q6_k_mmvq(out_w->ptr, q8_xn, ws_.logits, 4096, 128896, 1, st);
        step37dump::f32f(pos, ws_.logits, 128896, "result_output");
        }
        // 在线准入的搬运：放在 token 末尾，槽位不会被本 token 的后续层读到
        if (!pool_pending.empty()) {
            for (const auto& pk : pool_pending) {
                const ExpertSlice& es = experts_[pk.first / 288][pk.first % 288];
                memcpy(pool_pin, es.gate, 2785280);
                memcpy(pool_pin + 2785280, es.up, 2785280);
                memcpy(pool_pin + 2 * 2785280, es.down, 2785280);
                cudaMemcpyAsync(d_pool + (size_t)pk.second * kSlotBytes, pool_pin, kSlotBytes,
                                cudaMemcpyHostToDevice, st);
            }
            pool_pending.clear();
        }
        if ((pool_ev_events += 8 * 42) >= 32768) {   // LFU 老化（maya: 每 32768 事件整体减半）
            pool_ev_events = 0;
            for (auto& c : pool_cnt) c >>= 1;
        }
        pool_admit_budget = getenv("STEP37_POOL_ADMIT") ? atoi(getenv("STEP37_POOL_ADMIT")) : 8;
        return true;
    };

    // 暖机（= M1 朴素 prefill）：逐 token 前向 + 计时。resume_prompt = n_prompt 时
    // prompt 已由 prefill_run 前向（KV + ws_.logits 尾帧就位），直接 decode 接力
    const auto tp0 = std::chrono::steady_clock::now();
    if (resume_prompt < n_prompt) {
        for (int i = resume_prompt; i < n_prompt; ++i) {
            if (!forward_one(prompt_ids[i], base_pos + i)) { err = "forward failed"; return false; }
            cudaDeviceSynchronize();
        }
    }
    const double prefill_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
    fprintf(stderr, "PREFILL: %d tokens (resume=%d) in %.1fs\n", n_prompt - resume_prompt, resume_prompt, prefill_secs);
    // logits -> 采样（默认 greedy=argmax 不变，对拍口径不破；干活用 STEP37_TEMP>0 走
    // temperature/top-p/top-k CPU 采样：全量 logits D2H 515KB ≈ 0.3ms，可忽略）
    static int* d_am = nullptr; static int* h_am = nullptr;
    if (!d_am) { cudaMalloc(&d_am, 4); cudaMallocHost(&h_am, 4); }
    static float* h_logits = nullptr;
    // 采样参数：daemon 模式经 set_sampling() 逐请求下发；独立跑分模式回落 env（一次）
    const float s_temp   = samp_set_ ? samp_.temp   : (getenv("STEP37_TEMP")  ? atof(getenv("STEP37_TEMP"))  : 0.f);
    const float s_top_p  = samp_set_ ? samp_.top_p  : (getenv("STEP37_TOP_P") ? atof(getenv("STEP37_TOP_P")) : 1.f);
    const int   s_top_k  = samp_set_ ? samp_.top_k  : (getenv("STEP37_TOP_K") ? atoi(getenv("STEP37_TOP_K")) : 0);
    const int   s_eos    = samp_set_ ? samp_.eos    : (getenv("STEP37_EOS_ID") ? atoi(getenv("STEP37_EOS_ID")) : -1);
    // RP 手感三件（裸 temp+top-p 在 ablated 模型上实测退化循环；stock 用户人人都开 penalties）
    const float s_rep    = samp_set_ ? samp_.rep_pen : (getenv("STEP37_REP_PEN") ? atof(getenv("STEP37_REP_PEN")) : 1.1f);
    const int   s_rep_w  = samp_set_ ? samp_.rep_win : (getenv("STEP37_REP_WIN") ? atoi(getenv("STEP37_REP_WIN")) : 64);
    const float s_min_p  = samp_set_ ? samp_.min_p   : (getenv("STEP37_MIN_P")   ? atof(getenv("STEP37_MIN_P"))   : 0.05f);
    unsigned s_rng = samp_set_ ? samp_.seed : (getenv("STEP37_SEED") ? (unsigned)atoi(getenv("STEP37_SEED")) : 42u);
    samp_.seed = s_rng;   // LCG 状态写回，请求内连续
    auto next_token = [&]() -> int {
        if (s_temp <= 0.f) {
            device_argmax(ws_.logits, 128896, d_am, st);
            cudaMemcpy(h_am, d_am, 4, cudaMemcpyDeviceToHost);
            return h_am[0];
        }
        if (!h_logits) cudaMallocHost((void**)&h_logits, 128896 * 4);
        cudaMemcpy(h_logits, ws_.logits, 128896 * 4, cudaMemcpyDeviceToHost);
        // 重复惩罚（llama.cpp 语义：正 logit 除 penalty、负乘 penalty；prompt 尾 + 已生成
        // 都计入，重复出现重复罚）——裸 top-p 的循环退化实测靠它破
        const auto penalize = [&](const std::vector<int32_t>& v, int lo) {
            for (int j = lo; j < (int)v.size(); ++j) {
                const int t = v[j];
                if (t >= 0 && t < 128896) {
                    float& L = h_logits[t];
                    L = L > 0 ? L / s_rep : L * s_rep;
                }
            }
        };
        penalize(prompt_ids, std::max(0, (int)prompt_ids.size() - s_rep_w));
        penalize(out_tokens, std::max(0, (int)out_tokens.size() - s_rep_w));
        // softmax(x/T) → min_p + top-k/top-p 过滤 → 采样。
        // 全排序 128896 ≈ 15ms/token，吃 decode 预算；改 nth_element 选头 1024 候选
        // （temp≤1.2 时质量 0.95 远在此内），O(n) 选择 + 1024 内小排序 ≈ 2-3ms
        static std::vector<float> pr(128896);
        static std::vector<int> idx(128896);
        float mx = -1e30f;
        for (int i = 0; i < 128896; ++i) {
            const float x = h_logits[i];
            if (!std::isfinite(x)) h_logits[i] = -1e30f;   // NaN/inf 防护：一个 NaN 会让
            mx = std::max(mx, h_logits[i]);                // nth_element 的弱序比较器 UB
        }
        if (!std::isfinite(mx)) {
            static int nan_warn = 0;
            if (nan_warn++ < 8) fprintf(stderr, "SAMPLER: non-finite logits, fallback argmax\n");
            device_argmax(ws_.logits, 128896, d_am, st);
            cudaMemcpy(h_am, d_am, 4, cudaMemcpyDeviceToHost);
            return h_am[0];
        }
        float sum = 0.f;
        for (int i = 0; i < 128896; ++i) { pr[i] = std::exp((h_logits[i] - mx) / s_temp); sum += pr[i]; }
        const float inv = 1.f / sum;
        int nk = (s_top_k > 0 && s_top_k < 1024) ? s_top_k : 1024;
        for (int i = 0; i < 128896; ++i) idx[i] = i;
        std::nth_element(idx.begin(), idx.begin() + nk, idx.end(),
                         [&](int a, int b) { return pr[a] > pr[b]; });
        std::sort(idx.begin(), idx.begin() + nk, [&](int a, int b) { return pr[a] > pr[b]; });
        float cum = 0.f, pmax = -1.f; int cut = nk;
        for (int i = 0; i < nk; ++i) {
            const float p = pr[idx[(size_t)i]] * inv;
            if (i == 0) pmax = p;
            cum += p;
            if (p < s_min_p * pmax || cum >= s_top_p) { cut = i + 1; break; }   // min_p ∧ top-p
        }
        float r = (float)(s_rng = s_rng * 1664525u + 1013904223u) / 4294967296.f * cum;
        for (int i = 0; i < cut; ++i) {
            r -= pr[idx[(size_t)i]] * inv;
            if (r <= 0.f) return idx[(size_t)i];
        }
        return idx[0];
    };
    int tok = next_token();
    if (tok == s_eos) return true;   // 首 token 即 eos：空输出早停
    out_tokens.push_back(tok);
    ++n_gen;
    const auto t0 = std::chrono::steady_clock::now();
    const bool tok_ledger = getenv("STEP37_TOKLEDGER") != nullptr;
    double tl_fwd = 0, tl_sync = 0, tl_amx = 0;
    for (int i = 1; i < n_decode; ++i) {
        const auto t0 = tick();
        const double g0 = t_gpu, s0 = t_sync, c0 = t_cpu, a0 = t_tail;
        if (!forward_one(tok, base_pos + n_prompt - 1 + i)) { err = "forward failed"; return false; }
        const auto t1 = tick();
        cudaDeviceSynchronize();
        const auto t2 = tick();
        tok = next_token();
        const auto t3 = tick();
        if (tok == s_eos) break;   // eos 早停（干活模式：出 <|im_end|> 即收；跑分模式 env 不设则永不触发）
        out_tokens.push_back(tok);
        ++n_gen;
        tl_fwd += elapsed(t0, t1); tl_sync += elapsed(t1, t2); tl_amx += elapsed(t2, t3);
        if (tok_ledger && i <= 12) {
            const double w = elapsed(t0, t3);
            fprintf(stderr, "TOK %2d wall=%.2fms | enq=%.2f sync=%.2f cpu=%.2f tail=%.2f "
                            "| dsync=%.2f argmax=%.2f | unattributed=%.2f\n",
                    i, w * 1000, (t_gpu - g0) * 1000, (t_sync - s0) * 1000, (t_cpu - c0) * 1000,
                    (t_tail - a0) * 1000, elapsed(t1, t2) * 1000, elapsed(t2, t3) * 1000,
                    (w - (t_gpu - g0) - (t_sync - s0) - (t_cpu - c0) - (t_tail - a0)
                       - elapsed(t1, t2) - elapsed(t2, t3)) * 1000);
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    decode_tps = n_gen > 1 ? (n_gen - 1) / secs : 0.0;
    if (const char* po = getenv("STEP37_PROFILE_OUT")) {
        // 池开着就落"真实池的 LFU 计数"（含准入累积，对应 maya 的 expert_usage.txt：
        // 跨会话累积，下一轮起池直接继承本用户的热度分布）；否则落影子池计数
        const std::vector<uint32_t>& src = pool_cnt.empty() ? psim[4].cnt : pool_cnt;
        if (FILE* f = fopen(po, "w")) {
            int n = 0;
            for (int k = 0; k < 42 * 288; ++k)
                if (src[(size_t)k]) { fprintf(f, "%d %u\n", k, src[(size_t)k]); ++n; }
            fclose(f);
            fprintf(stderr, "PROFILE: written %s (%d keys)\n", po, n);
        }
    }
    {
        fprintf(stderr, "POOL-SIM hit-rate (slots -> hit%%, evictions):\n");
        for (const auto& ps : psim)
            fprintf(stderr, "  %5d slots -> %5.1f%%  hits=%llu miss=%llu evict=%llu\n",
                    ps.cap, ps.rate() * 100.0,
                    (unsigned long long)ps.hits, (unsigned long long)ps.miss,
                    (unsigned long long)ps.evict);
    }
    fprintf(stderr, "CPU-LANE: quant=%.1fms spin(work+idle)=%.1fms over %d tok\n",
            dbl_quant * 1000 / std::max(1, n_gen), dbl_spin * 1000 / std::max(1, n_gen), n_gen);
    // M1.5 phase report (seconds across this generate() call)
    fprintf(stderr, "PHASES: gpu_enqueue=%.3f sync=%.3f cpu_experts=%.3f tail=%.3f (n_tok=%d)\n",
            t_gpu, t_sync, t_cpu, t_tail, n_prompt + n_gen);
    return true;
}


bool Model::prefill_run(const std::vector<int32_t>& prompt_ids, double& pp_tps, std::string& err, int start_pos) {
    if (!loaded_) { err = "not loaded"; return false; }
    kv_wiped_ = 0;
    const int T = getenv("STEP37_UB") ? atoi(getenv("STEP37_UB")) : 512;
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);

    // ---- VRAM 预算：释放 decode 尾层驻留（结束时重载），KV cap 提到 prompt 长度 ----
    const int saved_tail = gpu_tail_;
    if (d_gpu_exp_) { cudaFree(d_gpu_exp_); d_gpu_exp_ = nullptr; gpu_tail_ = 0; }
    const int64_t need_pos = (int64_t)start_pos + (int64_t)prompt_ids.size();
    if ((int64_t)kv_cap_ < need_pos) {
        cudaFree(d_kcache_); cudaFree(d_vcache_);
        // 定容假设雷：prompt 长度恰为 4096 整数倍时 kv_cap == prompt 长度，
        // decode 第一个 token 要写第 kv_cap 行 → 越界（49152 崩、50000 因取整侥幸有余量）。
        // 预留一整块给 decode 的 KV append。
        // 环形寻址用位掩码（attention 热循环 & (cap-1)）→ cap 必须 2 的幂
        { uint32_t c = (uint32_t)(need_pos + 4096 + 4095); c >>= 1;
          while (c & (c - 1)) c &= c - 1; kv_cap_ = c << 1; }
        recompute_kv_layout();
        if (start_pos > 0) kv_wiped_ = 1;   // 重分配清空了前缀 KV——调用方必须全量重 prefill
        const uint64_t kv_bytes = (uint64_t)kv_base_bytes_[44] + (uint64_t)kv_lcap_[44] * 1088;  // q8_0
        if (cudaMalloc(&d_kcache_, kv_bytes) != cudaSuccess ||
            cudaMalloc(&d_vcache_, kv_bytes) != cudaSuccess) {
            err = "prefill KV realloc failed"; return false;
        }
    }
    // rope 表（50K 口径，prefill 专用）——跨请求缓存（daemon：每请求省 201MB 分配+建表）
    static float *cos_f = nullptr, *sin_f = nullptr, *cos_s = nullptr, *sin_s = nullptr;
    static uint32_t rope_cap = 0;
    if (rope_cap != kv_cap_) {
        if (cos_f) { cudaFree(cos_f); cudaFree(sin_f); cudaFree(cos_s); cudaFree(sin_s); }
        cudaError_t e1 = cudaMalloc(&cos_f, (size_t)kv_cap_ * 32 * 4), e2 = cudaMalloc(&sin_f, (size_t)kv_cap_ * 32 * 4);
        cudaError_t e3 = cudaMalloc(&cos_s, (size_t)kv_cap_ * 64 * 4), e4 = cudaMalloc(&sin_s, (size_t)kv_cap_ * 64 * 4);
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess || e4 != cudaSuccess) {
            err = "prefill rope table alloc failed"; 
            fprintf(stderr, "ROPEALLOC FAIL: %s %s %s %s\n", cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3), cudaGetErrorString(e4));
            return false;
        }
        rope_cap = kv_cap_;
    }
    if (getenv("STEP37_ROPEDBG"))
        fprintf(stderr, "ROPEDBG: cap=%u rope_cap=%u ptrs=%p/%p/%p/%p\n",
                kv_cap_, rope_cap, (void*)cos_f, (void*)sin_f, (void*)cos_s, (void*)sin_s);
    build_llama3_rope_table(64, 5e6, kv_cap_, cos_f, sin_f);
    build_llama3_rope_table(128, 10000.0, kv_cap_, cos_s, sin_s);
    // ---- MMQ context + 批量缓冲 ----
    static strata::prefill::mmq::Context mmq;
    if (!strata::prefill::mmq::built() || !strata::prefill::mmq::supported(23)) {
        err = "mmq not available"; return false;
    }
    // ---- P2: cuBLAS handle + FP16 权重 arena（maya hgemm_q 判例：投影 = dequant once → 16F TC GEMM）----
    // 尾层专家已释放（~24.6GB 空闲），整块 10.6GB FP16 投影缓存驻留整个 prefill，跨 chunk 复用
    cublasHandle_t cbh = nullptr;
    if (cublasCreate(&cbh) != CUBLAS_STATUS_SUCCESS) { err = "cublas create failed"; return false; }
    cublasSetStream(cbh, st);
    {   // 确定性：cuBLAS 位级复现承诺以"同 workspace 配置"为前提。默认池地址随进程
        // 分配历史漂移 → 启发式选核/对齐路径逐跑不同 = prefill 非确定根因
        // （2026-11-11：逐部件探针全确定、同输入同权重仍逐跑换层发散）。固定自管 workspace。
        static void* cw = nullptr;
        if (!cw) cudaMalloc(&cw, 128 << 20);
        cublasSetWorkspace(cbh, cw, 128 << 20);
    }
    if (getenv("STEP37_GEMMPROBE")) {
        // QPROBE5：hgemm_f16（cublasGemmEx FP16）跨进程确定性判据。
        // 三形状：SWA qkv [12288,22,4096] / full qkv [8192,22,4096] / o-proj [4096,22,8192]
        struct GShape { int N, T, K; const char* nm; };
        const GShape shapes[] = {{12288, 22, 4096, "swa-qkv"}, {8192, 22, 4096, "full-qkv"}, {4096, 22, 8192, "oproj"}};
        __half *A, *B; float* Y;
        cudaMalloc(&A, (size_t)12288 * 8192 * 2); cudaMalloc(&B, (size_t)8192 * 22 * 2);
        cudaMalloc(&Y, (size_t)12288 * 22 * 4);
        std::vector<__half> hA((size_t)12288 * 8192), hB((size_t)8192 * 22);
        unsigned s4 = 4242;
        for (auto& v : hA) { s4 = s4 * 1664525u + 1013904223u; v = __float2half((float)(int)(s4 >> 8) / 8388608.f - 1.f); }
        for (auto& v : hB) { s4 = s4 * 1664525u + 1013904223u; v = __float2half((float)(int)(s4 >> 8) / 8388608.f - 1.f); }
        cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice);
        for (auto& g : shapes) {
            const size_t yb = (size_t)g.N * g.T * 4;
            std::vector<float> ref(yb / 4), cur(yb / 4);
            int badit = 0;
            for (int it = 0; it < 30; ++it) {
                hgemm_f16(cbh, A, B, Y, g.N, g.T, g.K);
                cudaMemcpy(cur.data(), Y, yb, cudaMemcpyDeviceToHost);
                if (it == 0) { ref = cur; continue; }
                if (memcmp(cur.data(), ref.data(), yb) != 0) ++badit;
            }
            char pn[64]; snprintf(pn, sizeof pn, "/tmp/gprobe_%s.bin", g.nm);
            FILE* f = fopen(pn, "wb"); fwrite(ref.data(), 4, ref.size(), f); fclose(f);
            std::printf("GPROBE %s: %d/30 in-process differed\n", g.nm, badit);
        }
        return 0;
    }
    struct WProj16 { __half *q, *k, *v, *g, *o; };
    struct WFFN16 { __half *g, *u, *d; };
    std::vector<WProj16> wproj(45);
    std::vector<WFFN16> wffn(45);          // 0-2 dense(11264)，3-44 shexp(1280)
    const bool no_arena = getenv("STEP37_NO_ARENA") != nullptr;
    size_t arena_total = 0;
    __half* w16_arena = nullptr;
    {
        // STEP37_NO_ARENA=1：不驻留全模型 FP16 投影（10.4GB），改单层旋转缓冲（~0.45GB，
        // 每层进循环时重反量化——VRAM 流量 ~16ms/chunk，换 ub8192/16384 的激活空间）
        arena_total = 0;
        if (no_arena) {
            size_t mx = 0;
            for (int il = 0; il < 45; ++il) {
                const int nqh = (il % 4 == 0) ? 64 : 96, qb = nqh * 128;
                size_t attn = (size_t)qb * 4096 + 2 * 1024 * 4096 + (size_t)nqh * 4096 + (size_t)4096 * qb;
                size_t ffn = (il < 3) ? 2 * (size_t)11264 * 4096 + (size_t)4096 * 11264
                                      : 2 * (size_t)1280 * 4096 + (size_t)4096 * 1280;
                mx = std::max(mx, attn + ffn);
            }
            arena_total = mx;
        } else {
            for (int il = 0; il < 45; ++il) {
                const int nqh = (il % 4 == 0) ? 64 : 96, qb = nqh * 128;
                arena_total += (size_t)qb * 4096 + 2 * 1024 * 4096 + (size_t)nqh * 4096 + (size_t)4096 * qb;
            }
            for (int il = 0; il < 3; ++il)  arena_total += 2 * (size_t)11264 * 4096 + (size_t)4096 * 11264;
            for (int il = 3; il < 45; ++il) arena_total += 2 * (size_t)1280 * 4096 + (size_t)4096 * 1280;
        }
        if (cudaMalloc(&w16_arena, arena_total * 2) != cudaSuccess) {
            cublasDestroy(cbh);
            err = "fp16 weight arena OOM"; return false;
        }
        fprintf(stderr, "P2: fp16 proj mode = %s (%.2f GB)\n",
                no_arena ? "rotate-per-layer" : "full-resident", arena_total * 2.0 / (1 << 30));
    }
    if (!no_arena) {
        size_t off = 0;
        auto take = [&](size_t vals) { __half* p = w16_arena + off; off += vals; return p; };
        auto dq = [&](const char* name, __half* dst, int64_t vals, int type) {
            if (!strata::prefill::mmq::to_f16(type, dev(name)->ptr, (uint16_t*)dst, vals, st)) {
                fprintf(stderr, "STEP37 to_f16 unsupported type %d: %s\n", type, name);
                return false;
            }
            return true;
        };
        char nm[80];
        for (int il = 0; il < 45; ++il) {
            const int nqh = (il % 4 == 0) ? 64 : 96, qb = nqh * 128;
            wproj[il].q = take((size_t)qb * 4096);
            wproj[il].k = take((size_t)1024 * 4096);
            wproj[il].v = take((size_t)1024 * 4096);
            wproj[il].g = take((size_t)nqh * 4096);
            wproj[il].o = take((size_t)4096 * qb);
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", il);
            if (!dq(nm, wproj[il].q, (int64_t)qb * 4096, 23)) { err = "dq q"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", il);
            if (!dq(nm, wproj[il].k, (int64_t)1024 * 4096, 23)) { err = "dq k"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", il);
            if (!dq(nm, wproj[il].v, (int64_t)1024 * 4096, 13)) { err = "dq v (Q5_K)"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", il);
            if (!dq(nm, wproj[il].g, (int64_t)nqh * 4096, 23)) { err = "dq gate"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", il);
            if (!dq(nm, wproj[il].o, (int64_t)4096 * qb, 23)) { err = "dq o"; return false; }
        }
        for (int il = 0; il < 3; ++il) {
            const int nf = 11264;
            wffn[il].g = take((size_t)nf * 4096);
            wffn[il].u = take((size_t)nf * 4096);
            wffn[il].d = take((size_t)4096 * nf);
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", il);
            if (!dq(nm, wffn[il].g, (int64_t)nf * 4096, 23)) { err = "dq dense g"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", il);
            if (!dq(nm, wffn[il].u, (int64_t)nf * 4096, 23)) { err = "dq dense u"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", il);
            if (!dq(nm, wffn[il].d, (int64_t)4096 * nf, 23)) { err = "dq dense d"; return false; }
        }
        for (int il = 3; il < 45; ++il) {
            const int nf = 1280;
            wffn[il].g = take((size_t)nf * 4096);
            wffn[il].u = take((size_t)nf * 4096);
            wffn[il].d = take((size_t)4096 * nf);
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_shexp.weight", il);
            if (!dq(nm, wffn[il].g, (int64_t)nf * 4096, 23)) { err = "dq shexp g"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.ffn_up_shexp.weight", il);
            if (!dq(nm, wffn[il].u, (int64_t)nf * 4096, 23)) { err = "dq shexp u"; return false; }
            snprintf(nm, sizeof nm, "blk.%d.ffn_down_shexp.weight", il);
            if (!dq(nm, wffn[il].d, (int64_t)4096 * nf, 23)) { err = "dq shexp d"; return false; }
        }
        cudaStreamSynchronize(st);
        const cudaError_t de = cudaGetLastError();
        if (de != cudaSuccess) { err = "fp16 arena dequant failed"; return false; }
        fprintf(stderr, "P2: fp16 weight arena %.2f GB resident (%zu vals)\n",
                arena_total * 2.0 / (1 << 30), off);
    }
    const size_t R = (size_t)T * 8;
    auto alloc_f = [&](size_t n) { float* p = nullptr; cudaMalloc(&p, n * 4); return p; };
    float *bx = alloc_f((size_t)T * 4096), *bxn = alloc_f((size_t)T * 4096);
    float *bffn_x = alloc_f((size_t)T * 4096), *bffn_y = alloc_f((size_t)T * 4096), *bo = alloc_f((size_t)T * 4096);
    float *bq = alloc_f((size_t)T * 12288), *bq2 = alloc_f((size_t)T * 12288);
    float *bk = alloc_f((size_t)T * 1024), *bk2 = alloc_f((size_t)T * 1024);
    float *bv = alloc_f((size_t)T * 1024), *battn = alloc_f((size_t)T * 12288);
    float *bgate = alloc_f((size_t)T * 128), *blogits = alloc_f((size_t)T * 288);   // 128 宽：MMQ tile 列宽下限
    float *bscores = alloc_f((size_t)96 * 8192);   // decode 三 kernel 的 scores scratch
    float *bh1 = alloc_f((size_t)T * 11264), *bh2 = alloc_f((size_t)T * 11264);
    float *gu = alloc_f(R * 1280), *uu = alloc_f(R * 1280), *bh = alloc_f(R * 1280);
    float *gnorm_b = alloc_f((size_t)T * 4096);   // 平铺 gamma（norm kernel 的 gamma 是 [n_rows×n_cols] 平铺语义）
    float *parts = alloc_f(R * 4096), *bsh = alloc_f((size_t)T * 4096);
    float *gq_b = alloc_f((size_t)T * 96 * 128), *gk_b = alloc_f((size_t)T * 8 * 128);
    // P2: FP16 激活侧（x16=bxn/bffn_x [T,4096]；a16=battn [T,12288]；h16=bh1/gu [T,≤12288]）
    __half *x16 = nullptr, *a16 = nullptr, *h16 = nullptr;
    cudaMalloc(&x16, (size_t)T * 4096 * 2);
    cudaMalloc(&a16, (size_t)T * 12288 * 2);
    cudaMalloc(&h16, (size_t)T * 12288 * 2);
    void *xq_a = nullptr; cudaMalloc(&xq_a, strata::prefill::mmq::q8_bytes(T > (int64_t)R ? T : (int64_t)R, 4096));
    void *xq_h = nullptr; cudaMalloc(&xq_h, strata::prefill::mmq::q8_bytes((int64_t)R, 1280));
    void *xq_p = nullptr; cudaMalloc(&xq_p, strata::prefill::mmq::q8_bytes(T, 4096));
    int *d_bounds = nullptr; cudaMalloc(&d_bounds, (290 + 512) * 4);   // +IDS_PAD
    cudaMemset(d_bounds + 289 * 4, 0, (512 + 1) * 4);                  // pad 恒零（设备派发只写 [0..288]）
    int *d_rcnt = nullptr, *d_rofs = nullptr;
    cudaMalloc(&d_rcnt, 288 * 4); cudaMalloc(&d_rofs, 288 * 4);
    int *pf_eids = nullptr; float *pf_ew = nullptr;
    cudaMalloc(&pf_eids, (size_t)T * 8 * 4); cudaMalloc(&pf_ew, (size_t)T * 8 * 4);
    int *d_ids_tok = nullptr, *d_ids_row = nullptr, *d_inv = nullptr;
    cudaMalloc(&d_ids_tok, (size_t)(R + 512) * 4); cudaMalloc(&d_ids_row, (size_t)(R + 512) * 4);   // +IDS_PAD
    cudaMalloc(&d_inv, (size_t)T * 8 * 4);   // 逆映射 inv[路由i]=散射槽p（combine 按 rank 序确定性 gather）
    int *d_toks = nullptr, *d_p0 = nullptr, *d_posarr = nullptr;
    cudaMalloc(&d_toks, T * 4); cudaMalloc(&d_p0, 4); cudaMalloc(&d_posarr, T * 4);
    int *d_posq = nullptr, *d_posk = nullptr;   // rope 的行序 pos（[t][nqh] / [t][8]），每层重填
    cudaMalloc(&d_posq, (size_t)T * 96 * 4); cudaMalloc(&d_posk, (size_t)T * 8 * 4);
    int *d_unit = nullptr; cudaMalloc(&d_unit, 8);  // [0, T] bounds for plain products
    int *d_iota = nullptr; cudaMalloc(&d_iota, (size_t)(R + 512) * 4);   // 路由行号空间全长 iota + IDS_PAD（mmq.cuh 按
                                                                          // 全局行号读 ids，值 = dst 行；长度 T 的版本
                                                                          // 在 R=tN*8 > T 时读爆 → 垃圾行号写越界）
    // 专家整层 staging（gate|up|down 三段）
    const int nslot = getenv("STEP37_STAGE_SLOTS") ? atoi(getenv("STEP37_STAGE_SLOTS")) : 2;
    uint8_t* d_stage2[2] = {nullptr, nullptr};   // nslot=1 时只用 [0]
    const uint64_t layer_mat = 288ull * 2785280;      // 单矩阵整层 288 专家
    cudaMalloc(&d_stage2[0], 3 * layer_mat);          // P2.8: 双缓冲 stage——本层 DMA 与本层
    if (nslot > 1) cudaMalloc(&d_stage2[1], 3 * layer_mat);  // attention 在 copy engine/SM 上并行
                                                      // STEP37_STAGE_SLOTS=1：省 2.4GB（双实例模式）
    // host 侧 bucket（全 pinned + async：同步版 cudaMemcpy 在 legacy 流会隐式同步整条 st，
    // 每层等上越来越慢的 attention kernel —— 50K "other" 同斜率爆炸的根因）
    std::vector<int> h_ids(T * 8), h_wi(T * 8);
    std::vector<float> h_w(T * 8);
    int* pin_ids8 = nullptr; float* pin_w8 = nullptr;
    cudaMallocHost(&pin_ids8, (size_t)T * 8 * 4); cudaMallocHost(&pin_w8, (size_t)T * 8 * 4);
    int *pin_ids_tok = nullptr, *pin_ids_row = nullptr, *pin_bounds = nullptr;
    float* pin_w_route = nullptr;
    cudaMallocHost(&pin_ids_tok, R * 4); cudaMallocHost(&pin_ids_row, R * 4);
    cudaMallocHost(&pin_w_route, R * 4); cudaMallocHost(&pin_bounds, (290 + 512) * 4);
    std::vector<int> h_posarr(T);
    int h_unit[2] = {0, 0};
    cudaMemcpy(d_unit, h_unit, 8, cudaMemcpyHostToDevice);
    float* wq_dev = nullptr;  // plain product bounds dummy 不需要——用 d_unit
    h_unit[1] = T;
    cudaMemcpy(d_unit, h_unit, 8, cudaMemcpyHostToDevice);
    {   // iota（路由行号空间全长 [0..R)）+ IDS_PAD 尾巴清零（moe_mmq.hpp issue #645：tile 读不查界）
        std::vector<int> io(R + 512, 0);
        for (size_t i = 0; i < R; ++i) io[i] = (int)i;
        cudaMemcpy(d_iota, io.data(), (size_t)(R + 512) * 4, cudaMemcpyHostToDevice);
        cudaMemset(d_ids_tok, 0, (size_t)512 * 4);
        cudaMemset(d_ids_row, 0, (size_t)512 * 4);
    }
    const char* dmp = getenv("STEP37_DUMP_DIR");
    const float kq_scale = 0.08838834764831845f;
    const auto t_all0 = std::chrono::steady_clock::now();
    // 分项计时（CUDA events）：attention 段 / MoE 段（含 staging H2D）——50K 瓶颈定位用
    cudaEvent_t ev_at[45][2], ev_mo[42][2], ev_ck;
    for (auto& p : ev_at) { cudaEventCreate(&p[0]); cudaEventCreate(&p[1]); }
    for (auto& p : ev_mo) { cudaEventCreate(&p[0]); cudaEventCreate(&p[1]); }
    cudaEventCreate(&ev_ck);
    // P2.1: copy stream（staging H2D 与计算重叠）+ 交接事件；d_w_route 常驻
    cudaStream_t stc = nullptr; cudaStreamCreate(&stc);
    cudaEvent_t ev_stage_gu[2], ev_stage_dn[2];
    for (int i = 0; i < 2; ++i) { cudaEventCreate(&ev_stage_gu[i]); cudaEventCreate(&ev_stage_dn[i]); }
    float* d_w_route = nullptr; cudaMalloc(&d_w_route, R * 4);
    memset((char*)pin_bounds + 290 * 4, 0, 512 * 4);
    // rope pos 的 pinned 三版（64/96 头行序 + KV 8 头行序），chunk 开头填一次
    int *pin_pq64 = nullptr, *pin_pq96 = nullptr, *pin_pk = nullptr;
    cudaMallocHost(&pin_pq64, (size_t)T * 64 * 4);
    cudaMallocHost(&pin_pq96, (size_t)T * 96 * 4);
    cudaMallocHost(&pin_pk, (size_t)T * 8 * 4);
    // P2.2: 专家 staging pinned ring（maya 判例：mmap 源的 async H2D 走 driver staging 慢路径，
    // 实测 ~5GB/s；pinned landing + 后台预拷线程 → DMA 26GB/s 且与计算完全重叠）
    struct StageSlot {
        uint8_t *gu = nullptr, *dn = nullptr;   // gu = gate|up 拼段（2×layer_mat），dn = down
        int state = 0;                          // 0=free 2=ready
        int layer = -1, chunk = -1;             // 槽内货的 (chunk, layer)
    };
    StageSlot pslot[2];   // nslot=1 时只用 [0]
    cudaMallocHost(&pslot[0].gu, 2 * layer_mat); cudaMallocHost(&pslot[0].dn, layer_mat);
    if (nslot > 1) { cudaMallocHost(&pslot[1].gu, 2 * layer_mat); cudaMallocHost(&pslot[1].dn, layer_mat); }
    else { pslot[1].gu = pslot[1].dn = nullptr; }   // HOST 内存（不占 VRAM），但预拷线程会碰 → 置空防写
    cudaEvent_t ev_pin_free[2], ev_gpu_free[2];
    for (int i = 0; i < 2; ++i) { cudaEventCreate(&ev_pin_free[i]); cudaEventCreate(&ev_gpu_free[i]); }
    std::mutex pf_mu; std::condition_variable pf_cv;
    const int n_chunks_total = (int)((prompt_ids.size() + T - 1) / T);
    // 预拷线程：按全局 (chunk, layer) 序填双槽；gate/up 与 down 双路并行 memcpy。
    // pinned 槽复用 = 后台对 ev_pin_free 同步（上次同槽 DMA 完成）；
    // d_stage 复用 = stc 对 ev_gpu_free 的 wait（同槽上上层 MMQ down 完成）。
    std::thread prefetcher([&] {
        bool first2 = true;
        for (int ck = 0; ck < n_chunks_total; ++ck)
            for (int lj = 0; lj < 42; ++lj) {
                const int il = 3 + lj;
                StageSlot& s = pslot[lj % nslot];
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [&] { return s.state != 2; });   // 0=free 3=abandoned 都放行
                lk.unlock();
                if (!first2) cudaEventSynchronize(ev_pin_free[lj % nslot]);
                if (lj == 1 && ck == 0) first2 = false;
                const ExpertSlice& es = experts_[il - 3][0];
                std::thread c1([&] { memcpy(s.gu, es.gate, layer_mat); memcpy(s.gu + layer_mat, es.up, layer_mat); });
                memcpy(s.dn, es.down, layer_mat);
                c1.join();
                lk.lock();
                s.layer = il; s.chunk = ck; s.state = 2;
                lk.unlock(); pf_cv.notify_all();
            }
    });
    struct PJoin {   // 异常安全 joiner：异常穿越 prefill_run 时若不 join，std::thread
        std::thread* t;            // 析构 joinable = std::terminate（2026-11-12 深夜实证：
        std::mutex* mu;            // OOM 异常连锁杀引擎）。放行 abandoned 让线程自然退出。
        std::condition_variable* cv;
        StageSlot* slots;
        ~PJoin() {
            { std::lock_guard<std::mutex> lk(*mu);
              for (int i = 0; i < 2; ++i) if (slots[i].state == 2) slots[i].state = 3; }
            cv->notify_all();
            if (t->joinable()) t->join();
        }
    } pj{&prefetcher, &pf_mu, &pf_cv, pslot};

    double host_wait_pf = 0.0, host_wait_sync = 0.0;   // other 探针：等预拷槽 / 等 GPU 同步
    const auto tick = [] { return std::chrono::steady_clock::now(); };
    auto elapsed = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    float* bx_last = alloc_f(4096);   // 每 chunk 的尾 token 快照（尾 logits 用，bx 只有 T 行）
    for (int64_t q0 = 0; q0 < (int64_t)prompt_ids.size(); q0 += T) {
        const int64_t p0 = start_pos + q0;   // 绝对位置基（追加模式 = 已入 KV 的前缀长）
        const int tN = (int)std::min<int64_t>(T, (int64_t)prompt_ids.size() - q0);
        host_wait_pf = host_wait_sync = 0.0;
        for (int t = 0; t < tN; ++t) h_posarr[t] = (int)(p0 + t);
        for (int t = 0; t < tN; ++t) {
            for (int h = 0; h < 64; ++h) pin_pq64[(size_t)t * 64 + h] = (int)(p0 + t);
            for (int h = 0; h < 96; ++h) pin_pq96[(size_t)t * 96 + h] = (int)(p0 + t);
            for (int h = 0; h < 8; ++h) pin_pk[(size_t)t * 8 + h] = (int)(p0 + t);
        }
        cudaMemcpy(d_toks, prompt_ids.data() + q0, tN * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_posarr, h_posarr.data(), tN * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_p0, &p0, 4, cudaMemcpyHostToDevice);
        auto bdump = [&](const void* dev_p, int n_f32, const char* name) {
            if (!dmp || q0 != 0) return;
            cudaStreamSynchronize(st);
            std::vector<char> buf((size_t)n_f32 * 4);
            cudaMemcpy(buf.data(), dev_p, (size_t)n_f32 * 4, cudaMemcpyDeviceToHost);
            char pn[300]; snprintf(pn, sizeof pn, "%s/%s.bin", dmp, name);
            if (FILE* f = fopen(pn, "wb")) { fwrite(buf.data(), 1, buf.size(), f); fclose(f); }
        };
        iq4_xs_rows((const uint8_t*)dev("token_embd.weight")->ptr, d_toks, bx, tN, st);
        bdump(bx, tN * 4096, "tok_embd_b");
        for (int il = 0; il < 45; ++il) {
            if (il >= 3) {
                // P2.8: 本层 stage DMA 提前到层首——copy engine 与本层 attention（~300ms）
                // 并行，消灭"DMA 独占时间线"的 92ms/层 GPU 闲置
                const int lj0 = il - 3, par0 = lj0 % nslot;
                StageSlot& ps0 = pslot[lj0 % nslot];
                {
                    const auto w0 = std::chrono::steady_clock::now();
                    std::unique_lock<std::mutex> lk(pf_mu);
                    pf_cv.wait(lk, [&] { return ps0.state == 2 && ps0.layer == il && ps0.chunk == (int)(p0 / T); });
                    host_wait_pf += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                }
                cudaStreamWaitEvent(stc, ev_gpu_free[par0], 0);   // 同缓冲上上层 MMQ 完成
                cudaMemcpyAsync(d_stage2[par0], ps0.gu, 2 * layer_mat, cudaMemcpyHostToDevice, stc);
                cudaMemcpyAsync(d_stage2[par0] + 2 * layer_mat, ps0.dn, layer_mat, cudaMemcpyHostToDevice, stc);
                cudaEventRecord(ev_stage_gu[par0], stc);
                cudaEventRecord(ev_stage_dn[par0], stc);
                cudaEventRecord(ev_pin_free[par0], stc);          // pinned 槽释放（预拷线程可复用）
                { std::lock_guard<std::mutex> lk(pf_mu); ps0.state = 0; }
                pf_cv.notify_all();
            }
            const bool full_il = (il % 4 == 0);
            const int nqh = full_il ? 64 : 96;
            const int qb = nqh * 128;
            const int nrot = full_il ? 64 : 128;
            const float* cosT = full_il ? cos_f : cos_s;
            const float* sinT = full_il ? sin_f : sin_s;
            char nm[64];
            if (no_arena) {   // 旋转缓冲：本层投影就地重反量化（to_f16 现成件）
                size_t roff = 0;
                auto rot = [&](size_t vals) { __half* p = w16_arena + roff; roff += vals; return p; };
                const int nqh_ = full_il ? 64 : 96, qb_ = nqh_ * 128;
                wproj[il].q = rot((size_t)qb_ * 4096);
                wproj[il].k = rot((size_t)1024 * 4096);
                wproj[il].v = rot((size_t)1024 * 4096);
                wproj[il].g = rot((size_t)nqh_ * 4096);
                wproj[il].o = rot((size_t)4096 * qb_);
                snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", il);
                strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wproj[il].q, (int64_t)qb_ * 4096, st);
                snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", il);
                strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wproj[il].k, (int64_t)1024 * 4096, st);
                snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", il);
                strata::prefill::mmq::to_f16(13, dev(nm)->ptr, (uint16_t*)wproj[il].v, (int64_t)1024 * 4096, st);
                snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", il);
                strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wproj[il].g, (int64_t)nqh_ * 4096, st);
                snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", il);
                strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wproj[il].o, (int64_t)4096 * qb_, st);
                if (il < 3) {
                    wffn[il].g = rot((size_t)11264 * 4096);
                    wffn[il].u = rot((size_t)11264 * 4096);
                    wffn[il].d = rot((size_t)4096 * 11264);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", il);
                    strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wffn[il].g, (int64_t)11264 * 4096, st);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", il);
                    strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wffn[il].u, (int64_t)11264 * 4096, st);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", il);
                    strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wffn[il].d, (int64_t)4096 * 11264, st);
                } else {
                    wffn[il].g = rot((size_t)1280 * 4096);
                    wffn[il].u = rot((size_t)1280 * 4096);
                    wffn[il].d = rot((size_t)4096 * 1280);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_shexp.weight", il);
                    strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wffn[il].g, (int64_t)1280 * 4096, st);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_up_shexp.weight", il);
                    strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wffn[il].u, (int64_t)1280 * 4096, st);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_down_shexp.weight", il);
                    strata::prefill::mmq::to_f16(23, dev(nm)->ptr, (uint16_t*)wffn[il].d, (int64_t)4096 * 1280, st);
                }
            }
            // attn norm（批量）→ QKV O 投影（MMQ plain product）
            snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", il);
            tile_expand((const float*)dev(nm)->ptr, gnorm_b, tN, 1, 4096, st);
            strata::kernels::native_gr_rms_norm_weighted(bx, gnorm_b, bxn, 4096, tN, 1e-5, st);
            { char pn[64]; snprintf(pn, sizeof pn, "attn_norm-b%d", il); bdump(bxn, tN * 4096, pn); }
            if (il == 0) bdump(bxn, tN * 4096, "bxn_now");
            // P2: Q/K/V 投影 = 一次 f16 转换 + 3× GemmEx（TC）。V(Q5_K) 同走 to_f16 预反量化。
            f32_to_f16(bxn, x16, (long)tN * 4096, st);
            hgemm_f16(cbh, wproj[il].q, x16, bq, qb, tN, 4096);
            hgemm_f16(cbh, wproj[il].k, x16, bk, 1024, tN, 4096);
            hgemm_f16(cbh, wproj[il].v, x16, bv, 1024, tN, 4096);
            { char pn[64]; snprintf(pn, sizeof pn, "Qcur-b%d", il); bdump(bq, tN * qb, pn); }

            { char pn[64]; snprintf(pn, sizeof pn, "Vcur-b%d", il); bdump(bv, tN * 1024, pn); }
            // q/k norm（gamma 批量平铺 + 批量 norm，输出写回原位）
            snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", il);
            tile_expand((const float*)dev(nm)->ptr, gq_b, tN, nqh, 128, st);
            strata::kernels::native_gr_rms_norm_weighted(bq, gq_b, bq2, 128, tN * nqh, 1e-5, st);
            if (il == 0) { bdump(bq, tN * qb, "bq_in"); bdump(gq_b, tN * nqh * 128, "gq_b"); }
            if (il == 0) bdump(bq2, tN * qb, "qnorm-b0");
            cudaMemcpyAsync(bq, bq2, (size_t)tN * qb * 4, cudaMemcpyDeviceToDevice, st);
            snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", il);
            tile_expand((const float*)dev(nm)->ptr, gk_b, tN, 8, 128, st);
            strata::kernels::native_gr_rms_norm_weighted(bk, gk_b, bk2, 128, tN * 8, 1e-5, st);
            cudaMemcpyAsync(bk, bk2, (size_t)tN * 1024 * 4, cudaMemcpyDeviceToDevice, st);
            // rope（批量）：kernel = one-thread-per-row（stride=head_dim），行序 [t][heads]。
            // pos 值只依赖 (p0,t)：chunk 开头填两版 pinned（64/96 头），层内按头数选用
            {
                cudaMemcpyAsync(d_posq, full_il ? pin_pq64 : pin_pq96,
                                (size_t)tN * nqh * 4, cudaMemcpyHostToDevice, st);
                cudaMemcpyAsync(d_posk, pin_pk, (size_t)tN * 8 * 4, cudaMemcpyHostToDevice, st);
                strata::kernels::rope_neox_apply(bq, bq, (int64_t)tN * nqh, 128, nrot, cosT, sinT, d_posq, st);
                { char pn[64]; snprintf(pn, sizeof pn, "Qcur_pos-b%d", il); bdump(bq, tN * qb, pn); }
                strata::kernels::rope_neox_apply(bk, bk, (int64_t)tN * 8, 128, nrot, cosT, sinT, d_posk, st);
                { char pn[64]; snprintf(pn, sizeof pn, "Kcur_pos-b%d", il); bdump(bk, tN * 1024, pn); }
            }
            // KV append（批量，1 launch）+ warp-flash attention（批量，1 launch，双 PASS 认证）
            cudaEventRecord(ev_at[il][0], st);
            kv_append_batch(d_kcache_, d_vcache_, bk, bv, d_p0, tN, kv_base_bytes_[il], kv_lcap_[il], st);
            // P2.4: attention v3 = mma.sync flash（TC）；STEP37_ATTN=2 回退 v2 标量版
            static const int attn_ver = getenv("STEP37_ATTN") ? atoi(getenv("STEP37_ATTN")) : 3;
            if (attn_ver >= 3)
                attn_prefill_mma(d_kcache_, d_vcache_, bq, battn, d_p0, tN, !full_il, nqh, kq_scale,
                                 kv_base_bytes_[il], kv_lcap_[il], st);
            else
                attn_prefill(d_kcache_, d_vcache_, bq, battn, d_p0, tN, !full_il, nqh, kq_scale,
                             kv_base_bytes_[il], kv_lcap_[il], st);
            cudaEventRecord(ev_at[il][1], st);
            // gate（GemmEx，x16 仍是本层 bxn 的 f16）+ 批量 sigmoid 乘
            if (il == 0) bdump(battn, tN * qb, "attn_pre0");   // gate 前裸 attention（对拍二分用）
            hgemm_f16(cbh, wproj[il].g, x16, bgate, nqh, tN, 4096);
            gate_mul_batch(bgate, battn, tN, nqh, st);
            if (il == 0) { bdump(battn, tN * qb, "attn_raw0"); bdump(bgate, tN * nqh, "bgate0"); }
            { char pn[64]; snprintf(pn, sizeof pn, "attn_gated-b%d", il); bdump(battn, tN * qb, pn); }
            // O 投影（GemmEx，激活 = gated 后的 battn）+ 残差
            f32_to_f16(battn, a16, (long)tN * qb, st);
            hgemm_f16(cbh, wproj[il].o, a16, bo, 4096, tN, qb);
            strata::kernels::add_inplace(bx, bo, (size_t)tN * 4096, st);
            // ffn norm（批量）
            snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", il);
            tile_expand((const float*)dev(nm)->ptr, gnorm_b, tN, 1, 4096, st);
            strata::kernels::native_gr_rms_norm_weighted(bx, gnorm_b, bffn_x, 4096, tN, 1e-5, st);
            { char pn[64]; snprintf(pn, sizeof pn, "ffn_norm-b%d", il); bdump(bffn_x, tN * 4096, pn); }
            if (il < 3) {
                // dense FFN（GemmEx×3）：bffn_x → f16 复用 x16
                f32_to_f16(bffn_x, x16, (long)tN * 4096, st);
                hgemm_f16(cbh, wffn[il].g, x16, bh1, 11264, tN, 4096);
                hgemm_f16(cbh, wffn[il].u, x16, bh2, 11264, tN, 4096);
                swiglu_batch(bh1, bh2, bh1, (size_t)tN * 11264, st);
                f32_to_f16(bh1, h16, (long)tN * 11264, st);
                hgemm_f16(cbh, wffn[il].d, h16, bffn_y, 4096, tN, 11264);
            } else {
                // ---- MoE：批量 router → D2H bucket → 整层 H2D → 3×Product → combine ----
                const auto t_lay0 = tick();
                snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", il);
                {   // router：cuBLAS SGEMM（nsys：手写 router_gemm 15 GFLOPS = 2.56s/50K；
                    // maya 判例 router 强制 F32 SGEMM——TC 不用，纯 F32 也比手写快 20×）
                    const float one = 1.f, zero = 0.f;
                    cublasSgemm(cbh, CUBLAS_OP_T, CUBLAS_OP_N, 288, tN, 4096, &one,
                                (const float*)dev(nm)->ptr, 4096, bffn_x, 4096, &zero, blogits, 288);
                }
                snprintf(nm, sizeof nm, "blk.%d.exp_probs_b.bias", il);
                {   // 路由输出写设备内存（kernel 直写 host-mapped 每次过 PCIe = 1.49s/50K）；
                    // 设备端派发直接消费，host 不再回读
                    strata::kernels::router_sigmoid(blogits, (const float*)dev(nm)->ptr, tN, 288, 8, 3.0f, true,
                                                    pf_eids, pf_ew, st);
                }
                if (getenv("STEP37_DBG")) {
                    char pn[80]; snprintf(pn, sizeof pn, "%s/logits-%d.bin", getenv("STEP37_DUMP_DIR"), il);
                    FILE* f = fopen(pn, "wb");
                    static float hl[4 * 288];
                    cudaMemcpy(hl, blogits, tN * 288 * 4, cudaMemcpyDeviceToHost);
                    fwrite(hl, 4, tN * 288, f); fclose(f);
                    snprintf(pn, sizeof pn, "%s/rtr-%d.bin", getenv("STEP37_DUMP_DIR"), il);
                    f = fopen(pn, "wb"); fwrite(pin_w8, 4, 8, f); fwrite(pin_ids8, 4, 8, f); fclose(f);
                }
                // ---- 设备端路由派发（去每层全流 sync：GPU 流水线不断，maya "router 从不回读"同款）
                //      hist → bounds（含 ofs 清零）→ scatter，全在 st 上；Product.max_rows 用
                //      tN 安全上界（top-8 无放回 → 每专家 ≤ tN 路由），grid 多余块读 bounds 早退 ----
                cudaMemsetAsync(d_rcnt, 0, 288 * 4, st);
                route_hist(pf_eids, d_rcnt, tN * 8, st);
                route_bounds(d_rcnt, d_bounds, d_rofs, st);
                route_scatter(pf_eids, pf_ew, d_bounds, d_rofs, d_ids_tok, d_ids_row, d_w_route, d_inv, tN * 8, st);
                if (getenv("STEP37_DUMP_ROUTE")) {   // 路由对拍：eids/ew/rlogits/cnt 逐层落盘
                    cudaStreamSynchronize(st);
                    char pn[300];
                    const char* dd = getenv("STEP37_DUMP_DIR");
                    auto dmpv = [&](const void* p, int bytes, const char* fmt2, int L2) {
                        snprintf(pn, sizeof pn, "%s/%s-b%d.bin", dd, fmt2, L2);
                        std::vector<char> b((size_t)bytes);
                        cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(b.data(), 1, b.size(), f); fclose(f); }
                    };
                    dmpv(pf_eids, tN * 8 * 4, "eids", il);
                    dmpv(pf_ew, tN * 8 * 4, "ew", il);
                    dmpv(blogits, tN * 288 * 4, "rlogits", il);
                    dmpv(d_rcnt, 288 * 4, "cnt", il);
                }
                // combine 挪到 down-MMQ 之后（2026-11-11 重大修复）：旧序 combine 在
                // down-MMQ 前，同流执行读到的是上一层的 parts = 一层滞后 + 槽位错配的
                // 错注入 MoE（对 stock logits rel 0.61、think 退化、种子彩票的全部根源；
                // 自 9b010b0 "MMQ experts green" 起就存在——green 是库自测不是模型墙）。
                // MoE 专家：MMQ grouped；权重 DMA 已在层首发出（双缓冲，与 attention 并行）
                const int lj = il - 3, par = lj % nslot;
                cudaEventRecord(ev_mo[il - 3][0], st);
                strata::prefill::mmq::quantize(bffn_x, d_ids_tok, xq_a, 23, 4096, 4096, tN * 8, st);
                strata::prefill::mmq::Product pe;
                pe.type = 23; pe.w_cols = 4096; pe.expert_bytes = 2785280; pe.n = 288;
                pe.xq = xq_a; pe.bounds = d_bounds; pe.ids = d_iota;
                pe.total_rows = tN * 8; pe.max_rows = tN; pe.ld_dst = 1280;   // tN 上界：多余块读 bounds 早退
                cudaStreamWaitEvent(st, ev_stage_gu[par], 0);      // 等 gate/up 到货（层首 DMA）
                pe.w = d_stage2[par]; pe.w_rows = 1280; pe.dst = gu; mmq.run(pe, st);
                pe.w = d_stage2[par] + layer_mat; pe.dst = uu; mmq.run(pe, st);
                if (getenv("STEP37_DUMP_ROUTE") && il >= 3 && il <= 6) {   // MoE 解剖：staging 指纹 + mmq 输出
                    cudaStreamSynchronize(st);
                    char pn[300];
                    const char* dd = getenv("STEP37_DUMP_DIR");
                    auto dmpv = [&](const void* p, size_t bytes, size_t maxb, const char* nm2, int L2) {
                        snprintf(pn, sizeof pn, "%s/%s-b%d.bin", dd, nm2, L2);
                        const size_t nb2 = bytes < maxb ? bytes : maxb;
                        std::vector<char> b(nb2);
                        cudaMemcpy(b.data(), p, nb2, cudaMemcpyDeviceToHost);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(b.data(), 1, b.size(), f); fclose(f); }
                    };
                    if (getenv("STEP37_DUMP_FULLSTAGE")) dmpv(d_stage2[par], 3 * (size_t)layer_mat, (size_t)-1, "stagefull", il);  // 全量 2.4GB，独立 gate
                    dmpv(gu, (size_t)tN * 8 * 1280 * 4, (size_t)-1, "gu", il);
                    dmpv(uu, (size_t)tN * 8 * 1280 * 4, (size_t)-1, "uu", il);
                }
                swiglu_batch(gu, uu, bh, (size_t)tN * 8 * 1280, st);
                if (getenv("STEP37_DUMP_ROUTE") && il >= 3 && il <= 6) {   // down 路径解剖：dn 指纹 + bh
                    cudaStreamSynchronize(st);
                    char pn[300];
                    auto dmpv3 = [&](const void* p, size_t bytes, const char* nm2, int L2) {
                        snprintf(pn, sizeof pn, "%s/%s-b%d.bin", getenv("STEP37_DUMP_DIR"), nm2, L2);
                        std::vector<char> b(bytes);
                        cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(b.data(), 1, b.size(), f); fclose(f); }
                    };
                    if (getenv("STEP37_DUMP_FULLSTAGE")) dmpv3((const char*)d_stage2[par] + 2 * (size_t)layer_mat, (size_t)layer_mat, "dnfull", il);  // dn 全量 765MB
                    dmpv3(bh, (size_t)tN * 8 * 1280 * 4, "bh", il);
                }
                strata::prefill::mmq::quantize(bh, nullptr, xq_h, 23, 1280, 1280, tN * 8, st);
                if (getenv("STEP37_QPROBE2") && il >= 3 && il <= 6) {
                    // 管道内双量探针：同一 bh 再量一次到独立缓冲 + bh 前后两份。
                    // q2≠xqh（跑内）= kernel 级非确定；bh 前≠后 = 有人改写 bh；跨跑比 xqh。
                    cudaStreamSynchronize(st);
                    static void* q2 = nullptr;
                    static std::vector<char> bh_pre, bh_post, xh_a, xh_b;
                    const size_t qb = strata::prefill::mmq::q8_bytes(tN * 8, 1280);
                    if (!q2) { cudaMalloc(&q2, strata::prefill::mmq::q8_bytes((int64_t)T * 8, 1280));
                               bh_pre.resize((size_t)tN * 8 * 1280 * 4); xh_a.resize(qb); xh_b.resize(qb); }
                    bh_post.resize((size_t)tN * 8 * 1280 * 4);
                    cudaMemcpy(bh_pre.data(), bh, bh_pre.size(), cudaMemcpyDeviceToHost);
                    strata::prefill::mmq::quantize(bh, nullptr, q2, 23, 1280, 1280, tN * 8, st);
                    cudaStreamSynchronize(st);
                    cudaMemcpy(bh_post.data(), bh, bh_post.size(), cudaMemcpyDeviceToHost);
                    cudaMemcpy(xh_a.data(), xq_h, qb, cudaMemcpyDeviceToHost);
                    cudaMemcpy(xh_b.data(), q2, qb, cudaMemcpyDeviceToHost);
                    char pn[300]; const char* dd = getenv("STEP37_DUMP_DIR");
                    snprintf(pn, sizeof pn, "%s/bhpre-b%d.bin", dd, il);
                    FILE* f = fopen(pn, "wb"); fwrite(bh_pre.data(), 1, bh_pre.size(), f); fclose(f);
                    snprintf(pn, sizeof pn, "%s/bhpost-b%d.bin", dd, il);
                    f = fopen(pn, "wb"); fwrite(bh_post.data(), 1, bh_post.size(), f); fclose(f);
                    snprintf(pn, sizeof pn, "%s/xh2-b%d.bin", dd, il);
                    f = fopen(pn, "wb"); fwrite(xh_b.data(), 1, qb, f); fclose(f);
                    int dq = 0; for (size_t k = 0; k < qb; ++k) dq += xh_a[k] != xh_b[k];
                    int dbh = 0; for (size_t k = 0; k < bh_post.size(); ++k) dbh += bh_pre[k] != bh_post[k];
                    fprintf(stderr, "QPROBE2 L%d: xqh vs q2 diffbytes=%d/%zu  bh pre-vs-post diffbytes=%d/%zu\n",
                            il, dq, qb, dbh, bh_post.size());
                }
                if (getenv("STEP37_DUMP_ROUTE") && il >= 3 && il <= 6) {   // xq_h：down 路径的量化激活
                    cudaStreamSynchronize(st);
                    char pn[300];
                    snprintf(pn, sizeof pn, "%s/xqh-b%d.bin", getenv("STEP37_DUMP_DIR"), il);
                    const size_t qbytes = strata::prefill::mmq::q8_bytes(tN * 8, 1280);
                    std::vector<char> b(qbytes);
                    cudaMemcpy(b.data(), xq_h, qbytes, cudaMemcpyDeviceToHost);
                    FILE* f = fopen(pn, "wb"); fwrite(b.data(), 1, b.size(), f); fclose(f);
                }
                cudaStreamWaitEvent(st, ev_stage_dn[par], 0);      // 等 down 到货
                pe.w = d_stage2[par] + 2 * layer_mat; pe.w_rows = 4096; pe.w_cols = 1280;
                pe.xq = xq_h; pe.dst = parts; pe.ld_dst = 4096; mmq.run(pe, st);
                // combine（per-token gather，rank 序求和——零原子；旧 atomicAdd 版求和序
                // 是调度彩票：ULP 噪声 × 下一层 q8 舍入悬崖 → 行级翻转 → 路由漂移）
                cudaMemsetAsync(bffn_y, 0, (size_t)tN * 4096 * 4, st);
                moe_combine_routed(parts, d_inv, d_w_route, bffn_y, tN, 4096, st);
                if (getenv("STEP37_DUMP_ROUTE") && il >= 3 && il <= 6) {   // combine 输出（共享专家加之前）
                    cudaStreamSynchronize(st);
                    char pn[300];
                    auto dmpv2 = [&](const void* p, size_t bytes, const char* nm2, int L2) {
                        snprintf(pn, sizeof pn, "%s/%s-b%d.bin", getenv("STEP37_DUMP_DIR"), nm2, L2);
                        std::vector<char> b(bytes);
                        cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(b.data(), 1, b.size(), f); fclose(f); }
                    };
                    dmpv2(d_ids_row, (size_t)tN * 8 * 4, "idsrow", il);
                    dmpv2(parts, (size_t)tN * 8 * 4096 * 4, "parts", il);
                    dmpv2(bffn_y, (size_t)tN * 4096 * 4, "bffny", il);
                }
                if (getenv("STEP37_MMQPROBE") && il >= 3 && il <= 5) {
                    // 管道内双 mmq：同输入立即重跑 → 独立 dst。跑内等 = 输入可见性一致；
                    // 跨跑（canon）差 = launch 时刻真实输入与我 dump 的不同
                    static float* parts2 = nullptr;
                    if (!parts2) cudaMalloc(&parts2, (size_t)T * 8 * 4096 * 4);
                    pe.dst = parts2;
                    mmq.run(pe, st);
                    cudaStreamSynchronize(st);
                    char pn[300]; const char* dd = getenv("STEP37_DUMP_DIR");
                    auto dmp5 = [&](const void* pp, size_t bytes, const char* nm2, int L2) {
                        snprintf(pn, sizeof pn, "%s/%s-b%d.bin", dd, nm2, L2);
                        std::vector<char> b(bytes);
                        cudaMemcpy(b.data(), pp, bytes, cudaMemcpyDeviceToHost);
                        if (FILE* f = fopen(pn, "wb")) { fwrite(b.data(), 1, b.size(), f); fclose(f); }
                    };
                    dmp5(parts, (size_t)tN * 8 * 4096 * 4, "mp1", il);
                    dmp5(parts2, (size_t)tN * 8 * 4096 * 4, "mp2", il);
                    dmp5(xq_h, strata::prefill::mmq::q8_bytes(tN * 8, 1280), "mxq", il);
                }
                cudaEventRecord(ev_gpu_free[par], st);        // 同缓冲下下层 DMA 的保护
                cudaEventRecord(ev_mo[il - 3][1], st);
            }
            // shared expert（MoE 层，GemmEx×3）+ 残差
            if (il >= 3) {
                f32_to_f16(bffn_x, x16, (long)tN * 4096, st);   // router 用 F32 原件，shexp 用 f16
                hgemm_f16(cbh, wffn[il].g, x16, gu, 1280, tN, 4096);
                hgemm_f16(cbh, wffn[il].u, x16, uu, 1280, tN, 4096);
                swiglu_batch(gu, uu, gu, (size_t)tN * 1280, st);
                f32_to_f16(gu, h16, (long)tN * 1280, st);
                hgemm_f16(cbh, wffn[il].d, h16, bsh, 4096, tN, 1280);
                strata::kernels::add_inplace(bffn_y, bsh, (size_t)tN * 4096, st);
            }
            { char pn[64]; snprintf(pn, sizeof pn, "ffn_out-b%d", il); bdump(bffn_y, tN * 4096, pn); }
            strata::kernels::add_inplace(bx, bffn_y, (size_t)tN * 4096, st);
            { char pn[64]; snprintf(pn, sizeof pn, "l_out-b%d", il); bdump(bx, tN * 4096, pn); }
        }
        // chunk 分项账单（GPU events）；尾 token 快照
        cudaMemcpyAsync(bx_last, bx + (size_t)(tN - 1) * 4096, 4096 * 4, cudaMemcpyDeviceToDevice, st);
        cudaEventRecord(ev_ck, st);
        cudaEventSynchronize(ev_ck);
        float aa = 0.f, mm = 0.f;
        for (int i = 0; i < 45; ++i) { float d; cudaEventElapsedTime(&d, ev_at[i][0], ev_at[i][1]); aa += d; }
        for (int i = 0; i < 42; ++i) { float d; cudaEventElapsedTime(&d, ev_mo[i][0], ev_mo[i][1]); mm += d; }
        const double csec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all0).count();
        fprintf(stderr, "CHUNK p0=%ld tN=%d wall=%.1fs | attn=%.0fms moe(staging+MMQ)=%.0fms other=%.0fms"
                        " [host: wait_pf=%.1fs wait_sync=%.1fs err=%s]\n",
                (long)p0, tN, csec, aa, mm, (csec * 1000 - aa - mm), host_wait_pf, host_wait_sync,
                cudaGetErrorString(cudaGetLastError()));
    }
    // 通知后台收尾（最后一个槽可能无人消费）并 join
    { std::lock_guard<std::mutex> lk(pf_mu); }
    pf_cv.notify_all();
    {
        // 终局放行：把两个槽标记为消费完毕，让预拷线程的 wait 通过后自然退出
        std::lock_guard<std::mutex> lk(pf_mu);
        for (auto& s : pslot) if (s.state == 2) s.state = 3;   // 3=abandoned
    }
    pf_cv.notify_all();
    prefetcher.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all0).count();
    pp_tps = (double)prompt_ids.size() / secs;
    // 尾 token logits（对拍锚）+ decode 接力。注意 bx 只有 T 行：尾行须用最后 chunk 的快照
    // （修雷：多 chunk 时读 bx[(size-1)*4096] 是越界读——12288 落邻分配只错值、45K+ 落未映射页 illegal）
    {
        void* head_scratch = nullptr; cudaMalloc(&head_scratch, 262144);
        float* xn_last = alloc_f(4096);
        strata::kernels::native_gr_rms_norm_weighted(bx_last,
                                                     (const float*)dev("output_norm.weight")->ptr, xn_last, 4096, 1, 1e-5, st);
        // 尾 logits 落 ws_.logits：generate(resume_prompt) 接力时 device argmax 直接消费
        strata::kernels::native_q6_k_f32(dev("output.weight")->ptr, xn_last, head_scratch, ws_.logits, 4096, 128896, 1, st);
        std::vector<float> hl(128896);
        cudaMemcpy(hl.data(), ws_.logits, 128896 * 4, cudaMemcpyDeviceToHost);
        int best = 0; for (int v = 1; v < 128896; ++v) if (hl[v] > hl[best]) best = v;
        fprintf(stderr, "PREFILL-ARGMAX: %d\n", best);
        if (const char* dd = getenv("STEP37_DUMP_DIR")) {   // 对拍锚：尾 logits 全量落盘
            char pn[300]; snprintf(pn, sizeof pn, "%s/result_output-b0.bin", dd);
            FILE* f = fopen(pn, "wb"); fwrite(hl.data(), 4, hl.size(), f); fclose(f);
        }
        cudaFree(xn_last); cudaFree(head_scratch);
    }
    // ---- prefill 临时缓冲清场（给尾层重载腾 24.65GB）----
    cudaStreamDestroy(stc);
    for (int i = 0; i < 2; ++i) { cudaEventDestroy(ev_stage_gu[i]); cudaEventDestroy(ev_stage_dn[i]); }
    for (int i = 0; i < 2; ++i) { cudaEventDestroy(ev_pin_free[i]); cudaEventDestroy(ev_gpu_free[i]); }
    cudaFreeHost(pslot[0].gu); cudaFreeHost(pslot[0].dn); cudaFreeHost(pslot[1].gu); cudaFreeHost(pslot[1].dn);
    cudaFree(d_w_route);
    cudaFreeHost(pin_ids_tok); cudaFreeHost(pin_ids_row); cudaFreeHost(pin_w_route); cudaFreeHost(pin_bounds);
    cudaFreeHost(pin_pq64); cudaFreeHost(pin_pq96); cudaFreeHost(pin_pk);
    cudaFree(d_stage2[0]); cudaFree(d_stage2[1]);   // decode 无 staging，还给池
    cudaFree(w16_arena);
    cudaFree(x16); cudaFree(a16); cudaFree(h16);
    cublasDestroy(cbh);
    // rope 四表已改跨请求 static（rope_cap 门控），清场不再释放——曾致第二轮 op=1
    // memcpy 写悬垂指针 SEGV（2026-11-12 深夜，信箱复现 + gdb 栈钉死）
    cudaFree(bx); cudaFree(bxn); cudaFree(bffn_x); cudaFree(bffn_y); cudaFree(bo);
    cudaFree(bq); cudaFree(bq2); cudaFree(bk); cudaFree(bk2); cudaFree(bv); cudaFree(battn);
    cudaFree(bgate); cudaFree(blogits); cudaFree(bh1); cudaFree(bh2);
    cudaFree(gu); cudaFree(uu); cudaFree(bh); cudaFree(parts); cudaFree(bsh);
    cudaFree(bx_last);
    cudaFree(gq_b); cudaFree(gk_b); cudaFree(xq_a); cudaFree(xq_h); cudaFree(xq_p);
    cudaFree(d_bounds); cudaFree(d_ids_tok); cudaFree(d_ids_row);
    cudaFree(d_toks); cudaFree(d_p0); cudaFree(d_posarr); cudaFree(d_unit); cudaFree(d_iota);
    cudaFree(d_posq); cudaFree(d_posk);
    cudaFreeHost(pin_ids8); cudaFreeHost(pin_w8);
    // 重载 decode 尾层驻留（tail 自适应 v0：50K KV 下放不满 11 层——逐层退让，装几层是几层；
    // 修雷：cudaMalloc OOM 返回 NULL 未检查 → cuMemcpyHtoD 内部 SIGSEGV，49152/50000 复崩点）
    if (saved_tail > 0) {
        const uint64_t mat = layer_mat;
        int tail = saved_tail;
        while (tail > 0) {
            cudaGetLastError();   // 清残留错误态
            if (cudaMalloc(&d_gpu_exp_, 3 * mat * (uint64_t)tail) == cudaSuccess) break;
            --tail;
        }
        cudaGetLastError();
        gpu_tail_ = tail;
        if (tail < saved_tail)
            fprintf(stderr, "GPU_TAIL: reload shrunk %d -> %d layers (VRAM after prefill)\n", saved_tail, tail);
        for (int l = 0; l < gpu_tail_; ++l) {
            const int li = 42 - gpu_tail_ + l;
            uint8_t* dst = d_gpu_exp_ + 3 * mat * (uint64_t)l;
            cudaMemcpy(dst, experts_[li][0].gate, mat, cudaMemcpyHostToDevice);
            cudaMemcpy(dst + mat, experts_[li][0].up, mat, cudaMemcpyHostToDevice);
            cudaMemcpy(dst + 2 * mat, experts_[li][0].down, mat, cudaMemcpyHostToDevice);
        }
    }
    return true;
}

} // namespace step37