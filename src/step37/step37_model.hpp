// step37_model.hpp — maya-step37 T1: 装载编排与前向接口
//
// 全景比喻（BUILD-LOG.md §0）：中央车站的"调度室"。
//   收货登记处（step37_gguf.hpp 的 Manifest）交来货单，
//   这里把常驻货搬上工具架（VRAM arena），给专家货建主仓库指针表（mmap 页缓存），
//   并提供 decode（单件加急订单）与 prefill（整皮货批量加工）两条生产线。
//
// M1 边界（STEP37-PLAN.md）：
//   - 专家全驻 RAM（页缓存 in-place，不拷贝）——decode 走 CPU lane，prefill 走 H2D 流水
//   - 常驻集（~3.65 GiB）单一大 arena H2D
//   - 无槽池（M2）、无投机解码（M3 观察项）、无 GUI（后置）
#pragma once

#include "step37_gguf.hpp"
#include "step37_geo.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace step37 {

struct ModelOptions {
    std::string model_dir;
    int device = 0;
    uint32_t ubatch = 4096;   // prefill chunk；搬运量 = 层专家字节 × ceil(N/ubatch)
    uint32_t n_decode = 32;   // 默认 greedy 步数
};

// 装载账本（打印/测试用）。
struct LoadReport {
    size_t shards = 0;
    size_t tensors = 0;
    uint64_t expert_bytes = 0;    // RAM（mmap 页缓存）
    uint64_t resident_bytes = 0;  // VRAM arena
    uint64_t vram_arena_bytes = 0;
    double load_seconds = 0.0;
};

// 每个 (layer, expert) 的三件套指针——mmap 页缓存地址 + 字节偏移。
// GGML 布局: gate/up 张量 [n_embd, n_ff, n_exp]，dim0 最快 → 每专家是连续块。
// 常驻张量的 device 侧描述
struct DevTensor {
    const void* ptr;
    uint32_t type;
};

struct ExpertSlice {
    const uint8_t* gate; // IQ4_XS 数据起始（专家 e 在 gate_exps 张量内的块）
    const uint8_t* up;
    const uint8_t* down;
};

// mmap 只读映射（最小实现，析构在 cu 内）
struct MappedFile {
    uint8_t* base = nullptr;
    uint64_t bytes = 0;
    int fd = -1;
    bool open(const char* path);
};

class Model {
public:
    Model() = default;
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const ModelOptions& opts, std::string& err);
    const LoadReport& report() const { return report_; }

    // T2: decode 前向——prompt_ids 暖机（逐 token 填 KV），随后 greedy n_decode 步。
    // resume_prompt > 0：prompt 已由 prefill_run 前向（KV/尾 logits 就位），跳过暖机直接 decode。
    // out_tokens = 生成序列（不含 prompt）；decode_tps = 生成段 tokens/秒。
    bool generate(const std::vector<int32_t>& prompt_ids, int n_decode,
                  std::vector<int32_t>& out_tokens, double& decode_tps,
                  std::string& err, int resume_prompt = 0, int base_pos = 0);

    // T3: prefill 路径（chunk 批量）——M1 首版：GEMV 路线 + 层级 H2D
    // start_pos > 0：追加模式——prompt_ids 是增量 token，KV 前 start_pos 行已在
    // （会话前缀复用），rope/KV 位置 = start_pos + 序号。
    bool prefill_run(const std::vector<int32_t>& prompt_ids, double& pp_tps,
                     std::string& err, int start_pos = 0);

    // 采样参数（daemon 模式逐请求下发；默认值在首次 generate 时从 env 初始化）
    struct Sampling {
        float temp = -1.f, top_p = 1.f, min_p = 0.f, rep_pen = 1.f;
        int top_k = 0, rep_win = 64, eos = -1;
        unsigned seed = 42;
    };
    void set_sampling(const Sampling& s) { samp_ = s; samp_set_ = true; }
    bool kv_wiped() const { return kv_wiped_; }   // prefill_run 重分配了 KV（前缀缓存作废）

  private:
    void unload();
    // daemon 会话状态
    Sampling samp_{};
    bool samp_set_ = false;
    int kv_wiped_ = 0;

    Manifest manifest_;
    ModelOptions opts_;
    LoadReport report_;

    // 常驻张量的 device 地址表（name → ptr/type）
    std::vector<std::pair<std::string, DevTensor>> dev_map_;
    const DevTensor* dev(const char* name) const;

    // 常驻集 arena（device 端单一大缓冲 + host staging）
    uint8_t* d_arena_ = nullptr;
    uint64_t arena_bytes_ = 0;
    std::vector<uint64_t> d_offsets_; // 与 manifest_.tensors 平行（常驻张量的 device 偏移）

    // 专家切片表: [layer][expert]
    std::vector<std::vector<ExpertSlice>> experts_; // 42 MoE layer 0..41 (blk 3..44)
    std::vector<MappedFile> maps_;

    // KV cache（F16, [pos][hkv][hd]，45 层；F32 是 stock 参照口径的 2 倍 VRAM）
    void* d_kcache_ = nullptr; // q8_0：[45 层][cap_l][1088B]（SWA 层环形）
    void* d_vcache_ = nullptr;
    uint32_t kv_cap_ = 0;      // 全注意力层深度（=ctx）
    int swa_cap_ = 4096;       // SWA 层环形深度（STEP37_SWA_CAP）
    long kv_base_bytes_[45] = {};  // 逐层字节基址（q8 行 1088B/位置）
    int kv_lcap_[45] = {};         // 逐层环形 cap（全层=kv_cap_，SWA=swa_cap_）
    void recompute_kv_layout() {
        long off = 0;
        for (int il = 0; il < 45; ++il) {
            kv_lcap_[il] = (il % 4 == 0) ? (int)kv_cap_ : swa_cap_;
            kv_base_bytes_[il] = off * 1088L;
            off += kv_lcap_[il];
        }
    }

    // 工作缓冲 arena（decode/prefill 共用）
    uint8_t* d_ws_ = nullptr;
    uint64_t ws_bytes_ = 0;
    // 便捷切片（偏移表，见 cu 内 kWsLayout）
    struct WorkArea {
        float *x, *xn, *q, *k, *v, *attn, *gate, *o, *ffn_x, *h1, *h2, *ffn_y;
        float *logits, *scores, *router_logits, *exp_ids_f, *exp_w, *shared_y, *df_scratch, *gu_hi;
        float* parts; // [8][4096] down 输出堆叠
        uint16_t* f16_scratch;
        void* gemv_scratch;
        float* e_staging; // 命中专家 staging（24 切片）
        int* e_ids;
        float* cos_full;  // rope 表：全注意力层 (n_rot=64, theta 5M)
        float* sin_full;
        float* cos_swa;   // SWA 层 (n_rot=128, theta 10k)
        float* sin_swa;
    };
    WorkArea ws_{};


    bool loaded_ = false;
    // M1.5-d: tail-N layers with ALL experts VRAM-resident (stock's -ot tail config)
    uint8_t* d_gpu_exp_ = nullptr; // [gpu_tail_][3][288*2785280]
    int gpu_tail_ = 0;
    // per-head q/k norm gamma, tiled to kernel shape [rows*n]
    float* d_qn_gamma_ = nullptr; // [45][96*128]
    float* d_kn_gamma_ = nullptr; // [45][8*128]
};

} // namespace step37
