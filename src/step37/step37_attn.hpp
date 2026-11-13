// step37_attn.cu — T2: decode attention（CUDA graph 安全版：pos 从设备内存读）
// 数学契约（llama.cpp step35.cpp 对拍语义）：
//   scale = 1/sqrt(128)；GQA 映射 q_head h → kv_head h/(nqh/8)；
//   因果 = query 在位置 p 时 attend [lo..p]（lo = SWA-512 窗口下界，swa=false 时 0）；
//   F32 softmax（减 max）；attn_gate = sigmoid(g_proj(x)) 逐头广播乘（在 attention 之后）。
//   pos 经设备指针传入（graph 回放不冻结 token 位置）；swa 是层常量可安全入图。
#pragma once

#include <cstdint>

namespace step37 {

// scores[h][t] 行宽 = score_stride（kv_cap；arena 定容 8192，超容时调用方换动态缓冲——
// 修雷：pos ≥ 8192 时旧 8192 硬编码越界读写，50K ctx decode 全 NaN）
inline constexpr int kScoresMaxPos = 8192;

// A: scores[h][t] = dot(q[h], k_cache[t][h/(nqh/8)]) * scale, t in [lo, pos]
void attn_decode_scores(const void* k_cache, const float* q,
                        float* scores, const int* pos_dev, bool swa, int n_q_heads, int n_kv_heads,
                        int head_dim, float scale, int score_stride, int win_tiles, void* stream);

// B: per-head softmax（减 max，F32），in-place on scores, range [lo, pos]
void attn_decode_softmax(float* scores, const int* pos_dev, bool swa, int n_q_heads,
                         int score_stride, void* stream);

// C: out[h][d] = Σ_t p[h][t] * V[t][h/(nqh/8)][d], t in [lo, pos]
void attn_decode_weighted_v(const void* v_cache, const float* probs,
                            float* out, const int* pos_dev, bool swa, int n_q_heads, int n_kv_heads,
                            int head_dim, int score_stride, void* stream);

} // namespace step37

// ===== P1: 批量 prefill attention =====
// flash 风格 online-softmax；因果 + SWA 窗口（swa=true 时 [p0+t-511, p0+t]）。
// q: [T, nqh, 128] F32（rope 后）；out: [T, nqh*128] F32；KV cache F16。
// p0_dev: 设备端 chunk 起始全局位置（int）。
void attn_prefill(const void* k_cache, const void* v_cache,
                  const float* q, float* out, const int* p0_dev, int T,
                  bool swa, int n_q_heads, float scale,
                  long layer_row, int kv_cap, void* stream);

// P2.3: decode flash（单 query，pos 分段 + 二段规约）。win = 本次要看的窗口长度（host 已知，
// swa 层 min(pos+1,512)，全注意力层 pos+1）；scratch ≥ nqh × ceil(win/2048) × 130 floats。
void attn_decode_flash(const void* k_cache, const void* v_cache, const float* q, float* out,
                       const int* pos_dev, bool swa, int n_q_heads, int n_kv_heads, float scale,
                       long layer_row, int win, float* scratch, void* stream, int kv_cap);

// P2.4: prefill attention v3 — mma.sync flash（TC）。签名同 attn_prefill。
void attn_prefill_mma(const void* k_cache, const void* v_cache,
                      const float* q, float* out, const int* p0_dev, int T,
                      bool swa, int n_q_heads, float scale,
                      long layer_row, int kv_cap, void* stream);
