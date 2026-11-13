// step37_attn.cu — decode attention kernels (graph-safe: pos read from device memory)
#include "step37_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace step37 {
namespace {

// A: 每 (q_head, position) 一个线程，点积 128 维。pos 从设备内存读（CUDA graph 回放安全），
//    SWA 下界 lo 在核内从 pos 推出（swa 标志是层常量，可安全冻结进图）。
__global__ void attn_scores_kernel(const void* k_cache, const float* q,
                                   float* d_scores, const int* ppos, int swa, int n_qh, int n_kvh,
                                   int hd, float scale, int score_stride) {
    const int pos = *ppos;
    const int lo = swa ? (pos > 511 ? pos - 511 : 0) : 0;
    const int h = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (t > pos || t < lo) return;
    const int hkv = h / (n_qh / n_kvh);
    const float* qh = q + (size_t)h * hd;
    const __half* krow = (const __half*)k_cache + (size_t)t * n_kvh * hd + (size_t)hkv * hd;
    float s = 0.f;
    for (int d = 0; d < hd; ++d) s += __half2float(krow[d]) * qh[d];
    d_scores[(size_t)h * score_stride + t] = s * scale;
}

// B: per-head softmax（减 max，F32），in-place，range [lo, pos]。
__global__ void attn_softmax_kernel(float* d_scores, const int* ppos, int swa, int stride) {
    const int pos = *ppos;
    const int lo = swa ? (pos > 511 ? pos - 511 : 0) : 0;
    const int h = blockIdx.x;
    float* s = d_scores + (size_t)h * stride;
    // pass 1: max
    float m = -INFINITY;
    for (int t = lo + threadIdx.x; t <= pos; t += blockDim.x) {
        m = fmaxf(m, s[t]);
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < blockDim.x / 32 && w < 32; ++w) m = fmaxf(m, red[w]);
        red[0] = m;
    }
    __syncthreads();
    m = red[0];
    // pass 2: exp & sum
    float sum = 0.f;
    for (int t = lo + threadIdx.x; t <= pos; t += blockDim.x) {
        const float e = __expf(s[t] - m);
        s[t] = e;
        sum += e;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t2 = 0.f;
        for (int w = 0; w < blockDim.x / 32 && w < 32; ++w) t2 += red[w];
        red[0] = t2;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];
    for (int t = lo + threadIdx.x; t <= pos; t += blockDim.x) s[t] *= inv;
}

// C: 每 (head, dim) 一个线程，串行遍历 [lo, pos] 做 V 加权和。
__global__ void attn_weighted_v_kernel(const void* v_cache, const float* probs,
                                       float* out, const int* ppos, int swa, int n_qh, int n_kvh,
                                       int hd, int prob_stride) {
    const int pos = *ppos;
    const int lo = swa ? (pos > 511 ? pos - 511 : 0) : 0;
    const int h = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= hd) return;
    const int hkv = h / (n_qh / n_kvh);
    float acc = 0.f;
    const float* p = probs + (size_t)h * prob_stride;
    const __half* vcol = (const __half*)v_cache + (size_t)hkv * hd + d; // stride = n_kvh*hd per position
    for (int t = lo; t <= pos; ++t) {
        acc += p[t] * __half2float(vcol[(size_t)t * n_kvh * hd]);
    }
    out[(size_t)h * hd + d] = acc;
}

} // namespace

// ===== wrappers =====

void attn_decode_scores(const void* k_cache, const float* q, float* scores,
                        const int* pos_dev, bool swa, int n_q_heads, int n_kv_heads, int head_dim,
                        float scale, int score_stride, int win_tiles, void* stream) {
    // win_tiles: host 按 SWA 窗口/当前 pos 铺（grid 冻结是 graph 时代设计；graph 已弃，
    // 50K ctx 下按 kv_cap 铺会让 SWA 层白启 ~200 空 tile ≈ 66ms/token）
    const int tiles = win_tiles;
    dim3 grid(n_q_heads, tiles);
    attn_scores_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        k_cache, q, scores, pos_dev, swa ? 1 : 0, n_q_heads, n_kv_heads, head_dim, scale, score_stride);
}

void attn_decode_softmax(float* scores, const int* pos_dev, bool swa, int n_q_heads,
                         int score_stride, void* stream) {
    attn_softmax_kernel<<<n_q_heads, 256, 0, (cudaStream_t)stream>>>(scores, pos_dev, swa ? 1 : 0, score_stride);
}

void attn_decode_weighted_v(const void* v_cache, const float* probs, float* out,
                            const int* pos_dev, bool swa, int n_q_heads, int n_kv_heads, int head_dim,
                            int score_stride, void* stream) {
    dim3 grid(n_q_heads, (head_dim + 127) / 128);
    attn_weighted_v_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(
        v_cache, probs, out, pos_dev, swa ? 1 : 0, n_q_heads, n_kv_heads, head_dim, score_stride);
}

} // namespace step37

// ===== P1: 批量 prefill attention v2（4-warp 分段 online softmax，单遍）
// 每 block 128 线程 = 4 warps 处理一个 (t, h)：窗口连续 4 分段，每 warp 独立 online
// (m, l, acc)，终态 shared 规约。dot = lane 持 d 片 4 dims + butterfly 树归约（全员得 s，
// 无 divergent shfl）。v1（warp 串行全窗口 + 两遍 dot）在长上下文下是 GPU 95% 大头。
namespace {

__global__ void __launch_bounds__(128) attn_prefill_kernel(const uint8_t* kcache, const uint8_t* vcache,
                                    const float* q, float* out,
                                    const int* p0dev, int T, int swa, int n_qh,
                                    float scale, long layer_row, int kv_cap) {
    const int bh = blockIdx.x;
    const int nht = n_qh * T;
    if (bh >= nht) return;
    const int h = bh % n_qh;
    const int t = bh / n_qh;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p0 = *p0dev;
    const int pos_t = p0 + t;
    const int lo = swa ? max(0, pos_t - 511) : 0;
    const int hkv = h / (n_qh / 8);
    const float* qh = q + ((size_t)t * n_qh + h) * 128;
    const int span = pos_t - lo + 1;
    // 窗口连续 4 分段（warp 0 拿最长尾差，保证非空）
    const int seg = (span + 3) >> 2;
    const int p_lo = lo + w * seg;
    const int p_hi = min(p_lo + seg - 1, pos_t);
    const int cmask = kv_cap - 1;   // cap 恒 2 的幂（262144/4096）→ 环形取模 = 位与
    // q 片（lane 持 d ∈ [lane*4, lane*4+4)）
    const float q0 = qh[lane * 4 + 0], q1 = qh[lane * 4 + 1];
    const float q2 = qh[lane * 4 + 2], q3 = qh[lane * 4 + 3];
    float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int p = p_lo; p <= p_hi; ++p) {
        // q8_0 KV：行 = layer_row + (p % kv_cap)*1088；值 v = hkv*128 + lane*4 + j
        // → 块 v>>5（每块 34B = 32 qs + f16 d），4 值同块（v0&31 ≤ 28）
        const uint8_t* krow = kcache + layer_row + (size_t)(p & cmask) * 1088 + hkv * 136;
        const uint8_t* vrow = vcache + layer_row + (size_t)(p & cmask) * 1088 + hkv * 136;
        const int d0 = lane * 4, blk = d0 >> 5, off = d0 & 31;
        const float kd = __half2float(*(const __half*)(krow + blk * 34 + 32));
        float kf0 = (int8_t)krow[blk * 34 + off + 0] * kd;
        float kf1 = (int8_t)krow[blk * 34 + off + 1] * kd;
        float kf2 = (int8_t)krow[blk * 34 + off + 2] * kd;
        float kf3 = (int8_t)krow[blk * 34 + off + 3] * kd;
        float part = q0 * kf0 + q1 * kf1 + q2 * kf2 + q3 * kf3;
#pragma unroll
        for (int o = 16; o; o >>= 1) part += __shfl_xor_sync(~0u, part, o);
        const float s = part * scale;
        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);   // 首迭代 m=-inf → alpha=0
        const float e = __expf(s - m_new);
        const float vd = __half2float(*(const __half*)(vrow + blk * 34 + 32));
        l = l * alpha + e;
        a0 = a0 * alpha + e * (int8_t)vrow[blk * 34 + off + 0] * vd;
        a1 = a1 * alpha + e * (int8_t)vrow[blk * 34 + off + 1] * vd;
        a2 = a2 * alpha + e * (int8_t)vrow[blk * 34 + off + 2] * vd;
        a3 = a3 * alpha + e * (int8_t)vrow[blk * 34 + off + 3] * vd;
        m = m_new;
    }
    // 4-warp 终态规约：warp i 的 (m_i, l_i, acc_i) 合并 out = Σ acc_i·e^{m_i-M} / Σ l_i·e^{m_i-M}
    __shared__ float sm[4], sl[4], sacc[4][128];
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    sacc[w][lane * 4 + 0] = a0; sacc[w][lane * 4 + 1] = a1;
    sacc[w][lane * 4 + 2] = a2; sacc[w][lane * 4 + 3] = a3;
    __syncthreads();
    if (w == 0) {
        const float M = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
        float wg[4], L = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { wg[i] = __expf(sm[i] - M); L += sl[i] * wg[i]; }
        const float inv = 1.f / L;
        float* orow = out + ((size_t)t * n_qh + h) * 128;
        orow[lane * 4 + 0] = (sacc[0][lane*4+0]*wg[0] + sacc[1][lane*4+0]*wg[1]
                            + sacc[2][lane*4+0]*wg[2] + sacc[3][lane*4+0]*wg[3]) * inv;
        orow[lane * 4 + 1] = (sacc[0][lane*4+1]*wg[0] + sacc[1][lane*4+1]*wg[1]
                            + sacc[2][lane*4+1]*wg[2] + sacc[3][lane*4+1]*wg[3]) * inv;
        orow[lane * 4 + 2] = (sacc[0][lane*4+2]*wg[0] + sacc[1][lane*4+2]*wg[1]
                            + sacc[2][lane*4+2]*wg[2] + sacc[3][lane*4+2]*wg[3]) * inv;
        orow[lane * 4 + 3] = (sacc[0][lane*4+3]*wg[0] + sacc[1][lane*4+3]*wg[1]
                            + sacc[2][lane*4+3]*wg[2] + sacc[3][lane*4+3]*wg[3]) * inv;
    }
}

} // namespace

void attn_prefill(const void* k_cache, const void* v_cache,
                  const float* q, float* out, const int* p0_dev, int T,
                  bool swa, int n_q_heads, float scale,
                  long layer_row, int kv_cap, void* stream) {
    const int nht = n_q_heads * T;
    attn_prefill_kernel<<<nht, 128, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)k_cache, (const uint8_t*)v_cache, q, out, p0_dev, T, swa ? 1 : 0,
        n_q_heads, scale, layer_row, kv_cap);
}

extern "C" void step37_attn_prefill_direct(const void* k, const void* v, const float* q, float* o,
                                           const int* p0, int T, bool swa, int nqh, float scale,
                                           long lr, int cap, void* st) {
    attn_prefill(k, v, q, o, p0, T, swa, nqh, scale, lr, cap, st);
}

// ===== P2.3: decode flash attention（单 query，pos 分段并行 + 二段规约）=====
// 旧三件（scores 一 pos 一线程的标量 dot + weighted_v 每 (h,d) 一线程跨步 2KB 串行扫）
// 在 50K ctx 下 46.6 ms/token（实得 54GB/s = 7.4%，网格只有 96 block ≈12% occupancy，
// 延迟受限）。本 kernel：
//   · grid = (nqh, n_seg)，n_seg 由 host 按窗口算（graph 已判死，不必铺满设备端）；
//   · seg 是慢维（blockIdx.y）→ 同一段的 96 个头相邻调度，共享 8.4MB 段 KV（L2 复用）；
//   · 每 block 4 warp 管段内 4 个连续子段，各跑 online softmax（lane = d 片 4 dims +
//     butterfly 归约，全员得 s，无 divergent shfl）；
//   · 段态 (m, l, acc[128]) 落 scratch[h][seg][130]，merge kernel 二段合并。
namespace {

constexpr int kDFSeg = 2048;   // 每段 pos 数

__global__ void __launch_bounds__(128) attn_df_kernel(
        const uint8_t* kcache, const uint8_t* vcache, const float* q, float* out,
        const int* ppos, int swa, int n_qh, int n_kvh, float scale,
        long layer_row, float* scratch, int kv_cap) {
    const int h = blockIdx.x;
    const int seg = blockIdx.y;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = *ppos;
    const int lo = swa ? (pos > 511 ? pos - 511 : 0) : 0;
    const int hkv = h / (n_qh / n_kvh);
    const float* qh = q + (size_t)h * 128;
    const int seg_len = swa ? 128 : kDFSeg;
    const int p_lo = lo + seg * seg_len;
    const int p_hi = min(p_lo + seg_len - 1, pos);
    if (p_lo > pos) return;
    // 段内 4 分（warp 各管 1/4 连续子段）
    const int sub = (p_hi - p_lo + 4) >> 2;
    const int q_lo = p_lo + w * sub;
    const int q_hi = min(q_lo + sub - 1, p_hi);
    const int cmask = kv_cap - 1;
    const float q0 = qh[lane * 4 + 0], q1 = qh[lane * 4 + 1];
    const float q2 = qh[lane * 4 + 2], q3 = qh[lane * 4 + 3];
    float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int p = q_lo; p <= q_hi; ++p) {
        // q8_0 KV：4 值（lane*4..+3）同块；2×u16 读 qs（34B 块非 4 对齐，避开 uchar4）
        const uint8_t* krow = kcache + layer_row + (size_t)(p & cmask) * 1088 + hkv * 136;
        const uint8_t* vrow = vcache + layer_row + (size_t)(p & cmask) * 1088 + hkv * 136;
        const int d0 = lane * 4, blk = d0 >> 5, off = d0 & 31;
        const uint16_t kq = *(const uint16_t*)(krow + blk * 34 + off);
        const float kd = __half2float(*(const __half*)(krow + blk * 34 + 32));
        const float kf0 = (int8_t)(kq & 0xff) * kd, kf1 = (int8_t)(kq >> 8) * kd;
        const uint16_t kq2 = *(const uint16_t*)(krow + blk * 34 + off + 2);
        const float kf2 = (int8_t)(kq2 & 0xff) * kd, kf3 = (int8_t)(kq2 >> 8) * kd;
        float part = q0 * kf0 + q1 * kf1 + q2 * kf2 + q3 * kf3;
#pragma unroll
        for (int o = 16; o; o >>= 1) part += __shfl_xor_sync(~0u, part, o);
        const float s = part * scale;
        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);   // 首迭代 m=-inf → alpha=0
        const float e = __expf(s - m_new);
        const uint16_t vq = *(const uint16_t*)(vrow + blk * 34 + off);
        const float vd = __half2float(*(const __half*)(vrow + blk * 34 + 32));
        const uint16_t vq2 = *(const uint16_t*)(vrow + blk * 34 + off + 2);
        const float vf0 = (int8_t)(vq & 0xff) * vd, vf1 = (int8_t)(vq >> 8) * vd;
        const float vf2 = (int8_t)(vq2 & 0xff) * vd, vf3 = (int8_t)(vq2 >> 8) * vd;
        l = l * alpha + e;
        a0 = a0 * alpha + e * vf0;
        a1 = a1 * alpha + e * vf1;
        a2 = a2 * alpha + e * vf2;
        a3 = a3 * alpha + e * vf3;
        m = m_new;
    }
    // block 内 4-warp 规约 → 该段 (m, l, acc[128]) 落 scratch[h][seg]
    __shared__ float sm[4], sl[4], sacc[4][128];
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    sacc[w][lane * 4 + 0] = a0; sacc[w][lane * 4 + 1] = a1;
    sacc[w][lane * 4 + 2] = a2; sacc[w][lane * 4 + 3] = a3;
    __syncthreads();
    if (w == 0) {
        const float M = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
        float wg[4], L = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { wg[i] = __expf(sm[i] - M); L += sl[i] * wg[i]; }
        float* sc = scratch + ((size_t)h * gridDim.y + seg) * 130;
        if (lane == 0) { sc[128] = M; sc[129] = L; }
        sc[lane * 4 + 0] = sacc[0][lane*4+0]*wg[0] + sacc[1][lane*4+0]*wg[1]
                         + sacc[2][lane*4+0]*wg[2] + sacc[3][lane*4+0]*wg[3];
        sc[lane * 4 + 1] = sacc[0][lane*4+1]*wg[0] + sacc[1][lane*4+1]*wg[1]
                         + sacc[2][lane*4+1]*wg[2] + sacc[3][lane*4+1]*wg[3];
        sc[lane * 4 + 2] = sacc[0][lane*4+2]*wg[0] + sacc[1][lane*4+2]*wg[1]
                         + sacc[2][lane*4+2]*wg[2] + sacc[3][lane*4+2]*wg[3];
        sc[lane * 4 + 3] = sacc[0][lane*4+3]*wg[0] + sacc[1][lane*4+3]*wg[1]
                         + sacc[2][lane*4+3]*wg[2] + sacc[3][lane*4+3]*wg[3];
    }
}

// 段间合并：out[h][d] = Σ_seg acc·e^{m_seg−M} / Σ_seg l_seg·e^{m_seg−M}
__global__ void attn_df_merge(const float* scratch, float* out, int n_seg) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;   // 128 线程
    const float* base = scratch + (size_t)h * n_seg * 130;
    float M = -INFINITY;
    for (int s = 0; s < n_seg; ++s) M = fmaxf(M, base[(size_t)s * 130 + 128]);
    float acc = 0.f, L = 0.f;
    for (int s = 0; s < n_seg; ++s) {
        const float wg = __expf(base[(size_t)s * 130 + 128] - M);
        L += base[(size_t)s * 130 + 129] * wg;
        acc += base[(size_t)s * 130 + d] * wg;
    }
    out[(size_t)h * 128 + d] = acc / L;
}

} // namespace

void attn_decode_flash(const void* k_cache, const void* v_cache, const float* q, float* out,
                       const int* pos_dev, bool swa, int n_q_heads, int n_kv_heads, float scale,
                       long layer_row, int win, float* scratch, void* stream, int kv_cap) {
    // SWA 层 512 窗口拆 4×128 段（96→384 block，否则全机只有 96 block 占空比过低）；
    // 全注意力层 2048/段
    const int segsz = swa ? 128 : kDFSeg;
    const int n_seg = (win + segsz - 1) / segsz;
    if (n_seg <= 0) return;
    dim3 grid(n_q_heads, n_seg);
    attn_df_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)k_cache, (const uint8_t*)v_cache, q, out, pos_dev, swa ? 1 : 0,
        n_q_heads, n_kv_heads, scale, layer_row, scratch, kv_cap);
    attn_df_merge<<<n_q_heads, 128, 0, (cudaStream_t)stream>>>(scratch, out, n_seg);
}

// 隔离测试入口（tools/df_test.cu 用；不参与引擎路径）
extern "C" void step37_df_direct(const void* k, const void* v, const float* q, float* o,
                                 const int* pos_dev, int swa, int nqh, int nkv, float scale,
                                 long layer_row, int win, float* scratch, void* st, int kv_cap) {
    attn_decode_flash(k, v, q, o, pos_dev, swa != 0, nqh, nkv, scale, layer_row, win, scratch, st, kv_cap);
}

// ===== P2.4: prefill attention v3 — mma.sync flash（FA2 型）=====
// v2 是 F32 标量点积（实测 ~3 TFLOPS = FP32 峰值 3%——50K prefill 175s 的主项）。
// 本版走 tensor core：Q/K/V 全 F16、F32 累加（stock fattn 同口径），Br=64 查询 ×
// Bc=64 键/tile，4 warp × m16 行，online softmax（每线程 2 行 + quartet butterfly）。
// fragment 全部显式 u32 装载（PAD=8 halves；sQ/sK/sV 的 a/b-frag 装载银行冲突已逐条
// 验算为零；sV 转置写有 8 路冲突——写占 tile 时间 ~1/8，先正确后优化）。
//   sQ [64][136]  查询片（F32→F16）
//   sK [64][136]  K 片 [key][dim]（QK^T 的 B：col-major = dim 连续）
//   sV [136][72]  V 片转置 [dim][key]（PV 的 B：col-major = key 连续）
namespace {

constexpr int MA_Br = 64, MA_Bc = 64, MA_D = 128, MA_PAD = 8;
constexpr int MA_SMEM = (MA_Br * (MA_D + MA_PAD) + MA_Bc * (MA_D + MA_PAD)
                         + (MA_D + MA_PAD) * (MA_Bc + MA_PAD)) * (int)sizeof(__half);

__device__ __forceinline__ void mma16816(float& d0, float& d1, float& d2, float& d3,
                                         unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

__device__ __forceinline__ unsigned pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
}

__global__ void __launch_bounds__(128) attn_mma_kernel(
        const uint8_t* __restrict__ kcache, const uint8_t* __restrict__ vcache,
        const float* __restrict__ q, float* __restrict__ out,
        const int* __restrict__ p0dev, int T, int swa, int n_qh,
        float scale, long layer_row, int kv_cap) {
    extern __shared__ __half sm[];
    __half* sQ = sm;
    __half* sK = sQ + MA_Br * (MA_D + MA_PAD);
    __half* sV = sK + MA_Bc * (MA_D + MA_PAD);
    const int qt = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int p0 = *p0dev;
    const int hkv = h / (n_qh / 8);
    const int t0 = qt * MA_Br;
    const int r0 = lane >> 2, c0 = (lane & 3) * 2;


    // Q tile → sQ（F32→F16，float4 全局读合并）
    for (int i = threadIdx.x; i < MA_Br * (MA_D / 4); i += 128) {
        const int r = i >> 5, d4 = (i & 31) << 2;
        const int t = t0 + r;
        const float4 v = (t < T) ? *(const float4*)(q + ((size_t)t * n_qh + h) * MA_D + d4)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
        __half2* dst = (__half2*)(sQ + r * (MA_D + MA_PAD) + d4);
        dst[0] = __floats2half2_rn(v.x, v.y);
        dst[1] = __floats2half2_rn(v.z, v.w);
    }
    __syncthreads();

    // Q fragments（K 循环不变，一次装入 32 reg）
    unsigned qA[8][4];
#pragma unroll
    for (int di = 0; di < 8; ++di) {
        const __half* base = sQ + (w * 16 + r0) * (MA_D + MA_PAD) + di * 16 + c0;
        qA[di][0] = *(const unsigned*)base;
        qA[di][1] = *(const unsigned*)(base + (MA_D + MA_PAD) * 8);
        qA[di][2] = *(const unsigned*)(base + 8);
        qA[di][3] = *(const unsigned*)(base + (MA_D + MA_PAD) * 8 + 8);
    }

    const int kv_cmask = kv_cap - 1;   // cap 恒 2 的幂 → 位与
    const int qrow = w * 16 + r0;          // tile 内行基
    const int t_my0 = t0 + qrow;           // 全局 t（两行 = t_my0, +8）
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float O[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.f;

    const int kt_lo = swa ? max(0, p0 + t0 - 511) : 0;
    const int kt_hi = p0 + t0 + MA_Br - 1;
    const int qp0 = p0 + t_my0, qp1 = qp0 + 8;
    const bool rowv0 = t_my0 < T, rowv1 = t_my0 + 8 < T;

    for (int kt = kt_lo; kt <= kt_hi; kt += MA_Bc) {
        __syncthreads();                   // 上一 tile 的 fragment 读全部完成
        // K 片 [key][dim]（u32 合并读）+ V 片转置 [dim][key]（u16 写，8 路冲突已知）
        for (int i = threadIdx.x; i < MA_Bc * (MA_D / 2); i += 128) {
            const int j = i >> 6, d2 = (i & 63) << 1;
            // q8_0：值对 (d2,d2+1) 同块；行 = layer_row + ((kt+j) % kv_cap)*1088
            const long rrow = (long)((kt + j) & kv_cmask) * 1088;
            const int v0 = hkv * MA_D + d2;
            const int blk = v0 >> 5, off = v0 & 31;
            const uint8_t* kb = kcache + layer_row + rrow + blk * 34;
            const uint8_t* vbb = vcache + layer_row + rrow + blk * 34;
            const float kd = __half2float(*(const __half*)(kb + 32));
            const float vd = __half2float(*(const __half*)(vbb + 32));
            const __half2 kq = __floats2half2_rn((int8_t)kb[off] * kd, (int8_t)kb[off + 1] * kd);
            const __half2 vq = __floats2half2_rn((int8_t)vbb[off] * vd, (int8_t)vbb[off + 1] * vd);
            *(unsigned*)(sK + j * (MA_D + MA_PAD) + d2) = *(const unsigned*)&kq;
            *(__half*)(sV + d2 * (MA_Bc + MA_PAD) + j) = *(const __half*)&vq;
            *(__half*)(sV + (d2 + 1) * (MA_Bc + MA_PAD) + j) = *(const __half*)((const __half*)&vq + 1);
        }
        __syncthreads();

        // ---- S = Q·K^T（8 kstep × 8 ntile，F32 累加）----
        float S[8][4];
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) S[jn][0] = S[jn][1] = S[jn][2] = S[jn][3] = 0.f;
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
            for (int di = 0; di < 8; ++di) {
                const unsigned b0 = *(const unsigned*)(sK + (jn * 8 + (lane >> 2)) * (MA_D + MA_PAD) + di * 16 + c0);
                const unsigned b1 = *(const unsigned*)(sK + (jn * 8 + (lane >> 2)) * (MA_D + MA_PAD) + di * 16 + c0 + 8);
                mma16816(S[jn][0], S[jn][1], S[jn][2], S[jn][3],
                         qA[di][0], qA[di][1], qA[di][2], qA[di][3], b0, b1);
            }
        }
        // ---- scale + 因果/SWA mask + online softmax（quartet 归约，无 divergent shfl）----
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
            const int k0 = kt + jn * 8 + c0;
            const bool v00 = (k0     <= qp0) && (!swa || k0     > qp0 - 512) && rowv0;
            const bool v01 = (k0 + 1 <= qp0) && (!swa || k0 + 1 > qp0 - 512) && rowv0;
            const bool v10 = (k0     <= qp1) && (!swa || k0     > qp1 - 512) && rowv1;
            const bool v11 = (k0 + 1 <= qp1) && (!swa || k0 + 1 > qp1 - 512) && rowv1;
            S[jn][0] = v00 ? S[jn][0] * scale : -INFINITY;
            S[jn][1] = v01 ? S[jn][1] * scale : -INFINITY;
            S[jn][2] = v10 ? S[jn][2] * scale : -INFINITY;
            S[jn][3] = v11 ? S[jn][3] * scale : -INFINITY;
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
            tm0 = fmaxf(tm0, fmaxf(S[jn][0], S[jn][1]));
            tm1 = fmaxf(tm1, fmaxf(S[jn][2], S[jn][3]));
        }
#pragma unroll
        for (int o = 2; o; o >>= 1) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(~0u, tm0, o));
            tm1 = fmaxf(tm1, __shfl_xor_sync(~0u, tm1, o));
        }
        const float a0r = (m0 == -INFINITY || tm0 == -INFINITY) ? 1.f : __expf(m0 - tm0);
        const float a1r = (m1 == -INFINITY || tm1 == -INFINITY) ? 1.f : __expf(m1 - tm1);
        m0 = tm0; m1 = tm1;
        l0 *= a0r; l1 *= a1r;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            O[j][0] *= a0r; O[j][1] *= a0r;
            O[j][2] *= a1r; O[j][3] *= a1r;
        }
        // P = exp(S - m)（-inf 行守卫），存回 S 并按 PV A-frag 布局打包
        float psum0 = 0.f, psum1 = 0.f;
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
            S[jn][0] = (tm0 == -INFINITY) ? 0.f : __expf(S[jn][0] - tm0);
            S[jn][1] = (tm0 == -INFINITY) ? 0.f : __expf(S[jn][1] - tm0);
            S[jn][2] = (tm1 == -INFINITY) ? 0.f : __expf(S[jn][2] - tm1);
            S[jn][3] = (tm1 == -INFINITY) ? 0.f : __expf(S[jn][3] - tm1);
            psum0 += S[jn][0] + S[jn][1];
            psum1 += S[jn][2] + S[jn][3];
        }
#pragma unroll
        for (int o = 2; o; o >>= 1) {
            psum0 += __shfl_xor_sync(~0u, psum0, o);
            psum1 += __shfl_xor_sync(~0u, psum1, o);
        }
        l0 += psum0; l1 += psum1;
        unsigned pA[4][4];
#pragma unroll
        for (int j2 = 0; j2 < 4; ++j2) {
            pA[j2][0] = pack_h2(S[j2 * 2][0],     S[j2 * 2][1]);
            pA[j2][1] = pack_h2(S[j2 * 2][2],     S[j2 * 2][3]);
            pA[j2][2] = pack_h2(S[j2 * 2 + 1][0], S[j2 * 2 + 1][1]);
            pA[j2][3] = pack_h2(S[j2 * 2 + 1][2], S[j2 * 2 + 1][3]);
        }
        // ---- O += P·V（4 kstep × 16 ntile）----
#pragma unroll
        for (int jn = 0; jn < 16; ++jn) {
#pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) {
                const unsigned b0 = *(const unsigned*)(sV + (jn * 8 + (lane >> 2)) * (MA_Bc + MA_PAD) + j2 * 16 + c0);
                const unsigned b1 = *(const unsigned*)(sV + (jn * 8 + (lane >> 2)) * (MA_Bc + MA_PAD) + j2 * 16 + c0 + 8);
                mma16816(O[jn][0], O[jn][1], O[jn][2], O[jn][3],
                         pA[j2][0], pA[j2][1], pA[j2][2], pA[j2][3], b0, b1);
            }
        }
    }

    // ---- 归一化 + 写回（D-frag → 全局行）----
    if (rowv0) {
        const float inv = l0 > 0.f ? 1.f / l0 : 0.f;
#pragma unroll
        for (int jn = 0; jn < 16; ++jn) {
            float* orow = out + ((size_t)t_my0 * n_qh + h) * MA_D + jn * 8 + c0;
            orow[0] = O[jn][0] * inv;
            orow[1] = O[jn][1] * inv;
        }
    }
    if (rowv1) {
        const float inv = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
        for (int jn = 0; jn < 16; ++jn) {
            float* orow = out + ((size_t)(t_my0 + 8) * n_qh + h) * MA_D + jn * 8 + c0;
            orow[0] = O[jn][2] * inv;
            orow[1] = O[jn][3] * inv;
        }
    }
}

} // namespace

void attn_prefill_mma(const void* k_cache, const void* v_cache,
                      const float* q, float* out, const int* p0_dev, int T,
                      bool swa, int n_q_heads, float scale,
                      long layer_row, int kv_cap, void* stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(attn_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MA_SMEM);
        attr_set = true;
    }
    dim3 grid((T + MA_Br - 1) / MA_Br, n_q_heads);
    attn_mma_kernel<<<grid, 128, MA_SMEM, (cudaStream_t)stream>>>(
        (const uint8_t*)k_cache, (const uint8_t*)v_cache, q, out, p0_dev, T, swa ? 1 : 0,
        n_q_heads, scale, layer_row, kv_cap);
}

extern "C" void step37_attn_mma_direct(const void* k, const void* v, const float* q, float* o,
                                       const int* p0, int T, bool swa, int nqh, float scale,
                                       long lr, int cap, void* st) {
    attn_prefill_mma(k, v, q, o, p0, T, swa, nqh, scale, lr, cap, st);
}
