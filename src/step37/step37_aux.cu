// step37_aux.cu — decode 辅助 kernels（step37 ns 内）
#include "step37_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdio>
#include <cmath>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace step37 {


__global__ void iq4_xs_row_kernel(const uint8_t* table, int token, float* out) {
    const int ib = threadIdx.x;          // 0..127: 16 blocks x 8 sub-blocks
    const int blk_id = ib / 8, sub = ib % 8;
    const uint8_t* blk = table + (size_t)token * 2176 + (size_t)blk_id * 136;
    const float d = __half2float(*(const __half*)blk);
    const uint16_t sh = *(const uint16_t*)(blk + 2);
    const uint8_t* sl = blk + 4;
    const uint8_t* qs = blk + 8;
    float* o = out + (size_t)ib * 32;
    const int ls = ((sl[ib % 8 / 2] >> 4 * ((ib % 8) % 2)) & 0xf) | (((sh >> 2 * (ib % 8)) & 3) << 4);
    const float dl = d * (float)((int)ls - 32);
    const uint8_t* q4 = qs + 16 * (ib % 8);
    for (int j = 0; j < 16; ++j) {
        o[j] = dl * (float)kvalues_iq4nl[q4[j] & 0xf];
        o[j + 16] = dl * (float)kvalues_iq4nl[q4[j] >> 4];
    }
}

__global__ void f32_gemv_kernel(const float* w, const float* x, float* y, int n_in, int n_out) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    float acc = 0.f;
    const float* row = w + (size_t)o * n_in;
    for (int i = 0; i < n_in; ++i) acc += row[i] * x[i];
    y[o] = acc;
}

__global__ void gate_mul_kernel(const float* gate, float* attn, int hd, int n_qh) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= hd) return;
    attn[(size_t)h * hd + d] *= 1.0f / (1.0f + __expf(-gate[h]));
}

__global__ void swiglu_step37_kernel(const float* gate, const float* up, float* out, int n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = gate[i] / (1.f + expf(-gate[i])) * up[i];
}

void iq4_xs_row(const uint8_t* table, int token, float* out, void* stream) {
    iq4_xs_row_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(table, token, out);
}

void f32_gemv(const float* w, const float* x, float* y, int n_in, int n_out, void* stream) {
    f32_gemv_kernel<<<(n_out + 255) / 256, 256, 0, (cudaStream_t)stream>>>(w, x, y, n_in, n_out);
}

void attn_gate_mul(const float* gate, float* attn, int head_dim, int n_q_heads, void* stream) {
    gate_mul_kernel<<<n_q_heads, head_dim, 0, (cudaStream_t)stream>>>(gate, attn, head_dim, n_q_heads);
}

void swiglu_rows_step37(const float* gate, const float* up, float* out, int n, void* stream) {
    swiglu_step37_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(gate, up, out, n);
}

// device argmax：单 block 两遍（max 归约 + 找索引），写 4B 到 host pinned
__global__ void argmax_kernel(const float* logits, int n, int* out) {
    __shared__ float smax[256];
    __shared__ int sidx[256];
    const int tid = threadIdx.x;
    float best = -INFINITY; int bi = 0;
    for (int i = tid; i < n; i += 256) {
        const float v = logits[i];
        if (v > best) { best = v; bi = i; }
    }
    smax[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
        if (tid < o && smax[tid + o] > smax[tid]) { smax[tid] = smax[tid + o]; sidx[tid] = sidx[tid + o]; }
        __syncthreads();
    }
    if (tid == 0) *out = sidx[0];
}

void device_argmax(const float* logits, int n, int* out_dev, void* stream) {
    argmax_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(logits, n, out_dev);
}


// KV append (graph-safe): k/v [1024] F32 -> 写入本层 F16 cache 的第 pos 行，pos 从设备内存读
// ===== q8_0 KV（2026-11-11：262144×并发2 的容量前提）=====
// 布局：每位置 1024 值 = 32 块 × (32×int8 qs + f16 d) = 1088B；行 = layer_row + (pos % cap)×1088。
// 全层统一环形模（全层 cap=262144 ≥ max pos，模恒等；SWA 层 cap=4096 只留窗口）。
// 量化契约：d = amax/127（amax=0 → inv=0，qs 全 0、d=0，反量化恒 0）。
__device__ __forceinline__ void q8_store_row(uint8_t* row, const float* vals) {
    // vals: 1024 floats；全局线程号（decode 版 4 block×64 / batch 版 1 block×256），
    // 每线程 4 值，8 线程一组 = 1 块（shfl_xor width 8 组内 amax）
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 a = *(const float4*)(vals + i * 4);
    float am = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w)));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) am = fmaxf(am, __shfl_xor_sync(~0u, am, o, 8));
    const float inv = am > 0.f ? 127.f / am : 0.f;
    const int blk = i >> 3, off = (i & 7) * 4;
    uint8_t* qb = row + blk * 34 + off;
    qb[0] = (uint8_t)(int8_t)roundf(a.x * inv);
    qb[1] = (uint8_t)(int8_t)roundf(a.y * inv);
    qb[2] = (uint8_t)(int8_t)roundf(a.z * inv);
    qb[3] = (uint8_t)(int8_t)roundf(a.w * inv);
    if ((i & 7) == 0) *(uint16_t*)(row + blk * 34 + 32) = (uint16_t)__half_as_ushort(__float2half(am / 127.f));
}

__global__ void kv_append_kernel(uint8_t* kcache, uint8_t* vcache, const float* k, const float* v,
                                 const int* ppos, long layer_row, int cap) {
    const long pos = *ppos % cap;
    q8_store_row(kcache + layer_row + pos * 1088, k);
    q8_store_row(vcache + layer_row + pos * 1088, v);
}

void kv_append(void* kcache, void* vcache, const float* k, const float* v,
               const int* pos_dev, long layer_row, int cap, void* stream) {
    kv_append_kernel<<<4, 64, 0, (cudaStream_t)stream>>>(   // 恰 256 线程 × 4 值 = 1024 值；
        (uint8_t*)kcache, (uint8_t*)vcache, k, v, pos_dev, layer_row, cap);   // 4,256 会越界 4×
}

// doorbell (graph-safe): GPU 写主机映射标志位；值 = base*mul + add（base 经设备内存传入，
// mul/add 是层常量可烧进图）——每层门铃值唯一，主机自旋才不会读到上一层的旧路由
__global__ void ring_host_kernel(volatile int* flag, const int* pval, int mul, int add) {
    *flag = *pval * mul + add;
}

void ring_host(volatile int* flag, const int* pval, int mul, int add, void* stream) {
    ring_host_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(flag, pval, mul, add);
}

// ===== P1: 批量 prefill 小件 =====

// 批量 embedding：每 token 一个 block（128 线程 × 16 块 = 4096），复用 iq4_xs_row 的解包数学
__global__ void iq4_xs_rows_kernel(const uint8_t* table, const int* ids, float* out) {
    const int tok = blockIdx.x;
    const int ib = threadIdx.x;          // 0..127: 16 superblocks x 8 sub-blocks
    const int blk_id = ib / 8, sub = ib % 8;
    const int row = ids[tok];
    const uint8_t* blk = table + (size_t)row * 2176 + (size_t)blk_id * 136;
    const float d = __half2float(*(const __half*)blk);
    const uint16_t sh = *(const uint16_t*)(blk + 2);
    const uint8_t* sl = blk + 4;
    const uint8_t* qs = blk + 8;
    float* o = out + (size_t)tok * 4096 + (size_t)ib * 32;
    const int ls = ((sl[ib % 8 / 2] >> 4 * ((ib % 8) % 2)) & 0xf) | (((sh >> 2 * (ib % 8)) & 3) << 4);
    const float dl = d * (float)((int)ls - 32);
    const uint8_t* q4 = qs + 16 * (ib % 8);
    for (int j = 0; j < 16; ++j) {
        o[j] = dl * (float)kvalues_iq4nl[q4[j] & 0xf];
        o[j + 16] = dl * (float)kvalues_iq4nl[q4[j] >> 4];
    }
}

void iq4_xs_rows(const uint8_t* table, const int* ids_dev, float* out, int n_tok, void* stream) {
    iq4_xs_rows_kernel<<<n_tok, 128, 0, (cudaStream_t)stream>>>(table, ids_dev, out);
}

// 批量 KV append：每 token 一个 block 写一行（F32 -> F16 cache）
__global__ void kv_append_batch_kernel(uint8_t* kcache, uint8_t* vcache, const float* k, const float* v,
                                       const int* p0dev, int T, long layer_row, int cap) {
    const int t = blockIdx.x;
    const int i = threadIdx.x;           // 256 线程 × 4 值
    const long pos = ((long)*p0dev + t) % cap;
    float4 a = *(const float4*)(k + (size_t)t * 1024 + i * 4);
    float4 b = *(const float4*)(v + (size_t)t * 1024 + i * 4);
    // K、V 各自算 amax/尺度（V 用 K 尺度 = prefill 侧 V 全坏，2026-11-11 暂停点）
    float amk = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w)));
    float amv = fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w)));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) {
        amk = fmaxf(amk, __shfl_xor_sync(~0u, amk, o, 8));
        amv = fmaxf(amv, __shfl_xor_sync(~0u, amv, o, 8));
    }
    const float invk = amk > 0.f ? 127.f / amk : 0.f;
    const float invv = amv > 0.f ? 127.f / amv : 0.f;
    const int blk = i >> 3, off = (i & 7) * 4;
    uint8_t* kb = kcache + layer_row + pos * 1088 + blk * 34 + off;
    uint8_t* vb = vcache + layer_row + pos * 1088 + blk * 34 + off;
    kb[0] = (uint8_t)(int8_t)roundf(a.x * invk);
    kb[1] = (uint8_t)(int8_t)roundf(a.y * invk);
    kb[2] = (uint8_t)(int8_t)roundf(a.z * invk);
    kb[3] = (uint8_t)(int8_t)roundf(a.w * invk);
    vb[0] = (uint8_t)(int8_t)roundf(b.x * invv);
    vb[1] = (uint8_t)(int8_t)roundf(b.y * invv);
    vb[2] = (uint8_t)(int8_t)roundf(b.z * invv);
    vb[3] = (uint8_t)(int8_t)roundf(b.w * invv);
    if ((i & 7) == 0) {
        *(uint16_t*)(kcache + layer_row + pos * 1088 + blk * 34 + 32) = (uint16_t)__half_as_ushort(__float2half(amk / 127.f));
        *(uint16_t*)(vcache + layer_row + pos * 1088 + blk * 34 + 32) = (uint16_t)__half_as_ushort(__float2half(amv / 127.f));
    }
}

void kv_append_batch(void* kcache, void* vcache, const float* k, const float* v,
                     const int* p0_dev, int T, long layer_row, int cap, void* stream) {
    kv_append_batch_kernel<<<T, 256, 0, (cudaStream_t)stream>>>(
        (uint8_t*)kcache, (uint8_t*)vcache, k, v, p0_dev, T, layer_row, cap);
}

// 批量 router（F32 权重 GEMM）：blogits[t][e] = Σ bffn_x[t][:] · W[:, e]
__global__ void router_gemm_kernel(const float* x, const float* w, float* out, int T, int n_in, int n_out) {
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_out) return;
    const float* xr = x + (size_t)t * n_in;
    float acc = 0.f;
    for (int i = 0; i < n_in; ++i) acc += xr[i] * w[(size_t)i * n_out + e];
    out[(size_t)t * n_out + e] = acc;
}

void router_gemm(const float* x, const float* w, float* out, int T, int n_in, int n_out, void* stream) {
    dim3 grid((n_out + 127) / 128, T);
    router_gemm_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(x, w, out, T, n_in, n_out);
}

// 批量 gate 乘：battn[t][h][d] *= sigmoid(bgate[t][h])
// gate 行距 = n_qh（P2: hgemm_f16 ldc=nqh 紧凑布局；旧 GEMV 128 宽行距已随投影 cuBLAS 化退役）
__global__ void gate_mul_batch_kernel(const float* gate, float* attn, int T, int n_qh) {
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const float g = 1.0f / (1.0f + __expf(-gate[(size_t)t * n_qh + h]));
    attn[((size_t)t * n_qh + h) * 128 + d] *= g;
}

void gate_mul_batch(const float* gate, float* attn, int T, int n_qh, void* stream) {
    dim3 grid(n_qh, T);
    gate_mul_batch_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(gate, attn, T, n_qh);
}

// 批量 swiglu（分立 gate/up 缓冲）：h[t][j] = silu(gu[t][j]) * uu[t][j]
__global__ void swiglu_batch_kernel(const float* gu, const float* uu, float* h, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gu[i];
    h[i] = g / (1.f + expf(-g)) * uu[i];
}

void swiglu_batch(const float* gu, const float* uu, float* h, long n, void* stream) {
    const int blocks = (int)((n + 255) / 256);
    swiglu_batch_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(gu, uu, h, n);
}

// 批量 MoE combine：ffn_y[t] += Σ_s w[t][s] * parts[(t*8+s)][d]
__global__ void moe_combine_batch_kernel(const float* parts, const float* w, float* y, int T, int k, int n_embd) {
    const int t = blockIdx.x;
    const int d = threadIdx.x;           // 256 线程 × 16 = 4096
    float acc = 0.f;
    for (int s = 0; s < k; ++s) acc += w[(size_t)t * k + s] * parts[((size_t)t * k + s) * n_embd + d];
    y[(size_t)t * n_embd + d] += acc;
}

void moe_combine_batch(const float* parts, const float* w, float* y, int T, int k, int n_embd, void* stream) {
    moe_combine_batch_kernel<<<T, 256, 0, (cudaStream_t)stream>>>(parts, w, y, T, k, n_embd);
}

// gamma 批量平铺：gout[(t*nqh + h)*n + d] = gin[h*n + d]（q/k norm 批量化用）
__global__ void tile_expand_kernel(const float* gin, float* gout, int T, int rows, int n) {
    // gin 是单行 [n]（逐头共享的 norm gamma），复制到 gout 的全部 [T x rows] 行
    const int t = blockIdx.y;
    const int r = blockIdx.x;
    for (int d = threadIdx.x; d < n; d += blockDim.x)
        gout[((size_t)t * rows + r) * n + d] = gin[r * 0 + d];
}

void tile_expand(const float* gin, float* gout, int T, int rows, int n, void* stream) {
    dim3 grid(rows, T);
    tile_expand_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(gin, gout, T, rows, n);
}

// 路由序 combine（确定性版）：每 token 一个 block，按原路由序 k=0..7 gather 自己的
// 8 条路由求和——求和顺序 = rank 序，跨 run 位级确定（旧版 atomicAdd 的顺序是调度
// 彩票：ULP 噪声 × 下一层 q8 舍入悬崖 → 行级翻转 → 路由漂移 → 生成文本退化，
// 2026-11-11 对拍墙实证）。覆写 y，调用点的 memset 可省（保留也无害）。
__global__ void moe_combine_routed_kernel(const float* parts, const int* inv, const float* w_route,
                                          float* y, int n_embd) {
    const int t = blockIdx.x;
    const int d = threadIdx.x;   // 256 线程 × 循环 16 = 4096
    extern __shared__ float acc[];
    for (int dd = d; dd < n_embd; dd += 256) acc[dd] = 0.f;
    __syncthreads();
    for (int k = 0; k < 8; ++k) {
        const int p = inv[(size_t)t * 8 + k];
        const float w = w_route[p];
        const float* pr = parts + (size_t)p * n_embd;
        for (int dd = d; dd < n_embd; dd += 256) acc[dd] += w * pr[dd];
    }
    __syncthreads();
    float* yr = y + (size_t)t * n_embd;
    for (int dd = d; dd < n_embd; dd += 256) yr[dd] = acc[dd];
}

void moe_combine_routed(const float* parts, const int* inv, const float* w_route,
                        float* y, int T, int n_embd, void* stream) {
    moe_combine_routed_kernel<<<T, 256, n_embd * 4, (cudaStream_t)stream>>>(parts, inv, w_route, y, n_embd);
}



__device__ __constant__ int kvalues_iq4nl_dev[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// IQ4_XS 行内单元素解包（2176B 行 = 16 超块 × 136B；每元素 = d*(ls-32)*LUT[nib]）
__device__ __forceinline__ float deq_iq4xs(const uint8_t* row, int idx) {
    const int sb = (idx >> 8) & 15;                // 16 超块（256 值/超块）
    const int sub = (idx >> 5) & 7;                // 8 子块（每块 32 值）
    const int within = idx & 31;
    const uint8_t* blk = row + (size_t)sb * 136;
    const float d = __half2float(*(const __half*)blk);
    const uint16_t sh = *(const uint16_t*)(blk + 2);
    const int ls = ((blk[4 + (sub >> 1)] >> ((sub & 1) * 4)) & 0xf) | (((sh >> (2 * sub)) & 3) << 4);
    const float dl = d * (float)((int)ls - 32);
    const uint8_t* qs = blk + 8 + sub * 16;
    const uint8_t q = (within < 16) ? (qs[within] & 0xf) : (qs[within - 16] >> 4);
    return dl * (float)kvalues_iq4nl_dev[q];
}

// M2a: resident 层专家批算。grid(8, rows)——每 block 一行的 gate+up 双 dot（128 线程分片）。
// 权重直接命中 VRAM 常驻区（id 偏移），零 H2D 零拷贝。
__global__ void moe_gu8_kernel(const uint8_t* wg_base, const uint8_t* wu_base,
                               const float* x, const int* ids, float* gu, float* uu,
                               long slice, int n_out) {
    const int e = blockIdx.x;
    const int row = blockIdx.y;
    const int id = ids[e];
    const uint8_t* wg = wg_base + (size_t)id * slice + (size_t)row * 2176;
    const uint8_t* wu = wu_base + (size_t)id * slice + (size_t)row * 2176;
    const int tid = threadIdx.x;
    float sg = 0.f, su = 0.f;
    for (int i = tid; i < 4096; i += 128) {
        sg += deq_iq4xs(wg, i) * x[i];
        su += deq_iq4xs(wu, i) * x[i];
    }
    // warp xor 归约（128 线程 = 4 warp → 再 shared 归约）
    for (int o = 16; o; o >>= 1) { sg += __shfl_xor_sync(~0u, sg, o); su += __shfl_xor_sync(~0u, su, o); }
    __shared__ float rg[4], ru[4];
    if (tid % 32 == 0) { rg[tid / 32] = sg; ru[tid / 32] = su; }
    __syncthreads();
    if (tid == 0) {
        gu[(size_t)e * n_out + row] = rg[0] + rg[1] + rg[2] + rg[3];
        uu[(size_t)e * n_out + row] = ru[0] + ru[1] + ru[2] + ru[3];
    }
}

void moe_gu8(const uint8_t* wg_base, const uint8_t* wu_base, const float* x, const int* ids,
             float* gu, float* uu, long slice, int n_out, void* stream) {
    dim3 grid(8, n_out);
    moe_gu8_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(wg_base, wu_base, x, ids, gu, uu, slice, n_out);
}

// down 同构：输入 h（8 专家 × 1280），输出 parts（8 × 4096）
__global__ void moe_dn8_kernel(const uint8_t* wd_base, const float* h, const int* ids, float* parts,
                               long slice, int n_in) {
    const int e = blockIdx.x;
    const int row = blockIdx.y;      // 0..4095 (输出维)
    const int id = ids[e];
    const uint8_t* wd = wd_base + (size_t)id * slice + (size_t)row * 2176;
    const int tid = threadIdx.x;
    float s = 0.f;
    for (int i = tid; i < n_in; i += 128) s += deq_iq4xs(wd, i) * h[(size_t)e * n_in + i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    __shared__ float rs[4];
    if (tid % 32 == 0) rs[tid / 32] = s;
    __syncthreads();
    if (tid == 0) parts[(size_t)e * 4096 + row] = rs[0] + rs[1] + rs[2] + rs[3];
}

void moe_dn8(const uint8_t* wd_base, const float* h, const int* ids, float* parts,
             long slice, int n_in, void* stream) {
    dim3 grid(8, 4096);
    moe_dn8_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(wd_base, h, ids, parts, slice, n_in);
}

// ===== P2: 投影 cuBLAS 化（maya hgemm_q 同款：FP16 权重 × FP16 激活 → F32 累加 TC GEMM）=====

// f32 → f16 激活转换（GemmEx 的输入侧）
__global__ void f32_to_f16_kernel(const float* x, __half* y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
void f32_to_f16(const float* x, void* y, long n, void* stream) {
    f32_to_f16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        x, (__half*)y, n);
}

// Y[T,N] = X[T,K] · W[N,K]^T（W 行主序 = GGUF 行 = 输出单元；X 行主序 token 序）。
// cuBLAS 列主序翻译：Y_cm[N,T] = W_cm[K,N]^T · X_cm[K,T] → OP_T/OP_N，lda=ldb=K，ldc=N。
// 16F×16F 输入、COMPUTE_32F 累加、F32 输出、TENSOR_OP——maya glm_prefill.cu hgemm_q 同参数族。
void hgemm_f16(void* handle, const void* W16, const void* X16, float* Y, int N, int T, int K) {
    const float alpha = 1.f, beta = 0.f;
    cublasGemmEx((cublasHandle_t)handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 N, T, K, &alpha,
                 W16, CUDA_R_16F, K,
                 X16, CUDA_R_16F, K,
                 &beta, Y, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "STEP37 hgemm_f16 N=%d T=%d K=%d: %s\n", N, T, K, cudaGetErrorString(e));
}

// gate/up 合并布局的 swiglu：in 按"命中序"排（每专家 2560 = gate[1280] | up[1280]），
// out 写回专家的固定槽 out[e*1280]（e = mask 的第 j 个置位）。一次 launch 服务全部命中专家。
__global__ void swiglu_gu_masked_kernel(const float* in, float* out, unsigned mask, int n_ff) {
    int j = 0;
    int e = 0;
    // 找到本 block 负责的专家：blockIdx.x = 命中序 j
    for (int b = 0; b < 32; ++b) {
        if (!((mask >> b) & 1u)) continue;
        if (j == (int)blockIdx.x) { e = b; break; }
        ++j;
    }
    const float* g = in + (size_t)blockIdx.x * 2 * n_ff;
    const float* u = g + n_ff;
    float* o = out + (size_t)e * n_ff;
    for (int i = threadIdx.x; i < n_ff; i += blockDim.x)
        o[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}

void swiglu_gu_masked(const float* in, float* out, unsigned mask, int n_ff, void* stream) {
    int n = 0; for (unsigned b = mask; b; b >>= 1) n += (int)(b & 1u);
    if (n <= 0) return;
    swiglu_gu_masked_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(in, out, mask, n_ff);
}

// 批量 swiglu（gate|up 合并布局）：in[e*2*nff + i] = gate, in[e*2*nff + nff + i] = up，
// out[e*nff + i]。一次 launch 服务全部 8 槽（miss 槽的输出无人读，垃圾无害）。
__global__ void swiglu_gu8_kernel(const float* in, float* out, int n_ff) {
    const int e = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_ff) return;
    const float g = in[(size_t)e * 2 * n_ff + i];
    const float u = in[(size_t)e * 2 * n_ff + n_ff + i];
    out[(size_t)e * n_ff + i] = g / (1.f + expf(-g)) * u;
}

void swiglu_gu8(const float* in, float* out, int n_ff, void* stream) {
    dim3 grid(8, (n_ff + 255) / 256);
    swiglu_gu8_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(in, out, n_ff);
}

// ===== 设备端路由派发（prefill 去每层全流 sync：maya "router 从不回读"同款）=====
// router_sigmoid 的输出 (e_ids/exp_w) 留在设备；直方图 → bounds 扫描 → 散射，全在流上。
// 每专家行数上界 = tN（top-8 无放回，每 token 至多 1 路由/专家）→ Product.max_rows 安全上界。
__global__ void route_hist_kernel(const int* e_ids, int* cnt, int R) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    atomicAdd(&cnt[e_ids[i]], 1);
}

// 单线程顺序扫描（288 元素 ~2µs）：bounds = exclusive prefix，顺带清 ofs
__global__ void route_bounds_kernel(const int* cnt, int* bounds, int* ofs) {
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < 288; ++e) { bounds[e] = acc; acc += cnt[e]; ofs[e] = 0; }
        bounds[288] = acc;
    }
}

// 散射：路由 i → 专家序位置 p = bounds[e] + atomicAdd(ofs[e])
// （专家内顺序任意——行数学独立；inv[i]=p 供 combine 按"原路由序"确定性 gather）
__global__ void route_scatter_kernel(const int* e_ids, const float* e_w, const int* bounds, int* ofs,
                                     int* ids_tok, int* ids_row, float* w_route, int* inv, int R) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const int e = e_ids[i];
    const int p = bounds[e] + atomicAdd(&ofs[e], 1);
    ids_tok[p] = i / 8;
    ids_row[p] = i;
    w_route[p] = e_w[i];
    inv[i] = p;
}

void route_hist(const int* e_ids, int* cnt, int R, void* stream) {
    route_hist_kernel<<<(unsigned)((R + 255) / 256), 256, 0, (cudaStream_t)stream>>>(e_ids, cnt, R);
}
void route_bounds(const int* cnt, int* bounds, int* ofs, void* stream) {
    route_bounds_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(cnt, bounds, ofs);
}
void route_scatter(const int* e_ids, const float* e_w, const int* bounds, int* ofs,
                   int* ids_tok, int* ids_row, float* w_route, int* inv, int R, void* stream) {
    route_scatter_kernel<<<(unsigned)((R + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        e_ids, e_w, bounds, ofs, ids_tok, ids_row, w_route, inv, R);
}
} // namespace step37
