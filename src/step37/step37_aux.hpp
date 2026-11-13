// step37_aux.hpp — decode 辅助 kernels 的公共接口
#pragma once
namespace step37 {
void iq4_xs_row(const uint8_t* table, int token, float* out, void* stream);
void f32_gemv(const float* w, const float* x, float* y, int n_in, int n_out, void* stream);
void attn_gate_mul(const float* gate, float* attn, int head_dim, int n_q_heads, void* stream);
void swiglu_rows_step37(const float* gate, const float* up, float* out, int n, void* stream);
void build_llama3_rope_table(int n_rot, double base, int max_pos, float* cos_tab, float* sin_tab);

// KV append（graph 安全）：k/v 写入本层 cache 第 *pos_dev 行；layer_row = il*cap*1024（层常量）
void kv_append(void* kcache, void* vcache, const float* k, const float* v,
               const int* pos_dev, long layer_row, int cap, void* stream);

// doorbell：图内节点写主机 pinned 标志位；值 = base*mul + add（base 设备内存，mul/add 层常量）
void ring_host(volatile int* flag, const int* pval, int mul, int add, void* stream);

// ===== P1: 批量 prefill 小件 =====
void iq4_xs_rows(const uint8_t* table, const int* ids_dev, float* out, int n_tok, void* stream);
void kv_append_batch(void* kcache, void* vcache, const float* k, const float* v,
                     const int* p0_dev, int T, long layer_row, int cap, void* stream);
void router_gemm(const float* x, const float* w, float* out, int T, int n_in, int n_out, void* stream);
void gate_mul_batch(const float* gate, float* attn, int T, int n_qh, void* stream);
void swiglu_batch(const float* gu, const float* uu, float* h, long n, void* stream);
void moe_combine_batch(const float* parts, const float* w, float* y, int T, int k, int n_embd, void* stream);

void tile_expand(const float* gin, float* gout, int T, int rows, int n, void* stream);

void moe_combine_routed(const float* parts, const int* inv, const float* w_route,
                        float* y, int T, int n_embd, void* stream);

void device_argmax(const float* logits, int n, int* out_dev, void* stream);

// M2a: resident 层专家批算（单 launch 8 专家 gate+up / down）
void moe_gu8(const uint8_t* wg_base, const uint8_t* wu_base, const float* x, const int* ids,
             float* gu, float* uu, long slice, int n_out, void* stream);
void moe_dn8(const uint8_t* wd_base, const float* h, const int* ids, float* parts,
             long slice, int n_in, void* stream);

// ===== P2: 投影 cuBLAS 化 =====
void f32_to_f16(const float* x, void* y, long n, void* stream);
// Y[T,N] = X[T,K]·W[N,K]^T；handle = cublasHandle_t（已 SetStream），W/X 为 FP16
void hgemm_f16(void* handle, const void* W16, const void* X16, float* Y, int N, int T, int K);

// gate/up 合并布局的 swiglu（mask 的置位 = 本次命中的专家，按 mask 顺序编号）
void swiglu_gu_masked(const float* in, float* out, unsigned mask, int n_ff, void* stream);

// 批量 swiglu（gate|up 合并布局，8 槽一次 launch；miss 槽输出垃圾无害）
void swiglu_gu8(const float* in, float* out, int n_ff, void* stream);

// 设备端路由派发（prefill 去 sync）：hist → bounds(含 ofs 清零) → scatter
void route_hist(const int* e_ids, int* cnt, int R, void* stream);
void route_bounds(const int* cnt, int* bounds, int* ofs, void* stream);
void route_scatter(const int* e_ids, const float* e_w, const int* bounds, int* ofs,
                   int* ids_tok, int* ids_row, float* w_route, int* inv, int R, void* stream);
} // namespace step37
