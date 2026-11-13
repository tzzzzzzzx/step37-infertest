// 干净树裁剪：step37（step35/IQ4_XS 工作负载）永不分派的 strata 通用 kernel 桩。
// 仅满足链接；若未来引入这些类型/路径，请用 maya 树的真实实现替换。
#include <cstdio>
#include <stdint.h>
#include <cstdlib>
namespace strata::kernels {
void bf16_gemv_fp32_mmvf(const float*, const unsigned short*, float*, long, long, void*) { std::exit(1); }
void bf16_gemv_fp32_mmvf_multi(const float*, long, const unsigned short*, float*, long, long, long, int, void*) { std::exit(1); }
void s2_gemv_q8(const uint8_t*, const uint8_t*, const float*, float*, long, long, int, void*) { std::exit(1); }
void s_gemv_q8_0_split(const uint8_t*, const uint8_t*, const float*, const float*, float*, long, long, const struct SForm&, void*) { std::exit(1); }
void s_gemv_q8k_split(const uint8_t*, const uint8_t*, const float*, const float*, float*, long, long, const struct SForm&, void*) { std::exit(1); }
void quantize_q8_0(const float*, uint8_t*, long, void*) { std::exit(1); }
void quantize_q8_K(const float*, uint8_t*, long, void*) { std::exit(1); }
}
