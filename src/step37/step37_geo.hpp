// step37_geo.hpp — Step 3.7 Flash (step35 arch) 几何常量
// 实测来源（2026-11-05）：
//   - GGUF 解剖: notes/gguf-tensors-shard{1,2,3}.json + notes/gguf-mtp-draft.json
//     (权重路径见 README 配置节)
//   - HF config: stepfun-ai/Step-3.7-Flash (text_config, model_type "step3p5")
//   - llama.cpp 参照实现: workbench upstream @ 10d4846a, src/models/step35.cpp (551 行)
#pragma once

namespace step37 {

// ---- 主干 ----
constexpr int kTrunkLayers = 45;        // blk 0..44；blk 0-2 稠密(FFN 11264)，blk 3..44 MoE
constexpr int kDenseBlocks = 3;
constexpr int kDenseFFN    = 11264;
constexpr int kHidden      = 4096;
constexpr int kVocab       = 128896;
constexpr int kCtx         = 262144;
constexpr int kHeadDim     = 128;

// ---- 注意力（主干 64 头 GQA）----
constexpr int kQHeads  = 64;
constexpr int kKVHeads = 8;
// 每层部分 RoPE：全注意力层只用一半旋转维（upstream: n_rot_full/2）
constexpr int kRotFull = 64;            // 128 的一半
constexpr int kRotSwa  = 128;           // SWA 层全旋转维

// ---- MoE ----
constexpr int kNExperts      = 288;
constexpr int kTopK          = 8;
constexpr int kExpertFFN     = 1280;
constexpr int kSharedCount   = 1;
constexpr int kSharedFFN     = 1280;
constexpr float kRouterScale  = 3.0f;   // expert_weights_scale=3.0, weights_norm=true, sigmoid 路由 + probs_b 偏置

// ---- SWA ----
constexpr int kSwaWindow    = 512;
// 1:3 配比（~12 全注意力 + ~33 SWA），由 config 的 rope_theta 5M/10k 交替推断；
// 装载时以 GGUF 的 sliding_window_pattern 数组为准（装载器应读数组，不硬编码）
constexpr int kFullAttnLayersApprox = 12;

constexpr float kRopeBaseFull = 5000000.0f;
constexpr float kRopeBaseSwa  = 10000.0f;

// ---- MTP 草稿（独立草稿模 Step3.7-flash-mtp-Q8_0.gguf）----
constexpr int kMtpBlocks   = 3;         // blk 45..47
constexpr int kMtpQHeads   = 96;        // 草稿块 96 头（q 12288 = 96×128）
constexpr int kMtpKVHeads  = 8;
constexpr int kMtpFFN      = 11264;     // 草稿块是稠密 FFN，不是 MoE
constexpr bool kMtpOwnHead = true;      // 自带 shared_head_head [4096×128896]

// ---- 量化布局（ARA-Heretic IQ4_XS，UD 风格）----
// 专家 IQ4_XS (~95-100GB of 102GB 文件) | 注意力 Q5_K | norm/rope 因子 F32
// 常驻集 ~7B 参数 ≈ 4GB；路由专家 ≈ 190B 参数驻 RAM

} // namespace step37
