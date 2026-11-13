# step37-infertest

## 模型名

- Step-3.7-Flash（step35 架构：196.96B MoE / 11B 激活，288 专家 top-8 + shared，12 层全注意力 + 33 层 SWA-512）
- 权重：Step-3.7-Flash-ARA-Heretic（PeterCxy 解锁版，IQ4_XS 三分卷 97.78 GiB）

## 配置与代码构成

- 硬件：RTX 4080 SUPER 32G（sm_89）+ EPYC 7R12 48 核 / 8 通道 DDR4-2400，单卡单副本，全程未换硬件
- **代码血缘（如实声明）**：这不是从零写的引擎。**算术算子层大量复用现成实现**——量化算术
  （IQ4_XS/Q5_K/Q6_K/Q8_0 块布局与 vec_dot）来自 llama.cpp 的 ggml-cuda MMQ 子集（MIT，
  vendored 于 `vendor/`）；CPU 专家 lane 复用 strata 的 ggml-cpu 委托；嵌入/rope/norm 等
  基础 kernel 来自 strata kernels。**原创部分是调度与编排**：自定义 GGUF 直读、prefill
  staging 预取环、专家池 LFU、q8 KV 环形寻址、OP 分相 daemon、前缀会话复用、设备端路由
  派发——约 3.4K 行自写调度代码，把现成算子搬进一个反 ggml 调度模型的执行序列。
- 引擎运行时：CUDA 13.2 + cuBLAS
  - q8_0 KV 环形寻址：全注意力层 262144 / SWA 层 4096（位掩码，cap 恒 2 的幂）
  - prefill：cuBLAS FP16 投影 + mma.sync flash attention + 设备端路由派发 + 双槽 staging 预取
  - decode：分段 flash attention + 专家池 LFU（选装）+ CPU lane work-stealing
  - 服务：双实例常驻 daemon，OpenAI 兼容 :8080，SSE 流式（reasoning_content/content），前缀 KV 会话复用 + 会话亲和路由，prefill 全局串行 / decode 并行

## 结果

- prefill @50K：1072 t/s（ub16384）/ 823（ub8192）/ 571（ub4096，双实例档）
- decode @50K：25.29 t/s（带专家池 2048 槽，命中 77.3%）；F16 KV 对照：prefill 1222.8 / decode 33.04
- 对拍口径：per-op rel < 1e-3（多数 bit 级）；prompt ids 与 stock tokenize 逐位全等；终 logits top-8 与 stock 同序同值；同输入跨 run 字节级一致
- 服务验收：单发 / 双实例并发 / 多轮前缀复用（增量 prefill）三连通过；SSE 契约（include_usage）通过
- 优化明细见 [OPTIMIZATIONS.md](OPTIMIZATIONS.md)

## 参考项目

- [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）：MMQ INT8 TC GEMM、quantize、ggml-cpu AVX2（CPU lane）、RMSNorm、量化格式（IQ4_XS/Q5_K/Q6_K）
- [project-maya](https://github.com/mw00/project-maya)（strata 底座）：GGUF 直读、strata_kernels、专家池/doorbell 判例来源
- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)：prefill mma attention 的算法结构参考
- StepFun step35 官方实现（llama.cpp `src/models/step35.cpp`）：对拍参照系

## 生成模型

本项目实现过程由以下生成模型协助完成：**GLM 5.3 Flash**、**GLM 5.3**、**DS1017**。
