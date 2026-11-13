# 具体优化点

基线：llama.cpp stock（ngl 99 / 尾 35 层专家进 CPU / ub512）prefill 122.3 / decode 28.97 t/s @50K。
首跑（朴素路径）：4.2 / 4.2 t/s。方法：先量硬件峰值（FP16 TC ~104T / INT8 TC ~208T / PCIe pinned 26GB/s /
DRAM 98.9GB/s / VRAM 736GB/s），把每个工作单元搬到边际速率最高的部件上，每搬一次重签数值契约，每轮重测。

## prefill（4.2 → 1222.8 t/s，F16 KV 口径）

1. **投影 GEMV → cuBLAS FP16 TC**：自研标量 GEMV 只跑到 3 TFLOPS（3%）；other 段 69 → 1.0 s/chunk
2. **attention 两级**：CUDA-core 两遍 online softmax（16.0 → 1.72 s/chunk）→ mma.sync FA2 型 v3（Br64×Bc64，1719 → 147 ms/chunk，11.7×）
3. **专家权重流**：mmap 直连 async 只有 5.2 GB/s（driver 慢路径）→ 双槽 pinned landing + 后台预拷 + 事件交接（10.0 → 4.3 s/chunk）
4. **ubatch 拉档**：PCIe 地板公式 = chunks × 97.78 GiB ÷ 26 GB/s；ub512→16384 使 chunks 12.2→3，594 → 869 → 1128 t/s
5. **路由三件**：288 线程标量 GEMV（占 GPU 时间 43.6%）→ cublasSgemm；路由输出设备化（kernel 直写 host-mapped 每次过 PCIe = 1.49s）→ 设备端三 kernel 派发（hist / bounds 扫描 / atomic 散射），去每层全流 sync
6. **combine 确定性化**：路由序 per-token gather（rank 序求和、零原子）——atomicAdd 求和序逐跑漂移 × 下一层 q8 舍入悬崖曾致路由翻转与生成退化，修后与 stock top-8 逐位同序

## decode（4.2 → 33.04 t/s F16 口径 / 25.29 q8 口径）

1. **分段 flash attention**：pos 分段 + 二段规约（46.6 → 7.0 ms/token，带宽效率 7.4%→48%）+ uint2 键读 + SWA 4 段拆分（占空比摊薄）
2. **router cuBLAS**：14.8 → 0.3 ms/token
3. **一次预量化**：~1100 次/8tok 冗余 quantize → 一次量化全层共享（未归因 10.6 → 7.9 ms）
4. **gate|up 合并 mmvq**：17 → 2 launch/层
5. **专家池 LFU**（2048 槽 / 17.1GB，命中 77.3%）：CPU 段 28.5 → 12 ms；CPU lane work-stealing 贴 DRAM 地板 95%
6. **q8_0 KV 环形**：KV 25GB → 7.1GB/实例——262144×2 并发的容量前提，同帧解锁 50K 档专家池

## 服务层

- 双实例常驻 daemon（VRAM 常驻，前缀 KV 活过请求）；OP 分相：prefill 全局串行（单活跃 +6.2GB，双活跃 34GB>32.3）+ decode 真并行
- 前缀 KV 会话复用 + 会话亲和路由（最长公共前缀 diff → 增量 prefill）
- SSE 流式（reasoning_content/content/usage 全契约），自愈三路（DEAD/RESET/ERR）

## 对拍与测量纪律

- per-op 对拍墙：refdump 挂上游库采样 2818 节点 vs 引擎 20 工位 dump，numpy 裁判；终 logits top-8 与 stock 逐位同序
- 测量分层：env 旋钮扫描 → per-station 账本 → nsys → compute-sanitizer → SIGSEGV backtrace handler
- 账本未归因 >20% 必须停下查；混沌轨迹 A/B 前先隔离契约

## 负结果（同样入账）

- CUDA graph 本卡（4080S/CUDA 13.2）回放数值错且更慢，判死
- CCD 绑核不加带宽反挤通路；PCIe 拉 miss 同字节 3.8× 慢于 DRAM
- 单槽 staging 省显存 = 买断重叠（GPU 饿死在 CPU memcpy 上）；q8 df kernel load-count 4 vs F16 的 2（-23% 税，容量换速度的裁决）
