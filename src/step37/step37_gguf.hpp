// step37_gguf.hpp — maya-step37 M1: GGUF 直读装载器（trunk-only；MTP 草稿模默认不加载）
//
// 领地纪律: ../../CONFLICT-POLICY.md — 本目录是 step37 自有领地。
// 契约来源:
//   - reader: include/strata/artifact/gguf_reader.hpp (strata::GgufFile, header-only, bounds-checked)
//   - 独立期望值: workbench notes/gguf-tensors-shard{1,2,3}.json (gguf-py dump, 独立工具产生)
//   - 架构事实: src/step37/step37_geo.hpp (全部实测常量)
// 决策 D1 (STEP37-PLAN.md): GGUF 直读 — 本机 499G RAM 全覆盖专家, SSD 换页是纯兜底。
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace step37 {

// 一卷 GGUF 的清单条目（scan 阶段收集）。
struct ShardInfo {
    std::string path;
    uint32_t version = 0;
    uint64_t data_start = 0;
    uint64_t file_size = 0;
    size_t tensor_count = 0;
};

// 跨卷合并后的张量引用。offset 相对所在卷的 data_start。
struct TensorRef {
    std::string name;
    uint32_t shard = 0; // 分卷序号，与 Manifest::shards 下标对应
    uint64_t offset = 0;
    uint32_t type = 0;
    std::vector<uint64_t> shape; // GGUF 序: dim0 变化最快
    uint64_t elements = 0;

    // 字节数 = block 数 × block_geometry 字节。未知类型返回 0（调用方报错）。
    uint64_t bytes() const {
        int el = 0, by = 0;
        if (!strata::block_geometry(type, el, by)) return 0;
        return elements / (uint64_t)el * (uint64_t)by;
    }
    const char* type_name() const { return strata::ggml_type_name(type); }
};

// 布局归属。M1 只分两类；KV/工作区是运行时分配, 不来自文件。
enum class Home : uint8_t { VRAM_RESIDENT, EXPERT_POOL };

// 三卷合并清单 + 分类账。纯数据, scan 产出, layout/装载/槽池都吃它。
struct Manifest {
    std::vector<ShardInfo> shards;
    std::vector<TensorRef> tensors;
    std::vector<Home> home; // 与 tensors 平行

    uint64_t expert_bytes = 0;
    uint64_t resident_bytes = 0;
    size_t expert_count = 0;
    size_t routed_expert_slices = 0; // 42 层 × 288 专家(完整 FFN=gate/up/down 三件)

    const TensorRef* find(const char* name) const {
        for (const auto& t : tensors)
            if (t.name == name) return &t;
        return nullptr;
    }
};

// 扫描目录下全部 *.gguf 分卷并合并张量清单。
// MTP 草稿模跳过: 文件名含 "mtp"(大小写不敏感)。非 GGUF 文件报错终止。
Manifest scan_directory(const std::string& dir);

// 分类: 名字含 "_exps" (ffn_gate/up/down_exps) → EXPERT_POOL, 其余 → VRAM_RESIDENT。
// 返回的 Manifest.home 与 tensors 平行, 同时累加字节账。
void classify(Manifest& m);

} // namespace step37
