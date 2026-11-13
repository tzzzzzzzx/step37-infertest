// step37_gguf.cpp — Manifest 扫描与分类实现
// 独立期望值来源: workbench notes/gguf-tensors-shard{1,2,3}.json（gguf-py @10d4846a, 2026-11-05）
#include "step37_gguf.hpp"
#include "step37_geo.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace step37 {

namespace {

// MTP 草稿模跳过（决策 D1: trunk-only）。判断文件名（大小写不敏感）是否含 "mtp"。
bool looks_like_mtp_draft(const std::string& filename) {
    std::string low;
    low.reserve(filename.size());
    for (char c : filename) low.push_back((char)std::tolower((unsigned char)c));
    return low.find("mtp") != std::string::npos;
}

} // namespace

Manifest scan_directory(const std::string& dir) {
    Manifest m;

    // 1) 收集 *.gguf 分卷路径（跳过 MTP 草稿模），排序固定分卷序号。
    std::vector<std::string> paths;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (name.size() < 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) continue;
        if (looks_like_mtp_draft(name)) continue;
        paths.push_back(e.path().string());
    }
    if (paths.empty()) {
        throw std::runtime_error("step37: no trunk GGUF shards in " + dir);
    }
    std::sort(paths.begin(), paths.end());

    // 2) 逐卷打开，合并张量清单。
    for (uint32_t si = 0; si < paths.size(); ++si) {
        strata::GgufFile g(paths[si]);
        ShardInfo info;
        info.path = paths[si];
        info.version = g.version();
        info.data_start = g.data_start();
        info.file_size = g.file_size();
        info.tensor_count = g.tensors().size();
        m.shards.push_back(info);

        for (const auto& t : g.tensors()) {
            TensorRef r;
            r.name = t.name;
            r.shard = si;
            r.offset = t.offset;
            r.type = t.type;
            r.shape = t.shape;
            r.elements = t.elements();
            m.tensors.push_back(std::move(r));
        }
    }
    return m;
}

void classify(Manifest& m) {
    m.expert_bytes = 0;
    m.resident_bytes = 0;
    m.expert_count = 0;
    m.home.assign(m.tensors.size(), Home::VRAM_RESIDENT);

    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const TensorRef& t = m.tensors[i];
        // "_exps" 子串只命中 ffn_gate/up/down_exps；shexp/ffn_gate_inp 不含它。
        if (t.name.find("_exps") != std::string::npos) {
            m.home[i] = Home::EXPERT_POOL;
            m.expert_bytes += t.bytes();
            ++m.expert_count;
        } else {
            m.resident_bytes += t.bytes();
        }
    }
    // 路由专家槽位 = MoE 层数 × 每层专家数（geo 实测常量）。
    m.routed_expert_slices = (size_t)(kTrunkLayers - kDenseBlocks) * (size_t)kNExperts;
}

} // namespace step37
