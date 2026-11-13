// step37_dump.hpp - parity wall station dumps (env-gated, zero cost unset)
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <vector>
#include <cuda_runtime.h>

namespace step37dump {

// true while a per-layer CUDA graph is being captured: dump calls (cudaMemcpy = sync)
// are illegal inside capture, so they are skipped for the capturing token only
inline bool g_capturing = false;

inline bool pos_ok(int pos) {
    static int want = -2;
    if (want == -2) {
        const char* e = getenv("STEP37_DUMP_POS");
        want = e ? atoi(e) : 0;
    }
    return pos == want;
}

inline bool dump_on() {
    static const char* dir = getenv("STEP37_DUMP_DIR");
    return dir != nullptr;
}

inline void f32f(int pos, const void* p, int n_f32, const char* fmt, ...) {
    if (g_capturing) return;   // cudaMemcpy is illegal during graph capture
    if (!dump_on()) return;
    if (!pos_ok(pos)) return;
    char name[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    char path[600];
    snprintf(path, sizeof path, "%s/%s.bin", getenv("STEP37_DUMP_DIR"), name);
    std::vector<char> buf((size_t)n_f32 * 4);
    cudaMemcpy(buf.data(), p, (size_t)n_f32 * 4, cudaMemcpyDeviceToHost);
    if (FILE* f = fopen(path, "wb")) {
        fwrite(buf.data(), 1, buf.size(), f);
        fclose(f);
    }
}

} // namespace step37dump
