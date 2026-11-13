// ggml-cuda 的 ggml_abort 外部依赖 shim（干净树不携带整个 ggml.c）
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    fprintf(stderr, "ggml abort: %s:%d: ", file, line);
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); abort();
}
