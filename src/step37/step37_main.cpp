// step37_main.cpp — step37-load: T1 装载验收 + T2 generate 对拍
#include "step37_model.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <csignal>
#include <execinfo.h>
#include <cstdlib>
#include <unistd.h>
#include <thread>
#include <chrono>

static void segv_handler(int sig) {
    void* bt[24];
    int n = backtrace(bt, 24);
    fprintf(stderr, "\n=== SIGSEGV backtrace (%d) ===\n", sig);
    backtrace_symbols_fd(bt, n, 2);
    _exit(128 + sig);
}

int main(int argc, char** argv) {
    signal(SIGSEGV, segv_handler);
    if (argc < 2) {
        std::printf("usage: step37-load <model_dir> [--gen N]\n");
        --argc;
        return 2;
    }
    step37::Model m;
    step37::ModelOptions o;
    o.model_dir = argv[1];
    std::string err;
    if (!m.load(o, err)) {
        std::printf("FAIL: %s\n", err.c_str());
        return 1;
    }
    if (getenv("STEP37_QPROBE") && std::string(getenv("STEP37_QPROBE")) != "3") {
        // 微探针：固定输入 × N 次 quantize，逐次与第 0 次 diff——kernel 级非确定性判据
        const int ROWS = 176, COLS = 1280, ITER = 60;
        float* x; cudaMalloc(&x, (size_t)ROWS * COLS * 4);
        std::vector<float> hx((size_t)ROWS * COLS);
        unsigned s = 12345;
        for (size_t k2 = 0; k2 < hx.size(); ++k2) {
            s = s * 1664525u + 1013904223u;
            const float mag = (float[]){0.5f, 80.f, 1e-6f, 3.7f, 640.f, 1e-20f}[ (k2 / 1280) % 6 ];
            hx[k2] = ((float)(int)(s >> 8) / 8388608.f - 1.f) * mag;   // ±(0.5/80/1e-6/3.7/640/1e-20) 分行段
        }
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        void* q; cudaMalloc(&q, strata::prefill::mmq::q8_bytes(ROWS, COLS));
        cudaStream_t st; cudaStreamCreate(&st);
        const size_t qbytes = strata::prefill::mmq::q8_bytes(ROWS, COLS);
        std::vector<char> ref(qbytes), cur(qbytes);
        int badit = 0;
        for (int it = 0; it < ITER; ++it) {
            strata::prefill::mmq::quantize(x, nullptr, q, 23, COLS, COLS, ROWS, st);
            cudaMemcpy(cur.data(), q, qbytes, cudaMemcpyDeviceToHost);
            if (it == 0) { ref = cur; continue; }
            if (cur != ref) {
                int nd = 0; for (size_t k = 0; k < qbytes; ++k) nd += cur[k] != ref[k];
                if (badit < 5) std::printf("QPROBE it=%d diffbytes=%d/%zu\n", it, nd, qbytes);
                ++badit;
            }
        }
        std::printf("QPROBE: %d/%d iterations differed (bytes=%zu)\n", badit, ITER, qbytes);
        { FILE* f = fopen("/tmp/qprobe_ref.bin", "wb"); fwrite(ref.data(), 1, qbytes, f); fclose(f); }
        return 0;
    }
    if (getenv("STEP37_QPROBE") && std::string(getenv("STEP37_QPROBE")) == "3") {
        // mmq 探针：真实 bounds（L4）+ 合成权重 + 固定 xq → 60 次 mmq.run → 进程内 + 跨进程 diff
        namespace mmq = strata::prefill::mmq;
        const int ROWS = 176, COLS = 1280, NEXP = 288, ITER = 60;
        // bounds 来自真实路由（i1 的 cnt-b4）
        std::vector<int32_t> hb(NEXP + 1, 0);
        {   FILE* f = fopen("/tmp/eng-i1/cnt-b4.bin", "rb");
            std::vector<int32_t> cnt(NEXP);
            if (f) { fread(cnt.data(), 4, NEXP, f); fclose(f); }
            for (int e = 0; e < NEXP; ++e) hb[e + 1] = hb[e] + cnt[e];
            std::printf("QPROBE3: total routed = %d\n", hb[NEXP]);
        }
        // xq：先用 quantize 生成真实布局的 q8（合成 bh）
        float* x; cudaMalloc(&x, (size_t)ROWS * COLS * 4);
        std::vector<float> hx((size_t)ROWS * COLS);
        unsigned s2 = 777;
        for (auto& v : hx) { s2 = s2 * 1664525u + 1013904223u; v = (float)(int)(s2 >> 8) / 8388608.f - 1.f; }
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        void* xq; cudaMalloc(&xq, mmq::q8_bytes(ROWS, COLS));
        mmq::quantize(x, nullptr, xq, 23, COLS, COLS, ROWS, 0);
        // 权重：随机字节（确定性测试不需要合法 IQ4_XS 语义）
        void* w; cudaMalloc(&w, (size_t)NEXP * 2785280);
        {   std::vector<char> hw(2785280);
            unsigned s3 = 999;
            for (auto& b : hw) { s3 = s3 * 1664525u + 1013904223u; b = (char)(s3 >> 16); }
            for (int e = 0; e < NEXP; ++e) cudaMemcpy((char*)w + (size_t)e * 2785280, hw.data(), 2785280, cudaMemcpyHostToDevice);
        }
        int* d_bounds; cudaMalloc(&d_bounds, hb.size() * 4);
        cudaMemcpy(d_bounds, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
        // ids = iota（与引擎同约定）
        std::vector<int32_t> iota(ROWS);
        for (int i = 0; i < ROWS; ++i) iota[i] = i;
        int* d_ids; cudaMalloc(&d_ids, ROWS * 4);
        cudaMemcpy(d_ids, iota.data(), ROWS * 4, cudaMemcpyHostToDevice);
        float* dst; cudaMalloc(&dst, (size_t)ROWS * 4096 * 4);
        mmq::Context ctx2;
        std::vector<float> ref((size_t)ROWS * 4096), cur((size_t)ROWS * 4096);
        int badit = 0;
        for (int it = 0; it < ITER; ++it) {
            mmq::Product pe;
            pe.type = 23; pe.w_cols = COLS; pe.w_rows = 4096; pe.expert_bytes = 2785280; pe.n = NEXP;
            pe.xq = xq; pe.bounds = d_bounds; pe.ids = d_ids;
            pe.total_rows = ROWS; pe.max_rows = 22; pe.ld_dst = 4096;
            pe.w = w; pe.dst = dst;
            ctx2.run(pe, 0);
            cudaMemcpy(cur.data(), dst, cur.size() * 4, cudaMemcpyDeviceToHost);
            if (it == 0) { ref = cur; continue; }
            if (cur != ref) {
                int nd = 0; for (size_t k = 0; k < cur.size(); ++k) nd += std::memcmp(&cur[k], &ref[k], 4) != 0;
                if (badit < 5) std::printf("QPROBE3 it=%d difffloats=%d/%zu\n", it, nd, cur.size());
                ++badit;
            }
        }
        std::printf("QPROBE3: %d/%d iterations differed\n", badit, ITER);
        { FILE* f = fopen("/tmp/qprobe3_ref.bin", "wb"); fwrite(ref.data(), 4, ref.size(), f); fclose(f); }
        return 0;
    }
    // --prefill N: 批量 prefill 模式（N = prompt 长度）
    if (argc > 2 && std::string(argv[2]) == "--prefill") {
        const int n = argc > 3 ? std::atoi(argv[3]) : 4096;
        std::vector<int32_t> ids; ids.reserve(n + 32);
        // T0 参照 ids 循环填充到 N
        const int32_t t0[32] = {4972, 79656, 127999, 713, 303, 48339, 16976, 429,
                                16976, 8648, 5835, 301, 46015, 303, 1555, 63192,
                                6479, 79025, 109921, 4597, 5506, 2220, 303, 872,
                                4372, 1877, 17085, 63155, 5581, 6664, 3905, 320};
        for (int i = 0; i < n; ++i) ids.push_back(t0[i % 32]);
        ids.resize(n);
        double tps = 0.0;
        std::string err;
        if (!m.prefill_run(ids, tps, err)) { std::printf("PREFILL FAIL: %s\n", err.c_str()); return 1; }
        std::printf("prefill: %d tokens @ %.1f tok/s\n", n, tps);
        return 0;
    }
    const auto& r = m.report();
    std::printf("load: shards=%zu tensors=%zu expert=%.2f GiB resident=%.2f GiB %.1fs\n",
                r.shards, r.tensors,
                (double)r.expert_bytes / 1073741824.0,
                (double)r.resident_bytes / 1073741824.0, r.load_seconds);
    if (argc > 2 && std::string(argv[2]) == "--serve") {
        // 干活模式：prompt ids 从 STEP37_SERVE_IDS 读（host 写的二进制 int32 序列），
        // prefill → decode n=STEP37_SERVE_NGEN → 生成 ids 写 STEP37_SERVE_OUT。
        // 全程走 prefill_run（批量）+ generate(resume) 接力 = 与手感档同一代码路径。
        // ---- daemon 信箱模式（STEP37_MAILBOX 存在时）：常驻进程，逐请求处理。
        // 协议：req.bin = 头行 "NTOK NGEN START TEMP TOPP MINP REP TOPK REPWIN EOS SEED"
        //       + NTOK 个 int32 LE token；rename 成 req.ready 即发布。
        //       引擎写 out.bin + done.txt（OK <n> / RESET / ERR <msg>），删 req.ready。
        if (const char* mbox = getenv("STEP37_MAILBOX")) {
            const std::string dir = mbox;
            const std::string req = dir + "/req.bin", rdy = dir + "/req.ready",
                              out_path = dir + "/out.bin", done = dir + "/done.txt";
            std::fprintf(stderr, "serve: mailbox daemon up at %s\n", dir.c_str());
            for (;;) {
                while (access(rdy.c_str(), 0) != 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                FILE* fi = fopen(rdy.c_str(), "rb");   // rename 后实体就是 req.ready
                if (!fi) { FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR no req.bin\n"); fclose(d); remove(rdy.c_str()); continue; }
                long long op = 0, n_tok = 0, n_gen = 0, start = 0;
                float temp = 0, topp = 1, minp = 0, rep = 1;
                int topk = 0, repwin = 64, eos = -1; long long seed = 42;
                // 头行整行读入再 sscanf——fscanf 不吃行尾换行，fread 会从 '\n' 起读
                // token → 整体错一个字节（2026-11-11：21/22 嵌入行垃圾的根因）
                // OP: 0=全流程 1=仅 prefill（start_pos 追加） 2=仅 generate（ids 是全序
                //     列，位置绝对、base_pos=0——双实例并发下 prefill 全局串行、decode 并行）
                char hdrbuf[512];
                if (!fgets(hdrbuf, sizeof hdrbuf, fi) ||
                    sscanf(hdrbuf, "%lld %lld %lld %lld %g %g %g %g %d %d %d %lld",
                           &op, &n_tok, &n_gen, &start, &temp, &topp, &minp, &rep, &topk,
                           &repwin, &eos, &seed) != 12) {
                    fclose(fi); FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR bad header\n"); fclose(d); remove(rdy.c_str()); continue;
                }
                std::vector<int32_t> ids((size_t)(n_tok < 0 ? 0 : n_tok));
                if (n_tok > 0 && fread(ids.data(), 4, (size_t)n_tok, fi) != (size_t)n_tok) {
                    fclose(fi); FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR short tokens\n"); fclose(d); remove(rdy.c_str()); continue;
                }
                fclose(fi);
                if (n_tok < 0) { FILE* d = fopen(done.c_str(), "w"); fprintf(d, "BYE\n"); fclose(d); remove(rdy.c_str()); break; }
                step37::Model::Sampling sp;
                sp.temp = temp; sp.top_p = topp; sp.min_p = minp; sp.rep_pen = rep;
                sp.top_k = topk; sp.rep_win = repwin; sp.eos = eos; sp.seed = (unsigned)seed;
                if (!getenv("STEP37_NO_SETSAMP")) m.set_sampling(sp);   // bisect
                const auto t0 = std::chrono::steady_clock::now();
                if (start > 0 && m.kv_wiped()) {   // 上一请求触发了 KV 重分配
                    FILE* d = fopen(done.c_str(), "w"); fprintf(d, "RESET\n"); fclose(d); remove(rdy.c_str()); continue;
                }
                // OOM 等异常不许杀引擎（to_f16 OOM 曾 terminate 整进程 = 连锁重启）：
                // 捕获后回报 ERR，serve 层决策重试
                double pps = 0, dtps = 0;
                std::vector<int32_t> out;
                try {
                    if (op == 1) {   // 仅 prefill（KV 前缀 + delta 追加），decode 由 op=2 接力
                        if (!m.prefill_run(ids, pps, err, (int)start)) {
                            FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR prefill: %s\n", err.c_str()); fclose(d); remove(rdy.c_str()); continue;
                        }
                        if (start > 0 && m.kv_wiped()) {
                            FILE* d = fopen(done.c_str(), "w"); fprintf(d, "RESET\n"); fclose(d); remove(rdy.c_str()); continue;
                        }
                    } else if (op == 2) {   // 仅 generate：ids=全序列，位置绝对（base_pos=0）
                        if (!m.generate(ids, (int)n_gen, out, dtps, err, (int)ids.size(), 0)) {
                            FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR gen: %s\n", err.c_str()); fclose(d); remove(rdy.c_str()); continue;
                        }
                    } else {   // op=0 全流程（单发/回归用）
                        if (!m.prefill_run(ids, pps, err, (int)start)) {
                            FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR prefill: %s\n", err.c_str()); fclose(d); remove(rdy.c_str()); continue;
                        }
                        if (start > 0 && m.kv_wiped()) {
                            FILE* d = fopen(done.c_str(), "w"); fprintf(d, "RESET\n"); fclose(d); remove(rdy.c_str()); continue;
                        }
                        if (!m.generate(ids, (int)n_gen, out, dtps, err, (int)ids.size(), (int)start)) {
                            FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR gen: %s\n", err.c_str()); fclose(d); remove(rdy.c_str()); continue;
                        }
                    }
                } catch (const std::exception& e) {
                    FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR exception: %s\n", e.what()); fclose(d); remove(rdy.c_str()); continue;
                } catch (...) {
                    FILE* d = fopen(done.c_str(), "w"); fprintf(d, "ERR unknown\n"); fclose(d); remove(rdy.c_str()); continue;
                }
                FILE* fo = fopen(out_path.c_str(), "wb");
                for (int32_t t : out) fwrite(&t, 4, 1, fo);
                fclose(fo);
                FILE* d = fopen(done.c_str(), "w");
                fprintf(d, "OK %zu %.1f %.1f %.1f", out.size(), pps, dtps,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                fclose(d);
                remove(rdy.c_str());
            }
            return 0;
        }
const char* ids_p = getenv("STEP37_SERVE_IDS");
        const char* out_p = getenv("STEP37_SERVE_OUT");
        const int n_gen = getenv("STEP37_SERVE_NGEN") ? atoi(getenv("STEP37_SERVE_NGEN")) : 256;
        if (!ids_p || !out_p) { std::printf("serve mode needs STEP37_SERVE_IDS/OUT\n"); return 2; }
        
        FILE* fi = fopen(ids_p, "rb");
        if (!fi) { std::printf("cannot open %s\n", ids_p); return 2; }
        std::vector<int32_t> ids;
        int32_t v;
        while (fread(&v, 4, 1, fi) == 1) ids.push_back(v);
        fclose(fi);
        if (ids.empty()) { std::printf("empty prompt\n"); return 2; }
        double pps = 0.0;
        if (!m.prefill_run(ids, pps, err)) { std::printf("PREFILL FAIL: %s\n", err.c_str()); return 1; }
        std::fprintf(stderr, "serve: prefill %zu tokens @ %.1f tok/s\n", ids.size(), pps);
        std::vector<int32_t> out;
        double dtps = 0.0;
        if (!m.generate(ids, n_gen, out, dtps, err, /*resume_prompt=*/(int)ids.size())) {
            std::printf("GEN FAIL: %s\n", err.c_str()); return 1;
        }
        FILE* fo = fopen(out_p, "wb");
        if (!fo) { std::printf("cannot write %s\n", out_p); return 2; }
        for (int32_t t : out) fwrite(&t, 4, 1, fo);
        fclose(fo);
        std::fprintf(stderr, "serve: %zu tokens generated\n", out.size());
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "--pf-gen") {
        // 联合档：prefill_run（批量）→ generate 接力 decode（KV/尾 logits 续用）
        const int n = argc > 3 ? std::atoi(argv[3]) : 16;
        const int plen = argc > 4 ? std::atoi(argv[4]) : 50000;
        const int32_t t0[32] = {4972, 79656, 127999, 713, 303, 48339, 16976, 429,
                                16976, 8648, 5835, 301, 46015, 303, 1555, 63192,
                                6479, 79025, 109921, 4597, 5506, 2220, 303, 872,
                                4372, 1877, 17085, 63155, 5581, 6664, 3905, 320};
        std::vector<int32_t> ids;
        for (int i = 0; i < plen; ++i) ids.push_back(t0[i % 32]);
        double pps = 0.0; std::string err;
        if (!m.prefill_run(ids, pps, err)) { std::printf("PREFILL FAIL: %s\n", err.c_str()); return 1; }
        std::printf("prefill: %d tokens @ %.1f tok/s\n", plen, pps);
        std::vector<int32_t> out; double dtps = 0.0;
        if (!m.generate(ids, n, out, dtps, err, /*resume_prompt=*/plen)) {
            std::printf("GEN FAIL: %s\n", err.c_str()); return 1;
        }
        std::printf("greedy out:");
        for (int t : out) std::printf(" %d", t);
        std::printf("\ndecode: %.2f tok/s (%d tokens, ctx %d)\n", dtps, (int)out.size(), plen);
        return 0;
    }
    if (argc < 3 || std::string(argv[2]) != "--gen") return 0;
    const int n = argc > 3 ? std::atoi(argv[3]) : 16;
    const int plen = argc > 4 ? std::atoi(argv[4]) : 32;
    // T0 参照 ids（bench/t0-reference.sh 采集自 stock）
    std::vector<int32_t> ids = {4972, 79656, 127999, 713, 303, 48339, 16976, 429,
                                16976, 8648, 5835, 301, 46015, 303, 1555, 63192,
                                6479, 79025, 109921, 4597, 5506, 2220, 303, 872,
                                4372, 1877, 17085, 63155, 5581, 6664, 3905, 320};
    ids.resize(plen);
    std::vector<int32_t> out;
    double tps = 0.0;
    try {
    if (!m.generate(ids, n, out, tps, err)) {
        std::printf("GEN FAIL: %s\n", err.c_str());
        return 1;
    }
    std::printf("greedy out:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\ndecode: %.2f tok/s (%d tokens)\n", tps, (int)out.size());
    return 0;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
}
