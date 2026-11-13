"""step37_serve.py — step37 干活壳 v2（2026-11-11 傍晚）
OpenAI 兼容 /v1/chat/completions @ :8081，常驻引擎 daemon + 前缀 KV 会话复用。

架构（v2，三件事同时成立）：
- 常驻引擎：Popen 一次 load（VRAM 常驻 ~26GB 可见），逐请求经文件信箱下发
  （req.bin 头行带采样参数 + token；done.txt 带 OK/RESET/ERR）。引擎崩了自动重启。
- 前缀 KV 会话复用：缓存上一请求的 ids+gen，新请求 diff 最长公共前缀 →
  只 prefill 增量（start_pos=前缀长）。多轮对话第二轮起省掉全部重算。
  引擎 RESET（KV 重分配）/ 崩溃 → 缓存作废自动全量重 prefill。
- 采样：temp=0.7 / top_p=0.95 / min_p=0.05 / rep=1.0(关) 逐请求下发；
  种子每请求随机；eos(<|im_end|>=128007) 早停；think 段剥离。

用法:
  python3 step37_serve.py [--port 8081] [--max-tokens 768] [--ctx 8192]
"""
import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE / "../../tools"
sys.path.insert(0, str(TOOLS.resolve()))
from strata_tokenizer import Tokenizer  # noqa: E402

# 路径按本机布局调整（示例为默认挂载点）
GGUF_DIR = os.environ.get("STEP37_GGUF_DIR", "GGUF_PLACEHOLDER_SET_ENV")
GGUF = GGUF_DIR
ENGINE = os.environ.get("STEP37_ENGINE", "build/step37-load")
PROF = os.environ.get("STEP37_PROF", "")

_tok = None
def tok():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_gguf(os.environ.get("STEP37_GGUF_FILE", "MODEL-FILE-PLACEHOLDER"))
        print(f"[step37] tokenizer ready: {len(_tok.tokens)} tokens, pre={_tok.pre}", flush=True)
    return _tok

_CT = None
_BOS_TXT = None
_BID = 0
def chat_ids(messages):
    """官方 Jinja chat_template 渲染。BOS = 元数据 bos_token_id（id 0 =
    '<｜begin▁of▁sentence｜>'；曾错用 '<|im_start|>' → 双重起始标记 → 角色回声循环）。
    修复后与 stock llama.cpp /tokenize 逐位全等（22/22，2026-11-11 验证）。"""
    global _CT, _BOS_TXT, _BID
    import gguf as _gguf
    from jinja2 import Template
    if _CT is None:
        r = _gguf.GGUFReader(os.environ.get("STEP37_GGUF_FILE", "MODEL-FILE-PLACEHOLDER"))
        _CT = bytes(r.fields['tokenizer.chat_template'].parts[-1]).decode('utf-8', errors='replace')
        f = r.fields.get('tokenizer.ggml.bos_token_id')
        _BID = int(f.parts[-1][0]) if f is not None else 0
    t = tok()
    if _BOS_TXT is None:
        _BOS_TXT = t.tokens[_BID] if 0 <= _BID < len(t.tokens) else ""
    rendered = Template(_CT).render(messages=messages, bos_token=_BOS_TXT,
                                    add_generation_prompt=True, tools=None)
    ids = t.encode(rendered, parse_special=True)
    # think 短路（env 可开）：官方模板把 '<think>\n' 留给模型生成——combine 层滞后 bug
    # 修复后 think 已连贯，默认关闭短路。
    if os.environ.get("STEP37_THINK_BYPASS", "0") == "1":
        ids.extend(t.encode("</think>\n", parse_special=False))
    return ids



# prefill 相全局信号量：双实例并发下 VRAM 超限（单请求活跃 +6.2GB，双活跃 34GB > 32.3）。
# prefill 串行（峰值 10.9+6.2=17.1GB ✓），decode 并行（活跃 ~0.3GB）。真并发度=2 保留：
# 两路 decode 同时跑，prefill 排队（单卡物理约束，诚实账）。
PREFILL_SEM = threading.BoundedSemaphore(1)

class EngineDaemon:
    """常驻引擎进程 + 文件信箱客户端 + 前缀会话账本。"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.mail = pathlib.Path(tempfile.mkdtemp(prefix="step37_mail_"))
        self.proc = None
        self.errlog = open("/tmp/step37_engine.log", "ab", buffering=0)
        self.cache_ids = []      # 已入 KV 的 token 序列（prompt + gen[:-1]）
        self.cache_valid = False
        self.lock = threading.Lock()
        self._spawn()

    def _spawn(self):
        # 双实例 262144 档：ub4096 + staging 双槽（铁律：单槽 = 预拷 memcpy 在 CPU 上
        # 串行喂料，GPU 饿死 10-15s/请求——maya 线的 staging ring 深度是重叠的前提，
        # 2026-11-12 用户实证"显卡占用0全在CPU"后回滚）+ SWA 环 4096。
        # 显存账：idle 21.8 + prefill 峰值 8.5（双槽 4.6 + 激活 3.9）≈ 30.3GB ✓
        # 专家池改 STEP37_POOL=1 选装（320 槽 2.65GB 会让峰值顶到 33GB，默认关）。
        env = dict(os.environ,
                   STEP37_GPU_TAIL="0", STEP37_UB="4096", STEP37_NO_ARENA="1",
                   STEP37_SWA_CAP="4096", STEP37_STAGE_SLOTS="2", STEP37_KV_CAP="262144",
                   STEP37_MAILBOX=str(self.mail),
                   **(dict(STEP37_POOL_PROFILE=PROF, STEP37_POOL_SLOTS="320",
                           STEP37_POOL_ADMIT="8")
                      if os.environ.get("STEP37_POOL", "0") == "1" else {}))
        self.proc = subprocess.Popen([ENGINE, GGUF, "--serve"],
                                     stdout=self.errlog, stderr=self.errlog, env=env)
        self.cache_valid = False     # 新进程 KV 为空
        print(f"[step37] engine daemon pid={self.proc.pid} mailbox={self.mail}", flush=True)

    def _restart(self):
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        for f in ("req.bin", "req.ready", "done.txt", "out.bin"):
            try:
                (self.mail / f).unlink()
            except FileNotFoundError:
                pass
        self._spawn()

    def _wait_done(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return "DEAD"
            if (self.mail / "done.txt").exists():
                try:
                    return (self.mail / "done.txt").read_text().strip()
                except Exception:
                    pass
            time.sleep(0.004)
        return "TIMEOUT"

    def infer(self, ids, n_gen, temperature, top_p):
        """一次推理：前缀 diff → 增量 prefill → generate。返回 (gen_ids, meta)。"""
        with self.lock:
            if self.proc.poll() is not None:
                print("[step37] engine died — restarting", flush=True)
                self._restart()
            if not self.cache_valid:
                start, delta = 0, ids
            else:
                c = self.cache_ids
                m = 0
                lim = min(len(c), len(ids))
                while m < lim and c[m] == ids[m]:
                    m += 1
                start, delta = m, ids[m:]
            if self.ctx and start + len(delta) + n_gen + 64 > self.ctx:
                start, delta = 0, ids      # 超容量：回退全量（prefill_run 自行扩容）

            status = "ERR"
            for attempt in (0, 1):
                try:
                    (self.mail / "done.txt").unlink()
                except FileNotFoundError:
                    pass
                def hdr(ntok, ngen, st, op):
                    return (f"{op} {ntok} {ngen} {st} "
                            f"{temperature if temperature is not None else 0.7:g} "
                            f"{top_p if top_p is not None else 0.95:g} "
                            f"{os.environ.get('STEP37_MIN_P', '0.05')} "
                            f"{os.environ.get('STEP37_REP_PEN', '1.0')} "
                            f"{os.environ.get('STEP37_TOP_K', '0')} "
                            f"{os.environ.get('STEP37_REP_WIN', '64')} "
                            f"128007 {random.randrange(1, 2**31 - 1)}\n")
                def send(op, toks, st):
                    req = self.mail / "req.bin"
                    with open(req, "wb") as f:
                        f.write(hdr(len(toks), n_gen if op != 1 else 0, st, op).encode())
                        f.write(b"".join(i.to_bytes(4, "little") for i in toks))
                    # 陈旧 done 清除必须在 rename（发布）之前——rename 后引擎才可能写新 done，
                    # 此窗口内删掉的只能是上一请求的残留（曾致 op=2 读到 op=1 的 "OK 0" /
                    # 下一请求读到上一请求的 "OK 194"+其 out.bin = 答非所问）
                    try:
                        (self.mail / "done.txt").unlink()
                    except FileNotFoundError:
                        pass
                    os.rename(req, self.mail / "req.ready")   # 原子发布
                    return self._wait_done(600)

                # 相 1：prefill（全局信号量——单卡 VRAM 只容一路活跃 prefill）
                with PREFILL_SEM:
                    status = send(1, delta, start)
                if status.startswith("ERR"):
                    print(f"[step37] {status} — cache invalid, full prefill retry", flush=True)
                    self.cache_valid = False
                    start, delta = 0, ids
                    continue
                if status in ("RESET", "DEAD", "TIMEOUT"):
                    print(f"[step37] {status} — restart + full prefill", flush=True)
                    self._restart()
                    start, delta = 0, ids
                    continue
                # 相 2：generate（并发友好：decode 活跃峰值仅 ~0.3GB）。
                # ⚠ 位置 bug 修复：ids 必须给全序列、base_pos=0（位置本就绝对；
                #   旧版 base_pos=start 把前缀偏移重复计入 decode 位置）
                status = send(2, ids, 0)
                if status.startswith("ERR"):
                    print(f"[step37] {status} — cache invalid, full prefill retry", flush=True)
                    self.cache_valid = False
                    start, delta = 0, ids
                    continue
                if status in ("RESET", "DEAD", "TIMEOUT"):
                    print(f"[step37] {status} — restart + full prefill", flush=True)
                    self._restart()
                    start, delta = 0, ids
                    continue
                break

            gen = []
            meta = {"status": status, "start": start, "delta": len(delta)}
            if status.startswith("OK"):
                raw = (self.mail / "out.bin").read_bytes()
                gen = [int.from_bytes(raw[i:i+4], "little") for i in range(0, len(raw), 4)]
                # 缓存 = prompt + gen[:-1]（最后一个 token 没喂 forward，无 KV 行）
                self.cache_ids = ids + gen[:-1]
                self.cache_valid = True
                try:
                    (self.mail / "done.txt").unlink()
                except FileNotFoundError:
                    pass
            else:
                self.cache_valid = False
            return gen, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--max-tokens", type=int, default=768)  # reasoning：think+正文都要空间
    ap.add_argument("--ctx", type=int, default=262144)      # 全窗口（全注意力层 q8 KV）
    ap.add_argument("--instances", type=int, default=2)     # 并发度：常驻引擎实例数
    args = ap.parse_args()

    engs = [EngineDaemon(args.ctx) for _ in range(args.instances)]
    locks = [threading.Lock() for _ in range(args.instances)]
    eng_lock = threading.Lock()
    eng_state = {"next": 0}

    def pick(ids=None):
        # 会话亲和优先：谁的缓存与请求共享最长前缀（≥8 tok）就粘谁——多轮对话
        # 第二轮起命中该实例 KV 前缀（否则轮转会把同一会话打散到无缓存的实例，
        # start=0 全量重 prefill）。无亲和 → 轮转抢占空闲。返回【已抢占】的实例与锁
        # （threading.Lock 不可重入，调用方 release，不得二次 acquire）。
        with eng_lock:
            nxt = eng_state["next"]
            order = [(nxt + k) % args.instances for k in range(args.instances)]
            eng_state["next"] = (nxt + 1) % args.instances
        if ids:
            best, best_len = None, 0
            for k in order:
                e = engs[k]
                if e.cache_valid and e.cache_ids:
                    lim = min(len(e.cache_ids), len(ids))
                    m = 0
                    while m < lim and e.cache_ids[m] == ids[m]:
                        m += 1
                    if m > best_len:
                        best, best_len = k, m
            if best is not None and best_len >= 8:
                lk = locks[best]
                lk.acquire()
                return best, lk
        for k in order:
            lk = locks[k]
            if lk.acquire(blocking=False):
                return k, lk
        k = order[0]
        lk = locks[k]
        lk.acquire()
        return k, lk

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        def do_GET(self):
            if self.path == "/v1/models":
                alive = all(e.proc.poll() is None for e in engs)
                self._json(200, {"object": "list", "data": [
                    {"id": "step37-heretic", "object": "model",
                     "status": {"value": "loaded" if alive else "loading"},
                     "instances": len(engs)}]})
            else:
                self._json(404, {"error": "not found"})
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            msgs = req.get("messages", [])
            max_tokens = int(req.get("max_tokens") or args.max_tokens)
            if not msgs:
                return self._json(400, {"error": "messages required"})
            t0 = time.time()
            ids = chat_ids(msgs)
            if len(ids) + max_tokens + 64 > args.ctx:
                return self._json(400, {"error": f"context too long: {len(ids)} tok > ctx {args.ctx}"})
            k, lk = pick(ids)        # 亲和路由；返回【已持有】的锁（不可重入，勿再 acquire）
            try:
                gen, meta = engs[k].infer(ids, max_tokens, req.get("temperature"), req.get("top_p"))
            finally:
                lk.release()
            meta["inst"] = k
            dt = time.time() - t0
            text = tok().decode(gen)
            cut = text.find("<|im_end|>")
            if cut >= 0:
                text = text[:cut]
            # reasoning 模型：think 段 → reasoning_content（chatdeck 按 maya/GLM 语义计数），
            # 正文 = '</think>' 之后
            think_txt = ""
            if "</think>" in text:
                think_txt, _, text = text.partition("</think>")
                text = text.strip()
            else:
                text = text.strip()

            if req.get("stream"):
                # SSE 重放流：引擎批量出全量，按小片回放（TTFT=引擎时长；chatdeck stall_s=280）
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                cid = f"chatcmpl-step37-{random.randrange(1, 10**9)}"
                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                def chunk(delta, finish=None, usage=None):
                    o = {"id": cid, "object": "chat.completion.chunk", "model": "step37-heretic",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if usage is not None:
                        o["usage"] = usage
                    sse(o)
                try:
                    chunk({"role": "assistant"})
                    for i in range(0, len(think_txt), 12):
                        chunk({"reasoning_content": think_txt[i:i+12]})
                        self.wfile.flush(); time.sleep(0.012)
                    if think_txt:
                        chunk({"reasoning_content": ""})   # think 段闭合信号
                    for i in range(0, len(text), 12):
                        chunk({"content": text[i:i+12]})
                        self.wfile.flush(); time.sleep(0.012)
                    usage = {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                             "total_tokens": len(ids) + len(gen)}
                    chunk({}, finish="stop", usage=usage)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                print(f"[step37] {len(ids)} tok prompt → {len(gen)} gen in {dt:.1f}s "
                      f"(stream replay)  [inst={meta.get('inst')} kv-reuse start={meta['start']} "
                      f"delta={meta['delta']} {meta['status']}]", flush=True)
                return
            print(f"[step37] think {len(think_txt)} chars", flush=True)
            print(f"[step37] {len(ids)} tok prompt → {len(gen)} gen in {dt:.1f}s "
                  f"({len(gen)/max(dt,0.01):.1f} tok/s)  [inst={meta.get('inst')} "
                  f"kv-reuse start={meta['start']} delta={meta['delta']} {meta['status']}]", flush=True)
            self._json(200, {
                "id": "step37", "object": "chat.completion", "model": "step37-heretic",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text,
                                             "reasoning_content": think_txt}}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                          "total_tokens": len(ids) + len(gen)},
                "timing": {"engine_s": dt, "kv_reuse_from": meta["start"], "delta_tok": meta["delta"]}})

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"[step37] serving :{args.port} (daemon×{args.instances}, ctx={args.ctx}, "
          f"ub=4096 swa_cap=4096 q8kv, temp={os.environ.get('STEP37_TEMP', '0.7')} "
          f"min_p={os.environ.get('STEP37_MIN_P', '0.05')})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
