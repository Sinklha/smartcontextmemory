"""SCM Pro — серверный режим. Модель грузится ОДИН раз, дальше вопросы за мс.

Запуск:  python serve.py [--port 8000]
Вопрос:  python chat.py --remote "твой вопрос"
API:      POST /ask {"query": "...", "ask_ollama": false}
          POST /reindex {} | GET /health
Только stdlib, без новых зависимостей. Слушает 127.0.0.1.
"""
import sys
import os
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import chat

STATE = {"db": None, "docs": [], "skeletons": {},
         "total_raw": 0, "mode": "?", "model": None}
MAX_BODY = 1 << 20  # 1 МБ


def _json(handler, code, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    server_version = "SCMPro/1.0"

    def log_message(self, fmt, *args):  # тихий лог в одну строку
        sys.stdout.write(f"[HTTP] {self.command} {self.path}\n")

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return None

    def do_GET(self):
        if self.path == "/health":
            _json(self, 200, {"status": "ok",
                              "chunks": len(STATE["docs"]),
                              "mode": STATE["mode"],
                              "ollama": STATE["model"]})
        else:
            _json(self, 404, {"error": "unknown endpoint (GET /health)"})

    def do_POST(self):
        if self.path == "/ask":
            body = self._read_json()
            if not body or not str(body.get("query", "")).strip():
                _json(self, 400, {"error": "нужен JSON {'query': '...'} "})
                return
            q = str(body["query"]).strip()
            t0 = time.perf_counter()
            try:
                r = chat.retrieve(STATE["db"], STATE["docs"],
                                  STATE["skeletons"], STATE["total_raw"], q)
            except Exception as e:
                _json(self, 500, {"error": f"retrieve: {e}"})
                return
            out = {"query": q,
                   "retr_ms": round(r["retr_ms"], 1),
                   "found": r["found"],
                   "server_ms": round((time.perf_counter() - t0) * 1000, 1),
                   "hits": [{"score": round(s, 4), "file": m["file"],
                             "chunk": m["chunk_id"], "snippet": t[:300],
                             "entities": m.get("entities", []),
                             "has_negation": bool(m.get("has_negation"))}
                            for s, m, t in r["hits"]],
                   "ctx_bytes": r["ctx_bytes"], "tok_ctx": r["tok_ctx"],
                   "tok_raw": r["tok_raw"], "saving_pct": round(r["saving"], 1),
                   "used_files": r["used_files"], "context": r["context"]}
            if body.get("ask_ollama") and STATE["model"]:
                try:
                    resp, ms = chat.ollama_ask(STATE["model"], r["context"], q)
                    llm = {"model": STATE["model"], "ms": round(ms),
                           "text": resp}
                    # П.10: проверка ответом прямо на сервере.
                    try:
                        import scm_core as _scm
                        llm["verify"] = _scm.verify_answer(resp, r["context"])
                    except Exception as e:
                        llm["verify"] = {"error": str(e)[:200]}
                    out["llm"] = llm
                except Exception as e:
                    out["llm"] = {"error": str(e)}
            if body.get("verify_text"):
                try:
                    import scm_core as _scm
                    out["verify"] = _scm.verify_answer(str(body["verify_text"]),
                                                       r["context"])
                except Exception as e:
                    out["verify"] = {"error": str(e)[:200]}
            _json(self, 200, out)
        elif self.path == "/reindex":
            try:
                STATE["db"], STATE["mode"] = chat.build_db(STATE["docs"],
                                                           force_rebuild=True)
                _json(self, 200, {"status": "reindexed", "mode": STATE["mode"]})
            except Exception as e:
                _json(self, 500, {"error": f"reindex: {e}"})
        else:
            _json(self, 404, {"error": "unknown endpoint (/ask, /reindex)"})


def main():
    port = 8000
    if "--port" in sys.argv:
        try:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        except (ValueError, IndexError):
            pass
    print("--- SCM Pro server: прогрев (модель грузится один раз) ---")
    t0 = time.perf_counter()
    docs, skels, total = chat.load_files()
    db, mode = chat.build_db(docs)
    models = chat.ollama_models()
    STATE.update({"db": db, "docs": docs, "skeletons": skels,
                  "total_raw": total, "mode": mode,
                  "model": models[0] if models else None})
    print(f"[READY] {len(docs)} чанков, {mode}, ollama={STATE['model']} "
          f"за {(time.perf_counter()-t0):.0f}s -> http://127.0.0.1:{port}")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nСтоп.")


if __name__ == "__main__":
    main()
