"""SCM Pro — интерактивная консоль. Индексирует файлы папки, отвечает через real LLM (Ollama) если есть."""
import sys
import os
import time
import json
import urllib.request

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import scm_core

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX_FILES = ["input_data.txt", "scm_core.py", "main.py"]  # что индексируем
REPO_DIR = None  # если задан --repo PATH: индексируем папку вместо списка
REPO_EXTS = {".java", ".py", ".md", ".txt"}
SKIP_DIRS = {".git", "node_modules", "build", "target", "__pycache__",
             ".scm_index", ".gradle", ".idea", "out", "dist"}
MAX_FILES = 500
MAX_TOTAL_BYTES = 5 << 20  # 5 МБ честный предел демо-индекса
CHUNK = 800
K = 3
_INDEXED = []  # [(rel, abs)] последнего сбора — для отпечатка кэша


def collect_files():
    """Список файлов на индекс: папка --repo или 3 файла по умолчанию."""
    if not REPO_DIR:
        out = []
        for name in INDEX_FILES:
            p = os.path.join(BASE, name)
            if os.path.exists(p):
                out.append((name, p))
        return out
    out, total = [], 0
    for root, dirs, files in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in REPO_EXTS:
                continue
            ap = os.path.join(root, fn)
            try:
                sz = os.path.getsize(ap)
            except OSError:
                continue
            if len(out) >= MAX_FILES or total + sz > MAX_TOTAL_BYTES:
                continue
            total += sz
            out.append((os.path.relpath(ap, REPO_DIR), ap))
    return out


def load_files():
    global _INDEXED
    docs = []  # (file, chunk_id, text)
    skeletons = {}  # file -> skeleton_text
    total_raw = 0
    _INDEXED = collect_files()
    for rel, path in _INDEXED:
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        total_raw += len(content.encode("utf-8"))
        try:
            if path.endswith(".py"):
                skeletons[rel] = scm_core.ast_skeleton(content)["skeleton_text"]
            elif path.endswith(".java"):
                skeletons[rel] = scm_core.format_java_tree(
                    scm_core.java_parse(content))
        except Exception:
            pass
        for i in range(0, len(content), CHUNK):
            piece = content[i:i + CHUNK]
            if piece.strip():
                docs.append((rel, len([d for d in docs if d[0] == rel]), piece))
    return docs, skeletons, total_raw


INDEX_DIR = os.path.join(BASE, ".scm_index")
EMB_MODEL = "all-MiniLM-L6-v2"


def _index_fingerprint(docs):
    """Состав индексируемого: модель + файлы (имя/размер/mtime) + схема метаданных."""
    files = {}
    if REPO_DIR:
        for rel, ap in _INDEXED:
            try:
                st = os.stat(ap)
                files[rel] = [st.st_size, st.st_mtime_ns]
            except OSError:
                pass
        files[".repo"] = os.path.abspath(REPO_DIR)
    else:
        for name in INDEX_FILES:
            path = os.path.join(BASE, name)
            try:
                st = os.stat(path)
                files[name] = [st.st_size, st.st_mtime_ns]
            except OSError:
                pass
    return {"model": EMB_MODEL, "files": files, "chunks": len(docs),
            "graph_v": 1}  # bump при смене схемы метаданных -> пересборка


def _make_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMB_MODEL)


def build_db(docs, force_rebuild=False):
    """FAISS с кэшем на диске: пересборка только если файлы/модель поменялись."""
    import json as _json
    texts = [d[2] for d in docs]
    # FAISS-метаданные обогащены графом: сущности чанка для якорей к коду.
    metas = [{"file": d[0], "chunk_id": d[1],
              "entities": [e["name"] for e in
                           scm_core.extract_text_entities(d[2], top=8)]}
             for d in docs]
    fp = _index_fingerprint(docs)
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    t0 = time.perf_counter()
    try:
        emb = _make_embeddings()
    except Exception as e:
        print(f"[WARN] FAISS недоступен ({e}), работаю на keyword-поиске.")
        return None, "keyword-fallback"
    emb_ms = (time.perf_counter() - t0) * 1000

    if not force_rebuild and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                saved = _json.load(f)
            if saved == fp:
                from langchain_community.vectorstores import FAISS
                t1 = time.perf_counter()
                db = FAISS.load_local(INDEX_DIR, emb,
                                      allow_dangerous_deserialization=True)
                load_ms = (time.perf_counter() - t1) * 1000
                print(f"[INDEX] с диска за {load_ms:.0f}ms "
                      f"(+ модель {emb_ms:.0f}ms, {len(docs)} чанков, без пересборки)")
                return db, "faiss/" + EMB_MODEL + "+disk"
        except Exception as e:
            print(f"[INDEX] кэш бит ({e}), пересобираю...")

    try:
        from langchain_community.vectorstores import FAISS
        db = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metas)
        os.makedirs(INDEX_DIR, exist_ok=True)
        db.save_local(INDEX_DIR)
        with open(meta_path, "w", encoding="utf-8") as f:
            _json.dump(fp, f)
        print(f"[INDEX] построен за {(time.perf_counter()-t0)*1000:.0f}ms и сохранен на диск")
        return db, "faiss/" + EMB_MODEL
    except Exception as e:
        print(f"[WARN] FAISS недоступен ({e}), работаю на keyword-поиске.")
        return None, "keyword-fallback"


def keyword_search(docs, query, k=K):
    q = set(query.lower().split())
    scored = []
    for name, cid, text in docs:
        score = len(q & set(text.lower().split()))
        scored.append((score, name, cid, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(s, {"file": n, "chunk_id": c}, t) for s, n, c, t in scored[:k]]


def ollama_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_ask(model, system_ctx, query, timeout=90):
    body = json.dumps({
        "model": model,
        "prompt": f"Контекст проекта:\n{system_ctx}\n\nВопрос: {query}\nОтвет:",
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    ms = (time.perf_counter() - t0) * 1000
    return data.get("response", "(пусто)"), ms


def analyze_result(text, label="вставка"):
    """Считает анализ вставки, возвращает JSON-сериализуемый dict."""
    text = text.replace("\r\n", "\n")
    if not text.strip():
        return {"error": "empty", "label": label}
    raw_bytes = len(text.encode("utf-8"))
    tok_raw, tok_how = scm_core.estimate_tokens(text)

    code_p = scm_core.compress_code_block(text)
    tg = scm_core.build_text_graph(text)
    code_nodes = list(code_p.get("skeleton_nodes") or [])
    parsed = None
    if scm_core.is_java_like(text):
        try:
            parsed = scm_core.java_parse(_code_part(text))
            code_nodes = ([{"type": "class", "name": parsed["class"]}]
                          + [{"type": "method", "name": m["name"],
                              "parent": m["class"]} for m in parsed["methods"]]
                          + [{"type": "field", "name": f["name"]}
                              for f in parsed["fields"]])
        except Exception:
            parsed = None
    anchors = scm_core.anchor_to_code(tg, code_nodes) if tg["entities"] else []

    focus = None
    if parsed and parsed["methods"]:
        try:
            focus = scm_core.build_method_focus(parsed)
        except Exception:
            focus = None

    sk_bytes = code_p["skeleton_bytes"]
    out = {
        "label": label,
        "raw_bytes": raw_bytes, "tokens_raw": tok_raw,
        "tokens_how": tok_how,
        "skeleton": {"bytes": sk_bytes, "tokens": code_p["tokens_skeleton"],
                     "saving_pct": code_p["skeleton_saving_pct"],
                     "method": code_p["skeleton_method"],
                     "text": code_p["skeleton_text"]},
        "text_graph": {"entities": tg["entities"], "values": tg["values"],
                       "anchors": anchors},
        "code": None, "focus": None,
    }
    if parsed:
        out["code"] = {
            "class": parsed["class"], "parser": parsed.get("parser"),
            "partial": bool(parsed.get("has_error")),
            "methods": [{"name": m["name"], "params": m["params"],
                         "desc": scm_core._mdesc(m), "calls": m["calls"],
                         "imports": m["uses_imports"],
                         "fields": m["uses_fields"],
                         "event": m["event"]} for m in parsed["methods"]],
            "fields": parsed["fields"],
            "imports": [i["full"] for i in parsed["imports"]],
        }
    if focus:
        fb = focus["context_bytes"]
        out["focus"] = {
            "target": focus["target"], "bytes": fb,
            "tokens": focus["tokens"],
            "saving_pct": round((1 - fb / raw_bytes) * 100, 1) if raw_bytes else 0,
            "neighbors": focus["neighbors"],
            "dropped_neighbors": focus.get("dropped_neighbors", []),
            "imports": focus["imports"],
            "context": focus["context"],
        }
    return out


def analyze_paste(text, label="вставка"):
    """Компактный анализ: было -> стало + ключевая статистика."""
    r = analyze_result(text, label)
    if "error" in r:
        print("[ANALYZE] пусто, нечего считать.")
        return
    print(f"\n[ANALYZE] {r['label']}: было {r['raw_bytes']}B (~{r['tokens_raw']} tok)")
    s = r["skeleton"]
    print(f"[ANALYZE] скелет: {s['bytes']}B (~{s['tokens']} tok) = -{s['saving_pct']}% "
          f"[{s['method']}]")
    if r["focus"]:
        f = r["focus"]
        print(f"[ANALYZE] фокус ({f['target']}): {f['bytes']}B "
              f"(~{f['tokens']} tok) = -{f['saving_pct']}%")
    stats = []
    if r["code"]:
        c = r["code"]
        stats.append(f"методов: {len(c['methods'])}, "
                     f"полей: {len(c['fields'])}, "
                     f"импортов: {len(c['imports'])}")
    if r["text_graph"]["entities"]:
        t = r["text_graph"]
        stats.append(f"сущностей: {len(t['entities'])}, "
                     f"значений: {len(t['values'])}, "
                     f"якорей: {len(t['anchors'])}")
    if stats:
        print("[ANALYZE] " + " | ".join(stats))

    # --- Короткий показ дерева ---
    if r["code"]:
        print(f"[TREE] class {r['code']['class']}: "
              f"{len(r['code']['methods'])} методов "
              f"[{r['code']['parser']}]")
        for m in r["code"]["methods"][:15]:
            tail = f" [calls:{','.join(m['calls'])}]" if m["calls"] else ""
            desc = f"  # {m['desc']}" if m["desc"] else ""
            print(f"  {m['name']}({m['params']}): ...{desc}{tail}")
        if r["code"]["parser"] == "java-demo-regex":
            print("[TREE] разбор демо-regex (tree-sitter не справился).")
        elif r["code"]["partial"]:
            print("[TREE] частичный разбор (в коде есть ошибки синтаксиса).")


def paste_mode():
    print("[PASTE] вставляй свой код/текст. В конце отдельной строкой напиши END и Enter.")
    buf = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\n[PASTE] отменено.")
            return None
        if line.strip() == "END":
            break
        buf.append(line)
    return "\n".join(buf) + ("\n" if buf else "")


def _code_part(text):
    """Только кодовые строки вставки: прод-парсеру проза ломает дерево."""
    try:
        cl, _ = scm_core.smart_router(text)
        return "".join(cl) if cl else text
    except Exception:
        return text


def retrieve(db, docs, skeletons, total_raw, query):
    """Поиск + сборка контекста. Возвращает dict, ничего не печатает."""
    t0 = time.perf_counter()
    if db is not None:
        hits = db.similarity_search_with_score(query, k=K)
        norm = [(float(s), d.metadata, d.page_content) for d, s in hits]
    else:
        norm = keyword_search(docs, query)
    retr_ms = (time.perf_counter() - t0) * 1000

    # Контекст для LLM: скелеты затронутых .py + сырые чанки текстов
    used_files = []
    for _, meta, _ in norm:
        if meta["file"] not in used_files:
            used_files.append(meta["file"])
    ctx_parts = []
    for fn in used_files:
        if fn in skeletons and skeletons[fn]:
            ctx_parts.append(f"--- {fn} (SKELETON) ---\n{skeletons[fn]}")
        else:
            ctx_parts.append(f"--- {fn} (CHUNK) ---\n" + "\n".join(
                t[:500] for _, m, t in norm if m["file"] == fn)[:1500])
    context = "\n".join(ctx_parts)
    ctx_bytes = len(context.encode("utf-8"))
    tok_ctx, _ = scm_core.estimate_tokens(context)
    tok_raw, _ = scm_core.estimate_tokens(" ".join(d[2] for d in docs))
    saving = (1 - ctx_bytes / total_raw) * 100 if total_raw else 0
    return {"hits": norm, "context": context, "ctx_bytes": ctx_bytes,
            "tok_ctx": tok_ctx, "tok_raw": tok_raw, "saving": saving,
            "retr_ms": retr_ms, "used_files": used_files}


def answer_once(db, docs, skeletons, total_raw, query, model=None):
    r = retrieve(db, docs, skeletons, total_raw, query)
    norm, context = r["hits"], r["context"]
    retr_ms, tok_raw = r["retr_ms"], r["tok_raw"]
    ctx_bytes, tok_ctx, saving = r["ctx_bytes"], r["tok_ctx"], r["saving"]

    print(f"\n[RETRIEVE] {retr_ms:.0f}ms, top-{K}:")
    for score, meta, text in norm:
        ents = meta.get("entities") or []
        ent_str = f" | сущности: {', '.join(ents[:5])}" if ents else ""
        print(f"  score={score:.4f} {meta['file']}#{meta['chunk_id']}: {text[:100]!r}...{ent_str}")
    print(f"\n[METRICS] реальный размер всех файлов: {total_raw}B (~{tok_raw} tok)")
    print(f"[METRICS] контекст в LLM: {ctx_bytes}B (~{tok_ctx} tok) -> экономия {saving:.1f}%")
    print(f"[CONTEXT] (то, что уйдет в модель):\n{context[:1200]}")
    if len(context) > 1200:
        print("... [обрезано для экрана, полный контекст ушел бы в модель]")

    if model:
        print(f"\n[LLM] спрашиваю Ollama/{model} ...")
        try:
            resp, ms = ollama_ask(model, context, query)
            print(f"[LLM] ответ за {ms:.0f}ms:\n{resp}")
        except Exception as e:
            print(f"[LLM] ошибка: {e}")
    else:
        print("\n[LLM] Ollama не найдена (нет моделей на localhost:11434). "
              "Поставь Ollama + `ollama pull qwen2.5:3b`, и будет живой ответ. "
              "Контекст выше — честный, пощупать экономию уже можно.")


def remote_ask(query, port=8000):
    """Вопрос живому серверу (без прогрева модели): POST /ask."""
    import json as _json
    import urllib.request as _url
    body = _json.dumps({"query": query}).encode("utf-8")
    req = _url.Request(f"http://127.0.0.1:{port}/ask", data=body,
                       headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with _url.urlopen(req, timeout=120) as r:
            out = _json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"[REMOTE] сервер недоступен (запусти: python serve.py): {e}")
        return
    total_ms = (time.perf_counter() - t0) * 1000
    if "error" in out:
        print(f"[REMOTE] ошибка сервера: {out['error']}")
        return
    print(f"\n[RETRIEVE] {out['retr_ms']}ms на сервере, {total_ms:.0f}ms с сетью, top-{len(out['hits'])}:")
    for h in out["hits"]:
        print(f"  score={h['score']:.4f} {h['file']}#{h['chunk']}: {h['snippet'][:100]!r}...")
    print(f"\n[METRICS] контекст в LLM: {out['ctx_bytes']}B (~{out['tok_ctx']} tok) "
          f"-> экономия {out['saving_pct']}%")
    print(f"[CONTEXT]:\n{out['context'][:1200]}")
    if len(out["context"]) > 1200:
        print("... [обрезано для экрана]")


def analyze_batch(dirpath, as_json=False):
    """Пачка файлов за один вызов: общие итоги + per-file цифры и скелеты.

    Один битый файл пачку не роняет (считается в failed). Лимиты те же,
    что у индекса: MAX_FILES / MAX_TOTAL_BYTES, о пропуске пишется честно.
    """
    import json as _json
    if not os.path.isdir(dirpath):
        if as_json:
            print(_json.dumps({"error": "no such dir", "dir": dirpath}))
        else:
            print(f"[BATCH] нет папки: {dirpath}")
        return
    files, total, skipped = [], 0, 0
    for root, dirs, fns in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() not in REPO_EXTS:
                continue
            ap = os.path.join(root, fn)
            try:
                sz = os.path.getsize(ap)
            except OSError:
                continue
            if len(files) >= MAX_FILES or total + sz > MAX_TOTAL_BYTES:
                skipped += 1
                continue
            total += sz
            files.append(ap)
    items, raw_sum, sk_sum, fo_sum, failed = [], 0, 0, 0, 0
    for ap in files:
        rel = os.path.relpath(ap, dirpath)
        try:
            with open(ap, encoding="utf-8") as f:
                content = f.read()
            r = analyze_result(content, label=rel)
        except (OSError, UnicodeDecodeError, ValueError):
            failed += 1
            continue
        if "error" in r:
            failed += 1
            continue
        raw_sum += r["raw_bytes"]
        sk_sum += r["skeleton"]["bytes"]
        f = r.get("focus")
        if f:
            fo_sum += f["bytes"]
        c = r.get("code") or {}
        items.append({
            "file": rel, "raw_bytes": r["raw_bytes"],
            "tokens_raw": r["tokens_raw"],
            "skeleton": {"bytes": r["skeleton"]["bytes"],
                         "tokens": r["skeleton"]["tokens"],
                         "saving_pct": r["skeleton"]["saving_pct"],
                         "method": r["skeleton"]["method"],
                         "text": r["skeleton"]["text"]},
            "methods": len(c.get("methods", [])),
            "parser": (c.get("parser") if c else None),
            "focus": ({"target": f["target"], "bytes": f["bytes"],
                       "tokens": f["tokens"]} if f else None),
        })
    out = {"dir": os.path.abspath(dirpath), "files": len(items),
           "skipped_by_limit": skipped, "failed": failed,
           "raw_bytes": raw_sum, "tokens_raw": sum(i["tokens_raw"] for i in items),
           "skeleton_bytes": sk_sum,
           "skeleton_saving_pct": round((1 - sk_sum / raw_sum) * 100, 2) if raw_sum else 0,
           "focus_bytes": fo_sum, "items": items}
    if as_json:
        print(_json.dumps(out, ensure_ascii=False))
        return
    print(f"\n[BATCH] {out['dir']}: файлов: {len(items)} "
          f"(пропущено лимитом: {skipped}, битых: {failed})")
    print(f"[BATCH] было {raw_sum}B -> скелеты {sk_sum}B "
          f"= -{out['skeleton_saving_pct']}% | фокусы суммарно {fo_sum}B")
    for i in items:
        f = i["focus"]
        fl = f" | фокус {f['target']} {f['bytes']}B" if f else ""
        print(f"  {i['file']}: {i['raw_bytes']}B -> {i['skeleton']['bytes']}B "
              f"(-{i['skeleton']['saving_pct']}%) m={i['methods']}{fl}")


def main():
    global REPO_DIR
    as_json = "--json" in sys.argv
    sys.argv[:] = [a for a in sys.argv if a != "--json"]
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-file" and len(sys.argv) > 2:
        path = sys.argv[2]
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if as_json:
            import json as _json
            print(_json.dumps(analyze_result(content, label=path),
                              ensure_ascii=False))
        else:
            analyze_paste(content, label=path)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-batch" and len(sys.argv) > 2:
        analyze_batch(sys.argv[2], as_json=as_json)
        return

    port = 8000
    args = [a for a in sys.argv[1:] if a != "--reindex"]
    for _flag in ("--port", "--repo"):  # выкидываем флаги вместе со значениями
        if _flag in args:
            _i = args.index(_flag)
            if _flag == "--port":
                try:
                    port = int(args[_i + 1])
                except (ValueError, IndexError):
                    pass
            if _flag == "--repo":
                try:
                    REPO_DIR = args[_i + 1]
                except IndexError:
                    print("[INDEX] --repo требует путь к папке")
                    return
                if not os.path.isdir(REPO_DIR):
                    print(f"[INDEX] нет такой папки: {REPO_DIR}")
                    return
            del args[_i:_i + 2]

    if args and args[0] == "--remote":
        remote_ask(" ".join(args[1:]) or "Каковы лимиты памяти в коде?", port)
        return

    once = None
    if args and args[0] == "--once":
        once = " ".join(args[1:]) or "Каковы лимиты памяти в коде?"
    force = "--reindex" in sys.argv

    import contextlib as _ctx
    # В --json режиме stdout — только JSON: служебные строки уходят в stderr.
    _redir = _ctx.redirect_stdout(sys.stderr) if as_json else _ctx.nullcontext()
    with _redir:
        print("--- SCM Pro console ---")
        docs, skeletons, total_raw = load_files()
        src = f"папка {os.path.abspath(REPO_DIR)}" if REPO_DIR else "файлы проекта"
        print(f"[INDEX] {src}: файлов: {len(_INDEXED)}, чанков: {len(docs)}, сырье: {total_raw}B")
        for fn, sk in skeletons.items():
            print(f"  skeleton {fn}: {len(sk.encode('utf-8'))}B")
        db, mode = build_db(docs, force_rebuild=force)
        print(f"[INDEX] поиск: {mode}")
        models = ollama_models()
        model = models[0] if models else None
        print(f"[LLM] Ollama: {model if model else 'не найдена'}")

    if once is not None:
        if as_json:
            import json as _json
            r = retrieve(db, docs, skeletons, total_raw, once)
            print(_json.dumps({
                "query": once, "retr_ms": round(r["retr_ms"], 1),
                "hits": [{"score": round(s, 4), "file": m["file"],
                          "chunk": m.get("chunk_id"),
                          "entities": m.get("entities", []),
                          "snippet": t[:500]} for s, m, t in r["hits"]],
                "ctx_bytes": r["ctx_bytes"], "tok_ctx": r["tok_ctx"],
                "tok_raw": r["tok_raw"],
                "saving_pct": round(r["saving"], 1),
                "used_files": r["used_files"], "context": r["context"]},
                ensure_ascii=False))
        else:
            answer_once(db, docs, skeletons, total_raw, once, model)
        return

    print('Пиши вопрос, "exit" — выход.')
    print('"/code" — вставить свой код и посчитать экономию. "/help" — помощь.\n')
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "выход"):
            print("Пока.")
            break
        if q.lower() in ("/help", "help", "?"):
            print('  вопрос — поиск по файлам проекта\n  /code — вставить свой код (конец — строка END)\n  exit — выход')
            continue
        if q.lower() == "/code":
            pasted = paste_mode()
            if pasted is not None:
                analyze_paste(pasted)
            continue
        answer_once(db, docs, skeletons, total_raw, q, model)


if __name__ == "__main__":
    main()
