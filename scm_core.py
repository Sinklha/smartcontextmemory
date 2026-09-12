"""Smart Context Memory — ядро.

Честные правила, зафиксированные в этом модуле:
1. Размеры считаются через len(bytes), а НЕ через sys.getsizeof.
2. Разделяем две разные экономии:
   - context_saving: что реально увидит LLM (семантически очищенный текст);
   - storage_saving: сколько байт лежит в RAM (zlib-поверх очищенного).
   zlib-байты модель читать не может — перед генерацией нужен декомпресс.
3. VRAM здесь — СИМУЛЯЦИЯ на dict. Реальный KV-cache / cuda-память
   этот класс не трогает. Если нужен реальный замер — смотри main.py
   (torch.cuda.memory_allocated, если CUDA доступна).
"""

import zlib
import hashlib
import os
import sys
import ast
import asyncio
import time
import re

# Windows cp1251 падает на эмодзи — форсим utf-8 с заменой символов.
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def bytes_len(text: str) -> int:
    """Честный размер строки в байтах (utf-8)."""
    return len(text.encode("utf-8"))


def estimate_tokens(text: str) -> tuple[int, str]:
    """Грубая оценка токенов для LLM.

    Пытается использовать tiktoken (если установлен), иначе эвристика ~4 символа.
    Возвращает (кол-во, метод).
    """
    if not text:
        return 0, "empty"
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken/cl100k"
    except Exception:
        return max(1, len(text) // 4), "heuristic/chars/4"


# ---------------------------------------------------------------------------
# МОДУЛЬ 1: Semantic Router
# ---------------------------------------------------------------------------

_CODE_LINE_RE = re.compile(
    r"^\s*(def |class |import |from |return\b|public\b|private\b|protected\b"
    r"|static\b|void\b|int\b|if\b|for\b|while\b|try:|except|@\w+|System\.out|sqlite3)"
)
_CODE_SYMBOLS_RE = re.compile(r"[{};]")
_PY_FUNC_RE = re.compile(r"^\s*\w[\w\.]*\s*\(.*\)\s*:\s*(#.*)?$")


def _is_code_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _CODE_LINE_RE.match(line):
        return True
    if _CODE_SYMBOLS_RE.search(s):
        return True
    if _PY_FUNC_RE.match(line) and len(s.split()) <= 8:
        return True
    # типичный python-код с отступом и короткой строкой: "    pass", "    return ..."
    if line.startswith(("    ", "\t")) and len(s.split()) <= 6 and (
        "(" in s or s in ("pass",) or s.startswith(("return ", "import ", "from "))
    ):
        return True
    return False


def smart_router(raw_data: str) -> tuple[list[str], list[str]]:
    """Делит входной текст на строки кода и строки документации.

    Эвристика, не ML-классификатор. Ограничения честно:
    смешанные строки (код + комментарий в одной строке) уходят в код.
    """
    code_lines: list[str] = []
    text_lines: list[str] = []
    for line in raw_data.splitlines(keepends=True):
        if _is_code_line(line):
            code_lines.append(line)
        else:
            text_lines.append(line)
    return code_lines, text_lines


# ---------------------------------------------------------------------------
# МОДУЛЬ 2: Нормальное (семантическое + lossless) сжатие
# ---------------------------------------------------------------------------

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def semantic_clean_code(code: str) -> str:
    """Семантическая очистка кода — то, что реально экономит контекст LLM.

    - вырезает /*...*/ блоки,
    - удаляет полно-строчные # и // комментарии,
    - режет инлайн-комментарии вида '  # ...' и '  // ...',
    - убирает висячие пробелы и пустые строки.
    Структура кода сохраняется, поведение — нет гарантии для строковых
    литералов с '#' внутри (честное ограничение эвристики).
    """
    code = _BLOCK_COMMENT_RE.sub("", code)
    out: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        # инлайн-комментарии: только если перед # или // есть пробел
        # (чтобы не резать 'port=5432' или '://').
        cut = len(line)
        for marker in ("  #", " #", "\t#", "  //", " //", "\t//"):
            idx = line.find(marker)
            if idx != -1 and idx < cut:
                # не режем внутри кавычек грубо: считаем кавычки до маркера
                prefix = line[:idx]
                if prefix.count('"') % 2 == 0 and prefix.count("'") % 2 == 0:
                    cut = idx
        line = line[:cut].rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out) + ("\n" if out else "")


def compress_code_block(code_lines) -> dict:
    """Сжимает код: сначала семантика (для LLM), потом zlib (для RAM).

    Принимает list[str] или str. Возвращает dict с честной разбивкой метрик.
    """
    raw_text = "".join(code_lines) if isinstance(code_lines, list) else code_lines
    raw_bytes = bytes_len(raw_text)

    t0 = time.perf_counter()
    cleaned = semantic_clean_code(raw_text)
    cleaned_bytes = bytes_len(cleaned)
    compressed = zlib.compress(cleaned.encode("utf-8"), level=9)
    encode_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    # контрольная декомпрессия: доказываем lossless поверх очищенного
    assert zlib.decompress(compressed).decode("utf-8") == cleaned
    decode_ms = (time.perf_counter() - t1) * 1000

    semantic_saving = (1 - cleaned_bytes / raw_bytes) * 100 if raw_bytes else 0.0
    storage_saving = (1 - len(compressed) / raw_bytes) * 100 if raw_bytes else 0.0
    tok_raw, tok_method = estimate_tokens(raw_text)
    tok_clean, _ = estimate_tokens(cleaned)

    # AST-скелет поверх очищенного: это кандидат №1 для LLM-контекста,
    # полное тело едет вспышкой только по запросу.
    sk = ast_skeleton(cleaned)

    # Java-код: скелет строит прод-парсер (tree-sitter, фолбэк — демо-regex),
    # потому что он видит методы, поля и связи, а не только строки.
    # Смешанные файлы (Java + Python в одном): прод — база, недостающие
    # имена из regex-фолбэка добавляются с пометкой [regex], чтобы не терять.
    if is_java_like(cleaned):
        try:
            # Прод-парсеру отдаем только кодовые строки: проза ломает
            # верхний уровень дерева, а фолбэк теряет точность.
            _code_lines, _ = smart_router(cleaned)
            _jp = java_parse("".join(_code_lines) if _code_lines else cleaned)
            if _jp["methods"]:
                _names = {_m["name"] for _m in _jp["methods"]}
                _extra_lines, _extra_nodes = [], []
                for _n in ast_skeleton(cleaned)["nodes"]:
                    if _n.get("type") in ("method", "function") \
                            and _n["name"] not in _names:
                        _names.add(_n["name"])
                        _extra_lines.append(f"  {_n['name']}(...): ... [regex]")
                        _extra_nodes.append({"type": _n["type"],
                                             "name": _n["name"],
                                             "parent": _jp["class"]})
                _sk_text = format_java_tree(_jp)
                _tag = _jp["parser"]
                if _extra_lines:
                    _sk_text += "\n".join(_extra_lines) + "\n"
                    _tag += "+regex"
                _sk_bytes = bytes_len(_sk_text)
                _tok_sk, _ = estimate_tokens(_sk_text)
                _nodes = [{"type": "class", "name": _jp["class"]}]
                _nodes += [{"type": "import", "name": i["simple"]}
                           for i in _jp["imports"]]
                for _m in _jp["methods"]:
                    _nodes.append({"type": "method", "name": _m["name"],
                                   "parent": _m["class"]})
                _nodes += _extra_nodes
                sk = {"skeleton_text": _sk_text, "nodes": _nodes,
                      "method": _tag, "skeleton_bytes": _sk_bytes,
                      "saving_pct": round((1 - _sk_bytes / raw_bytes) * 100, 2)
                      if raw_bytes else 0.0, "tokens_skeleton": _tok_sk}
        except Exception:
            pass  # остался скелет от ast_skeleton — честный фолбэк

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,  # <-- полное тело (вспышка по запросу)
        "skeleton_text": sk["skeleton_text"],  # <-- коротко (в LLM по умолчанию)
        "skeleton_nodes": sk["nodes"],
        "skeleton_method": sk["method"],
        "skeleton_bytes": sk["skeleton_bytes"],
        "skeleton_saving_pct": sk["saving_pct"],
        "tokens_skeleton": sk["tokens_skeleton"],
        "compressed": compressed,  # <-- это лежит в RAM
        "raw_bytes": raw_bytes,
        "cleaned_bytes": cleaned_bytes,
        "compressed_bytes": len(compressed),
        "semantic_saving_pct": round(semantic_saving, 2),
        "storage_saving_pct": round(storage_saving, 2),
        "tokens_raw": tok_raw,
        "tokens_cleaned": tok_clean,
        "tokens_method": tok_method,
        "encode_ms": round(encode_ms, 3),
        "decode_ms": round(decode_ms, 3),
    }


def decompress_code_block(payload: dict) -> str:
    """Обратная операция: zlib -> очищенный код (то, что кормим модели)."""
    return zlib.decompress(payload["compressed"]).decode("utf-8")


# ---------------------------------------------------------------------------
# AST-скелет: граф вместо токенов (уровень 2 сжатия контекста)
# ---------------------------------------------------------------------------

_JAVA_CLASS_RE = re.compile(r"^\s*(public\s+|private\s+|protected\s+)?(class|interface)\s+(\w+)")
_JAVA_METHOD_RE = re.compile(
    r"^\s*(public\s+|private\s+|protected\s+|static\s+[\w<>\[\]]+\s+|[\w<>\[\]]+\s+)"
    r"(\w+)\s*\(([^)]*)\)\s*(\{|\;)?"
)

def _args_str(args: ast.arguments) -> str:
    parts = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    parts += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    # дефолты честно не разворачиваем полностью — только их количество
    if args.defaults or args.kw_defaults:
        return ", ".join(parts) + ", ..."
    return ", ".join(parts)


def ast_skeleton(code: str) -> dict:
    """Строит скелет кода: только сигнатуры, без тел функций.

    Python -> настоящий ast. Java/C-подобный -> regex-фолбэк.
    Возвращает скелет-текст (это едет в LLM для поиска),
    полное тело подгружается вспышкой отдельно.
    """
    raw_bytes = bytes_len(code)
    nodes: list[dict] = []
    lines: list[str] = []

    # --- попытка 1: настоящий Python AST ---
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ", ".join(a.asname or a.name for a in node.names)[:80]
                lines.append(f"import {names}")
                nodes.append({"type": "import", "name": names})
            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)[:60] if node.bases else ""
                decl = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
                lines.append(decl)
                nodes.append({"type": "class", "name": node.name})
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        doc = ast.get_docstring(item)
                        first = f"  # {doc.splitlines()[0][:60]}" if doc else ""
                        lines.append(f"  def {item.name}({_args_str(item.args)}): ...{first}")
                        nodes.append({"type": "method", "name": item.name,
                                      "parent": node.name})
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        lines.append(f"  {item.target.id}: ...")
                        nodes.append({"type": "field", "name": item.target.id,
                                      "parent": node.name})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                first = f"  # {doc.splitlines()[0][:60]}" if doc else ""
                lines.append(f"def {node.name}({_args_str(node.args)}): ...{first}")
                nodes.append({"type": "function", "name": node.name})
            elif isinstance(node, ast.Assign):
                targets = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))[:60]
                if targets:
                    lines.append(f"{targets} = ...")
                    nodes.append({"type": "const", "name": targets})
        method = "ast"
    else:
        # --- попытка 2: regex для Java/C (честный фолбэк, не полный парсер) ---
        found = False
        for line in code.splitlines():
            m = _JAVA_CLASS_RE.match(line)
            if m:
                lines.append(f"class {m.group(3)}:")
                nodes.append({"type": "class", "name": m.group(3)})
                found = True
                continue
            m = _JAVA_METHOD_RE.match(line)
            if m and len(line.strip()) < 120:
                lines.append(f"  {m.group(2)}({m.group(3).strip()[:60]}): ...")
                nodes.append({"type": "method", "name": m.group(2)})
                found = True
                continue
            s = line.strip()
            if s.startswith(("import ", "from ")) and len(s) < 120:
                lines.append(s)
                nodes.append({"type": "import", "name": s[:60]})
                found = True
        method = "regex-fallback" if found else "none"

    skeleton = "\n".join(lines) + ("\n" if lines else "")
    sk_bytes = bytes_len(skeleton)
    saving = (1 - sk_bytes / raw_bytes) * 100 if raw_bytes and skeleton else 0.0
    tok_raw, _ = estimate_tokens(code)
    tok_sk, tok_method = estimate_tokens(skeleton)
    return {
        "skeleton_text": skeleton,
        "nodes": nodes,
        "method": method,  # ast | regex-fallback | none
        "raw_bytes": raw_bytes,
        "skeleton_bytes": sk_bytes,
        "saving_pct": round(saving, 2),
        "tokens_raw": tok_raw,
        "tokens_skeleton": tok_sk,
        "tokens_method": tok_method,
    }


# ---------------------------------------------------------------------------
# Java-демо-парсер: дерево + связи (DEMO на регулярках, прод — ниже в чате)
# ---------------------------------------------------------------------------

_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(static\s+)?([\w\.]+)\s*;\s*$")
_JAVA_FIELD_RE = re.compile(
    r"^\s*(private|protected|public)?\s*(static\s+)?(final\s+)?"
    r"([\w\.<>\[\], ]+?)\s+(\w+)\s*(=\s*[^;]+)?;\s*$"
)
_JAVA_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "new",
                  "super", "this", "class", "void", "int", "long", "double",
                  "float", "boolean", "char", "byte", "short"}


def is_java_like(text: str) -> bool:
    return ("import " in text and ";" in text
            and "class " in text and "{" in text)


def _strip_java_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    out = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("//") or s.startswith("*"):
            continue
        out.append(line)
    return "\n".join(out)


def _java_simple_name(dotted: str) -> str:
    return dotted.split(".")[-1] if dotted else ""


def java_demo_parse(code: str) -> dict:
    """DEMO-парсер Java (только stdlib): импорты, классы, методы с телами,
    поля, описания из javadoc-строки выше, связи calls/uses-field/uses-import.

    Ограничения (честно): '{' на новой строке, лямбды, вложенные классы
    и сложный синтаксис разбираются грубо или пропускаются.
    Прод-версия: java_prod_parse() ниже (настоящий синтаксис через tree-sitter).
    """
    lines = code.splitlines()
    imports: list[dict] = []
    for line in lines:
        m = _JAVA_IMPORT_RE.match(line)
        if m:
            full = m.group(2)
            imports.append({"full": full, "simple": _java_simple_name(full)})

    # описания: javadoc-однострочники /** ... */ и // над сигнатурой
    pending_desc = ""
    pending_ann: list[str] = []
    cls_name = ""
    for m in (_JAVA_CLASS_RE.match(l) for l in lines):
        if m:
            cls_name = m.group(3)
            break

    methods: list[dict] = []
    fields: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if s.startswith("/**") and s.endswith("*/") and len(s) > 6:
            pending_desc = s[3:-2].strip()[:120]
            i += 1
            continue
        if s.startswith("@"):
            pending_ann.append(s.split("(")[0][:40])
            i += 1
            continue
        if not s or s.startswith("//") or s.startswith("*"):
            i += 1
            continue
        # поле: ';' на конце, без '{' (инициализатор вроде new X<>() допустим)
        if s.endswith(";") and "{" not in s and cls_name:
            fm = _JAVA_FIELD_RE.match(line)
            if fm and len(s) < 160:
                fields.append({"name": fm.group(5), "type": fm.group(4)[:40]})
            i += 1
            pending_desc, pending_ann = "", []
            continue
        # метод: есть '(' ... ')' ... '{' на той же строке
        if "(" in s and ")" in s and "{" in s and cls_name:
            name_m = re.search(r"(\w+)\s*\([^)]*\)\s*(throws\s+[\w,\s]+)?\{\s*\}?\s*$", s)
            one_liner = False
            if not name_m and s.count("{") == s.count("}") and s.rstrip().endswith("}"):
                # однострочник с телом: private void g() { ...; }
                name_m = re.search(r"(\w+)\s*\([^)]*\)\s*\{", s)
                one_liner = name_m is not None
            if name_m and name_m.group(1) not in _JAVA_KEYWORDS:
                mname = name_m.group(1)
                params = s[s.find("(") + 1:s.find(")")][:100]
                if one_liner:
                    body, j = line, i + 1
                else:
                    # тело по балансу скобок
                    depth = 0
                    body_lines: list[str] = []
                    j = i
                    started = False
                    while j < n:
                        for ch in lines[j]:
                            if ch == "{":
                                depth += 1
                                started = True
                            elif ch == "}":
                                depth -= 1
                        body_lines.append(lines[j])
                        j += 1
                        if started and depth <= 0:
                            break
                        if j - i > 600:  # страховка от рассинхрона
                            break
                    body = "\n".join(body_lines)
                methods.append({
                    "name": mname, "class": cls_name,
                    "params": params, "annotations": list(pending_ann),
                    "desc": pending_desc,
                    "sig": f"{mname}({params}): ...",
                    "body": body, "body_bytes": bytes_len(body),
                })
                i = j
                pending_desc, pending_ann = "", []
                continue
        # пустая/прочая строка сбрасывает только аннотации через разрыв
        if s in ("{", "}"):
            pending_ann = []
        i += 1

    names = [m["name"] for m in methods]
    fnames = [f["name"] for f in fields]
    for m in methods:
        body = m["body"]
        m["calls"] = sorted({w for w in re.findall(r"(\w+)\s*\(", body)
                             if w in names and w != m["name"]})[:10]
        m["uses_fields"] = sorted({w for w in re.findall(r"\b\w+\b", body)
                                   if w in fnames})[:10]
        m["uses_imports"] = sorted(
            {imp["full"] for imp in imports
             if re.search(r"\b" + re.escape(imp["simple"]) + r"\b",
                          m["sig"] + body)})[:8]
        m["event"] = ""
        if any("EventHandler" in a for a in m["annotations"]):
            pm = re.match(r"\s*([\w<>\[\]]+)\s+\w+\s*$", m["params"].strip())
            if pm:
                m["event"] = pm.group(1)[:60]

    # Автоописания (ступень 2: детерминированная эвристика, без LLM —
    # ничего не выдумывает, только факты из разбора).
    _java_auto_descs(methods)

    return {"class": cls_name, "imports": imports, "methods": methods,
            "fields": fields, "parser": "java-demo-regex"}


def _java_auto_descs(methods: list[dict]) -> None:
    """Проставляет m['auto_desc'] по фактам: событие, вызовы, поля. Без LLM."""
    for m in methods:
        bits = []
        if m["event"]:
            bits.append(f"обработчик {m['event']}")
        if m["calls"]:
            bits.append("вызывает: " + ", ".join(m["calls"]))
        if m["uses_fields"]:
            bits.append("поля: " + ", ".join(m["uses_fields"]))
        m["auto_desc"] = "; ".join(bits) if bits else "листовой метод"


def _java_desc_above(lines: list[str], row: int) -> str:
    """Описание над строкой row (0-based): /** ... */ однострочник или // блок."""
    j = row - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j < 0:
        return ""
    s = lines[j].strip()
    if s.startswith("/**") and s.endswith("*/") and len(s) > 6:
        return s[3:-2].strip()[:120]
    buf: list[str] = []
    while j >= 0 and lines[j].strip().startswith("//"):
        buf.append(lines[j].strip()[2:].strip())
        j -= 1
    buf.reverse()
    return " ".join(b for b in buf if b)[:120]


_TS_LANG = None
_TS_PARSER = None
_TS_QUERIES: dict[str, object] = {}


def _ts_java():
    """Ленивый tree-sitter-java. Бросает исключение, если пакет не установлен."""
    global _TS_LANG, _TS_PARSER
    if _TS_PARSER is None:
        import tree_sitter_java as tsjava
        from tree_sitter import Language, Parser
        _TS_LANG = Language(tsjava.language())
        _TS_PARSER = Parser(_TS_LANG)
    return _TS_LANG, _TS_PARSER


def _ts_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _ts_field_text(src: bytes, node, field: str) -> str:
    child = node.child_by_field_name(field)
    return _ts_text(src, child) if child is not None else ""


def java_prod_parse(code: str) -> dict:
    """ПРОД-парсер Java на tree-sitter: настоящий синтаксис, а не регулярки.

    Понимает Java 16+ (pattern matching instanceof, record), '{' на новой
    строке, лямбды; методы вложенных классов внешнему не приписывает.
    Битый код не роняет: помечается has_error=True, разбор продолжается.
    Выход — тот же формат, что у java_demo_parse (списки calls/uses_imports
    здесь полные, без обрезки).

    Надежность: свежие Windows-сборки py-tree-sitter иногда роняют процесс
    (access violation при массовых разборах — баг нативного слоя, не наш).
    Поэтому разбор едет в отдельном воркере (_ts_worker.py): его падение
    превращается в исключение и фолбэк, а не в смерть хоста. Плюс кэш
    разобранных деревьев на диске (.scm_index/trees) — повторы бесплатны.
    Бросает исключение, если разобрать нельзя, — вызывай через java_parse().
    """
    import hashlib as _hl
    import json as _json
    import subprocess as _sp
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             ".scm_index", "trees")
    key = _hl.sha1(code.encode("utf-8")).hexdigest() + ".json"
    cp = os.path.join(cache_dir, key)
    if os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                hit = _json.load(f)
            if isinstance(hit, dict) and hit.get("parser") == "tree-sitter-java":
                return hit
        except Exception:
            pass
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_ts_worker.py")
    try:
        r = _sp.run([sys.executable, worker], input=code.encode("utf-8"),
                    capture_output=True, timeout=60)
    except Exception as e:
        raise RuntimeError(f"ts-worker: {e}")
    if r.returncode != 0:
        raise RuntimeError(f"ts-worker died: {r.stderr.decode()[:200]}")
    try:
        out = _json.loads(r.stdout.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"ts-worker bad json: {e}")
    if not isinstance(out, dict) or "methods" not in out:
        raise RuntimeError("ts-worker bad shape")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as f:
            _json.dump(out, f, ensure_ascii=False)
        _ts_cache_evict(cache_dir)
    except Exception:
        pass
    return out


def _ts_cache_evict(cache_dir: str, limit: int = 500) -> None:
    """Чтобы кэш деревьев не рос бесконечно: держим свежие 500."""
    try:
        files = [(os.path.getmtime(os.path.join(cache_dir, n)), n)
                 for n in os.listdir(cache_dir) if n.endswith(".json")]
    except OSError:
        return
    if len(files) > limit:
        for _, n in sorted(files)[:len(files) - limit]:
            try:
                os.remove(os.path.join(cache_dir, n))
            except OSError:
                pass


def _java_prod_parse_local(code: str) -> dict:
    from tree_sitter import Query, QueryCursor

    lang, parser = _ts_java()
    src = code.encode("utf-8")
    lines = code.splitlines()
    tree = parser.parse(src)
    root = tree.root_node

    def caps(pattern: str, node):
        # Свежий Query на вызов: переиспользование скомпилированных запросов
        # на подузлах роняло процесс (баг нативного слоя на Windows).
        return QueryCursor(Query(lang, pattern)).captures(node)

    # --- импорты ---
    imports: list[dict] = []
    for n in caps("(import_declaration) @i", root).get("i", []):
        t = _ts_text(src, n).strip()
        full = t[len("import"):].strip().rstrip(";").strip()
        if full.startswith("static "):
            full = full[len("static "):].strip()
        full = full[:-2] if full.endswith(".*") else full
        if full:
            imports.append({"full": full, "simple": _java_simple_name(full)})

    # --- класс (первый верхнего уровня): class / record / interface / enum ---
    cls_node = None
    cls_name = ""
    for child in root.children:
        if child.type in ("class_declaration", "record_declaration",
                          "interface_declaration", "enum_declaration"):
            name_n = child.child_by_field_name("name")
            if name_n is not None:
                cls_node, cls_name = child, _ts_text(src, name_n)
                break
    if cls_node is None:
        raise ValueError("класс не найден")

    declared_types = {cls_name}
    for c in caps("(record_declaration name: (identifier) @r)", cls_node).get("r", []):
        declared_types.add(_ts_text(src, c))

    def _owned_walk():
        """Свежий обход методов класса (без вложенных типов).

        Узлы оберткок нативного дерева НЕ храним между стадиями: повторное
        чтение старых оберток роняло процесс (баг нативного слоя Windows).
        Каждый проход — заново от cls_node, тексты забираем сразу.
        """
        out: list = []
        start = cls_node.child_by_field_name("body") or cls_node
        stack = [start]
        while stack:
            node = stack.pop()
            for ch in node.children:
                if ch.type in ("class_declaration", "record_declaration",
                               "interface_declaration", "enum_declaration",
                               "annotation_type_declaration"):
                    continue
                if ch.type == "method_declaration":
                    out.append(ch)
                else:
                    stack.append(ch)
        return out

    # Проход 1 — только имена (строки, не узлы): нужны для резолва вызовов.
    names: list[str] = []
    for m in _owned_walk():
        names.append(_ts_field_text(src, m, "name") or "?")

    # Проход 2 — каждый метод целиком за один заход: узел не переживает итерацию.
    methods: list[dict] = []
    for m in _owned_walk():
        name = _ts_field_text(src, m, "name") or "?"
        params_n = m.child_by_field_name("parameters")
        params = ""
        if params_n is not None:
            params = re.sub(r"\s+", " ", _ts_text(src, params_n).strip())
            if params.startswith("(") and params.endswith(")"):
                params = params[1:-1].strip()
            params = params[:100]
        anns: list[str] = []
        for a in caps("(marker_annotation name: (identifier) @a)", m).get("a", []):
            anns.append(_ts_text(src, a)[:40])
        for a in caps("(annotation name: (identifier) @a)", m).get("a", []):
            t = _ts_text(src, a)[:40]
            if t not in anns:
                anns.append(t)
        body_n = m.child_by_field_name("body")
        # Тело = весь узел метода (аннотации + сигнатура + блок): LLM видит
        # метод целиком, а не голый '{...}' без заголовка.
        body = _ts_text(src, m)
        invoked: set[str] = set()
        created: set[str] = set()
        if body_n is not None:
            for c in caps("(method_invocation name: (identifier) @c)",
                          body_n).get("c", []):
                invoked.add(_ts_text(src, c))
            for c in caps("(object_creation_expression type: (type_identifier) @c)",
                          body_n).get("c", []):
                created.add(_ts_text(src, c))
        methods.append({
            "name": name, "class": cls_name,
            "params": params, "annotations": anns,
            "desc": _java_desc_above(lines, m.start_point.row),
            "sig": f"{name}({params}): ...",
            "body": body, "body_bytes": bytes_len(body),
            "_calls_raw": sorted(invoked) + sorted("new:" + c for c in created),
        })

    # --- поля (свежий обход, тексты сразу) ---
    fields: list[dict] = []
    start = cls_node.child_by_field_name("body") or cls_node
    stack = [start]
    while stack:
        node = stack.pop()
        for ch in node.children:
            if ch.type in ("class_declaration", "record_declaration",
                           "interface_declaration", "enum_declaration",
                           "annotation_type_declaration"):
                continue
            if ch.type == "field_declaration":
                t = _ts_field_text(src, ch, "type")[:40]
                for v in caps("(variable_declarator name: (identifier) @v)",
                              ch).get("v", []):
                    fields.append({"name": _ts_text(src, v), "type": t})
            else:
                stack.append(ch)
    fnames = [f["name"] for f in fields]

    # --- связи (чистый Python по уже извлеченным строкам) ---
    for m in methods:
        raw = m.pop("_calls_raw")
        raw_calls = [c for c in raw if not c.startswith("new:")]
        raw_news = [c[4:] for c in raw if c.startswith("new:")]
        body = m["body"]
        m["calls"] = sorted((set(raw_calls) & set(names) - {m["name"]})
                            | (set(raw_news) & declared_types))
        m["uses_fields"] = sorted({w for w in re.findall(r"\b\w+\b", body)
                                   if w in fnames})
        m["uses_imports"] = sorted(
            {imp["full"] for imp in imports
             if re.search(r"\b" + re.escape(imp["simple"]) + r"\b",
                          m["sig"] + body)})
        m["event"] = ""
        if any("EventHandler" in a for a in m["annotations"]):
            pm = re.match(r"\s*([\w<>\[\]]+)\s+\w+\s*$", m["params"].strip())
            if pm:
                m["event"] = pm.group(1)[:60]

    _java_auto_descs(methods)
    return {"class": cls_name, "imports": imports, "methods": methods,
            "fields": fields, "parser": "tree-sitter-java",
            "has_error": root.has_error}


def java_parse(code: str) -> dict:
    """Диспетчер: сначала прод (tree-sitter), при любой проблеме — демо-regex.

    Поле 'parser' в результате честно говорит, кто реально разобрал.
    """
    try:
        return java_prod_parse(code)
    except Exception:
        return java_demo_parse(code)


def _mdesc(m: dict) -> str:
    """Авторское описание (javadoc) или автоописание-запасной вариант."""
    return m["desc"] or m.get("auto_desc", "")


def format_java_tree(parsed: dict, max_methods: int = 15) -> str:
    out = [f"class {parsed['class'] or '?'}: "
           f"{len(parsed['methods'])} методов, "
           f"{len(parsed['fields'])} полей, "
           f"{len(parsed['imports'])} импортов [{parsed['parser']}]"]
    for m in parsed["methods"][:max_methods]:
        tags = []
        if m["event"]:
            tags.append(f"event:{m['event']}")
        if m["calls"]:
            tags.append("calls:" + ",".join(m["calls"]))  # полный список, без обрезки
        if m["uses_imports"]:
            tags.append(f"imports:{len(m['uses_imports'])}")
        tail = f" [{'; '.join(tags)}]" if tags else ""
        desc = f"  # {_mdesc(m)}" if _mdesc(m) else ""
        out.append(f"  {m['sig']}{desc}{tail}")
    if len(parsed["methods"]) > max_methods:
        out.append(f"  ... +{len(parsed['methods']) - max_methods} методов")
    return "\n".join(out) + "\n"


def build_method_focus(parsed: dict, target: str | None = None) -> dict:
    """Фокус-контекст на 1 метод: его тело + сигнатуры соседей +
    только его импорты + описания. Это и едет в LLM на фикс."""
    methods = parsed["methods"]
    if not methods:
        return {"target": "", "context": "", "context_bytes": 0,
                "tokens": 0, "imports": [], "neighbors": []}
    tgt = next((m for m in methods if m["name"] == target), None)
    if tgt is None:
        tgt = next((m for m in methods
                    if any("EventHandler" in a for a in m["annotations"])),
                   methods[0])
    neigh_all = [m for m in methods
                 if m["name"] in tgt["calls"]]
    neigh, dropped = neigh_all[:15], neigh_all[15:]  # сигнатуры дешевые, режем поздно и вслух
    parts = [f"TARGET {tgt['class']}.{tgt['name']}({tgt['params']}):"]
    if _mdesc(tgt):
        parts.append(f"// {_mdesc(tgt)}")
    parts.append(tgt["body"][:3000])
    parts.append("\n-- соседи (сигнатуры + описания) --")
    for m in neigh:
        d = f"  # {_mdesc(m)}" if _mdesc(m) else ""
        parts.append(f"{m['sig']}{d}")
    if dropped:
        parts.append(f"-- скрыто соседей: {len(dropped)} ({', '.join(m['name'] for m in dropped)}) --")
    parts.append("-- импорты этого метода --")
    for imp in tgt["uses_imports"]:
        parts.append(f"import {imp};")
    if tgt["uses_fields"]:
        parts.append("-- поля этого метода --")
        parts.append(", ".join(tgt["uses_fields"]))
    ctx = "\n".join(parts) + "\n"
    tok, _ = estimate_tokens(ctx)
    return {"target": f"{tgt['class']}.{tgt['name']}",
            "context": ctx, "context_bytes": bytes_len(ctx),
            "tokens": tok, "imports": tgt["uses_imports"],
            "neighbors": [m["name"] for m in neigh],
            "dropped_neighbors": [m["name"] for m in dropped],
            "target_body_bytes": tgt["body_bytes"]}


# ---------------------------------------------------------------------------
# Граф текстов: сущности из документации + якоря к коду (без LLM)
# ---------------------------------------------------------------------------

_TEXT_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    r"|[a-z]+[A-Z][A-Za-z0-9]*"
    r"|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+"
    r"|[A-Z][A-Z0-9_]{2,}"
    r"|[a-z]+(?:_[a-z0-9]+)+"
)
_TEXT_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)")
_PROSE_STOP = {"the", "and", "for", "with", "this", "that", "from", "into",
               "java", "python", "code"}


def extract_text_entities(text: str, top: int = 20) -> list[dict]:
    """Кандидаты-сущности из текста: CamelCase, UPPER_SNAKE, snake_case,
    dotted.path. Обычные слова прозы (кириллица и одиночные lowercase)
    не извлекаются — для них нужен NER/LLM (честное ограничение)."""
    counts: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for m in _TEXT_TOKEN_RE.finditer(text):
        w = m.group(0).strip("._")
        wl = w.lower()
        if wl in _PROSE_STOP or len(w) < 3:
            continue
        if "." in w:
            kind = "dotted"
        elif "_" in w and w == w.upper():
            kind = "upper"
        elif "_" in w:
            kind = "snake"
        else:
            kind = "camel"
        counts[wl] = counts.get(wl, 0) + 1
        kinds.setdefault(wl, (w, kind))
    items = [{"name": kinds[k][0], "kind": kinds[k][1], "count": c}
             for k, c in counts.items()]
    items.sort(key=lambda e: (-e["count"], e["name"]))
    return items[:top]


def build_text_graph(text: str) -> dict:
    """Граф текста: сущности + значения вида name=123. Детерминирован."""
    values: dict[str, str] = {}
    for m in _TEXT_VALUE_RE.finditer(text):
        values[m.group(1)] = m.group(2)
    return {"entities": extract_text_entities(text),
            "values": values, "parser": "text-demo-regex"}


def anchor_to_code(graph: dict, code_nodes: list[dict]) -> list[dict]:
    """Якоря: сущность текста -> узел кода (exact: точное имя, partial:
    вхождение от 4 символов). code_nodes: skeleton_nodes или методы/поля."""
    by_name: dict[str, dict] = {}
    for n in code_nodes:
        by_name.setdefault(n["name"].lower(), n)
    anchors: list[dict] = []
    for e in graph["entities"]:
        el = e["name"].lower()
        if el in by_name:
            t = by_name[el]
            anchors.append({"entity": e["name"],
                            "target": {"type": t["type"], "name": t["name"],
                                       "parent": t.get("parent", "")},
                            "rel": "exact"})
            continue
        for nl, t in by_name.items():
            if len(el) >= 4 and len(nl) >= 4 and (el in nl or nl in el):
                anchors.append({"entity": e["name"],
                                "target": {"type": t["type"], "name": t["name"],
                                           "parent": t.get("parent", "")},
                                "rel": "partial"})
                break
    return anchors


def suggest_focus(query: str, anchors: list[dict]) -> str:
    """Целевой метод из вопроса через якоря: exact=2, partial=1. '' если мимо."""
    terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.lower()))
    scores: dict[str, int] = {}
    order: list[str] = []
    for a in anchors:
        t = a["target"]
        if t["type"] not in ("method", "function"):
            continue
        hit = any(term == a["entity"].lower() or
                  (len(term) >= 4 and (term in a["entity"].lower()
                                       or a["entity"].lower() in term))
                  for term in terms)
        if hit:
            key = t["parent"] + "." + t["name"] if t.get("parent") else t["name"]
            scores[key] = scores.get(key, 0) + (2 if a["rel"] == "exact" else 1)
            if key not in order:
                order.append(key)
    if not scores:
        return ""
    best = max(scores.values())
    return next(k for k in order if scores[k] == best)


def compress_text_semantic(text_lines) -> dict:
    """Легкая нормализация документации (для LLM-контекста).

    - схлопывает пробелы, убирает пустые строки-дубли,
    - удаляет точные дубли строк (часто встречается в доках).
    Без LLM-суммаризации: смысл не теряется вообще.
    """
    raw_text = "".join(text_lines) if isinstance(text_lines, list) else text_lines
    raw_bytes = bytes_len(raw_text)
    seen: set[str] = set()
    out: list[str] = []
    for line in raw_text.splitlines():
        norm = re.sub(r"\s+", " ", line).strip()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    cleaned = "\n".join(out) + ("\n" if out else "")
    cleaned_bytes = bytes_len(cleaned)
    saving = (1 - cleaned_bytes / raw_bytes) * 100 if raw_bytes else 0.0
    tok_raw, tok_method = estimate_tokens(raw_text)
    tok_clean, _ = estimate_tokens(cleaned)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "raw_bytes": raw_bytes,
        "cleaned_bytes": cleaned_bytes,
        "saving_pct": round(saving, 2),
        "tokens_raw": tok_raw,
        "tokens_cleaned": tok_clean,
        "tokens_method": tok_method,
    }


# ---------------------------------------------------------------------------
# МОДУЛЬ 3: Passport
# ---------------------------------------------------------------------------

def create_block_passport(cross_hash_id: str, kind: str, size_bytes: int,
                          extra: dict | None = None) -> dict:
    """Легкий паспорт блока. location — ЛОГИЧЕСКОЕ поле симуляции."""
    passport = {
        "hash_id": cross_hash_id,
        "kind": kind,
        "size_ram_bytes": int(size_bytes),
        "status": "OFFLOADED_TO_RAM",
        "location": "RAM",  # симуляция, не реальная cuda-память
    }
    if extra:
        passport.update(extra)
    return passport


# ---------------------------------------------------------------------------
# Класс-оркестратор (совместим со старым API route_and_pack)
# ---------------------------------------------------------------------------

class SmartContextMemory:
    def __init__(self, vram_limit_bytes=4096):
        # 1. RAM — сжатые тяжелые данные
        self.ram_storage = {}
        # 2. VRAM (СИМУЛЯЦИЯ) — только паспорта
        self.vram_passports = {}
        # 3. Активный буфер VRAM (СИМУЛЯЦИЯ) — подгруженные "вспышкой" блоки
        self.active_vram_buffer = {}

        self.vram_limit = vram_limit_bytes
        self.current_vram_usage = 0  # в БАЙТАХ очищенного текста (utf-8)

    # === МОДУЛЬ 1: Semantic Router & Cross-Modal Hash ===
    def _generate_hash_id(self, content: str) -> str:
        """Уникальный 8-символьный Cross-Hash ID (демо; для прода — 16+ символов)."""
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def route_and_pack(self, raw_data: dict) -> str:
        """Разделяет данные, чистит, сжимает и складывает в RAM (симуляция)."""
        code_content = raw_data.get("code", "")
        text_content = raw_data.get("text", "")

        hash_id = self._generate_hash_id(code_content + text_content)

        # Честное сжатие: семантика + zlib
        code_payload = compress_code_block(code_content)
        text_payload = compress_text_semantic(text_content)

        # Реальные узлы из AST-скелета (уникальные имена, по порядку).
        seen_names: list[str] = []
        for n in code_payload["skeleton_nodes"]:
            if n["name"] not in seen_names:
                seen_names.append(n["name"])
        graph_web = {
            "anchor_id": hash_id,
            "entities": seen_names[:20],
            "nodes": code_payload["skeleton_nodes"][:20],
            "skeleton_method": code_payload["skeleton_method"],
            "text_summary": text_payload["cleaned_text"][:200],
        }

        self.ram_storage[hash_id] = {
            "code_zlib": code_payload["compressed"],
            "code_cleaned_bytes": code_payload["cleaned_bytes"],
            "skeleton_text": code_payload["skeleton_text"],
            "graph_web": graph_web,
            "raw_text": text_payload["cleaned_text"],
        }

        passport = {
            "hash_id": hash_id,
            "size_ram_bytes": code_payload["compressed_bytes"],
            "size_cleaned_bytes": code_payload["cleaned_bytes"],
            "semantic_saving_pct": code_payload["semantic_saving_pct"],
            "storage_saving_pct": code_payload["storage_saving_pct"],
            "status": "OFFLOADED_TO_RAM",
        }
        self.vram_passports[hash_id] = passport

        print(f"[PACK] ID: [{hash_id}] | code raw={code_payload['raw_bytes']}B "
              f"-> cleaned={code_payload['cleaned_bytes']}B "
              f"(-{code_payload['semantic_saving_pct']}%) "
              f"-> zlib={code_payload['compressed_bytes']}B "
              f"(-{code_payload['storage_saving_pct']}%) [RAM, sim]")
        return hash_id

    # === МОДУЛЬ 4: Asynchronous Flash-Engine (симуляция) ===
    async def prefetch_to_vram(self, hash_id: str) -> float:
        """Predictive Pre-fetching (СИМУЛЯЦИЯ PCIe: sleep + декомпресс)."""
        t0 = time.perf_counter()
        await asyncio.sleep(0.01)  # симуляция передачи по PCIe

        if hash_id in self.ram_storage and hash_id not in self.active_vram_buffer:
            t1 = time.perf_counter()
            decompressed_code = zlib.decompress(
                self.ram_storage[hash_id]["code_zlib"]).decode()
            decode_ms = (time.perf_counter() - t1) * 1000
            block_size = bytes_len(decompressed_code)

            if self.current_vram_usage + block_size > self.vram_limit:
                print("[VRAM GUARD] Лимит VRAM (sim). Экстренная очистка...")
                self._evict_oldest_vram_block()

            self.active_vram_buffer[hash_id] = {
                "code": decompressed_code,
                "graph": self.ram_storage[hash_id]["graph_web"],
            }
            self.current_vram_usage += block_size
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[PRE-FETCH] Блок [{hash_id}] -> VRAM (sim). "
                  f"VRAM={self.current_vram_usage}B, "
                  f"decode={decode_ms:.2f}ms, total={total_ms:.2f}ms")
            return total_ms
        return 0.0

    async def execute_query(self, hash_id: str, query: str):
        """Micro-Swap ('вспышка') + Event-Driven Cleanup по [EOS]."""
        print(f"\n[QUERY]: '{query}'")

        if hash_id not in self.active_vram_buffer:
            print("[SWAP] Нет в VRAM (sim). Micro-Swap...")
            await self.prefetch_to_vram(hash_id)

        active_block = self.active_vram_buffer[hash_id]
        print(f"[GEN] Контекст [{hash_id}]:")
        print(f"  -> code: {active_block['code'][:80]!r}...")
        print(f"  -> graph entities (ast): {active_block['graph']['entities']}")

        await asyncio.sleep(0.1)

        self.event_driven_cleanup(hash_id)

    def event_driven_cleanup(self, hash_id: str):
        """Очистка VRAM-симуляции по сигналу [EOS]."""
        if hash_id in self.active_vram_buffer:
            block_size = bytes_len(self.active_vram_buffer[hash_id]["code"])
            del self.active_vram_buffer[hash_id]
            self.current_vram_usage -= block_size
            print(f"[CLEANUP] [EOS]! Блок [{hash_id}] удален из VRAM (sim). "
                  f"VRAM={self.current_vram_usage}B.")

    def _evict_oldest_vram_block(self):
        """Принудительное освобождение при переполнении (FIFO)."""
        if self.active_vram_buffer:
            oldest_hash = next(iter(self.active_vram_buffer))
            self.event_driven_cleanup(oldest_hash)


# === ТЕСТОВЫЙ ПРОГОН ===
async def main():
    scm = SmartContextMemory(vram_limit_bytes=2048)

    sample_data = {
        "code": "class DatabaseConnector:\n" + "    def connect(self):\n        pass\n" * 20,
        "text": "Документация по подключению к базе данных и настройкам драйвера.",
    }

    print("=== 1. СЖАТИЕ И ВЫГРУЗКА В RAM (sim) ===")
    hash_id = scm.route_and_pack(sample_data)

    print("\n=== 2. PRE-FETCHING (sim) ===")
    await scm.prefetch_to_vram(hash_id)

    print("\n=== 3. ВСПЫШКА И ОЧИСТКА (sim) ===")
    # блок уже в VRAM после шага 2 — execute_query пойдет без swap (честно)
    await scm.execute_query(hash_id, "Как устроена база данных?")

if __name__ == "__main__":
    asyncio.run(main())
