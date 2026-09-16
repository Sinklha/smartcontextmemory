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

# П.1 PLAN: умные комменты не режем. Дешево, спасает предупреждения.
_SMART_COMMENT_RE = re.compile(
    r"\b(TODO|FIXME|WARNING|WARN|NOTE|XXX|HACK|BUG|CAUTION|IMPORTANT|SECURITY)\b",
    re.IGNORECASE,
)

# Эвристика "сложной" строки: пояснение над ней стоит сохранить.
_COMPLEX_HINTS = (
    "lambda", "yield", "await ", "re.", "Pattern", "<<", ">>", ">>>", "::",
    "->", "=>", "synchronized", "volatile", "transient", "instanceof",
    "isinstance", "eval(", "exec(", "__", "?.", "??", "reflection", "unsafe",
    "bitwise", "битов", "сдвиг",
)


def _is_complex_code_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    low = s.lower()
    for h in _COMPLEX_HINTS:
        if h.lower() in low:
            return True
    # длинный вызов с вложенностью: вероятно нетривиален
    if len(s) > 70 and "(" in s and ")" in s:
        return True
    if s.count("(") >= 2 and s.count(")") >= 2 and len(s) > 50:
        return True
    return False


def _has_smart_marker(text: str) -> bool:
    return bool(_SMART_COMMENT_RE.search(text))


def _block_sub_keep_smart(m: "re.Match") -> str:
    block = m.group(0)
    if _has_smart_marker(block):
        # Сжимаем блок в одну строку, чтобы не терять предупреждение.
        inner = re.sub(r"\s+", " ", block[2:-2].strip())[:200]
        return f"\n# SMART: {inner}\n" if inner else "\n"
    return ""


def semantic_clean_code(code: str) -> str:
    """Семантическая очистка кода — то, что реально экономит контекст LLM.

    - вырезает /*...*/ блоки (КРОМЕ содержащих TODO/FIXME/... — они
      сжимаются в одну строку `# SMART: ...`),
    - удаляет полно-строчные # и // комментарии (КРОМЕ умных и пояснений
      над сложными строками — п.1 PLAN),
    - режет инлайн-комментарии вида '  # ...' и '  // ...'
      (КРОМЕ умных — их оставляем целиком),
    - убирает висячие пробелы и пустые строки.
    Структура кода сохраняется, поведение — нет гарантии для строковых
    литералов с '#' внутри (честное ограничение эвристики).
    """
    code = _BLOCK_COMMENT_RE.sub(_block_sub_keep_smart, code)
    lines = code.splitlines()
    out: list[str] = []
    in_triple: str | None = None  # '"""' или "'''" — внутри многострочной строки
    for idx, line in enumerate(lines):
        # Трекинг triple-quoted строк: внутри них комменты не режем вообще,
        # иначе рвем docstring и ломаем ast.parse (было: regex-fallback вместо ast).
        if in_triple:
            out.append(line.rstrip())
            # закрытие: нечетное число делимитеров в строке
            try:
                if line.count(in_triple) % 2 == 1:
                    in_triple = None
            except Exception:
                in_triple = None
            continue
        else:
            triple = None
            for delim in ('"""', "'''"):
                try:
                    if line.count(delim) % 2 == 1:
                        triple = delim
                        break
                except Exception:
                    continue
            if triple:
                out.append(line.rstrip())
                in_triple = triple
                continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            if _has_smart_marker(stripped):
                out.append(line.rstrip())
                continue
            # Пояснение над сложной строкой: смотрим следующую кодовую строку.
            j = idx + 1
            nxt = ""
            while j < len(lines):
                s2 = lines[j].strip()
                if s2:
                    nxt = lines[j]
                    break
                j += 1
            if nxt and _is_complex_code_line(nxt):
                out.append(line.rstrip())
                continue
            continue
        # инлайн-комментарии: только если перед # или // есть пробел
        # (чтобы не резать 'port=5432' или '://').
        # ВАЖНО: '//' в Python — это floor-division (len(text) // 4), а не
        # коммент (в Python коммент — '#'). Режем '//' только в Java-подобных
        # строках (есть ';'/'{'/'}'), иначе ломаем синтаксис и ast.parse падает.
        cut = len(line)
        is_javaish = (";" in line or "{" in line or "}" in line)
        for marker in ("  #", " #", "\t#", "  //", " //", "\t//"):
            if "//" in marker and not is_javaish:
                continue
            midx = line.find(marker)
            if midx != -1 and midx < cut:
                # не режем внутри кавычек грубо: считаем кавычки до маркера
                prefix = line[:midx]
                if prefix.count('"') % 2 == 0 and prefix.count("'") % 2 == 0:
                    cut = midx
        if cut != len(line):
            comment_part = line[cut:]
            if _has_smart_marker(comment_part):
                # Умный инлайн-коммент оставляем целиком.
                out.append(line.rstrip())
                continue
            line = line[:cut].rstrip()
        else:
            line = line.rstrip()
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
    # Чистый Python (ast разобрал) в Java-ветку не идет — иначе ';'/'{'
    # в строках дают ложное is_java_like и мусор вроде метода "g".
    if is_java_like(cleaned) and sk.get("method") != "ast":
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

def _unparse_short(node, limit: int = 30) -> str:
    """Безопасный ast.unparse с обрезкой. Пусто при любой проблеме."""
    try:
        s = ast.unparse(node).strip()
    except Exception:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s[:limit] if len(s) > limit else s


def _decor_name(d) -> str:
    try:
        s = ast.unparse(d).strip().split("(")[0].split(".")[-1]
        return "@" + s[:30]
    except Exception:
        return ""


def _args_str(args: ast.arguments) -> str:
    """Сигнатура с реальными значениями (п.2 PLAN): дефолты, аннотации едут
    в LLM даже когда тело выкинуто. Обрезка честная — длинные значения
    режутся до 30 символов, общий список — до ~140."""
    parts: list[str] = []
    pos = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    # defaults относятся к последним N позиционным
    nd = len(defaults)
    start = len(pos) - nd if nd else len(pos)
    for i, a in enumerate(pos):
        ann = f": {_unparse_short(a.annotation, 20)}" if a.annotation else ""
        if i >= start:
            d = _unparse_short(defaults[i - start], 30)
            parts.append(f"{a.arg}{ann}={d}" if d else f"{a.arg}{ann}")
        else:
            parts.append(f"{a.arg}{ann}")
        if a.arg == "/" or False:
            pass
    if args.vararg:
        ann = f": {_unparse_short(args.vararg.annotation, 20)}" if args.vararg.annotation else ""
        parts.append("*" + args.vararg.arg + ann)
    for j, a in enumerate(args.kwonlyargs):
        ann = f": {_unparse_short(a.annotation, 20)}" if a.annotation else ""
        try:
            kd = args.kw_defaults[j]
        except Exception:
            kd = None
        if kd is not None:
            d = _unparse_short(kd, 30)
            parts.append(f"{a.arg}{ann}={d}" if d else f"{a.arg}{ann}")
        else:
            parts.append(f"{a.arg}{ann}")
    if args.kwarg:
        ann = f": {_unparse_short(args.kwarg.annotation, 20)}" if args.kwarg.annotation else ""
        parts.append("**" + args.kwarg.arg + ann)
    s = ", ".join(parts)
    return s[:140]


def ast_skeleton(code: str) -> dict:
    """Строит скелет кода: только сигнатуры, без тел функций.

    Python -> настоящий ast. Java/C-подобный -> regex-фолбэк.
    П.2 PLAN: значения всегда в скелете — константы, дефолты параметров,
    аннотации (@EventHandler, лимиты) едут в LLM даже когда тело выкинуто.
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
                for d in node.decorator_list:
                    dn = _decor_name(d)
                    if dn:
                        lines.append(dn)
                bases = ", ".join(ast.unparse(b) for b in node.bases)[:60] if node.bases else ""
                decl = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
                lines.append(decl)
                nodes.append({"type": "class", "name": node.name})
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for d in item.decorator_list:
                            dn = _decor_name(d)
                            if dn:
                                lines.append(f"  {dn}")
                        doc = ast.get_docstring(item)
                        first = f"  # {doc.splitlines()[0][:60]}" if doc else ""
                        ret = f" -> {_unparse_short(item.returns, 20)}" if item.returns else ""
                        lines.append(f"  def {item.name}({_args_str(item.args)}){ret}: ...{first}")
                        nodes.append({"type": "method", "name": item.name,
                                      "parent": node.name})
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        ann = _unparse_short(item.annotation, 30) if item.annotation else ""
                        val = _unparse_short(item.value, 30) if item.value else ""
                        tail = f": {ann}" if ann else ""
                        tail += f" = {val}" if val else ""
                        lines.append(f"  {item.target.id}{tail if tail else ': ...'}")
                        nodes.append({"type": "field", "name": item.target.id,
                                      "parent": node.name})
                    elif isinstance(item, ast.Assign):
                        targets = ", ".join(t.id for t in item.targets if isinstance(t, ast.Name))[:60]
                        if targets:
                            val = _unparse_short(item.value, 40)
                            lines.append(f"  {targets} = {val}" if val else f"  {targets} = ...")
                            nodes.append({"type": "field", "name": targets,
                                          "parent": node.name})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in node.decorator_list:
                    dn = _decor_name(d)
                    if dn:
                        lines.append(dn)
                doc = ast.get_docstring(node)
                first = f"  # {doc.splitlines()[0][:60]}" if doc else ""
                ret = f" -> {_unparse_short(node.returns, 20)}" if node.returns else ""
                lines.append(f"def {node.name}({_args_str(node.args)}){ret}: ...{first}")
                nodes.append({"type": "function", "name": node.name})
            elif isinstance(node, ast.Assign):
                targets = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))[:60]
                if targets:
                    val = _unparse_short(node.value, 40)
                    lines.append(f"{targets} = {val}" if val else f"{targets} = ...")
                    nodes.append({"type": "const", "name": targets})
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                ann = _unparse_short(node.annotation, 30) if node.annotation else ""
                val = _unparse_short(node.value, 30) if node.value else ""
                tail = f": {ann}" if ann else ""
                tail += f" = {val}" if val else ""
                if tail:
                    lines.append(f"{node.target.id}{tail}")
                    nodes.append({"type": "const", "name": node.target.id})
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
            if fm and len(s) < 200:
                raw_val = (fm.group(6) or "").strip()
                if raw_val.startswith("="):
                    raw_val = raw_val[1:].strip()
                if raw_val.endswith(";"):
                    raw_val = raw_val[:-1].strip()
                fields.append({"name": fm.group(5), "type": fm.group(4).strip()[:40],
                               "value": re.sub(r"\s+", " ", raw_val)[:40]})
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


def _extract_contract(body: str, params: str = "") -> str:
    """П.4 PLAN: контракты соседей — сигнатура + предусловия/возврат.

    Детерминированная эвристика без LLM, только факты из тела:
    - `if x == null return/throw` → «требует не-null: x»,
    - `Objects.requireNonNull(x)` / `assert x` → то же,
    - `if x is None` (Python) → то же,
    - `throw new X` / `raise X` → «бросает X»,
    - `if len/size <op> N return` → «предусловие: ...».
    Пусто, если guard-паттернов нет (честно, не выдумываем).
    """
    if not body:
        return ""
    bits: list[str] = []
    # null / None guard-ы
    for m in re.finditer(r"(\w+)\s*==\s*null", body):
        bits.append(f"требует не-null: {m.group(1)}")
    for m in re.finditer(r"(\w+)\s+is\s+None\b", body):
        bits.append(f"требует не-None: {m.group(1)}")
    for m in re.finditer(r"(?:Objects\s*\.\s*requireNonNull|requireNonNull)\s*\(\s*(\w+)", body):
        bits.append(f"требует не-null: {m.group(1)}")
    for m in re.finditer(r"assert\s+(\w+)(?:\s+is\s+not\s+None)?", body):
        v = m.group(1)
        if v not in ("True", "False", "None"):
            bits.append(f"assert: {v}")
    # бросаемые исключения (уникально, до 3)
    throws: list[str] = []
    for m in re.finditer(r"throw\s+new\s+(\w+)", body):
        if m.group(1) not in throws:
            throws.append(m.group(1))
    for m in re.finditer(r"raise\s+(\w+)", body):
        if m.group(1) not in throws:
            throws.append(m.group(1))
    for t in throws[:3]:
        bits.append(f"бросает {t}")
    # числовые guard-ы вида if (size <= 0) return / throw (первые 5 строк с if)
    for line in body.splitlines()[:25]:
        s = line.strip()
        if not s.startswith("if"):
            continue
        if not any(k in s for k in ("return", "throw", "raise", "assert")):
            continue
        cond = re.sub(r"^\s*if\s*\(?\s*", "", s)[:60].rstrip("){}: ")
        if cond and len(cond) >= 3:
            bits.append(f"предусловие: {cond}")
            if len(bits) >= 6:
                break
    # дедуп с сохранением порядка, до 4 штук (контракт — ступень, не тело)
    seen: set[str] = set()
    uniq: list[str] = []
    for b in bits:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
        if len(uniq) >= 4:
            break
    return "; ".join(uniq)


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
        try:
            m["contract"] = _extract_contract(m.get("body", ""), m.get("params", ""))
        except Exception:
            m["contract"] = ""


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
            # Связью считаем только свои вызовы: голые (foo()), this/super
            # и статику своего класса. map.put(), log.info() и new X().k()
            # с чужим получателем — не наши методы (раньше склеивались по имени).
            for inv in caps("(method_invocation) @inv", body_n).get("inv", []):
                nm = inv.child_by_field_name("name")
                if nm is None:
                    continue
                ob = inv.child_by_field_name("object")
                recv = _ts_text(src, ob).strip() if ob is not None else ""
                if not recv or recv in ("this", "super", cls_name):
                    invoked.add(_ts_text(src, nm))
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
                    # значение инициализатора: ищем value у variable_declarator
                    val = ""
                    try:
                        parent = v.parent
                        if parent is not None:
                            vv = parent.child_by_field_name("value")
                            if vv is not None:
                                val = re.sub(r"\s+", " ", _ts_text(src, vv).strip())[:40]
                        if not val:
                            # запасной вариант: всё после '=' в деклараторе
                            full_decl = _ts_text(src, parent) if parent is not None else ""
                            if "=" in full_decl:
                                val = re.sub(r"\s+", " ", full_decl.split("=", 1)[1].strip().rstrip(";"))[:40]
                    except Exception:
                        val = ""
                    fields.append({"name": _ts_text(src, v), "type": t, "value": val})
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


def python_parse(code: str) -> dict:
    """Парсер Python через ast (для анализа .py без Java-ложных срабатываний).

    Возвращает структуру как java_parse: class/imports/methods/fields/parser.
    Связи calls/uses_fields для Python — честно пустые (нужен dataflow),
    контракты извлекаются той же эвристикой. Бросает ValueError, если
    Python-методов/классов нет.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"не Python: {e}")
    imports: list[dict] = []
    methods: list[dict] = []
    fields: list[dict] = []
    cls_name = ""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                full = a.name
                imports.append({"full": full, "simple": full.split(".")[-1][:40]})
        elif isinstance(node, ast.ClassDef):
            if not cls_name:
                cls_name = node.name
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    try:
                        body = ast.get_source_segment(code, item) or ""
                    except Exception:
                        body = ""
                    anns = []
                    for d in item.decorator_list:
                        dn = _decor_name(d)
                        if dn:
                            anns.append(dn)
                    doc = ast.get_docstring(item)
                    params = _args_str(item.args)
                    methods.append({
                        "name": item.name, "class": node.name,
                        "params": params, "annotations": anns,
                        "desc": (doc.splitlines()[0][:120] if doc else ""),
                        "sig": f"{item.name}({params}): ...",
                        "body": body[:4000], "body_bytes": bytes_len(body[:4000]),
                        "calls": [], "uses_fields": [], "uses_imports": [],
                        "event": "",
                    })
                elif isinstance(item, (ast.AnnAssign, ast.Assign)):
                    try:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            fields.append({"name": item.target.id,
                                           "type": _unparse_short(item.annotation, 40) if item.annotation else "",
                                           "value": _unparse_short(item.value, 40) if item.value else ""})
                        elif isinstance(item, ast.Assign):
                            tg = ", ".join(t.id for t in item.targets if isinstance(t, ast.Name))[:60]
                            if tg:
                                fields.append({"name": tg, "type": "",
                                               "value": _unparse_short(item.value, 40)})
                    except Exception:
                        pass
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                body = ast.get_source_segment(code, node) or ""
            except Exception:
                body = ""
            anns = []
            for d in node.decorator_list:
                dn = _decor_name(d)
                if dn:
                    anns.append(dn)
            doc = ast.get_docstring(node)
            params = _args_str(node.args)
            methods.append({
                "name": node.name, "class": "",
                "params": params, "annotations": anns,
                "desc": (doc.splitlines()[0][:120] if doc else ""),
                "sig": f"{node.name}({params}): ...",
                "body": body[:4000], "body_bytes": bytes_len(body[:4000]),
                "calls": [], "uses_fields": [], "uses_imports": [],
                "event": "",
            })
    if not methods and not cls_name:
        raise ValueError("Python-методов не найдено")
    _java_auto_descs(methods)
    return {"class": cls_name, "imports": imports, "methods": methods,
            "fields": fields, "parser": "ast-python"}


def _mdesc(m: dict) -> str:
    """Авторское описание (javadoc) или автоописание-запасной вариант."""
    return m["desc"] or m.get("auto_desc", "")


def format_java_tree(parsed: dict, max_methods: int = 15) -> str:
    out = [f"class {parsed['class'] or '?'}: "
           f"{len(parsed['methods'])} методов, "
           f"{len(parsed['fields'])} полей, "
           f"{len(parsed['imports'])} импортов [{parsed['parser']}]"]
    # П.2 PLAN: поля со значениями тоже в скелете (дешево, спасает лимиты).
    for f in (parsed.get("fields") or [])[:10]:
        fv = f" = {f['value']}" if f.get("value") else ""
        out.append(f"  field {f['name']}: {f.get('type','')}{fv}")
    for m in parsed["methods"][:max_methods]:
        tags = []
        if m.get("annotations"):
            anns = [a for a in m["annotations"] if a][:2]
            if anns:
                tags.append("@" + ",".join(a.lstrip("@") for a in anns))
        if m["event"]:
            tags.append(f"event:{m['event']}")
        if m["calls"]:
            tags.append("calls:" + ",".join(m["calls"]))  # полный список, без обрезки
        if m["uses_imports"]:
            tags.append(f"imports:{len(m['uses_imports'])}")
        tail = f" [{'; '.join(tags)}]" if tags else ""
        desc = f"  # {_mdesc(m)}" if _mdesc(m) else ""
        contract = f"  // контракт: {m['contract']}" if m.get("contract") else ""
        out.append(f"  {m['sig']}{desc}{tail}{contract}")
    if len(parsed["methods"]) > max_methods:
        out.append(f"  ... +{len(parsed['methods']) - max_methods} методов")
    return "\n".join(out) + "\n"


def build_method_focus(parsed: dict, target: str | None = None) -> dict:
    """Фокус-контекст на 1 метод: его тело + сигнатуры соседей +
    только его импорты + описания. Это и едет в LLM на фикс.

    П.3 PLAN: к соседям по вызовам добавляются писатели/читатели тех же
    полей (ловят событийные связи Bukkit-хендлеров, которых нет в графе).
    П.4 PLAN: у соседей показываем контракты (ступень между скелетом и телом).
    """
    methods = parsed["methods"]
    if not methods:
        return {"target": "", "context": "", "context_bytes": 0,
                "tokens": 0, "imports": [], "neighbors": []}
    tgt = next((m for m in methods if m["name"] == target), None)
    if tgt is None:
        tgt = next((m for m in methods
                    if any("EventHandler" in a for a in m["annotations"])),
                   methods[0])
    call_names = set(tgt.get("calls") or [])
    tgt_fields = set(tgt.get("uses_fields") or [])
    neigh_calls = [m for m in methods if m["name"] in call_names]
    # П.3: те же поля, но не вызовы — событийные соседи.
    neigh_fields = [m for m in methods
                    if m["name"] != tgt["name"]
                    and m["name"] not in call_names
                    and tgt_fields and (set(m.get("uses_fields") or []) & tgt_fields)]
    neigh_all = neigh_calls + neigh_fields
    neigh, dropped = neigh_all[:15], neigh_all[15:]  # сигнатуры дешевые, режем поздно и вслух
    tgt_head = f"{tgt['class']}.{tgt['name']}" if tgt.get("class") else tgt["name"]
    parts = [f"TARGET {tgt_head}({tgt['params']}):"]
    if _mdesc(tgt):
        parts.append(f"// {_mdesc(tgt)}")
    if tgt.get("contract"):
        parts.append(f"// контракт: {tgt['contract']}")
    parts.append(tgt["body"][:3000])
    parts.append("\n-- соседи (сигнатуры + описания + контракты) --")
    for m in neigh:
        d = f"  # {_mdesc(m)}" if _mdesc(m) else ""
        c = f"  // контракт: {m['contract']}" if m.get("contract") else ""
        # помечаем, откуда сосед: вызов или общее поле
        how = "calls" if m["name"] in call_names else "field"
        parts.append(f"[{how}] {m['sig']}{d}{c}")
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
    tgt_label = f"{tgt['class']}.{tgt['name']}" if tgt.get("class") else tgt["name"]
    return {"target": tgt_label,
            "context": ctx, "context_bytes": bytes_len(ctx),
            "tokens": tok, "imports": tgt["uses_imports"],
            "neighbors": [m["name"] for m in neigh],
            "call_neighbors": [m["name"] for m in neigh_calls if m["name"] in [n["name"] for n in neigh]],
            "field_neighbors": [m["name"] for m in neigh_fields if m["name"] in [n["name"] for n in neigh]],
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

# П.7 PLAN: флаг отрицания — отличаем факт от условия.
_NEG_RE = re.compile(
    r"\b(не|нет|ни|без|нельзя|невозможно|только\s+когда|только\s+если|"
    r"only\s+if|only\s+when|unless|except\b|if\s+not\b|\bnot\b|n't\b|"
    r"\bnever\b|\bno\b|\bnone\b|\bnobody\b)\b",
    re.IGNORECASE,
)


def detect_negation(text: str) -> dict:
    """Есть ли в куске отрицание/условие. Возвращает флаг + найденные маркеры."""
    if not text:
        return {"has_negation": False, "terms": []}
    terms: list[str] = []
    for m in _NEG_RE.finditer(text):
        t = re.sub(r"\s+", " ", m.group(0).strip().lower())[:20]
        if t and t not in terms:
            terms.append(t)
        if len(terms) >= 5:
            break
    return {"has_negation": bool(terms), "terms": terms}


def _sentence_with(text: str, name: str, limit: int = 140) -> str:
    """П.6 PLAN: якорь с контекстом — предложение, где встретилась сущность."""
    if not text or not name:
        return ""
    # грубое деление на предложения: по .!?… + переводы строк
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    nl = name.lower()
    for p in parts:
        if nl in p.lower():
            s = re.sub(r"\s+", " ", p.strip())
            return s[:limit]
    # запасной вариант: окно ±60 символов вокруг первого вхождения
    try:
        i = text.lower().index(nl)
        s = re.sub(r"\s+", " ", text[max(0, i - 60):i + 80].strip())
        return s[:limit]
    except ValueError:
        return ""


def _anchor_weight(node_type: str, rel: str) -> float:
    """П.9 PLAN: вес источникам. Javadoc/имя метода > случайное упоминание."""
    if rel == "partial":
        return 1.0
    return {
        "method": 3.0,
        "function": 3.0,
        "class": 2.5,
        "field": 2.0,
        "const": 1.8,
        "import": 1.5,
    }.get(node_type, 1.2)


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
    """Граф текста: сущности + значения вида name=123 + флаг отрицания.

    Детерминирован. П.7: has_negation помечает чанки с «не/нет/only if».
    """
    values: dict[str, str] = {}
    for m in _TEXT_VALUE_RE.finditer(text):
        values[m.group(1)] = m.group(2)
    neg = detect_negation(text)
    return {"entities": extract_text_entities(text),
            "values": values, "parser": "text-demo-regex",
            "has_negation": neg["has_negation"], "negation_terms": neg["terms"]}


def anchor_to_code(graph: dict, code_nodes: list[dict], text: str = "") -> list[dict]:
    """Якоря: сущность текста -> узел кода (exact: точное имя, partial:
    вхождение от 4 символов). code_nodes: skeleton_nodes или методы/поля.

    П.6: каждый якорь несет `context` — предложение, где встретилась сущность
    («только когда все 5» сохраняет условия почти бесплатно).
    П.9: каждый якорь несет `weight` — приоритет по месту, не только по имени.
    `text` опционален для совместимости со старым API.
    """
    by_name: dict[str, dict] = {}
    for n in code_nodes:
        try:
            by_name.setdefault(str(n["name"]).lower(), n)
        except Exception:
            continue
    anchors: list[dict] = []
    for e in (graph or {}).get("entities", []):
        el = str(e["name"]).lower()
        ctx = _sentence_with(text, e["name"]) if text else ""
        if el in by_name:
            t = by_name[el]
            anchors.append({"entity": e["name"],
                            "target": {"type": t["type"], "name": t["name"],
                                       "parent": t.get("parent", "")},
                            "rel": "exact",
                            "weight": _anchor_weight(t.get("type", ""), "exact"),
                            "context": ctx})
            continue
        for nl, t in by_name.items():
            if len(el) >= 4 and len(nl) >= 4 and (el in nl or nl in el):
                anchors.append({"entity": e["name"],
                                "target": {"type": t["type"], "name": t["name"],
                                           "parent": t.get("parent", "")},
                                "rel": "partial",
                                "weight": _anchor_weight(t.get("type", ""), "partial"),
                                "context": ctx})
                break
    return anchors


def suggest_focus(query: str, anchors: list[dict]) -> str:
    """Целевой метод из вопроса через якоря (вес: exact по типу, partial=1).

    '' если мимо.
    """
    terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.lower()))
    scores: dict[str, float] = {}
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
            w = float(a.get("weight", 2.0 if a.get("rel") == "exact" else 1.0))
            scores[key] = scores.get(key, 0.0) + w
            if key not in order:
                order.append(key)
    if not scores:
        return ""
    best = max(scores.values())
    return next(k for k in order if scores[k] == best)


def resolve_focus_target(parsed: dict, anchors: list[dict], arg) -> str:
    """Имя целевого метода: точное имя > якоря из вопроса > '' (авто по умолчанию).

    arg: None/'' — авто; 'onHit' — точное имя; 'где чинится урон?' — вопрос.
    """
    methods = (parsed or {}).get("methods", [])
    if not methods:
        return ""
    if arg and str(arg).strip():
        a = str(arg).strip()
        for m in methods:
            if m["name"].lower() == a.lower():
                return m["name"]
        sug = suggest_focus(a, anchors or [])
        base = sug.split(".")[-1]
        for m in methods:
            if m["name"] == base:
                return m["name"]
    return ""


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
# П.8 PLAN: чанки по предложениям с нахлестом (вместо реза по 800 символов)
# ---------------------------------------------------------------------------

def split_prose_chunks(text: str, max_chars: int = 800, overlap_chars: int = 150) -> list[str]:
    """Проза (.md/.txt): режем по границам предложений, с нахлестом.

    Нахлест — последние 1-2 предложения предыдущего чанка (до overlap_chars),
    чтобы связь на границе не рвалась. Детерминировано, без зависимостей.
    """
    if not text or not text.strip():
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n{2,}|\n", text) if s.strip()]
    if not sents:
        return [text[:max_chars]] if text.strip() else []
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        # очень длинное предложение — режем жестко, иначе чанк распухнет
        while len(s) > max_chars:
            if cur:
                chunks.append(" ".join(cur))
                # нахлест для следующего
                ov: list[str] = []
                ov_len = 0
                for prev in reversed(cur):
                    if ov_len + len(prev) + 1 > overlap_chars:
                        break
                    ov.insert(0, prev)
                    ov_len += len(prev) + 1
                cur, cur_len = list(ov), ov_len
            chunks.append(s[:max_chars])
            s = s[max_chars:]
        if cur_len + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            ov = []
            ov_len = 0
            for prev in reversed(cur):
                if ov_len + len(prev) + 1 > overlap_chars:
                    break
                ov.insert(0, prev)
                ov_len += len(prev) + 1
            cur, cur_len = list(ov), ov_len
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


def split_code_chunks(text: str, max_chars: int = 800, overlap_lines: int = 5) -> list[str]:
    """Код (.java/.py): режем по строкам с нахлестом, чтобы не рвать метод."""
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for ln in lines:
        ll = len(ln) + 1
        if cur_len + ll > max_chars and cur:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_lines:] if overlap_lines > 0 else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(ln)
        cur_len += ll
    if cur and any(x.strip() for x in cur):
        chunks.append("\n".join(cur))
    return [c for c in chunks if c.strip()]


def split_text_chunks(text: str, ext: str = ".txt", max_chars: int = 800,
                      overlap_chars: int = 150) -> list[str]:
    """Диспетчер чанкинга: проза — по предложениям, код — по строкам."""
    if (ext or "").lower() in (".md", ".txt", ".rst"):
        out = split_prose_chunks(text, max_chars, overlap_chars)
    else:
        out = split_code_chunks(text, max_chars, overlap_lines=5)
    if not out and text.strip():
        out = [text[:max_chars]]
    return out


# ---------------------------------------------------------------------------
# П.5 PLAN: тело по требованию + П.10: проверка ответом
# ---------------------------------------------------------------------------

def _python_method_bodies(content: str) -> dict:
    """Тела Python-методов через ast (для --get-body). Безопасно: пусто при ошибке."""
    try:
        tree = ast.parse(content)
    except Exception:
        return {}
    out: dict[str, str] = {}
    try:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    seg = ast.get_source_segment(content, node)
                except Exception:
                    seg = None
                if seg:
                    out.setdefault(node.name, seg[:4000])
    except Exception:
        pass
    return out


def extract_method_body(text: str, method: str) -> dict:
    """П.5: полный текст конкретного метода — вместо подгрузки всего файла.

    Пробует Java-парсер, затем Python-AST. Возвращает dict:
    {found, method, class, body, body_bytes, parser}.
    """
    name = (method or "").strip()
    if not name or not text:
        return {"found": False, "method": name, "class": "",
                "body": "", "body_bytes": 0, "parser": "none"}
    # Java-путь: пробуем при любом Java-подобном коде (class + { + ;),
    # не только при строгом is_java_like (там требуется import — мелкие
    # примеры без импортов иначе не находятся).
    looks_java = is_java_like(text) or ("class " in text and "{" in text and ";" in text)
    if looks_java:
        try:
            cl, _ = smart_router(text)
            parsed = java_parse("".join(cl) if cl else text)
            for m in parsed.get("methods", []):
                if m["name"].lower() == name.lower():
                    return {"found": True, "method": m["name"],
                            "class": m.get("class", ""), "body": m.get("body", ""),
                            "body_bytes": m.get("body_bytes", bytes_len(m.get("body", ""))),
                            "parser": parsed.get("parser", ""),
                            "contract": m.get("contract", ""),
                            "annotations": m.get("annotations", [])}
        except Exception:
            pass
    # Python-путь
    try:
        bodies = _python_method_bodies(text)
        for k, v in bodies.items():
            if k.lower() == name.lower():
                return {"found": True, "method": k, "class": "",
                        "body": v, "body_bytes": bytes_len(v),
                        "parser": "ast", "contract": _extract_contract(v, ""),
                        "annotations": []}
        # частичное совпадение от 4 символов (честно помечаем)
        for k, v in bodies.items():
            if len(name) >= 4 and (name.lower() in k.lower() or k.lower() in name.lower()):
                return {"found": True, "method": k, "class": "",
                        "body": v, "body_bytes": bytes_len(v),
                        "parser": "ast", "contract": _extract_contract(v, ""),
                        "annotations": [], "rel": "partial"}
    except Exception:
        pass
    return {"found": False, "method": name, "class": "",
            "body": "", "body_bytes": 0, "parser": "none"}


def verify_answer(answer: str, sources) -> dict:
    """П.10 PLAN: все имена/числа из ответа обязаны найтись в исходниках.

    sources: str | list[str] | dict с 'context'/'skeleton_text'/etc.
    Возвращает {ok, unknown_names, unknown_numbers, checked_*}.
    Ловит галлюцинации о деталях, не про смысл.
    """
    if isinstance(sources, dict):
        parts: list[str] = []
        for k in ("context", "skeleton_text", "cleaned_text", "raw_text", "body"):
            v = sources.get(k)
            if isinstance(v, str) and v:
                parts.append(v)
        # вложенные items для batch
        items = sources.get("items")
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    sk = ((it.get("skeleton") or {}) if isinstance(it.get("skeleton"), dict) else {})
                    if isinstance(sk.get("text"), str):
                        parts.append(sk["text"])
        src_text = "\n".join(parts)
    elif isinstance(sources, (list, tuple)):
        src_text = "\n".join(s for s in sources if isinstance(s, str))
    else:
        src_text = sources if isinstance(sources, str) else ""
    src_low = src_text.lower()
    names: list[str] = []
    seen: set[str] = set()
    for m in _TEXT_TOKEN_RE.finditer(answer or ""):
        w = m.group(0).strip("._")
        if len(w) < 3 or w.lower() in _PROSE_STOP:
            continue
        if w.lower() not in seen:
            seen.add(w.lower())
            names.append(w)
    nums: list[str] = []
    seen_n: set[str] = set()
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", answer or ""):
        if m.group(0) not in seen_n:
            seen_n.add(m.group(0))
            nums.append(m.group(0))
    unknown_names = [w for w in names if w.lower() not in src_low]
    # числа проверяем мягко: ищем как подстроку (1024 найдется в "maxLimit = 1024")
    unknown_numbers = [n for n in nums if n not in src_text]
    return {"ok": not unknown_names and not unknown_numbers,
            "unknown_names": unknown_names[:20],
            "unknown_numbers": unknown_numbers[:20],
            "checked_names": len(names), "checked_numbers": len(nums)}


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
