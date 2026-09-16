"""Воркер tree-sitter: разбирает Java из stdin, отдает JSON в stdout.

Зачем отдельным процессом: свежие Windows-сборки py-tree-sitter могут
ронять процесс (access violation при массовых разборах). Падение воркера =
ненулевой exit, родитель (java_prod_parse) ловит и уходит в демо-фолбэк,
а хост (консоль/скилл) продолжает жить.
Протокол: stdin = исходник utf-8, stdout = parsed-dict JSON.
"""
import sys

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    import json
    try:
        import scm_core
    except Exception as e:
        sys.stderr.write(f"import: {e}")
        return 2
    try:
        code = sys.stdin.buffer.read().decode("utf-8")
        out = scm_core._java_prod_parse_local(code)
    except Exception as e:
        sys.stderr.write(f"{type(e).__name__}: {str(e)[:300]}")
        return 1
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
