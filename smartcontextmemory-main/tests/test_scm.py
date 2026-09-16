"""SCM Pro — регрессионные тесты (stdlib only, без FAISS/torch/Ollama).

Запуск без установки:  python tests/test_scm.py
Запуск через pytest:   python -m pytest tests/ -q
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import scm_core


def test_smart_comments_kept():
    code = "# TODO: починить лимит\nx = 1\n// FIXME: гонка\nint a = 1;\n"
    out = scm_core.semantic_clean_code(code)
    assert "TODO" in out, f"TODO потерян: {out!r}"
    assert "FIXME" in out, f"FIXME потерян: {out!r}"


def test_smart_inline_kept():
    code = "x = 1  # TODO: проверить переполнение\n"
    out = scm_core.semantic_clean_code(code)
    assert "TODO" in out, f"инлайн TODO потерян: {out!r}"


def test_plain_comments_still_cut():
    code = "# обычный коммент\nx = 1\n// обычный си\nint a = 1;\n"
    out = scm_core.semantic_clean_code(code)
    assert "обычный" not in out, f"мусор не вырезан: {out!r}"
    assert "x = 1" in out


def test_block_smart_kept():
    code = "/* TODO: важно */\nint a = 1;\n/* мусорный блок */\nint b = 2;\n"
    out = scm_core.semantic_clean_code(code)
    assert "TODO" in out, f"блок TODO потерян: {out!r}"
    assert "мусорный" not in out


def test_skeleton_has_defaults_and_consts():
    code = "PORT = 5432\ndef connect(port=5432, host='localhost'):\n    pass\n"
    sk = scm_core.ast_skeleton(code)
    assert "5432" in sk["skeleton_text"], f"значение потеряно: {sk['skeleton_text']!r}"
    assert "localhost" in sk["skeleton_text"] or "port" in sk["skeleton_text"]


def test_skeleton_has_annotations():
    code = "@app.route('/x')\ndef handler():\n    pass\n"
    sk = scm_core.ast_skeleton(code)
    assert "@" in sk["skeleton_text"], f"аннотация потеряна: {sk['skeleton_text']!r}"


def test_skeleton_field_value():
    code = "class C:\n    maxLimit: int = 1024\n    def f(self):\n        pass\n"
    sk = scm_core.ast_skeleton(code)
    assert "1024" in sk["skeleton_text"], f"поле без значения: {sk['skeleton_text']!r}"


def test_java_field_value_and_contract():
    code = ("public class C {\n"
            "    int maxLimit = 1024;\n"
            "    public void check(String x) {\n"
            "        if (x == null) return;\n"
            "        System.out.println(x);\n"
            "    }\n"
            "}\n")
    p = scm_core.java_demo_parse(code)
    assert p["fields"] and p["fields"][0]["name"] == "maxLimit"
    assert "1024" in (p["fields"][0].get("value") or ""), f"нет value: {p['fields']!r}"
    m = next(m for m in p["methods"] if m["name"] == "check")
    assert "null" in (m.get("contract") or "").lower(), f"нет контракта: {m!r}"


def test_field_neighbors():
    code = ("public class C {\n"
            "    int hp = 100;\n"
            "    public void damage(int d) { hp -= d; }\n"
            "    public void heal(int h) { hp += h; }\n"
            "    public void unrelated() { int z = 1; }\n"
            "}\n")
    p = scm_core.java_demo_parse(code)
    f = scm_core.build_method_focus(p, "damage")
    assert "heal" in f.get("field_neighbors", []), f"нет соседа по полю: {f!r}"
    assert "heal" in f["neighbors"]


def test_format_tree_shows_contract_and_fields():
    code = ("public class C {\n"
            "    int maxLimit = 1024;\n"
            "    public void check(String x) {\n"
            "        if (x == null) return;\n"
            "    }\n"
            "}\n")
    p = scm_core.java_demo_parse(code)
    t = scm_core.format_java_tree(p)
    assert "1024" in t, f"поле без значения в дереве: {t!r}"
    assert "контракт" in t.lower(), f"нет контракта в дереве: {t!r}"


def test_anchor_context_and_weight():
    tg = scm_core.build_text_graph("Урон чинится только когда все 5 флагов в checkDamage.")
    nodes = [{"type": "method", "name": "checkDamage", "parent": "C"}]
    anc = scm_core.anchor_to_code(tg, nodes, "Урон чинится только когда все 5 флагов в checkDamage.")
    assert anc, "якорей нет"
    a = anc[0]
    assert a.get("context"), "нет контекста якоря"
    assert float(a.get("weight", 0)) >= 2.0, f"вес мал: {a!r}"


def test_negation_flag():
    assert scm_core.detect_negation("только когда все 5")["has_negation"] is True
    assert scm_core.detect_negation("only if ready")["has_negation"] is True
    assert scm_core.detect_negation("простой факт про урон")["has_negation"] is False
    g = scm_core.build_text_graph("не готов, only if x")
    assert g["has_negation"] is True


def test_prose_chunks_overlap():
    text = "Первое предложение. Второе предложение! Третье? Четвертое. Пятое. Шестое."
    chunks = scm_core.split_prose_chunks(text, max_chars=40, overlap_chars=20)
    assert len(chunks) >= 2, f"мало чанков: {chunks!r}"
    # нахлест: конец первого встречается в начале второго
    assert any(w in chunks[1] for w in chunks[0].split()[-3:]), f"нет нахлеста: {chunks!r}"


def test_code_chunks_overlap():
    text = "\n".join(f"line{i} = {i};" for i in range(20))
    chunks = scm_core.split_code_chunks(text, max_chars=60, overlap_lines=2)
    assert len(chunks) >= 2
    assert "line" in chunks[0] and "line" in chunks[1]


def test_get_body_java_and_python():
    j = "public class C {\n public void hit() {\n int x = 1;\n }\n}\n"
    r = scm_core.extract_method_body(j, "hit")
    assert r["found"] and "int x = 1" in r["body"], f"java body: {r!r}"
    py = "def foo():\n    return 42\n"
    r2 = scm_core.extract_method_body(py, "foo")
    assert r2["found"] and "42" in r2["body"], f"py body: {r2!r}"
    r3 = scm_core.extract_method_body(py, "nope")
    assert not r3["found"]


def test_verify_answer():
    src = "class MemoryManager:\n    maxLimit = 1024\n    def checkMemory(): ..."
    ok = scm_core.verify_answer("Вызови checkMemory, лимит maxLimit.", src)
    assert ok["ok"], f"ложное срабатывание: {ok!r}"
    bad = scm_core.verify_answer("Вызови deleteEverything, лимит 9999.", src)
    assert not bad["ok"]
    assert "deleteEverything" in bad["unknown_names"]
    assert "9999" in bad["unknown_numbers"]


def test_zlib_roundtrip_still_ok():
    p = scm_core.compress_code_block("x = 1\n# коммент\n")
    assert scm_core.decompress_code_block(p) == p["cleaned_text"]


def test_floor_division_not_cut():
    code = "return max(1, len(text) // 4)\n"
    out = scm_core.semantic_clean_code(code)
    assert "// 4" in out, f"floor-division порезано: {out!r}"
    import ast as _ast
    _ast.parse(out)


def test_python_file_stays_ast():
    import pathlib
    base = pathlib.Path(__file__).resolve().parent.parent
    txt = (base / "scm_core.py").read_text(encoding="utf-8")
    cp = scm_core.compress_code_block(txt)
    assert cp["skeleton_method"] == "ast", f"Python ушел в Java: {cp['skeleton_method']}"


TESTS = [
    test_smart_comments_kept,
    test_smart_inline_kept,
    test_plain_comments_still_cut,
    test_block_smart_kept,
    test_skeleton_has_defaults_and_consts,
    test_skeleton_has_annotations,
    test_skeleton_field_value,
    test_java_field_value_and_contract,
    test_field_neighbors,
    test_format_tree_shows_contract_and_fields,
    test_anchor_context_and_weight,
    test_negation_flag,
    test_prose_chunks_overlap,
    test_code_chunks_overlap,
    test_get_body_java_and_python,
    test_verify_answer,
    test_zlib_roundtrip_still_ok,
    test_floor_division_not_cut,
    test_python_file_stays_ast,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
