"""SCM Pro — минимальный регресс (stdlib unittest, без модели и сети).

Запуск из корня проекта:  python -m unittest discover -s tests -v
Падавший раньше TypeConverter-кейс покрыт через Bukkit-образец ниже.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scm_core

SHOP_JAVA = """import java.util.HashMap;
import org.bukkit.Bukkit;

public class Shop {
    private HashMap<String, Long> last = new HashMap<>();
    private int limit = 5;

    /** Called on buy. */
    public void onBuy(String name) {
        last.put(name, 1L);
        check(name);
        Bukkit.getLogger().info(name);
        this.check(name);
    }

    private boolean check(String name) {
        return name.length() < limit;
    }

    private void a() {}
    private void g() { Bukkit.getLogger().info("g"); }
}
"""

NESTED_JAVA = """public class Outer {
    static int counter = 0;
    class Nested { void hidden() {} }
    interface Inner { default void dflt() {} }
    enum Color { RED, GREEN; }
    public <T> T work(T x) { counter++; return x; }
}
"""

PY_CODE = '"""Doc."""\nimport sqlite3\n\n\ndef connect(port=5432):\n    """Connect."""\n    return sqlite3.connect("x.db")\n'


class TestMetrics(unittest.TestCase):
    def test_bytes_len(self):
        self.assertEqual(scm_core.bytes_len("abc"), 3)
        self.assertGreater(scm_core.bytes_len("привет"), 6)

    def test_estimate_tokens_shape(self):
        n, how = scm_core.estimate_tokens("hello world")
        self.assertGreater(n, 0)
        self.assertTrue(how)


class TestRouterClean(unittest.TestCase):
    def test_router_splits(self):
        code, text = scm_core.smart_router("def f():\npass\nПросто текст\n")
        self.assertTrue(any("def" in line for line in code))
        self.assertTrue(any("Просто" in line for line in text))

    def test_clean_keeps_structure(self):
        out = scm_core.semantic_clean_code("# full\ndef f():  # inline\n    x = 1\n\n")
        self.assertNotIn("# full", out)
        self.assertIn("def f():", out)
        self.assertIn("    x = 1", out)

    def test_clean_keeps_todo_fixme(self):
        # Пункт из PLAN: метки не режем — они едут в LLM.
        out = scm_core.semantic_clean_code(
            "# обычный коммент\n# TODO: починить урон\n// FIXME: краш\ndef f():\n    pass\n")
        self.assertNotIn("обычный коммент", out)
        self.assertIn("TODO", out)
        self.assertIn("FIXME", out)

    def test_compress_roundtrip(self):
        p = scm_core.compress_code_block("def f():\n    pass\n")
        self.assertEqual(scm_core.decompress_code_block(p), p["cleaned_text"])
        self.assertLessEqual(p["compressed_bytes"], p["raw_bytes"] + 8)


class TestPythonSkeleton(unittest.TestCase):
    def test_ast(self):
        sk = scm_core.ast_skeleton(PY_CODE)
        self.assertEqual(sk["method"], "ast")
        names = [n["name"] for n in sk["nodes"]]
        self.assertIn("connect", names)


class TestJavaProd(unittest.TestCase):
    def test_shop_tree(self):
        p = scm_core.java_parse(SHOP_JAVA)
        self.assertEqual(p["parser"], "tree-sitter-java")
        names = [m["name"] for m in p["methods"]]
        self.assertEqual(names, ["onBuy", "check", "a", "g"])
        onbuy = next(m for m in p["methods"] if m["name"] == "onBuy")
        # map.put / Bukkit.info с чужим получателем — не связи
        self.assertEqual(onbuy["calls"], ["check"])
        self.assertIn("org.bukkit.Bukkit", onbuy["uses_imports"])
        self.assertIn("last", onbuy["uses_fields"])

    def test_nested_excluded(self):
        p = scm_core.java_parse(NESTED_JAVA)
        names = [m["name"] for m in p["methods"]]
        self.assertIn("work", names)
        self.assertNotIn("hidden", names)
        self.assertNotIn("dflt", names)

    def test_broken_falls_back(self):
        p = scm_core.java_parse("public class A { void f( { ;;;")
        self.assertIn(p["parser"], ("java-demo-regex", "tree-sitter-java"))

    def test_focus_target(self):
        p = scm_core.java_parse(SHOP_JAVA)
        f = scm_core.build_method_focus(p, "check")
        self.assertEqual(f["target"], "Shop.check")
        self.assertIn("check", f["context"])


class TestTextGraph(unittest.TestCase):
    def test_entities_values_anchors(self):
        g = scm_core.build_text_graph("Лимит maxLimit = 1024 для MemoryManager.")
        names = [e["name"] for e in g["entities"]]
        self.assertIn("maxLimit", names)
        self.assertIn("MemoryManager", names)
        self.assertEqual(g["values"].get("maxLimit"), "1024")
        a = scm_core.anchor_to_code(
            g, [{"type": "field", "name": "maxLimit"},
                {"type": "class", "name": "MemoryManager"}])
        self.assertTrue(all(x["rel"] == "exact" for x in a))
        self.assertEqual(len(a), 2)

    def test_suggest(self):
        g = scm_core.build_text_graph("Как чинить checkMemory?")
        a = scm_core.anchor_to_code(
            g, [{"type": "method", "name": "checkMemory", "parent": "M"}])
        self.assertEqual(scm_core.suggest_focus("Как чинить checkMemory?", a),
                         "M.checkMemory")
        self.assertEqual(scm_core.suggest_focus("привет", a), "")


class TestAnalyzeResult(unittest.TestCase):
    def test_json_shape(self):
        import chat
        r = chat.analyze_result(SHOP_JAVA, label="t")
        self.assertNotIn("error", r)
        self.assertIn("skeleton", r)
        self.assertIn("focus", r)
        self.assertTrue(r["focus"]["target"].endswith("onBuy"))
        json.dumps(r, ensure_ascii=False)  # сериализуемость

    def test_focus_arg(self):
        import chat
        r = chat.analyze_result(SHOP_JAVA, focus_arg="check")
        self.assertEqual(r["focus"]["target"], "Shop.check")


class TestHybridSearch(unittest.TestCase):
    DOCS = [
        ("a.java", 0, "public void onDamage(EntityDamageByEntityEvent e) { check(e); }"),
        ("b.java", 0, "Привет! Документация про лимиты памяти и подключения."),
        ("c.java", 0, "class MemoryManager { int maxLimit = 1024; }"),
    ]

    def test_bm25_exact_name_wins(self):
        hits = __import__("chat").bm25_search(self.DOCS, "maxLimit", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1]["file"], "c.java")

    def test_fuse_agreement(self):
        import chat
        vec = [(0.3, {"file": "a.java", "chunk_id": 0}, "t1"),
               (1.9, {"file": "b.java", "chunk_id": 0}, "t2")]
        bm = [(2.0, {"file": "a.java", "chunk_id": 0}, "t1")]
        fused = chat.hybrid_fuse(vec, bm, k=2)
        self.assertEqual(fused[0][1]["file"], "a.java")
        self.assertTrue(all(isinstance(s, float) for s, _, _ in fused))

    def test_retrieve_found_flag(self):
        import chat
        r = chat.retrieve(None, self.DOCS, {}, 100, "maxLimit")
        self.assertTrue(r["found"])
        self.assertIn("found", r)
        r2 = chat.retrieve(None, self.DOCS, {}, 100, "xyzzy qqqqq zzzzz")
        self.assertFalse(r2["found"])


class TestAstHonesty(unittest.TestCase):
    """Связи только из AST-узлов, а не regex по тексту (п.3 аудита)."""

    COMMENT_TRAP = """import org.bukkit.Bukkit;

public class Trap {
    private int limit = 5;

    public void onHit() {
        // check() в комментарии — не вызов, "limit" в строке — не поле
        String s = "limit check Bukkit";
        real();
    }

    private void real() {}
    private void check() {}
}
"""

    BRACE_NEXT_LINE = """public class Next {
    public void onHit()
    {
        heal();
    }

    private void heal() {}
}
"""

    def test_comment_and_string_are_not_links(self):
        p = scm_core.java_parse(self.COMMENT_TRAP)
        self.assertEqual(p["parser"], "tree-sitter-java")
        onhit = next(m for m in p["methods"] if m["name"] == "onHit")
        # check только в комментарии/строке → не связь; real — да
        self.assertEqual(onhit["calls"], ["real"])
        # limit только в строке → не поле; Bukkit только в строке → не импорт
        self.assertEqual(onhit["uses_fields"], [])
        self.assertEqual(onhit["uses_imports"], [])

    def test_brace_on_next_line(self):
        p = scm_core.java_parse(self.BRACE_NEXT_LINE)
        names = [m["name"] for m in p["methods"]]
        self.assertIn("onHit", names)
        self.assertIn("heal", names)
        onhit = next(m for m in p["methods"] if m["name"] == "onHit")
        self.assertEqual(onhit["calls"], ["heal"])


class TestJavaFocus(unittest.TestCase):
    """Фокус на метод: автовыбор хендлера, соседи, импорты, глубина."""

    MULTI_HANDLER = """import org.bukkit.event.EventHandler;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerJoinEvent;

public class Combat {
    public void onHit() {
        applyDamage();
    }

    @EventHandler
    public void onDamage(EntityDamageByEntityEvent e) {
        applyDamage();
    }

    @EventHandler
    public void onJoin(PlayerJoinEvent e) {
    }

    private void applyDamage() {}
}
"""

    CHAIN = """public class Chain {
    public void onHit() {
        applyDamage();
    }

    private void applyDamage() {
        healIfNeeded();
    }

    private void healIfNeeded() {}
}
"""

    def test_auto_focus_picks_first_handler(self):
        p = scm_core.java_parse(self.MULTI_HANDLER)
        self.assertEqual(p["parser"], "tree-sitter-java")
        f = scm_core.build_method_focus(p, None)
        # авто: первый @EventHandler, а не первый метод файла
        self.assertEqual(f["target"], "Combat.onDamage")

    def test_explicit_focus_second_handler(self):
        p = scm_core.java_parse(self.MULTI_HANDLER)
        f = scm_core.build_method_focus(p, "onJoin")
        self.assertEqual(f["target"], "Combat.onJoin")
        self.assertIn("PlayerJoinEvent", f["context"])

    def test_focus_neighbors_are_direct_calls(self):
        p = scm_core.java_parse(self.MULTI_HANDLER)
        f = scm_core.build_method_focus(p, "onDamage")
        self.assertIn("applyDamage", f["neighbors"])
        self.assertIn("applyDamage", f["context"])

    def test_focus_depth_is_one_step(self):
        # Текущее поведение: фокус берет прямых соседей (1 шаг).
        # 2 шага (onHit -> applyDamage -> healIfNeeded) — пункт из PLAN,
        # тест зафиксирует момент изменения поведения.
        p = scm_core.java_parse(self.CHAIN)
        f = scm_core.build_method_focus(p, "onHit")
        self.assertIn("applyDamage", f["neighbors"])
        self.assertNotIn("healIfNeeded", f["neighbors"])

    def test_event_type_extracted(self):
        p = scm_core.java_parse(self.MULTI_HANDLER)
        ondamage = next(m for m in p["methods"] if m["name"] == "onDamage")
        self.assertIn("EntityDamageByEntityEvent", ondamage["event"])

    def test_focus_imports_only_own(self):
        p = scm_core.java_parse(self.MULTI_HANDLER)
        f = scm_core.build_method_focus(p, "onJoin")
        joined = "\n".join(f["imports"])
        self.assertIn("PlayerJoinEvent", joined)
        self.assertNotIn("EntityDamageByEntityEvent", joined)


if __name__ == "__main__":
    unittest.main()
