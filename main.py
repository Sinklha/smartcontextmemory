"""SCM Pro — демо-точка входа. Все метрики честные (len(bytes), без getsizeof)."""
import sys
import asyncio
import hashlib
import time
import warnings as _warnings

_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", message=".*LangChain.*")

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import scm_core


def cuda_status() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (f"CUDA доступна: {torch.cuda.get_device_name(0)} | "
                    f"allocated={torch.cuda.memory_allocated()}B")
        return "CUDA недоступна -> VRAM-метрики = СИМУЛЯЦИЯ на dict"
    except Exception as e:
        return f"torch нет/ошибка ({e}) -> VRAM-метрики = СИМУЛЯЦИЯ"


def build_vector_db(chunks: list[str], cross_hash: str, embeddings):
    from langchain_community.vectorstores import FAISS
    metadatas = [{"cross_hash": cross_hash, "chunk_id": i} for i in range(len(chunks))]
    return FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)


def keyword_fallback(chunks: list[str], query: str) -> tuple[int, str]:
    """Честный фолбэк без эмбеддингов: пересечение слов."""
    q = set(query.lower().split())
    best_i, best_score = 0, -1
    for i, ch in enumerate(chunks):
        score = len(q & set(ch.lower().split()))
        if score > best_score:
            best_i, best_score = i, score
    return best_i, chunks[best_i] if chunks else (0, "")


async def run_smart_context_memory():
    input_file = "input_data.txt"
    print("--- Запуск Smart Context Memory (SCM Pro) ---")
    print(f"[ENV] {cuda_status()}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = f.read()
    except FileNotFoundError:
        raw_data = "def process_data(): return True\n" * 50 + "Документация к функциям.\n"

    raw_bytes = len(raw_data.encode("utf-8"))
    tok_raw, tok_method = scm_core.estimate_tokens(raw_data)
    print(f"[INPUT] {input_file}: {raw_bytes}B, ~{tok_raw} tok ({tok_method})")

    # --- Модуль 1: Router + Cross-Hash ---
    code_lines, text_lines = scm_core.smart_router(raw_data)
    cross_hash_id = hashlib.sha256(raw_data.encode()).hexdigest()[:8]
    print(f"[Router] строк кода: {len(code_lines)}, текста: {len(text_lines)} "
          f"| Cross-Hash: [{cross_hash_id}]")

    # --- Модуль 2: нормальное сжатие ---
    t0 = time.perf_counter()
    code_p = scm_core.compress_code_block(code_lines)
    text_p = scm_core.compress_text_semantic(text_lines)
    compress_ms = (time.perf_counter() - t0) * 1000

    # Что реально увидит LLM: СКЕЛЕТ + текст (полное тело — только вспышкой).
    # zlib — только для хранения в RAM, модель его не читает.
    llm_context_bytes = code_p["skeleton_bytes"] + text_p["cleaned_bytes"]
    llm_tokens = scm_core.estimate_tokens(code_p["skeleton_text"] + text_p["cleaned_text"])[0]
    ram_storage_bytes = code_p["compressed_bytes"] + text_p["cleaned_bytes"]
    context_saving = (1 - llm_context_bytes / raw_bytes) * 100 if raw_bytes else 0
    storage_saving = (1 - ram_storage_bytes / raw_bytes) * 100 if raw_bytes else 0

    print(f"[Compress] code: {code_p['raw_bytes']}B -> cleaned {code_p['cleaned_bytes']}B "
          f"(-{code_p['semantic_saving_pct']}%) -> zlib {code_p['compressed_bytes']}B "
          f"(-{code_p['storage_saving_pct']}%) | encode {code_p['encode_ms']}ms")
    print(f"[Skeleton:{code_p['skeleton_method']}] {code_p['skeleton_bytes']}B "
          f"(-{code_p['skeleton_saving_pct']}%) | в LLM едет скелет, тело — вспышкой")
    print(f"[Skeleton] {code_p['skeleton_text'][:200]!r}")
    print(f"[Compress] text: {text_p['raw_bytes']}B -> cleaned {text_p['cleaned_bytes']}B "
          f"(-{text_p['saving_pct']}%) | tokens {text_p['tokens_raw']}->{text_p['tokens_cleaned']}")

    # --- Модуль 3: FAISS по чанкам + паспорт ---
    chunks = [c for c in text_p["cleaned_text"].splitlines() if len(c.strip()) > 3]
    if not chunks:
        chunks = [text_p["cleaned_text"] or "(пусто)"]
    print(f"[FAISS] чанков для индекса: {len(chunks)}")

    vector_db = None
    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # новый пакет
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # legacy
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = build_vector_db(chunks, cross_hash_id, embeddings)
        print("[FAISS] индекс построен (all-MiniLM-L6-v2)")
    except Exception as e:
        print(f"[FAISS] WARN: эмбеддинги недоступны ({e}). Использую keyword-фолбэк.")

    code_passport = scm_core.create_block_passport(
        cross_hash_id, "CODE_ZLIB", code_p["compressed_bytes"])
    print(f"[Passport] {code_passport}")

    # --- Модуль 4: Pre-fetch + EOS (симуляция, честно помечена) ---
    print("\n--- Симуляция Модуля 4 (Pre-fetch & EOS, SIM) ---")
    query = "Каковы лимиты памяти в коде?"
    print(f"Запрос: '{query}'")

    retrieved = ""
    if vector_db is not None:
        t1 = time.perf_counter()
        docs = vector_db.similarity_search_with_score(query, k=1)
        retr_ms = (time.perf_counter() - t1) * 1000
        doc, score = docs[0]
        retrieved = doc.page_content
        found_hash = doc.metadata.get("cross_hash", "?")
        print(f"[RETRIEVE] score={score:.4f} (L2, меньше=ближе), "
              f"hash=[{found_hash}], {retr_ms:.1f}ms")
    else:
        bi, retrieved = keyword_fallback(chunks, query)
        print(f"[RETRIEVE] keyword-фолбэк, chunk_id={bi}")
    print(f"[RETRIEVE] текст: {retrieved[:200]!r}")

    # Декомпресс = то, что реально уйдет в модель
    t2 = time.perf_counter()
    code_for_llm = scm_core.decompress_code_block(code_p)
    decode_ms = (time.perf_counter() - t2) * 1000
    assert code_for_llm == code_p["cleaned_text"], "zlib round-trip нарушен!"
    print(f"[PRE-FETCH SIM] декомпресс {code_p['compressed_bytes']}B -> "
          f"{code_p['cleaned_bytes']}B за {decode_ms:.2f}ms (по шине PCIe — симуляция)")
    code_passport["location"] = "VRAM (sim)"
    print(f"[GEN SIM] фрагмент для LLM: {code_for_llm[:160]!r}...")

    await asyncio.sleep(0.1)
    code_passport["location"] = "RAM"
    print("[EVENT CLEANUP SIM] [EOS] получен: блок удален из VRAM-симуляции.")

    # --- Честный отчет ---
    print("\n--- Метрики (честные, len(bytes)) ---")
    print(f"RAW файл:              {raw_bytes}B (~{tok_raw} tok)")
    print(f"Контекст для LLM:      {llm_context_bytes}B (~{llm_tokens} tok) "
          f"-> экономия контекста {context_saving:.1f}% (скелет+текст, тело вспышкой)")
    print(f"Хранение в RAM:        {ram_storage_bytes}B "
          f"-> экономия хранения {storage_saving:.1f}% (zlib только для хранения!)")
    print(f"Время сжатия:          {compress_ms:.1f}ms "
          f"(encode {code_p['encode_ms']}ms / decode {code_p['decode_ms']}ms)")
    print("VRAM в покое:          симуляция ~0B реального GPU (паспорт only). "
          "Реальный KV-cache меряй через torch.cuda.memory_allocated().")
    print("ВАЖНО: zlib-байты модель не читает. Выигрыш для LLM = скелет; "
          "zlib = выигрыш для RAM/диска.")


if __name__ == "__main__":
    asyncio.run(run_smart_context_memory())
