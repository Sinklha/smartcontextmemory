# SCM Pro — структурный компрессор контекста для кода

Индексирует код, строит скелеты/дерево (tree-sitter для Java, ast для Python),
в LLM отправляет только нужное: скелет для обзора, тело 1 метода + его
зависимости для фикса. Метрики честные (`len(bytes)`, без `getsizeof`).

## Установка

```bat
install.bat
```
Ставит зависимости, кладет скилл `scm-context` в OpenCode.
Другой хост (Claude Code, Cursor, Codex): 
```bat
python install.py --host claude
```
SKILL.md — стандартный Agent Skills, переделок не требует.
Быстрое демо без модели (10 секунд):
```bat
demo.bat
```

## Запуск вручную

Демо пайплайна:
```bat
python main.py
```

Интерактивная консоль (вопросы по проекту, индекс кэшируется на диске):
```bat
python chat.py
```
Принудительная пересборка индекса (если что-то пошло не так):
```bat
python chat.py --reindex
```

Один вопрос без интерактива:
```bat
python chat.py --once "Каковы лимиты памяти в коде?"
```

Индексация своей папки с кодом (.java/.py/.md/.txt, до 2000 файлов и 20 МБ):
```bat
python chat.py --repo path\to\project --once "где обработка урона?"
```

Анализ своего кода/текста с экономией токенов:
```bat
python chat.py --analyze-file path\to\File.java
```

Анализ целой папки одним вызовом (итоги + скелеты по файлам):
```bat
python chat.py --analyze-batch path\to\project
```

Тело конкретного метода без всего файла (П.5, машинный — с `--json`):
```bat
python chat.py --analyze-file path\to\File.java --get-body checkMemory
```

Проверка ответа модели по источникам (П.10, ловля галлюцинаций):
```bat
python chat.py --analyze-file path\to\File.java --verify "лимит 1024 в maxLimit"
python chat.py --repo path\to\project --verify "ответ модели" --against path\to\File.java
```

Бенчмарк потерь ретривера (П.11, recall@1/recall@k без LLM):
```bat
python chat.py --repo path\to\project --bench
python chat.py --bench --json
```

Тесты (stdlib only, без FAISS/torch):
```bat
python tests/test_scm.py
python -m pytest tests/ -q
```

Внутри консоли: `/code` — вставить свой код (конец — строка `END`),
`exit` — выход. Живые ответы — через Ollama на `localhost:11434`
(`ollama pull qwen2.5:3b`), без нее консоль показывает честный контекст
и метрики без выдуманного ответа.

## Файлы

- `scm_core.py` — ядро: роутер, чистка (TODO/FIXME сохраняются), скелеты
  со значениями/дефолтами/аннотациями, Java-дерево (tree-sitter +
  regex-фолбэк), контракты, якоря с контекстом, флаг отрицания,
  чанки по предложениям с нахлестом, verify, zlib-хранение, VRAM-симуляция
- `main.py` — демо end-to-end на `input_data.txt`
- `chat.py` — консоль: FAISS-поиск, фокус на метод (соседи по вызовам +
  по полям), анализ вставок, `--get-body` / `--verify` / `--bench`
- `serve.py` — серверный режим (модель грузится раз): `POST /ask`
  (ключи hits/context/saving_pct + `verify`, `has_negation` в хитах)
- `tests/test_scm.py` — 19 регрессионных тестов (stdlib only)
- `SKILL.md` — описание скилла для моделей: воркфлоу, команды, правила
  (машинный вывод — те же команды с `--json`)
- `input_data.txt` — тестовые данные для демо
- `.scm_index/` — кэш FAISS-индекса (создается сам, можно удалять)

## Честные ограничения

- VRAM — симуляция на dict, не real CUDA (смотри `torch.cuda` сам)
- zlib — только хранение в RAM, модель его не читает
- Демо-Java-парсер пропускает сложный синтаксис; прод — tree-sitter
- Токены без `tiktoken` считаются эвристикой ~4 символа
