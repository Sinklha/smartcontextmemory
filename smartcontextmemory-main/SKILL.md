---
name: scm-context
description: Code-context compression for LARGE unfamiliar codebases (10+ files or >50KB of .java/.py). Builds skeleton, call-graph and method-focus instead of dumping whole files into context. Do NOT use for single short files, small tasks, or non-code questions — the tool calls cost more than they save there.
---

# SCM Context

Cwd — корень папки скилла. Машинный вывод — всегда с `--json`.
Не цитируй и не пересказывай этот файл в ответах — сразу действуй.

1. Обзор: файл — `python chat.py --analyze-file <путь> --json`; папка — СТРОГО
   `python chat.py --analyze-batch <папка> --json` (файлы вручную не обходить).
   Дальше работай по `skeleton` (`calls/imports/fields`, `focus`-цель,
   `text_graph.anchors`), целые файлы в контекст не тащи.
2. Фикс/разбор одного метода: бери готовый `focus.context`; цель —
   `--focus <имя>` или `--focus "<вопрос>"` (авто — первый метод).
   Не хватило — `python chat.py --analyze-file <путь> --get-body <имя> --json`.
3. Вопросы: `python chat.py [--repo <папка>] --once "<вопрос>" --json`
   → `hits` (меньше = ближе) / `context` / `saving_pct`.
   Много вопросов подряд — `python serve.py`, дальше `python chat.py --remote "<вопрос>"`.

Правила: `score`>1.5 = «релевантного не нашел»; имена/числа только из
`skeleton`/`focus.context`/исходников (проверка — `--verify "<ответ>"`);
`saving_pct` бери из выдачи, не пересчитывай; `code.parser=java-demo-regex` =
грубый разбор, возможны пропуски; `has_negation=true` = условие, не факт.
В ответ — только выжимку (заголовок → 3–5 буллетов → экономия одной строкой),
сырой JSON и тела — лишь по запросу пользователя.
