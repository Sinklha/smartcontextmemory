# SCM Pro — AST-driven prompt compression (skill core).
# Linux-based image: тот же код что на Windows, без .bat.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# faiss-cpu нужен компилятор на slim — ставим минимум для сборки.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY scm_core.py chat.py serve.py main.py _ts_worker.py SKILL.md input_data.txt ./
COPY tests/ ./tests/

# Кэш индекса живет внутри контейнера, наружу не торчит.
VOLUME ["/app/.scm_index"]

EXPOSE 8000

# Проверка: парсер + скелет без модели и сети.
RUN python -m unittest discover -s tests

CMD ["python", "serve.py", "--port", "8000"]
