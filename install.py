"""SCM Pro — установщик: зависимости + скилл для AI-хоста.

Использование:
  install.bat                  — зависимости + OpenCode (по умолчанию)
  python install.py --host claude|codex|cursor|opencode|agents|all
SKILL.md — стандартный Agent Skills (name/description), понимается
Claude Code, Cursor, Codex, OpenCode без переделок.
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

HOSTS = {
    "opencode": (".config", "opencode", "skills"),
    "claude": (".claude", "skills"),
    "codex": (".codex", "skills"),
    "cursor": (".cursor", "skills"),
    "agents": (".agents", "skills"),
}


def run(*args):
    r = subprocess.run(list(args), cwd=BASE)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(args)}")


def install_skill(host):
    parts = HOSTS[host]
    dst_dir = os.path.join(os.path.expanduser("~"), *parts, "scm-context")
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(BASE, "SKILL.md"), encoding="utf-8") as f:
        text = f.read()
    text = text.replace("из корня папки скилла",
                        f"из рабочей папки `{BASE}`")
    with open(os.path.join(dst_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return dst_dir


def main():
    args = sys.argv[1:]
    host = "opencode"
    if "--host" in args:
        try:
            host = args[args.index("--host") + 1].lower()
        except IndexError:
            sys.exit("Использование: install.py --host claude|codex|cursor|opencode|agents|all")
    if host != "all" and host not in HOSTS:
        sys.exit(f"Неизвестный хост: {host} ({'/'.join(list(HOSTS) + ['all'])})")
    if sys.version_info < (3, 10):
        sys.exit("Нужен Python 3.10+")
    print("[1/3] Python", sys.version.split()[0])
    print("[2/3] Зависимости...")
    run(sys.executable, "-m", "pip", "install", "-r", "requirements.txt")
    targets = list(HOSTS) if host == "all" else [host]
    print("[3/3] Скилл scm-context ->", ", ".join(targets))
    for h in targets:
        print("  -", install_skill(h))
    print("Готово. Проверка: python chat.py --analyze-file input_data.txt")
    print("Демо без хоста вообще: demo.bat")


if __name__ == "__main__":
    main()
