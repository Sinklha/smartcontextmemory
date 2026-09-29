#!/bin/sh
# SCM Pro installer (Linux): install.sh -> install.py
set -e
cd "$(dirname "$0")"
python3 install.py "$@"
