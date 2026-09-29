#!/bin/sh
# SCM Pro — 10-second demo, no model or internet needed.
set -e
cd "$(dirname "$0")"
python3 chat.py --analyze-file input_data.txt
echo ""
echo "Full folder analysis: python3 chat.py --analyze-batch /path/to/folder"
