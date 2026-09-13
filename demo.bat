@echo off
REM SCM Pro - 10-second demo, no model or internet needed.
cd /d "%~dp0"
python chat.py --analyze-file input_data.txt
echo.
echo Full folder analysis: python chat.py --analyze-batch "Path\To\Folder"
