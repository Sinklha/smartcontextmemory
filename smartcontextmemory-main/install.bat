@echo off
REM SCM Pro installer: install.bat -> install.py
chcp 65001 >nul
cd /d "%~dp0"
python install.py || exit /b 1
