@echo off
chcp 65001 > nul
cd /d "%~dp0"
python run_gui.py
if errorlevel 1 (
  echo Khong the khoi dong ung dung. Hay chay setup_windows.bat truoc.
  pause
)
