@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo === Cai dat thu vien ImageEnhancement ===
python -m pip install --upgrade pip
if errorlevel 1 goto error
python -m pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Cai dat hoan tat. Chay run_windows.bat de mo ung dung.
pause
exit /b 0
:error
echo.
echo Cai dat that bai. Hay kiem tra Python va ket noi Internet.
pause
exit /b 1
