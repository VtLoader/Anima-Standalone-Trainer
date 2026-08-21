@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set DISTUTILS_USE_SDK=1
"D:\anima_trainer_ref\venv\Scripts\python.exe" setup.py bdist_wheel > "%CD%\rebuild.log" 2>&1
