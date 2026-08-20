@echo off
cd /d "%~dp0"
echo Starting Anima Training UI...
call npm start -- %*
echo.
echo Application exited (check for errors above).
pause
