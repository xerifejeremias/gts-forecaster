@echo off
rem Run one automatic job of GTS Forecaster (called by Windows Task Scheduler; see config\schedule.yaml).
rem Usage: scripts\scheduled_run.bat --frequency monthly --job forecast
rem The output is appended to runs\scheduler.log; the run itself is recorded in runs\<frequency>\<run id>\.
cd /d "%~dp0\.."
set PYTHONUTF8=1
if not exist runs mkdir runs
echo. >> runs\scheduler.log
echo ===== %DATE% %TIME%  %* ===== >> runs\scheduler.log
".venv\Scripts\python.exe" -m gts_forecaster.ops.jobs %* >> runs\scheduler.log 2>&1
set CODE=%ERRORLEVEL%
echo ===== exit code %CODE% ===== >> runs\scheduler.log
exit /b %CODE%
