@echo off
rem First real run of GTS Forecaster: installs the fast model libraries, runs the tests, then a
rem monthly backtest, forecast, frozen-models forecast and HTML report. Everything is written to first_run_log.txt in the project root,
rem so the log can be read afterwards. Prophet and Orbit are NOT installed or run.
rem Usage: double-click, or in a terminal from the project root:  scripts\first_run.bat

cd /d "%~dp0\.."
set PYTHONUTF8=1
set LOG=first_run_log.txt
echo GTS Forecaster first run, started %DATE% %TIME% > %LOG%

where poetry >nul 2>&1
if errorlevel 1 (
  echo Poetry was not found on PATH. Install it first: https://python-poetry.org/docs/ >> %LOG%
  echo Poetry was not found on PATH. See %LOG%
  exit /b 1
)

set "STEP=poetry --version"
call :run poetry --version
set "STEP=installed Python versions (3.12 or 3.13 needed)"
call :run py -0p
rem Point Poetry at Python 3.13 or 3.12 if the py launcher knows one (the project needs 3.12+).
set "PYEXE="
for /f "delims=" %%P in ('py -3.13 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if defined PYEXE (
  set "STEP=poetry env use %PYEXE%"
  call :run poetry env use "%PYEXE%"
) else (
  echo No Python 3.12 or 3.13 found by the py launcher. Install one from python.org first. >> %LOG%
)
set "STEP=poetry env info"
call :run poetry env info
set "STEP=install (stats ml foundation)"
call :run poetry install --extras "stats ml foundation"
set "STEP=pytest"
call :run poetry run pytest -q
set "STEP=validate-input monthly"
call :run poetry run validate-input --frequency monthly
set "STEP=backtest monthly"
call :run poetry run backtest --frequency monthly
set "STEP=forecast monthly"
call :run poetry run forecast --frequency monthly
set "STEP=train monthly"
call :run poetry run train --frequency monthly
set "STEP=forecast monthly, frozen models"
call :run poetry run forecast --frequency monthly --mode frozen_models
set "STEP=report monthly"
call :run poetry run report --frequency monthly
set "STEP=run-pipeline monthly (all steps with a run record in runs\monthly)"
call :run poetry run run-pipeline --frequency monthly
set "STEP=benchmark monthly, 1000 series (timing estimate)"
call :run poetry run benchmark --frequency monthly --series 1000

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo Done. Log: %CD%\%LOG%
exit /b 0

:run
echo. >> %LOG%
echo ===== %STEP%  [%TIME%] ===== >> %LOG%
%* >> %LOG% 2>&1
echo ===== exit code: %ERRORLEVEL% ===== >> %LOG%
exit /b 0
