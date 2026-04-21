@echo off
setlocal

REM Auto stage, commit, and push current branch.
REM Usage:
REM   auto_commit_push.bat
REM   auto_commit_push.bat Your commit message here

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Not inside a Git repository.
  exit /b 1
)

for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set BRANCH=%%b
if not defined BRANCH (
  echo [ERROR] Could not detect current branch.
  exit /b 1
)

if "%~1"=="" (
  set "MSG=Auto commit %DATE% %TIME%"
) else (
  set "MSG=%~1"
)

echo [INFO] Branch: %BRANCH%
echo [INFO] Staging changes...
git add -A
if errorlevel 1 (
  echo [ERROR] git add failed.
  exit /b 1
)

git diff --cached --quiet
if errorlevel 1 (
  echo [INFO] Creating commit...
  git commit -m "%MSG%"
  if errorlevel 1 (
    echo [ERROR] git commit failed.
    exit /b 1
  )
) else (
  echo [INFO] No staged changes to commit. Skipping commit.
)

echo [INFO] Pushing to origin/%BRANCH%...
git push origin %BRANCH%
if errorlevel 1 (
  echo [ERROR] git push failed.
  exit /b 1
)

echo [OK] Done.
exit /b 0
