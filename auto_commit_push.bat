@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

where git >nul 2>&1
if errorlevel 1 (
    echo Git is not installed or not available in PATH.
    exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo This script must be run inside a Git repository.
    exit /b 1
)

for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "BRANCH=%%i"
if not defined BRANCH (
    echo Could not determine the current branch.
    exit /b 1
)

set "COMMIT_MSG=%~1"
if not defined COMMIT_MSG (
    set /p "COMMIT_MSG=Enter commit message: "
)

if not defined COMMIT_MSG (
    echo Commit message is required.
    exit /b 1
)

echo.
echo Staging changes...
git add -A
if errorlevel 1 (
    echo Failed to stage changes.
    exit /b 1
)

git diff --cached --quiet
if not errorlevel 1 (
    goto commit_changes
)

echo No staged changes to commit.
exit /b 0

:commit_changes
echo Creating commit on branch %BRANCH%...
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
    echo Commit failed.
    exit /b 1
)

echo Pushing to origin/%BRANCH%...
git push origin %BRANCH%
if errorlevel 1 (
    echo Push failed.
    exit /b 1
)

echo.
echo Commit and push completed successfully.
exit /b 0
