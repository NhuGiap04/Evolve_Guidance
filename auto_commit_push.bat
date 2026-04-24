@echo off
setlocal EnableExtensions EnableDelayedExpansion

git add .
set /p commit_message=Enter commit message:
git commit -m "!commit_message!"
git push

