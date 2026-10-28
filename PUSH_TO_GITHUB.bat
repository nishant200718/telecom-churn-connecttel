@echo off
title ConnectTel - Push to GitHub
color 0b
echo =======================================================================
echo          CONNECTTEL - 1-CLICK GITHUB REPOSITORY UPLOADER
echo =======================================================================
echo.
echo Step 1: Open https://github.com/new in your browser.
echo Step 2: Create a new repository named: telecom-churn-connecttel
echo         (Do NOT initialize with README, .gitignore, or license)
echo Step 3: Copy the repository URL.
echo.
echo =======================================================================
set /p REPO_URL="Enter your GitHub Repository URL: "

if "%REPO_URL%"=="" (
    echo.
    echo [ERROR] No repository URL was entered.
    pause
    exit /b 1
)

echo.
echo [1/3] Configuring Git Remote...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo [2/3] Verifying Git Status...
git add .
git commit -m "Add GitHub helper script and update repository" 2>nul

echo [3/3] Pushing all files, reports, PPT and source code to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================================
    echo   [SUCCESS] Your entire ConnectTel project is now live on GitHub!
    echo =======================================================================
) else (
    echo.
    echo [INFO] If GitHub prompted you to sign in, complete the browser prompt
    echo        and then re-run this script.
)

echo.
pause
