@echo off
title ConnectTel Churn Prediction - Live Server & Tunnel
cls
echo =================================================================
echo   CONNECTTEL CHURN ENGINE - STARTING LIVE INTERNET SERVICE
echo =================================================================
echo.

set PROJ_DIR=%~dp0telecom-churn-connecttel
if not exist "%PROJ_DIR%" set PROJ_DIR=%~dp0

set CLOUDFLARED="C:\Users\NAMAN\OneDrive\Desktop\nishant\cloudflared.exe"

echo [1/2] Starting ConnectTel Production App (Port 5000)...
start /b python "%PROJ_DIR%\production\app.py" > nul 2>&1
timeout /t 2 /nobreak > nul

echo [2/2] Launching Public HTTPS Cloudflare Tunnel...
echo.
echo Look for the link ending in: trycloudflare.com
echo.
%CLOUDFLARED% tunnel --url http://127.0.0.1:5000
pause
