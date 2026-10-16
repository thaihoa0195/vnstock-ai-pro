@echo off
title VN-Stock AI Pro - Phan Tich Chung Khoan Viet Nam
chcp 65001 >nul
echo ============================================================
echo   VN-STOCK AI PRO - PHAN TICH DUONG GIA CHUNG KHOAN VN
echo ============================================================
echo Dang khoi dong may chu web va tao link online...
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe

if not exist "%PYTHON_PATH%" (
    echo Khong tim thay Python tai: %PYTHON_PATH%
    pause
    exit /b 1
)

"%PYTHON_PATH%" run_server.py
pause
