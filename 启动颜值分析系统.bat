@echo off
chcp 65001 >nul
title 颜值矩阵分析系统 v53.6
cd /d "%~dp0"

echo ============================================
echo   颜值矩阵分析系统 v53.6
echo ============================================
echo.

REM -- 1. 清理 Python 字节码缓存 --
echo [清理] 正在清除 __pycache__ 缓存...
python -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]" 2>nul
echo [完成] 缓存已清理
echo.

REM -- 1.5. 警告: 请使用本 bat 启动, 不要运行 dist/ 里的旧 exe --
if exist "dist\*.exe" echo [提醒] dist/ 目录存在旧版 exe [v38~v48], 请勿直接运行
if exist "dist\*.exe" echo        应使用本 bat 启动最新版本 v53.6
if exist "dist\*.exe" echo.

REM -- 2. 检查 Python 环境 --
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python, 请先安装 Python 3.10+
    echo   下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [检查] Python 环境: 已就绪

REM -- 3. 全量依赖检查 --
echo [检查] 正在验证依赖库...

set MISSING=0

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo   [缺失] numpy
    set MISSING=1
) else (
    echo   [通过] numpy
)

python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo   [缺失] opencv-python
    set MISSING=1
) else (
    echo   [通过] opencv-python
)

python -c "from PIL import Image" >nul 2>&1
if errorlevel 1 (
    echo   [缺失] Pillow
    set MISSING=1
) else (
    echo   [通过] Pillow
)

python -c "import customtkinter" >nul 2>&1
if errorlevel 1 (
    echo   [缺失] customtkinter
    set MISSING=1
) else (
    echo   [通过] customtkinter
)

python -c "import sklearn" >nul 2>&1
if errorlevel 1 (
    echo   [缺失] scikit-learn
) else (
    echo   [通过] scikit-learn
)

if "%MISSING%"=="1" (
    echo.
    echo ============================================
    echo [操作] 检测到缺失依赖, 正在自动安装...
    echo ============================================
    pip install numpy opencv-python Pillow customtkinter scikit-learn -q
    if errorlevel 1 (
        echo.
        echo [错误] 自动安装失败, 请手动运行:
        echo   pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [完成] 依赖安装成功
    echo.
)

REM -- 4. 启动程序 --
echo [启动] 正在加载颜值矩阵分析系统...
echo.
echo [日志] 错误输出将保存到 crash_log.txt
python -u beauty_gui_desktop.py 2>crash_log.txt
set EXIT_CODE=%errorlevel%

if %EXIT_CODE% neq 0 (
    echo.
    echo ============================================
    echo [错误] 程序异常退出 (代码: %EXIT_CODE%)
    echo ============================================
    if exist crash_log.txt (
        echo 最新错误日志 (crash_log.txt):
        type crash_log.txt
    )
    echo.
    echo 常见原因:
    echo   1. 依赖版本不兼容: pip install -r requirements.txt
    echo   2. Python 版本过低: 需要 3.10 以上
    echo   3. 模型文件缺失: 确保 stats_output/ 目录完整
    echo.
    pause
) else (
    echo.
    echo [完成] 程序已正常退出
    echo.
    pause
)
