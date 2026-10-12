@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem  本地 SQLite 模式启动（本机没有 MySQL 时用这个）
rem  先执行一次：init_local_db.py 生成数据库
rem ============================================================

set "VENV_PY=%~dp0..\ml\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [start] 使用 ML 环境: %VENV_PY%
) else (
    set "PY=python"
    echo [start] 未找到 ml\.venv，回退系统 python（模型功能可能不可用）
)

set "DB_FILE=%~dp0quantmind_local.db"
if not exist "%DB_FILE%" (
    echo [start] 未找到 quantmind_local.db，正在初始化 ...
    set "QUANTMIND_DB_URL=sqlite:///%DB_FILE:\=/%"
    "%PY%" init_local_db.py
)

set "QUANTMIND_DB_URL=sqlite:///%DB_FILE:\=/%"
set "HF_ENDPOINT=https://hf-mirror.com"
rem 新版 huggingface_hub 默认走 Xet 协议，hf-mirror 不支持，必须禁用
set "HF_HUB_DISABLE_XET=1"
set "QUANTMIND_SENTIMENT_WARMUP=1"
set "PYTHONIOENCODING=utf-8"

echo [start] 数据库: %QUANTMIND_DB_URL%
"%PY%" app.py

endlocal
pause
