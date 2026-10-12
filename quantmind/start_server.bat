@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem  QuantMind 启动脚本
rem  优先使用 ml\.venv（已含 torch/transformers + Web 依赖），
rem  找不到时回退系统 python。
rem ============================================================

set "VENV_PY=%~dp0..\ml\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [start] 使用 ML 环境: %VENV_PY%
) else (
    set "PY=python"
    echo [start] 未找到 ml\.venv，回退系统 python（模型功能可能不可用）
)

rem 国内网络需走 HF 镜像才能下载模型/语料
set "HF_ENDPOINT=https://hf-mirror.com"
rem 新版 huggingface_hub 默认走 Xet 协议，hf-mirror 不支持，必须禁用
set "HF_HUB_DISABLE_XET=1"
rem 启动即加载情感模型，避免首个请求等待；不需要可改为 0
set "QUANTMIND_SENTIMENT_WARMUP=1"
set "PYTHONIOENCODING=utf-8"

"%PY%" app.py

endlocal
pause
