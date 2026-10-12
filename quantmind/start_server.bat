@echo off
setlocal
cd /d "%~dp0"
title QuantMind - MySQL 模式

rem ============================================================
rem  QuantMind 启动脚本 · MySQL 模式
rem
rem  本机没有 MySQL 时请改用 start_server_local.bat（SQLite，零配置）。
rem
rem  使用前请确认：
rem    1. 已执行 quantmind_schema.sql 建表
rem    2. quantmind\config.py 里的连接串与你的 MySQL 一致
rem       （也可用环境变量 QUANTMIND_DB_URL 覆盖，不必改代码）
rem ============================================================

set "ROOT=%~dp0.."
set "VENV_PY=%ROOT%\ml\.venv\Scripts\python.exe"
set "MERGED_CFG=%ROOT%\ml\models\lora_dapt_flame\merged\config.json"
set "ADAPTER_CFG=%ROOT%\ml\models\lora_dapt_flame\adapter\adapter_config.json"

echo ============================================================
echo   QuantMind 智能量化分析平台（MySQL 模式）
echo ============================================================
echo.

rem ---------- 1. Python 解释器 ----------
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [1/4] Python : 使用虚拟环境 ml\.venv
) else (
    set "PY=python"
    echo [1/4] Python : [警告] 未找到 ml\.venv，已回退系统 python
    echo              ^-^> 界面功能可用，但 AI 情感分析不可用。
    echo              ^-^> 建环境方法见 README「二、步骤 2」。
)
echo.

rem ---------- 2. Web 依赖 ----------
"%PY%" -c "import flask, pandas, sqlalchemy, pymysql" >nul 2>&1
if errorlevel 1 (
    echo [2/4] 依赖   : [警告] 缺少依赖（flask / pandas / sqlalchemy / pymysql）
    echo              ^-^> 请在 ml\.venv 下执行：
    echo                 pip install -r ..\quantmind\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)
echo [2/4] 依赖   : Web 依赖已安装
echo.

rem ---------- 3. AI 模型 ----------
if exist "%MERGED_CFG%" (
    echo [3/4] AI 模型: 已就绪（合并模型 merged）
    goto :env
)
if exist "%ADAPTER_CFG%" (
    echo [3/4] AI 模型: [提示] 只有 LoRA adapter，没有合并模型
    echo              ^-^> 想启用「AI 模型」引擎，请先执行一次：
    echo                 cd ..\ml
    echo                 python setup_model.py
    echo              ^-^> 不执行也能启动，情感分析会自动走关键词规则。
    goto :env
)
echo [3/4] AI 模型: [警告] 未找到模型目录 ml\models\lora_dapt_flame\
echo              ^-^> 请确认代码完整，然后执行：cd ..\ml ^&^& python setup_model.py --check

:env
echo.
if defined QUANTMIND_DB_URL (
    echo [4/4] 数据库 : 使用环境变量 QUANTMIND_DB_URL
) else (
    echo [4/4] 数据库 : 使用 config.py 中的连接串
)
rem 国内网络需走 HF 镜像才能下载模型/语料
set "HF_ENDPOINT=https://hf-mirror.com"
rem 新版 huggingface_hub 默认走 Xet 协议，hf-mirror 不支持，必须禁用
set "HF_HUB_DISABLE_XET=1"
rem 启动即加载情感模型，避免首个请求等待；不需要可改为 0
set "QUANTMIND_SENTIMENT_WARMUP=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo ============================================================
echo   浏览器打开 http://127.0.0.1:5000
echo   进入「舆情决策」页，可切换 关键词规则 / AI 模型
echo   按 Ctrl+C 停止服务
echo ============================================================
echo.

"%PY%" app.py

echo.
echo [服务已退出]
pause
endlocal
