@echo off
setlocal
cd /d "%~dp0"
title QuantMind - 本地模式 (SQLite)

rem ============================================================
rem  QuantMind 启动脚本 · 本地 SQLite 模式（推荐，无需 MySQL）
rem
rem  全新电脑上第一次运行前，请先看 README 的
rem  「二、从一台全新电脑部署」，做完步骤 2（建环境）
rem  与步骤 4（准备 AI 模型）。
rem
rem  本脚本会自动完成：
rem    1. 选择 Python 解释器（优先 ml\.venv）
rem    2. 检查 Web 依赖是否安装
rem    3. 检查 AI 模型是否就绪，缺了会告诉你怎么补
rem    4. 检查数据库，不存在则自动建库 + 灌示例数据
rem    5. 启动服务
rem ============================================================

set "ROOT=%~dp0.."
set "VENV_PY=%ROOT%\ml\.venv\Scripts\python.exe"
set "MERGED_CFG=%ROOT%\ml\models\lora_dapt_flame\merged\config.json"
set "ADAPTER_CFG=%ROOT%\ml\models\lora_dapt_flame\adapter\adapter_config.json"
set "DB_FILE=%~dp0quantmind_local.db"

echo ============================================================
echo   QuantMind 智能量化分析平台
echo ============================================================
echo.

rem ---------- 1. Python 解释器 ----------
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [1/5] Python : 使用虚拟环境 ml\.venv
) else (
    set "PY=python"
    echo [1/5] Python : [警告] 未找到 ml\.venv，已回退系统 python
    echo              ^-^> 界面功能可用，但 AI 情感分析不可用。
    echo              ^-^> 建环境方法见 README「二、步骤 2」。
)
echo.

rem ---------- 2. Web 依赖 ----------
"%PY%" -c "import flask, pandas, sqlalchemy" >nul 2>&1
if errorlevel 1 (
    echo [2/5] 依赖   : [警告] 缺少 Web 依赖（flask / pandas / sqlalchemy）
    echo              ^-^> 请在 ml\.venv 下执行：
    echo                 pip install -r ..\quantmind\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)
echo [2/5] 依赖   : Web 依赖已安装
echo.

rem ---------- 3. AI 模型 ----------
if exist "%MERGED_CFG%" (
    echo [3/5] AI 模型: 已就绪（合并模型 merged）
    goto :db
)
if exist "%ADAPTER_CFG%" (
    echo [3/5] AI 模型: [提示] 只有 LoRA adapter，没有合并模型
    echo              ^-^> 想启用「AI 模型」引擎，请先执行一次：
    echo                 cd ..\ml
    echo                 python setup_model.py
    echo              ^-^> 不执行也能启动，情感分析会自动走关键词规则。
    goto :db
)
echo [3/5] AI 模型: [警告] 未找到模型目录 ml\models\lora_dapt_flame\
echo              ^-^> 请确认代码完整，然后执行：cd ..\ml ^&^& python setup_model.py --check

:db
echo.

rem ---------- 4. 数据库 ----------
if exist "%DB_FILE%" (
    echo [4/5] 数据库 : 已有 quantmind_local.db
    goto :env
)
echo [4/5] 数据库 : 未找到，正在初始化（建表 + 灌入示例股票）...
set "QUANTMIND_DB_URL=sqlite:///%DB_FILE:\=/%"
"%PY%" init_local_db.py
if errorlevel 1 (
    echo              [错误] 初始化失败，请检查上面的报错
    echo.
    pause
    exit /b 1
)

:env
set "QUANTMIND_DB_URL=sqlite:///%DB_FILE:\=/%"
rem 国内网络需走 HF 镜像才能下载模型/语料
set "HF_ENDPOINT=https://hf-mirror.com"
rem 新版 huggingface_hub 默认走 Xet 协议，hf-mirror 不支持，必须禁用
set "HF_HUB_DISABLE_XET=1"
rem 启动即加载情感模型，避免首个请求等待；不需要可改为 0
set "QUANTMIND_SENTIMENT_WARMUP=1"
set "PYTHONIOENCODING=utf-8"

echo [5/5] 启动   : http://127.0.0.1:5000
echo.
echo ============================================================
echo   浏览器打开 http://127.0.0.1:5000
echo   首次使用请先登录：demo / demo123456（也可自行注册）
echo   进入「舆情决策」页，可切换 关键词规则 / AI 模型
echo   按 Ctrl+C 停止服务
echo ============================================================
echo.

"%PY%" app.py

echo.
echo [服务已退出]
pause
endlocal
