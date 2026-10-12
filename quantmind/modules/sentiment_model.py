"""
情感分析模型后端（可插拔 ML 引擎）
==================================
把 LoRA 微调后的 BERT 三分类模型包装成情感分析后端，供 ``sentiment.py`` 调用。

设计原则（实训重点：模型挂了系统也要能跑）
------------------------------------------
1. **懒加载 + 单例**：模型在首次真正被用到时才加载，全局只加载一次，
   不在导入期做任何重活（否则 Flask 启动会卡十几秒）。
2. **优雅降级**：torch / transformers 未安装、模型目录不存在、权重损坏、
   推理抛异常 —— 一律把 ``available`` 置为 False，由调用方回退到关键词规则。
3. **线程安全**：Flask 以 ``threaded=True`` 运行，加载与推理都在锁内进行。
4. **标签词表对齐**：输出中文标签 ``看涨 / 看跌 / 中性``（高置信度时可输出
   ``强烈看涨 / 强烈看跌``），与 ``sentiment.analyze_sentiment`` 的词表一致，
   保证 ``analyze_sentiments`` 里 ``'看涨' in sentiment`` 的子串聚合逻辑直接可用。

配置（环境变量，均可选）
------------------------
``QUANTMIND_SENTIMENT_MODEL``   模型目录，默认 ``<repo>/ml/models/lora_guba/merged``
``QUANTMIND_SENTIMENT_DEVICE``  ``cuda`` / ``cpu``，默认自动探测
``QUANTMIND_SENTIMENT_MAX_LEN`` 最大 token 长度，默认 64（该数据集标题最长 79 字）
``QUANTMIND_SENTIMENT_BATCH``   推理批大小，默认 32
``QUANTMIND_SENTIMENT_STRONG``  强烈标签阈值，默认 0.9；设为 0 则禁用
"""

import os
import threading
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent          # quantmind/modules
REPO_ROOT = MODULE_DIR.parent.parent                  # 仓库根目录

# 依次探测：优先 DAPT 版（若已训练），其次默认 LoRA 版
DEFAULT_MODEL_CANDIDATES = [
    REPO_ROOT / "ml" / "models" / "lora_dapt" / "merged",
    REPO_ROOT / "ml" / "models" / "lora_guba" / "merged",
    REPO_ROOT / "ml" / "models" / "lora_guba" / "adapter",
]

LABEL_ORDER = ("看跌", "看涨", "中性")


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def normalize_label(raw):
    """把模型任意标签写法归一化成中文三分类标签。"""
    s = str(raw)
    low = s.lower()
    if "看涨" in s or "positive" in low or "bull" in low:
        return "看涨"
    if "看跌" in s or "看空" in s or "negative" in low or "bear" in low:
        return "看跌"
    return "中性"


def _looks_like_model_dir(p: Path) -> bool:
    if not p or not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_w = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    has_adapter = (p / "adapter_config.json").exists()
    return (has_cfg and has_w) or has_adapter


class SentimentModelBackend:
    """BERT+LoRA 情感分类模型的推理封装（线程安全的懒加载单例）。"""

    def __init__(self, model_dir=None, device=None, max_len=None,
                 batch_size=None, strong_threshold=None):
        env_dir = os.environ.get("QUANTMIND_SENTIMENT_MODEL")
        explicit = model_dir or env_dir
        self._explicit_dir = Path(explicit) if explicit else None
        self._device_pref = device or os.environ.get("QUANTMIND_SENTIMENT_DEVICE")
        self.max_len = max_len if max_len is not None else _env_int("QUANTMIND_SENTIMENT_MAX_LEN", 64)
        self.batch_size = batch_size if batch_size is not None else _env_int("QUANTMIND_SENTIMENT_BATCH", 32)
        self.strong_threshold = (
            strong_threshold if strong_threshold is not None
            else _env_float("QUANTMIND_SENTIMENT_STRONG", 0.9)
        )

        self._lock = threading.RLock()
        self._loaded = False
        self._error = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self._model_path = None
        self._id2label = {}

    # ------------------------------------------------------------------ 路径
    def resolve_path(self):
        """按优先级探测可用模型目录，找不到返回 None。"""
        if self._explicit_dir:
            return self._explicit_dir if _looks_like_model_dir(self._explicit_dir) else None
        for cand in DEFAULT_MODEL_CANDIDATES:
            if _looks_like_model_dir(cand):
                return cand
        return None

    # ------------------------------------------------------------------ 加载
    def load(self):
        """加载模型。返回 True 表示可用；False 时错误原因见 status()['error']。"""
        with self._lock:
            if self._loaded:
                return True
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except Exception as e:                       # ImportError 及其它加载期异常
                self._error = f"缺少推理依赖 torch/transformers：{e}"
                return False

            path = self.resolve_path()
            if path is None:
                if self._explicit_dir:
                    self._error = (f"指定的模型目录不可用：{self._explicit_dir}"
                                   "（需包含 config.json 与权重文件）")
                else:
                    self._error = (
                        "未找到情感模型目录，已回退关键词规则。可选路径："
                        + ", ".join(str(p) for p in DEFAULT_MODEL_CANDIDATES)
                    )
                return False

            try:
                tokenizer = AutoTokenizer.from_pretrained(str(path))
                model = AutoModelForSequenceClassification.from_pretrained(str(path))
                model.eval()

                if self._device_pref:
                    device = torch.device(self._device_pref)
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                self._torch = torch
                self._tokenizer = tokenizer
                self._model = model
                self._device = str(device)
                self._model_path = str(path)
                self._id2label = {
                    int(i): normalize_label(lbl)
                    for i, lbl in getattr(model.config, "id2label", {}).items()
                }
                self._loaded = True
                self._error = None
                return True
            except Exception as e:
                self._error = f"加载模型失败（{path}）：{e}"
                return False

    # ------------------------------------------------------------------ 状态
    @property
    def available(self):
        return self.load()

    def status(self, probe=False):
        """状态快照。

        ``probe=False``（默认）只看已知状态，**不会触发加载**，
        因此可以安全地放进高频接口的返回体里；
        ``probe=True`` 会真正尝试加载一次。
        """
        if probe:
            self.load()
        if self._loaded:
            avail = True
        elif self._error:
            avail = False
        else:
            avail = None                                # 尚未尝试加载，状态未知
        return {
            "available": avail,
            "loaded": self._loaded,
            "model_path": self._model_path,
            "device": self._device,
            "max_len": self.max_len,
            "batch_size": self.batch_size,
            "strong_threshold": self.strong_threshold,
            "labels": list(LABEL_ORDER),
            "error": self._error,
        }

    # ------------------------------------------------------------------ 推理
    def predict(self, titles):
        """批量预测。

        返回 ``[{'label': str, 'confidence': float, 'probs': {label: float}}]``；
        不可用时返回 None（由调用方回退），不抛异常。
        """
        if not titles:
            return []
        if not self.load():
            return None

        batch_size = max(1, int(self.batch_size))
        out = []
        try:
            with self._lock:
                torch = self._torch
                for start in range(0, len(titles), batch_size):
                    chunk = [str(t or "") for t in titles[start:start + batch_size]]
                    enc = self._tokenizer(
                        chunk, truncation=True, max_length=self.max_len,
                        padding=True, return_tensors="pt",
                    )
                    enc = {k: v.to(self._model.device) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = self._model(**enc).logits
                    probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                    for row in probs:
                        out.append(self._format_row(row))
        except Exception as e:
            self._error = f"推理失败：{e}"
            return None

        if len(out) != len(titles):
            self._error = "推理结果数量与输入不一致"
            return None
        return out

    def _format_row(self, row):
        best = int(max(range(len(row)), key=lambda i: row[i]))
        raw = self._id2label.get(best)
        if raw is None:
            raw = LABEL_ORDER[best] if best < len(LABEL_ORDER) else "中性"
        conf = float(row[best])

        label = raw
        if self.strong_threshold > 0 and conf >= self.strong_threshold:
            if raw == "看涨":
                label = "强烈看涨"
            elif raw == "看跌":
                label = "强烈看跌"

        probs = {}
        for i, p in enumerate(row):
            name = self._id2label.get(i)
            if name is None:
                name = LABEL_ORDER[i] if i < len(LABEL_ORDER) else f"label_{i}"
            probs[name] = round(float(p), 4)

        return {"label": label, "confidence": round(conf, 4), "probs": probs}


# ---------------------------------------------------------------- 单例入口
_BACKEND = None
_BACKEND_LOCK = threading.Lock()


def get_backend():
    """进程内单例。"""
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = SentimentModelBackend()
    return _BACKEND


def warmup():
    """可选预热：服务启动时提前加载，避免首个请求变慢。"""
    return get_backend().load()


if __name__ == "__main__":
    import json
    import sys

    be = get_backend()
    ok = be.load()
    print(json.dumps({**be.status(), "engine": "model" if ok else "keyword"},
                     ensure_ascii=False, indent=2))
    if not ok:
        sys.exit(1)
    demo = [
        "公司发布业绩预告，净利润同比增长80%",
        "大股东拟减持不超过3%股份，股价承压",
        "明天怎么玩？2000多家绿盘！指数收红，真牛，超级牛！",
        "关于召开2025年第一次临时股东大会的通知",
    ]
    for t, r in zip(demo, be.predict(demo)):
        print(f"  {r['label']:6} {r['confidence']:.3f}  {t}")
