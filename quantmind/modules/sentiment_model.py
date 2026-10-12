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

import json
import os
import threading
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent          # quantmind/modules
REPO_ROOT = MODULE_DIR.parent.parent                  # 仓库根目录

# 依次探测：优先 DAPT 版（若已训练），其次默认 LoRA 版（合并态），
# 最后是仓库里自带的 LoRA adapter（需要用底座现场合并，见 _resolve_adapter_base）
DEFAULT_MODEL_CANDIDATES = [
    REPO_ROOT / "ml" / "models" / "lora_dapt" / "merged",
    REPO_ROOT / "ml" / "models" / "lora_guba" / "merged",
    REPO_ROOT / "ml" / "models" / "lora_guba" / "adapter",
]

# 底座候选：从仓库内已下载好的目录里找（clone 后先跑 ml/setup_model.py 生成）
BASE_MODEL_CANDIDATES = [
    REPO_ROOT / "ml" / "models" / "bert-base-chinese",
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


def _looks_like_base_dir(p: Path) -> bool:
    """是不是一个「完整可加载」的 transformers 模型目录（底座/合并模型）。"""
    if not p or not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_w = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    return has_cfg and has_w


def _is_peft_adapter_dir(p: Path) -> bool:
    """是否只有 LoRA adapter、没有完整权重（需要现场与底座合并）。"""
    return (p / "adapter_config.json").exists() and not _looks_like_base_dir(p)


def _looks_like_local_path(s: str) -> bool:
    """判断字符串更像本地路径而不是 HF 仓库 id。

    训练时 ``adapter_config.json`` 里写的是 ``models/bert-base-chinese`` 这类
    相对仓库的路径。若直接当 HF repo id 用，会去联网请求一个不存在的仓库，
    导致首次请求白等一次网络超时 —— 所以这里要提前识别出来。
    """
    if not s:
        return False
    if "\\" in s:
        return True
    t = s.replace("\\", "/")
    if t.startswith(("./", "../", "/", "~")):
        return True
    if len(t) > 1 and t[1] == ":":           # Windows 盘符，如 C:/...
        return True
    return t.startswith(("models/", "ml/", "checkpoints/", "checkpoint/"))


def _resolve_adapter_base(adapter_dir: Path):
    """解析 LoRA adapter 对应的底座模型路径。

    优先级：环境变量 > adapter_config.json 里的 base_model_name_or_path
    > 仓库内默认底座目录 > 直接返回原始值（当作 HF id，需联网）。

    返回 ``(base, note)``。若 adapter 指向的是本地路径但本地找不到，
    抛 ``FileNotFoundError`` 并给出可操作的指引（而不是去联网空等）。
    """
    env_base = os.environ.get("QUANTMIND_SENTIMENT_BASE_MODEL")
    if env_base:
        return env_base, "环境变量 QUANTMIND_SENTIMENT_BASE_MODEL"

    raw = None
    cfg_file = adapter_dir / "adapter_config.json"
    try:
        raw = json.loads(cfg_file.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except Exception:
        raw = None

    if raw:
        p = Path(raw)
        if p.is_dir():
            return str(p), "adapter_config 中的绝对路径"
        # 训练时写的是相对路径（如 models/bert-base-chinese），相对仓库根 / ml / ml/models 各试一次
        for root in (REPO_ROOT, REPO_ROOT / "ml", REPO_ROOT / "ml" / "models"):
            cand = (root / raw)
            if _looks_like_base_dir(cand):
                return str(cand), "adapter_config 相对路径解析"
        for cand in BASE_MODEL_CANDIDATES:
            if _looks_like_base_dir(cand):
                return str(cand), "仓库内默认底座目录"
        if _looks_like_local_path(raw):
            raise FileNotFoundError(
                f"adapter 指定的底座是本地路径「{raw}」，但本地不存在。"
                f"请先执行：python ml/setup_model.py（会自动下载底座到 {BASE_MODEL_CANDIDATES[0]}）"
            )

    for cand in BASE_MODEL_CANDIDATES:
        if _looks_like_base_dir(cand):
            return str(cand), "仓库内默认底座目录"

    return (raw or "bert-base-chinese"), "HF 模型 id（需要联网，建议先跑 ml/setup_model.py）"


def _build_id2label(config):
    """从模型 config 取 id→标签映射。

    注意：直接 ``merge_and_unload()`` 出来的模型，其 config 继承自底座
    ``bert-base-chinese``，``id2label`` 是 ``LABEL_0/1/2`` 这种占位符，
    ``normalize_label`` 会把它们全判成「中性」。因此这里做一次校验，
    不可识别时按训练时的固定顺序（看跌/看涨/中性）回填。
    """
    raw = getattr(config, "id2label", None) or {}
    try:
        mapped = {int(i): normalize_label(lbl) for i, lbl in raw.items()}
    except Exception:
        mapped = {}
    if len(mapped) != len(LABEL_ORDER) or set(mapped.values()) != set(LABEL_ORDER):
        return {i: LABEL_ORDER[i] for i in range(len(LABEL_ORDER))}
    return mapped


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
        self._load_note = None

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

            # 轻量预检：adapter 场景下先确认底座在不在。
            # 这一步只读文件系统，**故意放在 import torch 之前** ——
            # 否则光是导入 torch/transformers 就要十几秒，用户点一下「AI 模型」
            # 要白等半天才被告知模型没准备好。
            if _is_peft_adapter_dir(path):
                try:
                    _resolve_adapter_base(path)
                except Exception as e:
                    self._error = f"模型不可用：{e}"
                    return False

            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except Exception as e:                       # ImportError 及其它加载期异常
                self._error = f"缺少推理依赖 torch/transformers：{e}"
                return False

            try:
                if _is_peft_adapter_dir(path):
                    tokenizer, model, note = self._load_peft_adapter(
                        path, torch,
                        AutoModelForSequenceClassification, AutoTokenizer,
                    )
                    self._load_note = note
                else:
                    tokenizer = AutoTokenizer.from_pretrained(str(path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(path))
                    self._load_note = "完整模型目录"

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
                self._id2label = _build_id2label(model.config)
                self._loaded = True
                self._error = None
                return True
            except Exception as e:
                self._error = f"加载模型失败（{path}）：{e}"
                return False

    def _load_peft_adapter(self, adapter_dir, torch, AutoModel, AutoTokenizer):
        """只有 LoRA adapter 时：找底座 → 挂 adapter → 合并成完整模型。

        返回 ``(tokenizer, model, note)``；任何一步失败都抛异常，由调用方降级。
        """
        from peft import PeftModel

        base, note = _resolve_adapter_base(adapter_dir)

        # tokenizer：仓库内 adapter 目录自带 tokenizer.json，优先用它，失败再退回底座
        tokenizer = None
        for src in (str(adapter_dir), str(base)):
            try:
                tokenizer = AutoTokenizer.from_pretrained(src)
                break
            except Exception:
                continue
        if tokenizer is None:
            raise RuntimeError(f"tokenizer 加载失败（尝试过 {adapter_dir} 与 {base}）")

        base_model = AutoModel.from_pretrained(
            str(base),
            num_labels=len(LABEL_ORDER),
            id2label={i: v for i, v in enumerate(LABEL_ORDER)},
            label2id={v: i for i, v in enumerate(LABEL_ORDER)},
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.merge_and_unload()          # 合并后就是普通 BERT，推理更省事
        return tokenizer, model, f"LoRA adapter + 底座（{note}）"

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
            "load_note": self._load_note,
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
