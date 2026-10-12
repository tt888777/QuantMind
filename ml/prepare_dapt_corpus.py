"""
构建 DAPT 语料：从 CN-Market-Corpus 取头部片段，转成"一行一条"的 txt
=====================================================================
为什么不直接用 `datasets` 的流式加载：
  该数据集不是 parquet，而是**单个 13.8GB 的 JSON 数组文件**（`d2.json`）。
  `datasets` 的 streaming 在这种结构上没有分片可切，实测会退化成整文件下载而卡住。
  这里改为发 HTTP Range 请求，只取文件头部若干 MB，解析完整行即可 —— 快且可断点续传。

产出：data/cn_market_dapt.txt   默认 20 万条（源数据约 0.9GB）
每条 = 标题 + 正文前 N 字（正文默认截 150 字，`--title_only` 可只取标题）

用法：
  python prepare_dapt_corpus.py                       # 默认 20 万条
  python prepare_dapt_corpus.py --max_records 20000   # 小样本快速验证
  python prepare_dapt_corpus.py --check               # 只看看现有语料状态
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DATASET = "mulfsa/CN-Market-Corpus"
FILENAME = "d2.json"
URL = f"{ENDPOINT}/datasets/{DATASET}/resolve/main/{FILENAME}"
# 镜像站对空 UA 会返回 403，必须带一个 User-Agent
UA = "Mozilla/5.0 (compatible; QuantMind-DAPT/1.0)"


def fetch_range(start: int, end: int, retries: int = 4, timeout: int = 180) -> bytes:
    """取 [start, end] 闭区间的字节。urllib 会自动跟随镜像的 302 并保留 Range 头。"""
    hdr = {"User-Agent": UA, "Range": f"bytes={start}-{end}"}
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # 网络抖动 / 429 限流：退避重试
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"下载 bytes={start}-{end} 失败：{last}")


def to_text(rec: dict, body_chars: int) -> str:
    title = (rec.get("title") or "").strip()
    body = (rec.get("body") or "").strip()
    if body_chars <= 0 or not body:
        return title
    return f"{title}。{body[:body_chars]}"


def build(max_records: int, out: Path, cache: Path, body_chars: int,
          chunk_mb: int, min_len: int) -> int:
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 复用上次下好的片段，避免重复流量
    cached = cache.read_bytes() if cache.exists() else b""
    if cached:
        print(f"  复用缓存 : {cache.name}  {len(cached)/2**20:.1f} MB")
    if not cache.exists():
        cache.touch()

    texts, carry, pos, total = [], b"", len(cached), None
    chunk = chunk_mb * 2**20
    t0 = time.time()

    def consume(blob: bytes):
        """按行切分，最后一段可能被截断，留给下一轮。"""
        nonlocal carry
        parts = (carry + blob).split(b"\n")
        carry = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "ignore").strip().rstrip(",")
            if not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = to_text(rec, body_chars)
            if len(txt) >= min_len:
                texts.append(txt)
            if len(texts) >= max_records:
                return True
        return False

    if cached and consume(cached):
        pass

    while len(texts) < max_records:
        got = fetch_range(pos, pos + chunk - 1)
        if not got:
            break
        if total is None:
            print(f"  源文件共 : {13.78 * 1024:.0f} MB（按需取头部）")
        pos += len(got)
        with open(cache, "ab") as f:
            f.write(got)
        print(f"  已下载   : {pos/2**20:.1f} MB  已收集 {len(texts):,} 条"
              f"  ({time.time()-t0:.0f}s)")
        if consume(got):
            break

    print(f"\n  写出语料 : {out}")
    with open(out, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    avg = sum(len(t) for t in texts) / max(len(texts), 1)
    print(f"  条数     : {len(texts):,}")
    print(f"  平均字数 : {avg:.1f}")
    print(f"  文件大小 : {out.stat().st_size/2**20:.1f} MB")
    print(f"  耗时     : {time.time()-t0:.0f}s")
    return len(texts)


def check(out: Path):
    if not out.exists():
        print(f"[未生成] {out}")
        return 1
    n = 0
    tot = 0
    with open(out, encoding="utf-8") as f:
        for line in f:
            n += 1
            tot += len(line)
    print(f"[已存在] {out}")
    print(f"  条数     : {n:,}")
    print(f"  平均字数 : {tot/max(n,1):.1f}")
    print(f"  文件大小 : {out.stat().st_size/2**20:.1f} MB")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_records", type=int, default=200000)
    p.add_argument("--out", default=str(ROOT / "data" / "cn_market_dapt.txt"))
    p.add_argument("--cache", default=str(ROOT / ".cache" / "d2_head.json"))
    p.add_argument("--body_chars", type=int, default=150, help="正文截取字数，0=只用标题")
    p.add_argument("--title_only", action="store_true")
    p.add_argument("--chunk_mb", type=int, default=16)
    p.add_argument("--min_len", type=int, default=4)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if args.check:
        return check(out)

    if args.title_only:
        args.body_chars = 0

    print("=" * 64)
    print("构建 DAPT 语料 (CN-Market-Corpus 头部片段)")
    print("=" * 64)
    print(f"  镜像     : {ENDPOINT}")
    print(f"  目标条数 : {args.max_records:,}")
    print(f"  文本构成 : 标题{'+正文前%d字' % args.body_chars if args.body_chars else '（仅标题）'}\n")
    n = build(args.max_records, out, Path(args.cache), args.body_chars,
              args.chunk_mb, args.min_len)
    if n == 0:
        print("\n[错误] 没收集到任何文本")
        return 1
    print("\n完成。下一步：python train_dapt.py --corpus " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
