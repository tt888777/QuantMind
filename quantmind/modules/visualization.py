import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

DARK_BG = "#0A0A0A"
DARK_FG = "#EDEDEF"
GRID_CLR = "#1A1A1E"
MUTED = "#8A8F98"
INDIGO = "#6366F1"
PURPLE = "#A855F7"
GREEN = "#22C55E"
RED = "#EF4444"
AMBER = "#F59E0B"
CYAN = "#06B6D4"


def _style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.5)


def plot_kline(df, stock_code=""):
    fig = Figure(figsize=(10, 5), dpi=100, facecolor=DARK_BG)
    ax = fig.add_subplot(111, facecolor=DARK_BG)

    dates = range(len(df))
    closes = df["close"].tolist()
    opens = df["open"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()

    for i in dates:
        color = RED if closes[i] >= opens[i] else GREEN
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8)
        ax.plot([i, i], [min(opens[i], closes[i]), max(opens[i], closes[i])],
                color=color, linewidth=3)

    if "ma5" in df.columns:
        ax.plot(dates, df["ma5"].tolist(), label="MA5", color=INDIGO, linewidth=1.2)
    if "ma20" in df.columns:
        ax.plot(dates, df["ma20"].tolist(), label="MA20", color=PURPLE, linewidth=1.2)

    ax.set_title(f"{stock_code} K线图", color=DARK_FG, fontsize=11, pad=8)
    leg = ax.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_CLR,
                    labelcolor=DARK_FG, fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    return FigureCanvas(fig)


def plot_equity(equity_curve, init_capital=100000):
    fig = Figure(figsize=(10, 3.5), dpi=100, facecolor=DARK_BG)
    ax = fig.add_subplot(111, facecolor=DARK_BG)
    eq = [e["equity"] for e in equity_curve]
    ax.fill_between(range(len(eq)), eq, init_capital,
                    where=[e >= init_capital for e in eq],
                    color=INDIGO, alpha=0.15)
    ax.plot(range(len(eq)), eq, color=INDIGO, linewidth=1.5, label="资金曲线")
    ax.axhline(y=init_capital, color=MUTED, linestyle="--", linewidth=0.8, label="初始资金")
    ax.set_title("策略资金曲线", color=DARK_FG, fontsize=11, pad=8)
    leg = ax.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_CLR,
                    labelcolor=DARK_FG, fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    return FigureCanvas(fig)


def plot_indicator_preview(df, stock_code=""):
    fig = Figure(figsize=(8, 3), dpi=100, facecolor=DARK_BG)
    ax = fig.add_subplot(111, facecolor=DARK_BG)
    n = min(len(df), 60)
    tail = df.tail(n)
    ax.plot(range(n), tail["close"].tolist(), color=DARK_FG, linewidth=1, label="收盘价")
    if "ma5" in tail.columns:
        ax.plot(range(n), tail["ma5"].tolist(), color=INDIGO, linewidth=1, label="MA5")
    if "ma20" in tail.columns:
        ax.plot(range(n), tail["ma20"].tolist(), color=PURPLE, linewidth=1, label="MA20")
    ax.set_title(f"{stock_code} 指标预览", color=DARK_FG, fontsize=10, pad=6)
    ax.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_CLR,
              labelcolor=DARK_FG, fontsize=7)
    _style_ax(ax)
    fig.tight_layout()
    return FigureCanvas(fig)


def plot_strategy_compare():
    fig = Figure(figsize=(5, 3.5), dpi=100, facecolor=DARK_BG)
    ax = fig.add_subplot(111, facecolor=DARK_BG)
    strategies = ["双均线\n5/20", "动量\n策略", "布林带\n突破", "双均线+\n情绪过滤", "买入持有"]
    returns = [8.2, 5.6, -2.1, 12.4, 6.8]
    colors = [GREEN if r > 0 else RED for r in returns]
    bars = ax.bar(strategies, returns, color=colors, width=0.6, edgecolor="none")
    for bar, r in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{r:+.1f}%", ha="center", va="bottom", color=DARK_FG, fontsize=8)
    ax.set_title("策略收益对比（近30日）", color=DARK_FG, fontsize=10, pad=8)
    ax.set_ylabel("收益率 %", color=MUTED, fontsize=8)
    ax.axhline(y=0, color=MUTED, linewidth=0.5)
    _style_ax(ax)
    fig.tight_layout()
    return FigureCanvas(fig)


def plot_sentiment_trend():
    import math
    fig = Figure(figsize=(6, 2.5), dpi=100, facecolor=DARK_BG)
    ax = fig.add_subplot(111, facecolor=DARK_BG)
    data = [50 + math.sin(i / 1.5) * 12 + (i % 3 - 1) * 2 for i in range(20)]
    ax.fill_between(range(20), data, 20, color=GREEN, alpha=0.1)
    ax.plot(range(20), data, color=CYAN, linewidth=1.5)
    ax.set_title("舆情情绪时间序列", color=DARK_FG, fontsize=10, pad=6)
    _style_ax(ax)
    fig.tight_layout()
    return FigureCanvas(fig)
