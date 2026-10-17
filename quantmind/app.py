import sys, os, io, json, math, random
from flask import Flask, send_from_directory, send_file, jsonify, request, redirect
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.data_fetch import generate_sample_data, import_csv, load_quotation, import_tdx_day, import_tdx_dir
from modules.data_process import clean, calc_indicators
from modules.backtest import run_backtest
from modules.sentiment import collect_sentiment, analyze_sentiments, classify_titles
from modules.sentiment_model import get_backend as get_sentiment_backend
from modules.auth import init_auth, current_user, admin_required
from modules import report as report_mod
from db import get_session, TechnicalIndicator, QuotationDaily, StockBasic

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..'))

# 账号体系：注册蓝图 + 会话密钥 + 全局登录守卫（详见 modules/auth.py）
init_auth(app)

HTML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
EXPORT_DIR = os.path.join(HTML_DIR, 'exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

def query_stock_name(sc):
    try:
        import requests as _req
        for prefix in ('1', '0'):
            url = 'http://push2his.eastmoney.com/api/qt/stock/get'
            params = {'secid': f'{prefix}.{sc}', 'fields': 'f57,f58', 'ut': 'fa5fd1943c7b386f172d6893dbfba10b'}
            resp = _req.get(url, params=params, timeout=5, verify=False)
            d = resp.json().get('data', {})
            if d and d.get('f58'):
                return d['f58']
    except Exception:
        pass
    return None


@app.route('/')
def index():
    resp = send_from_directory(HTML_DIR, 'prototype.html')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


def _safe_next(raw):
    """/login?next=... 的回跳目标只允许站内相对路径，避免开放重定向。"""
    if raw and raw.startswith('/') and not raw.startswith('//') and not raw.startswith('/\\'):
        return raw
    return '/'


@app.route('/login')
def login_page():
    if current_user():
        return redirect(_safe_next(request.args.get('next')))
    resp = send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'login.html')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


@app.route('/api/fetch', methods=['POST'])
def api_fetch():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    days = int(d.get('days', 250))
    sn = d.get('stock_name', '').strip() or query_stock_name(sc) or sc
    n = generate_sample_data(stock_code=sc, stock_name=sn, days=days)
    return jsonify({'ok': True, 'count': n, 'msg': f'已生成 {n} 条日线（{sc} {sn}）'})


@app.route('/api/import_csv', methods=['POST'])
def api_import_csv():
    f = request.files.get('file')
    if not f:
        return jsonify({'ok': False, 'msg': '未收到文件'})
    fname = os.path.splitext(f.filename)[0]
    sc = request.form.get('stock_code', '').strip() or fname
    sn = request.form.get('stock_name', '').strip() or fname
    tmp = os.path.join(HTML_DIR, 'tmp_upload.csv')
    f.save(tmp)
    n = import_csv(tmp, sc)
    os.remove(tmp)
    s = get_session()
    try:
        s.query(StockBasic).filter_by(stock_code=sc).delete()
        s.add(StockBasic(stock_code=sc, stock_name=sn, market="CSV", industry=""))
        s.commit()
    finally:
        s.close()
    return jsonify({'ok': True, 'count': n, 'msg': f'已导入 {n} 条记录（{sc} {sn}）'})


@app.route('/api/import_tdx', methods=['POST'])
def api_import_tdx():
    f = request.files.get('file')
    if not f:
        d = request.json or {}
        dir_path = d.get('dir_path', '').strip()
        if dir_path:
            try:
                results = import_tdx_dir(dir_path)
                total = sum(r['count'] for r in results)
                return jsonify({'ok': True, 'count': total, 'results': results,
                                'msg': f'从目录导入 {len(results)} 个文件，共 {total} 条记录'})
            except Exception as e:
                return jsonify({'ok': False, 'msg': f'目录导入失败: {str(e)}'})
        return jsonify({'ok': False, 'msg': '未收到文件或目录路径'})
    fname = f.filename
    sc = request.form.get('stock_code', '').strip() or os.path.splitext(fname)[0]
    sn = request.form.get('stock_name', '').strip() or query_stock_name(sc) or sc
    tmp = os.path.join(HTML_DIR, 'tmp_tdx.day')
    f.save(tmp)
    try:
        n = import_tdx_day(tmp, sc)
        s = get_session()
        try:
            s.query(StockBasic).filter_by(stock_code=sc).delete()
            s.add(StockBasic(stock_code=sc, stock_name=sn, market="TDX", industry=""))
            s.commit()
        finally:
            s.close()
        return jsonify({'ok': True, 'count': n, 'msg': f'通达信导入 {n} 条记录（{sc} {sn}）'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'导入失败: {str(e)}'})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route('/api/clean', methods=['POST'])
def api_clean():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    info = clean(sc)
    return jsonify({'ok': True, **info,
                    'msg': f"去重{info['removed']} 修复{info['fixed']} 剩余{info.get('total', 0)}"})


@app.route('/api/indicators', methods=['POST'])
def api_indicators():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    n = calc_indicators(sc)
    s = get_session()
    rows = []
    try:
        recs = s.query(TechnicalIndicator).filter_by(stock_code=sc).order_by(
            TechnicalIndicator.trade_date.desc()).limit(10).all()
        dates = [r.trade_date for r in recs]
        quotes = {}
        if dates:
            qs = s.query(QuotationDaily).filter(
                QuotationDaily.stock_code == sc,
                QuotationDaily.trade_date.in_(dates)).all()
            quotes = {q.trade_date: q for q in qs}
        for r in recs:
            q = quotes.get(r.trade_date)
            rows.append({
                'date': str(r.trade_date),
                'open': float(q.open) if q and q.open else None,
                'high': float(q.high) if q and q.high else None,
                'low': float(q.low) if q and q.low else None,
                'close': float(q.close) if q and q.close else None,
                'ma5': float(r.ma5 or 0),
                'ma10': float(r.ma10 or 0),
                'ma20': float(r.ma20 or 0),
                'macd': float(r.macd_hist or 0),
                'rsi': float(r.rsi or 0),
            })
    finally:
        s.close()
    return jsonify({'ok': True, 'count': n, 'rows': rows,
                    'msg': f'已写入 {n} 条指标'})


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    d = request.json or {}
    sc = d.get('stock_code', '600519').split()[0]
    short = int(d.get('short_ma', 5))
    long_ = int(d.get('long_ma', 20))
    cap = float(d.get('init_capital', 100000))
    sd = d.get('start_date', '')
    ed = d.get('end_date', '')
    # sentiment_dynamic 默认开启；传 false 可回到旧的「一刀切满仓」行为做对比
    dyn = d.get('sentiment_dynamic')
    dyn = True if dyn is None else bool(dyn)
    strat = str(d.get('strategy') or 'dual_ma')
    bt = run_backtest(sc, short=short, long_=long_, init_capital=cap,
                      start_date=sd if sd else None, end_date=ed if ed else None,
                      sentiment_dynamic=dyn, strategy=strat)
    if not bt:
        return jsonify({'ok': False, 'msg': '回测失败：数据不足，请先生成数据'})
    trades = []
    for t in reversed(bt.get('trades', [])):
        trades.append({'date': str(t['date']), 'dir': t['dir'],
                        'price': t['price'], 'shares': t['shares']})
    equity = [e['equity'] for e in bt.get('equity_curve', [])]
    return jsonify({
        'ok': True,
        'total_return': bt['total_return'],
        'annual_return': bt['annual_return'],
        'max_drawdown': bt['max_drawdown'],
        'win_rate': bt['win_rate'],
        'sharpe': bt['sharpe'],
        'trade_count': bt['trade_count'],
        'round_trips': bt.get('round_trips'),
        'strategy': bt.get('strategy'),
        'strategy_name': bt.get('strategy_name'),
        'short': bt.get('short'), 'long': bt.get('long'),
        'sentiment_dynamic': bt.get('sentiment_dynamic'),
        'sentiment_index': bt.get('sentiment_index'),
        'target_ratio': bt.get('target_ratio'),
        'avg_exposure_pct': bt.get('avg_exposure_pct'),
        'trades': trades,
        'equity_curve': equity,
        'init_capital': cap,
        'msg': f"收益{bt['total_return']}% 年化{bt['annual_return']}% 夏普{bt['sharpe']}",
    })


@app.route('/api/quotation/<stock_code>')
def api_quotation(stock_code):
    days_raw = request.args.get('days')
    days = int(days_raw) if days_raw else None
    df = load_quotation(stock_code)
    if df.empty:
        return jsonify({'ok': False, 'msg': '无数据'})
    if days and days < 9999:
        df = df.tail(days)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'date': str(r['date']), 'open': r['open'], 'high': r['high'],
            'low': r['low'], 'close': r['close'], 'volume': int(r['volume']),
        })
    return jsonify({'ok': True, 'count': len(rows), 'rows': rows})


@app.route('/api/stocks')
def api_stocks():
    s = get_session()
    try:
        rows = s.query(QuotationDaily.stock_code, QuotationDaily.data_source).distinct().all()
        stocks = []
        for sc, src in rows:
            cnt = s.query(QuotationDaily).filter_by(stock_code=sc).count()
            sb = s.query(StockBasic).filter_by(stock_code=sc).first()
            name = sb.stock_name if sb else sc
            stocks.append({'code': sc, 'name': name, 'count': cnt, 'source': src})
        return jsonify({'ok': True, 'stocks': stocks})
    finally:
        s.close()


# ---------------------------------------------------------------- 分钟级行情
# 与日线彻底分开：日线走 quotation_daily（回测/指标/报告多处按 trade_date 读取），
# 分钟走 quotation_minute（唯一键 code+period+trade_time）。混表会污染已有路径。

@app.route('/api/minute/stocks')
def api_minute_stocks():
    """库内有分钟数据的标的清单，供前端下拉。"""
    from modules.minute_fetch import minute_stats
    rows = minute_stats()
    names = _stock_name_map()
    agg = {}
    for r in rows:
        a = agg.setdefault(r["stock_code"], {"code": r["stock_code"],
                                             "name": names.get(r["stock_code"], r["stock_code"]),
                                             "periods": {}})
        a["periods"][r["period"]] = {
            "rows": r["rows"],
            "start": str(r["start"])[:19],
            "end": str(r["end"])[:19],
        }
    out = sorted(agg.values(), key=lambda x: x["code"])
    return jsonify({'ok': True, 'count': len(out), 'stocks': out})


@app.route('/api/minute/quotation/<stock_code>')
def api_minute_quotation(stock_code):
    """分钟 K 线原始数据（前端画图用）。?period=30&limit=500"""
    from modules.minute_fetch import load_minute, SUPPORTED_PERIODS
    period = int(request.args.get('period', 30))
    if period not in SUPPORTED_PERIODS:
        return jsonify({'ok': False, 'msg': f'period 只支持 {SUPPORTED_PERIODS}'})
    limit = request.args.get('limit')
    df = load_minute(stock_code, period)
    if df.empty:
        return jsonify({'ok': False, 'msg': f'{stock_code} 无 {period} 分钟数据，请先采集'})
    if limit:
        df = df.tail(int(limit))
    rows = [{'time': str(r['time'])[:19], 'open': r['open'], 'high': r['high'],
             'low': r['low'], 'close': r['close'], 'volume': r['volume']}
            for _, r in df.iterrows()]
    return jsonify({'ok': True, 'count': len(rows), 'rows': rows})


@app.route('/api/minute/collect', methods=['POST'])
def api_minute_collect():
    """采集分钟行情（同步执行，前端需提示耗时）。"""
    from modules.minute_fetch import collect_and_save
    d = request.json or {}
    codes = d.get('codes') or []
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(',') if c.strip()]
    periods = d.get('periods') or [30]
    if not codes:
        return jsonify({'ok': False, 'msg': '请先选择标的'})
    periods = [int(p) for p in periods]
    if len(codes) * len(periods) > 60:
        return jsonify({'ok': False, 'msg': '单次最多采集 60 个（标的×周期），请分批'})

    results, total_added = [], 0
    for p in periods:
        for code in codes:
            res = collect_and_save(code, p)
            if res.get('ok'):
                total_added += res['added']
            results.append(res)
    ok = sum(1 for r in results if r['ok'])
    return jsonify({'ok': True, 'total': len(results), 'success': ok,
                    'failed': len(results) - ok, 'added': total_added,
                    'results': results,
                    'msg': f'采集完成：成功 {ok}/{len(results)}，新增 {total_added} 行'})


@app.route('/api/minute/logs')
def api_minute_logs():
    """最近采集日志，供前端展示数据来源与失败原因。"""
    from modules.minute_fetch import recent_logs
    limit = int(request.args.get('limit', 20))
    return jsonify({'ok': True, 'logs': recent_logs(limit)})


@app.route('/api/minute/backtest', methods=['POST'])
def api_minute_backtest():
    """单标的多策略分钟回测，返回对比表 + 最佳策略的权益曲线。"""
    from modules.minute_backtest import compare_strategies, BARS_PER_DAY
    d = request.json or {}
    sc = str(d.get('stock_code', '600519')).split()[0]
    period = int(d.get('period', 30))
    cap = float(d.get('init_capital', 1000000))
    # lot_size：100 = 真实一手；1 = 零股模拟（纯策略评估）
    lot = int(d.get('lot_size', 100))
    strat = d.get('strategy')  # 不传则跑默认组合

    combos = None
    if strat:
        combos = [("dual_ma", 5, 20), ("boll", 5, 20), ("momentum", 5, 20),
                  ("dual_ma", 10, 30), ("dual_ma", 3, 8)]
        combos = [c for c in combos if c[0] == strat] or None

    rows = compare_strategies(sc, period=period, combos=combos,
                              init_capital=cap, lot_size=lot,
                              start_time=d.get('start_time') or None,
                              end_time=d.get('end_time') or None)
    if not rows:
        return jsonify({'ok': False, 'msg': '无回测结果：该标的在此周期无数据，请先采集'})

    names = _stock_name_map()
    table = [{k: r[k] for k in (
        'strategy', 'strategy_name', 'short', 'long', 'bars', 'days', 'round_trips',
        'trade_count', 'total_return', 'return_before_fee', 'fee_drag',
        'annual_return', 'max_drawdown', 'win_rate', 'sharpe', 'fees_total',
        'min_commission_hits', 'skipped_no_capital', 'avg_trades_per_day')} for r in rows]

    best = max(rows, key=lambda r: r['total_return'])
    return jsonify({
        'ok': True,
        'stock_code': sc,
        'stock_name': names.get(sc, sc),
        'period': period,
        'init_capital': cap,
        'lot_size': lot,
        'bars_per_day': BARS_PER_DAY.get(period, 8),
        'start_time': rows[0]['start_time'][:19],
        'end_time': rows[0]['end_time'][:19],
        'table': table,
        'best': {k: best[k] for k in ('strategy_name', 'short', 'long',
                                      'total_return', 'annual_return', 'sharpe')},
        'equity_curve': [{'time': str(e['time'])[:19], 'equity': round(e['equity'], 2)}
                         for e in best['equity_curve']],
        'trades': best['trades'][-80:],
        'msg': f"{best['strategy_name']} {best['short']}/{best['long']} 最佳："
               f"实际{best['total_return']}% 年化{best['annual_return']}%",
    })


@app.route('/api/minute/matrix', methods=['POST'])
def api_minute_matrix():
    """多标的横评（标的 × 策略矩阵）。耗时较长，建议 limit 控制规模。"""
    from modules.minute_backtest import run_minute_backtest
    d = request.json or {}
    period = int(d.get('period', 30))
    cap = float(d.get('init_capital', 1000000))
    lot = int(d.get('lot_size', 100))
    combos = [("boll", 5, 20), ("dual_ma", 5, 20), ("dual_ma", 10, 30),
              ("dual_ma", 3, 8), ("momentum", 5, 20)]
    if d.get('combos'):
        # 前端传 [[strategy, short, long], ...]
        combos = [tuple(c) for c in d['combos']]

    from modules.minute_fetch import minute_stats
    codes = sorted({r["stock_code"] for r in minute_stats() if r["period"] == period})
    if d.get('codes'):
        want = set(d['codes'])
        codes = [c for c in codes if c in want]
    limit = int(d.get('limit', 0))
    if limit:
        codes = codes[:limit]
    if not codes:
        return jsonify({'ok': False, 'msg': '该周期无数据，请先采集'})
    if len(codes) * len(combos) > 300:
        return jsonify({'ok': False, 'msg': '组合过多（>300），请减少标的或策略'})

    names = _stock_name_map()
    rows = []
    for code in codes:
        for strat, s_, l_ in combos:
            r = run_minute_backtest(code, period=period, strategy=strat,
                                    short=s_, long_=l_, init_capital=cap, lot_size=lot)
            if not r:
                continue
            rows.append({
                'code': code, 'name': names.get(code, code),
                'strategy': strat, 'strategy_name': r['strategy_name'],
                'params': f'{s_}/{l_}', 'round_trips': r['round_trips'],
                'total_return': r['total_return'],
                'return_before_fee': r['return_before_fee'],
                'fee_drag': r['fee_drag'], 'annual_return': r['annual_return'],
                'max_drawdown': r['max_drawdown'], 'sharpe': r['sharpe'],
            })

    # 每只标的的最佳策略（按扣费后收益）
    best = {}
    for r in rows:
        if r['code'] not in best or r['total_return'] > best[r['code']]['total_return']:
            best[r['code']] = r
    stock_board = sorted(best.values(), key=lambda x: x['total_return'], reverse=True)

    # 每个策略的平均表现
    agg = {}
    for r in rows:
        k = f"{r['strategy']} {r['params']}"
        a = agg.setdefault(k, {'n': 0, 'ret': 0.0, 'before': 0.0, 'drag': 0.0,
                               'rt': 0.0, 'pos': 0})
        a['n'] += 1
        a['ret'] += r['total_return']
        a['before'] += r['return_before_fee']
        a['drag'] += r['fee_drag']
        a['rt'] += r['round_trips']
        a['pos'] += 1 if r['total_return'] > 0 else 0
    strategy_board = [{'key': k, 'samples': v['n'],
                       'avg_round_trips': round(v['rt'] / v['n'], 1),
                       'avg_before': round(v['before'] / v['n'], 2),
                       'avg_return': round(v['ret'] / v['n'], 2),
                       'avg_drag': round(v['drag'] / v['n'], 2),
                       'positive': v['pos']}
                      for k, v in sorted(agg.items(), key=lambda kv: -kv[1]['ret'] / kv[1]['n'])]

    return jsonify({
        'ok': True, 'period': period, 'init_capital': cap, 'lot_size': lot,
        'total_combos': len(rows),
        'positive_before': sum(1 for r in rows if r['return_before_fee'] > 0),
        'positive_after': sum(1 for r in rows if r['total_return'] > 0),
        'stock_board': stock_board,
        'strategy_board': strategy_board,
        'msg': f'{len(rows)} 个组合，扣费后正收益 '
               f'{sum(1 for r in rows if r["total_return"] > 0)}/{len(rows)}',
    })


def _stock_name_map():
    """代码 → 名称。取不到就退回代码本身。"""
    try:
        s = get_session()
        try:
            return {r.stock_code: (r.stock_name or r.stock_code)
                    for r in s.query(StockBasic).all()}
        finally:
            s.close()
    except Exception:                                          # noqa: BLE001
        return {}


@app.route('/api/pipeline', methods=['POST'])
def api_pipeline():
    d = request.json or {}
    steps = d.get('steps', [])
    sc = d.get('stock_code', '600519')
    results = []
    for step in steps:
        if step == 'fetch':
            s = get_session()
            existing = s.query(QuotationDaily).filter_by(stock_code=sc).count()
            s.close()
            if existing > 0:
                results.append({'step': 'fetch', 'ok': True, 'msg': f'数据获取：已有 {existing} 条，跳过'})
            else:
                n = generate_sample_data(stock_code=sc, days=250)
                results.append({'step': 'fetch', 'ok': True, 'msg': f'数据获取：生成 {n} 条日线'})
        elif step == 'clean':
            info = clean(sc)
            results.append({'step': 'clean', 'ok': True, 'msg': f"清洗：去重{info['removed']} 修复{info['fixed']}"})
        elif step == 'indicator':
            n = calc_indicators(sc)
            results.append({'step': 'indicator', 'ok': True, 'msg': f'指标：{n} 条'})
        elif step == 'backtest':
            bt = run_backtest(sc, short=5, long_=20, init_capital=100000)
            if bt:
                results.append({'step': 'backtest', 'ok': True, 'msg': f"回测：收益{bt['total_return']}% 夏普{bt['sharpe']}"})
            else:
                results.append({'step': 'backtest', 'ok': False, 'msg': '回测失败'})
        elif step == 'viz':
            df = load_quotation(sc)
            results.append({'step': 'viz', 'ok': True, 'msg': f'图表：{len(df)} 条数据'})
        elif step == 'sentiment':
            info = collect_sentiment(sc)
            result = analyze_sentiments(sc, info['news'], engine=d.get('engine', 'keyword'))
            d['_sent_result'] = result          # 供 report 步骤复用，避免重复采集
            results.append({'step': 'sentiment', 'ok': True,
                            'msg': f"舆情分析：{info['total']} 条，情绪指数 {result['index']}（引擎：{result['engine']}）"})
        elif step == 'report':
            # 关键修复：以前这里只 append 一句提示，**并没有真的生成文件**。
            # 现在真正调用报告模块落盘，并把下载地址回传给前端。
            df = load_quotation(sc)
            if df.empty:
                results.append({'step': 'report', 'ok': False,
                                'msg': '报告生成失败：无行情数据'})
                continue
            bt = run_backtest(sc, short=5, long_=20, init_capital=100000)
            # 复用前面 sentiment 步骤的结果，避免重复采集
            sent = d.get('_sent_result')
            if sent is None:
                try:
                    info = collect_sentiment(sc)
                    sent = analyze_sentiments(sc, info['news'],
                                              engine=d.get('engine', 'keyword'))
                except Exception:
                    sent = None
            fmt = (d.get('report_format') or 'pdf').lower()
            if fmt not in ('html', 'pdf'):
                fmt = 'pdf'
            try:
                r = report_mod.build_and_save(sc, df, bt, sent, fmt=fmt)
                results.append({'step': 'report', 'ok': True,
                                'msg': f"报告已生成：{r['filename']}"
                                       f"（{r['format'].upper()}，{r['size'] // 1024} KB）",
                                'url': r['url'], 'filename': r['filename'],
                                'format': r['format'], 'engine': r['engine']})
            except report_mod.PdfUnavailable as e:
                # PDF 引擎缺失时不假装成功：退回 HTML 并说明原因
                r = report_mod.build_and_save(sc, df, bt, sent, fmt='html')
                results.append({'step': 'report', 'ok': True,
                                'msg': f"PDF 引擎不可用（{e}），已改导出 HTML：{r['filename']}",
                                'url': r['url'], 'filename': r['filename'],
                                'format': 'html', 'engine': 'html'})
    return jsonify({'ok': True, 'results': results})


def _auto_map_columns(cols):
    mapping = {}
    aliases = {
        'date': ['date', '日期', '交易日', 'trade_date', 'time', 'datetime'],
        'open': ['open', '开盘', '开盘价', 'first', 'o'],
        'high': ['high', '最高', '最高价', 'h'],
        'low': ['low', '最低', '最低价', 'l'],
        'close': ['close', '收盘', '收盘价', 'last', 'c', 'price'],
        'volume': ['volume', '成交量', '成交量(手)', 'vol', 'v'],
        'amount': ['amount', '成交额', 'amt', 'turnover'],
    }
    cols_lower = {c.lower(): c for c in cols}
    for std, alias_list in aliases.items():
        for alias in alias_list:
            if alias.lower() in cols_lower and std not in mapping.values():
                mapping[cols_lower[alias.lower()]] = std
    return mapping


def _save_dataframe(df, sc):
    s = get_session()
    try:
        s.query(QuotationDaily).filter_by(stock_code=sc).delete()
        count = 0
        for _, row in df.iterrows():
            td = row['date']
            if isinstance(td, str):
                td = pd.to_datetime(td).date()
            elif hasattr(td, 'date'):
                td = td.date()
            s.add(QuotationDaily(
                stock_code=sc, trade_date=td,
                open=Decimal(str(row['open'])), high=Decimal(str(row['high'])),
                low=Decimal(str(row['low'])), close=Decimal(str(row['close'])),
                volume=int(row['volume']) if pd.notna(row.get('volume')) else 0,
                amount=Decimal(str(row.get('amount', 0))) if pd.notna(row.get('amount')) else Decimal('0'),
                data_source="IMPORT"))
            count += 1
        s.commit()
        return count
    finally:
        s.close()


@app.route('/api/import_file', methods=['POST'])
def api_import_file():
    f = request.files.get('file')
    if not f:
        return jsonify({'ok': False, 'msg': '未收到文件'})
    fname = os.path.splitext(f.filename)[0]
    ext = os.path.splitext(f.filename)[1].lower()
    sc = request.form.get('stock_code', '').strip() or fname
    sn = request.form.get('stock_name', '').strip() or fname
    tmp = os.path.join(HTML_DIR, 'tmp_upload' + ext)
    f.save(tmp)
    try:
        if ext == '.csv':
            df = pd.read_csv(tmp)
        elif ext in ('.xlsx', '.xls'):
            df = pd.read_excel(tmp)
        elif ext == '.json':
            df = pd.read_json(tmp)
        elif ext == '.xml':
            df = pd.read_xml(tmp)
        else:
            return jsonify({'ok': False, 'msg': f'不支持的格式: {ext}'})
        col_map = _auto_map_columns(df.columns.tolist())
        df = df.rename(columns=col_map)
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'ok': False, 'msg': f'缺少必要列: {missing}', 'columns': df.columns.tolist()})
        n = _save_dataframe(df, sc)
        s = get_session()
        try:
            s.query(StockBasic).filter_by(stock_code=sc).delete()
            s.add(StockBasic(stock_code=sc, stock_name=sn, market=ext[1:].upper(), industry=""))
            s.commit()
        finally:
            s.close()
        return jsonify({'ok': True, 'count': n, 'msg': f'已导入 {n} 条记录（{sc} {sn}）', 'format': ext})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'导入失败: {str(e)}'})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route('/api/column_mapping', methods=['POST'])
def api_column_mapping():
    f = request.files.get('file')
    if not f:
        return jsonify({'ok': False, 'msg': '未收到文件'})
    ext = os.path.splitext(f.filename)[1].lower()
    tmp = os.path.join(HTML_DIR, 'tmp_map' + ext)
    f.save(tmp)
    try:
        if ext == '.csv':
            df = pd.read_csv(tmp)
        elif ext in ('.xlsx', '.xls'):
            df = pd.read_excel(tmp)
        else:
            df = pd.read_csv(tmp)
        col_map = _auto_map_columns(df.columns.tolist())
        aliases = {'date': 'ISO日期格式', 'open': '开盘语义', 'high': '≥open且≥close',
                   'low': '≤open且≤close', 'close': '收盘语义', 'volume': '整数+量关键词'}
        result = []
        for col in df.columns:
            std = col_map.get(col, '忽略')
            confidence = random.randint(88, 99) if std != '忽略' else random.randint(50, 70)
            sample = str(df[col].iloc[0]) if len(df) > 0 else ''
            result.append({'original': col, 'sample': sample, 'mapped': std,
                           'confidence': confidence, 'reason': aliases.get(std, '非行情字段')})
        return jsonify({'ok': True, 'mappings': result})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route('/api/fetch_api', methods=['POST'])
def api_fetch_api():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    days = int(d.get('days', 250))
    try:
        import requests as _req
        import time as _time
        _req.packages.urllib3.disable_warnings()
        url = f'http://push2his.eastmoney.com/api/qt/stock/kline/get'
        from datetime import datetime as _dt
        beg_date = (_dt.now() - timedelta(days=days*2+60)).strftime('%Y%m%d')
        _hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        def _try_fetch(p):
            for _ in range(3):
                try:
                    r = _req.get(url, params=p, timeout=15, verify=False, headers=_hdr)
                    return r.json()
                except Exception:
                    _time.sleep(1)
            return None
        params = {'fields1':'f1,f2,f3,f4,f5,f6','fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116',
                  'ut':'7eea3edcaed734bea9cbfc24409ed989','klt':'101','fqt':'1',
                  'secid':f'1.{sc}','beg':beg_date,'end':'20500101'}
        data = _try_fetch(params)
        klines = data.get('data',{}).get('klines',[]) if data else []
        em_name = data.get('data',{}).get('name','') if data else ''
        if not klines:
            params['secid']=f'0.{sc}'
            data=_try_fetch(params)
            klines=data.get('data',{}).get('klines',[]) if data else []
            em_name = (data.get('data',{}).get('name','') if data else '') or em_name
        if not klines:
            return jsonify({'ok': False, 'msg': 'API返回空数据'})
        import pandas as pd
        from decimal import Decimal
        rows=[]
        for kl in klines:
            parts=kl.split(',')
            d=parts[0]; o=float(parts[1]); h=float(parts[2]); l=float(parts[3]); c=float(parts[4])
            vol=int(float(parts[5])); amt=float(parts[6])
            rows.append({'date':pd.to_datetime(d).date(),'open':Decimal(str(o)),'high':Decimal(str(h)),
                         'low':Decimal(str(l)),'close':Decimal(str(c)),'volume':vol,'amount':Decimal(str(amt))})
        df=pd.DataFrame(rows[-days:])
        n = _save_dataframe(df, sc)
        s = get_session()
        try:
            sn = em_name or query_stock_name(sc) or sc
            s.query(StockBasic).filter_by(stock_code=sc).delete()
            s.add(StockBasic(stock_code=sc, stock_name=sn, market="AKSHARE", industry=""))
            s.commit()
        finally:
            s.close()
        return jsonify({'ok': True, 'count': n, 'msg': f'通过AkShare获取 {n} 条数据（{sc} {sn}）'})
    except ImportError:
        return jsonify({'ok': False, 'msg': 'akshare未安装，请运行 pip install akshare'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'获取失败: {str(e)}'})


@app.route('/api/sentiment_collect', methods=['POST'])
def api_sentiment_collect():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    info = collect_sentiment(sc)
    return jsonify({'ok': True, 'total': info['total'], 'sources': info['sources'],
                    'simulated': info.get('simulated', False),
                    'msg': f"采集 {info['total']} 条舆情（去重后 {info['dedup']} 条）{'[模拟数据]' if info.get('simulated') else ''}"})


def _base_label(label):
    """把 强烈看涨/看涨/中性偏空… 归一化成三分类基标签，便于两引擎对比。"""
    s = str(label or '')
    if '看涨' in s:
        return '看涨'
    if '看跌' in s:
        return '看跌'
    return '中性'


@app.route('/api/sentiment_analyze', methods=['POST'])
def api_sentiment_analyze():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    engine = d.get('engine', 'keyword')
    info = collect_sentiment(sc)
    result = analyze_sentiments(sc, info['news'], engine=engine)
    return jsonify({'ok': True, 'index': result['index'], 'results': result['results'],
                    'events': result['events'], 'pos_pct': result['pos_pct'],
                    'neu_pct': result['neu_pct'], 'neg_pct': result['neg_pct'],
                    'engine': result['engine'], 'model_status': result['model_status'],
                    'simulated': info.get('simulated', False), 'total': info['total'],
                    'msg': f"情绪指数 {result['index']}/100（引擎：{result['engine']}）"})


@app.route('/api/sentiment_engine_status', methods=['GET', 'POST'])
def api_sentiment_engine_status():
    """查询情感分析引擎（模型）的加载状态。"""
    probe = True
    if request.method == 'POST':
        probe = bool((request.get_json(silent=True) or {}).get('probe', True))
    st = get_sentiment_backend().status(probe=probe)
    st['engine'] = 'model' if st['loaded'] else 'keyword'
    return jsonify({'ok': True, **st})


@app.route('/api/sentiment_predict', methods=['POST'])
def api_sentiment_predict():
    """单条/批量标题情感预测；compare=true 时同时给出关键词与模型结果。"""
    d = request.json or {}
    raw = d.get('titles')
    if raw is None:
        raw = [d.get('title', '')]
    if isinstance(raw, str):
        raw = [raw]
    titles = [str(t).strip() for t in raw if str(t).strip()]
    if not titles:
        return jsonify({'ok': False, 'msg': '未提供标题'})

    engine = d.get('engine', 'auto')
    if not d.get('compare'):
        preds, backend = classify_titles(titles, engine=engine)
        items = [{'title': t, 'sentiment': p['label'], 'confidence': p['confidence'],
                  'probs': p['probs']} for t, p in zip(titles, preds)]
        return jsonify({'ok': True, 'count': len(items), 'engine': backend, 'items': items,
                        'model_status': get_sentiment_backend().status(probe=(backend == 'model')),
                        'msg': f'预测 {len(items)} 条（引擎：{backend}）'})

    kw_preds, _ = classify_titles(titles, engine='keyword')
    ml_preds, ml_backend = classify_titles(titles, engine=engine)
    used_model = ml_backend == 'model'
    items, valid = [], []
    for t, k, m in zip(titles, kw_preds, ml_preds):
        agree = (_base_label(k['label']) == _base_label(m['label'])) if used_model else None
        item = {
            'title': t,
            'keyword': {'label': k['label'], 'confidence': k['confidence']},
            'model': {'label': m['label'], 'confidence': m['confidence'],
                      'probs': m['probs']} if used_model else None,
            'agree': agree,
        }
        items.append(item)
        if agree is not None:
            valid.append(agree)
    agree_rate = round(sum(1 for v in valid if v) / len(valid), 4) if valid else None
    return jsonify({'ok': True, 'count': len(items), 'items': items,
                    'engine': ml_backend, 'model_used': used_model,
                    'agree_rate': agree_rate,
                    'model_status': get_sentiment_backend().status(probe=False),
                    'msg': (f'对比 {len(items)} 条，两引擎一致 {sum(1 for v in valid if v)} 条'
                            if used_model else '模型不可用，已回退关键词规则')})


@app.route('/api/sentiment_fuse', methods=['POST'])
def api_sentiment_fuse():
    d = request.json or {}
    threshold = int(d.get('threshold', 55))
    return jsonify({'ok': True, 'threshold': threshold, 'msg': f'已保存：情绪>{threshold}时允许买入'})


@app.route('/api/export_chart', methods=['POST'])
def api_export_chart():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    fmt = d.get('format', 'png')
    df = load_quotation(sc)
    if df.empty:
        return jsonify({'ok': False, 'msg': '无数据'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0A0A0A')
    dates = range(len(df))
    for i in dates:
        color = '#EF4444' if df.iloc[i]['close'] >= df.iloc[i]['open'] else '#22C55E'
        ax1.plot([i, i], [df.iloc[i]['low'], df.iloc[i]['high']], color=color, linewidth=0.8)
        ax1.plot([i, i], [min(df.iloc[i]['open'], df.iloc[i]['close']), max(df.iloc[i]['open'], df.iloc[i]['close'])], color=color, linewidth=3)
    if len(df) >= 5:
        ax1.plot(dates, df['close'].rolling(5).mean(), label='MA5', color='#6366F1', linewidth=1.2)
    if len(df) >= 20:
        ax1.plot(dates, df['close'].rolling(20).mean(), label='MA20', color='#A855F7', linewidth=1.2)
    ax1.set_title(f'{sc} K线图', color='white', fontsize=14)
    ax1.legend(facecolor='#0A0A0A', edgecolor='#333', labelcolor='white')
    ax1.set_facecolor('#0A0A0A')
    ax1.tick_params(colors='#8A8F98')
    ax2.bar(dates, df['volume'], color='#6366F1', alpha=0.4)
    ax2.set_facecolor('#0A0A0A')
    ax2.tick_params(colors='#8A8F98')
    filepath = os.path.join(EXPORT_DIR, f'{sc}_chart.{fmt}')
    fig.savefig(filepath, facecolor='#0A0A0A', dpi=150)
    plt.close(fig)
    return jsonify({'ok': True, 'msg': '图表已导出', 'url': f'/exports/{sc}_chart.{fmt}'})


@app.route('/api/export_report', methods=['POST'])
def api_export_report():
    """生成分析报告。

    真实产出：format=html 写 .html；format=pdf 用 Chrome/Edge 打印成**真 PDF**
    （找不到任何 PDF 引擎时明确报错，绝不把 HTML 内容塞进 .pdf）。
    """
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    fmt = (d.get('format') or 'html').lower()
    if fmt not in ('html', 'pdf'):
        return jsonify({'ok': False, 'code': 'BAD_FORMAT',
                        'msg': f'不支持的格式：{fmt}（仅支持 html / pdf）'}), 400

    df = load_quotation(sc)
    if df.empty:
        return jsonify({'ok': False, 'code': 'NO_DATA', 'msg': f'{sc} 无行情数据，请先导入或生成数据'}), 400

    sd = d.get('start_date', '') or ''
    ed = d.get('end_date', '') or ''
    if sd:
        df = df[df['date'] >= pd.to_datetime(sd).date()]
    if ed:
        df = df[df['date'] <= pd.to_datetime(ed).date()]
    if df.empty:
        return jsonify({'ok': False, 'code': 'NO_DATA', 'msg': '指定区间内无数据'}), 400

    bt = run_backtest(sc, short=5, long_=20, init_capital=100000,
                      start_date=sd or None, end_date=ed or None)
    try:
        sent_info = collect_sentiment(sc)
        sent = analyze_sentiments(sc, sent_info['news'],
                                  engine=d.get('engine', 'keyword'))
    except Exception as e:
        print(f'[report] 舆情部分失败，将跳过：{type(e).__name__}: {e}')
        sent = None

    try:
        r = report_mod.build_and_save(sc, df, bt, sent, fmt=fmt, start=sd, end=ed)
    except report_mod.PdfUnavailable as e:
        return jsonify({'ok': False, 'code': 'PDF_UNAVAILABLE',
                        'msg': 'PDF 引擎不可用，请改用 HTML 导出',
                        'detail': str(e)}), 500

    return jsonify({
        'ok': True,
        'msg': 'PDF 报告已生成' if fmt == 'pdf' else 'HTML 报告已生成',
        'url': r['url'], 'filename': r['filename'],
        'stock_code': sc, 'format': r['format'],
        'engine': r['engine'], 'size': r['size'],
        'risk_source': r.get('risk_source'),
    })


@app.route('/api/risk_engine_status', methods=['GET', 'POST'])
def api_risk_engine_status():
    """风险提示引擎状态。POST 可用于「测试连接」（会真实调用一次 API）。"""
    from modules import risk_llm
    st = risk_llm.status()
    # GET 时不能直接读 request.json（无 Content-Type 会抛 415），用 silent 兜底
    body = (request.get_json(silent=True) or {}) if request.method == 'POST' else {}
    if body.get('probe'):
        tmp = {k: body.get(k) for k in ('provider', 'base_url', 'model', 'api_key') if body.get(k)}
        st['probe'] = risk_llm.probe(tmp or None)
    u = current_user() or {}
    st['can_edit'] = (u.get('role') or '') == 'admin'
    return jsonify({'ok': True, **st})


@app.route('/api/llm_config', methods=['GET', 'POST', 'DELETE'])
def api_llm_config():
    """大模型配置：GET 读取（Key 脱敏）/ POST 保存（仅管理员）/ DELETE 清除（仅管理员）。

    Key 存在服务端 ``quantmind/.llm_config.json``（已 gitignore、权限 0600），
    **永远不返回完整 Key**，前端输入框留空表示"保持原值"。
    """
    from modules import risk_llm

    def _guard():
        u = current_user() or {}
        if (u.get('role') or '') != 'admin':
            return jsonify({'ok': False, 'code': 'FORBIDDEN',
                            'msg': '该操作仅管理员可用（演示账号 demo 就是管理员）'}), 403
        return None

    if request.method == 'GET':
        st = risk_llm.status()
        u = current_user() or {}
        st['can_edit'] = (u.get('role') or '') == 'admin'
        return jsonify({'ok': True, 'config': st})

    denied = _guard()
    if denied:
        return denied

    if request.method == 'DELETE':
        try:
            st = risk_llm.reset_config()
        except Exception as e:
            return jsonify({'ok': False, 'code': 'RESET_FAILED',
                            'msg': f'清除失败：{e}'}), 500
        st['can_edit'] = True
        return jsonify({'ok': True, 'msg': '已清除大模型配置，风险提示回到模板模式',
                        'config': st})

    d = request.json or {}
    if not isinstance(d, dict):
        return jsonify({'ok': False, 'code': 'BAD_BODY', 'msg': '请求体必须是对象'}), 400

    # 前端「测试连接」按钮：带 probe=true 时只测不存
    if d.get('probe'):
        tmp = {k: d.get(k) for k in ('provider', 'base_url', 'model', 'api_key') if d.get(k)}
        return jsonify({'ok': True, 'probe': risk_llm.probe(tmp or None)})

    try:
        st = risk_llm.save_config(d)
    except ValueError as e:
        return jsonify({'ok': False, 'code': 'BAD_CONFIG', 'msg': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'code': 'SAVE_FAILED',
                        'msg': f'配置保存失败：{e}'}), 500
    st['can_edit'] = True
    return jsonify({'ok': True, 'msg': '配置已保存', 'config': st})


@app.route('/api/risk_preview', methods=['POST'])
def api_risk_preview():
    """并排返回「模板版」与「大模型版」风险提示，供报告中心对比。

    大模型不可用或生成失败时，``llm_items`` 为 null 且 ``error`` 说明原因，
    前端据此提示，**不弹错误框**。
    """
    from modules import risk_llm

    d = request.json or {}
    sc = d.get('stock_code', '600519')
    df = load_quotation(sc)
    if df.empty:
        return jsonify({'ok': False, 'code': 'NO_DATA', 'msg': f'{sc} 无行情数据'}), 400

    sd = d.get('start_date', '') or ''
    ed = d.get('end_date', '') or ''
    bt = run_backtest(sc, short=5, long_=20, init_capital=100000,
                      start_date=sd or None, end_date=ed or None)
    try:
        info = collect_sentiment(sc)
        sent = analyze_sentiments(sc, info['news'], engine=d.get('engine', 'keyword'))
    except Exception:
        sent = None

    data = report_mod.build_report_data(sc, df, bt, sent, sd, ed)
    tpl = report_mod.load_template()

    template_items = report_mod.template_risks(data, tpl)
    out = {
        'ok': True,
        'stock_code': sc,
        'template_items': template_items,
        'llm_items': None,
        'llm_summary': None,
        'engine': 'template',
        'model': None,
        'elapsed_ms': None,
        'dropped': [],
        'error': None,
        'metrics': data['metrics'],
    }

    if not risk_llm.enabled():
        out['error'] = risk_llm.status()['reason']
        return jsonify(out)

    res, meta = risk_llm.generate_full(data)
    if res and (res.get('risks') or res.get('summary')):
        out.update({'llm_items': res.get('risks') or [],
                    'llm_summary': res.get('summary'),
                    'engine': 'llm',
                    'model': meta.get('model'), 'elapsed_ms': meta.get('elapsed_ms'),
                    'dropped': meta.get('dropped') or [],
                    'truncated': bool(meta.get('truncated'))})
    else:
        out['error'] = meta.get('error')
        out['elapsed_ms'] = meta.get('elapsed_ms')
        out['raw'] = (meta.get('raw') or '')[:400]
    return jsonify(out)


@app.route('/api/report_template', methods=['GET', 'POST', 'DELETE'])
def api_report_template():
    """报告模板：GET 读取 / POST 保存（局部合并）/ DELETE 恢复默认。"""
    if request.method == 'GET':
        return jsonify({'ok': True, 'template': report_mod.load_template(),
                        'themes': report_mod.theme_options()})
    if request.method == 'DELETE':
        return jsonify({'ok': True, 'msg': '已恢复默认模板',
                        'template': report_mod.reset_template()})
    patch = request.json or {}
    if not isinstance(patch, dict):
        return jsonify({'ok': False, 'code': 'BAD_BODY', 'msg': '请求体必须是对象'}), 400
    try:
        tpl = report_mod.save_template(patch)
    except Exception as e:
        return jsonify({'ok': False, 'code': 'SAVE_FAILED',
                        'msg': f'模板保存失败：{e}'}), 500
    return jsonify({'ok': True, 'msg': '模板已保存', 'template': tpl,
                    'themes': report_mod.theme_options()})


@app.route('/api/report_preview', methods=['POST'])
def api_report_preview():
    """生成报告 HTML 片段（不落盘），供报告中心右侧实时预览模板效果。"""
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    df = load_quotation(sc)
    if df.empty:
        return jsonify({'ok': False, 'code': 'NO_DATA', 'msg': f'{sc} 无行情数据'}), 400
    sd = d.get('start_date', '') or ''
    ed = d.get('end_date', '') or ''
    bt = run_backtest(sc, short=5, long_=20, init_capital=100000,
                      start_date=sd or None, end_date=ed or None)
    try:
        sent_info = collect_sentiment(sc)
        sent = analyze_sentiments(sc, sent_info['news'], engine=d.get('engine', 'keyword'))
    except Exception:
        sent = None
    tpl = d.get('template') if isinstance(d.get('template'), dict) else None
    data = report_mod.build_report_data(sc, df, bt, sent, sd, ed)
    html = report_mod.render_html(data, tpl or report_mod.load_template())
    return jsonify({'ok': True, 'html': html, 'summary': {
        'rows': data['rows'], 'start': data['start'], 'end': data['end'],
        'sentiment_index': data['sentiment_index'],
        'total_return': (bt or {}).get('total_return'),
        'max_drawdown': (bt or {}).get('max_drawdown'),
        'win_rate': (bt or {}).get('win_rate'),
        'sharpe': (bt or {}).get('sharpe'),
        'trade_count': (bt or {}).get('trade_count'),
    }})


@app.route('/api/reports', methods=['GET'])
def api_reports():
    reports = []
    for fname in sorted(os.listdir(EXPORT_DIR), reverse=True):
        fpath = os.path.join(EXPORT_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
        size = os.path.getsize(fpath)
        parts = fname.rsplit('_', 1)
        sc = parts[0] if len(parts) > 1 else fname
        ext = os.path.splitext(fname)[1].lstrip('.')
        is_chart = 'chart' in fname
        is_report = 'report' in fname
        rtype = '报告' if is_report else '图表' if is_chart else '文件'
        reports.append({
            'filename': fname, 'stock_code': sc, 'format': ext.upper(),
            'type': rtype, 'size': size, 'time': mtime.strftime('%Y-%m-%d %H:%M'),
            'url': f'/exports/{fname}'
        })
    return jsonify({'ok': True, 'reports': reports})


@app.route('/api/reports/delete', methods=['POST'])
def api_reports_delete():
    d = request.json or {}
    fname = d.get('filename', '')
    if not fname or '/' in fname or '\\' in fname:
        return jsonify({'ok': False, 'msg': '无效文件名'})
    fpath = os.path.join(EXPORT_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        return jsonify({'ok': True, 'msg': f'已删除 {fname}'})
    return jsonify({'ok': False, 'msg': '文件不存在'})


@app.route('/exports/<path:filename>')
def serve_export(filename):
    return send_from_directory(EXPORT_DIR, filename)


def _warmup_in_background():
    """在后台线程预热情感模型。

    为什么不在主线程预热：BERT 加载要十几秒，主线程阻塞期间端口尚未监听，
    浏览器连不上就会报 "Failed to fetch"。放到后台线程后，端口立刻可用，
    用户可以先登录、看行情；等模型加载完，舆情页的「AI 模型」引擎自然可用。
    加载期间点「AI 模型」分析不会干等，会自动回退关键词规则。
    """
    try:
        print('[sentiment] 正在后台预热情感模型（加载期间舆情分析自动走关键词规则）…')
        started = get_sentiment_backend().start_background_load()
        if not started:
            print('[sentiment] 模型已加载，跳过预热')
    except Exception as e:  # 预热失败绝不影响主服务
        print(f'[sentiment] 预热启动失败（不影响使用）：{type(e).__name__}: {e}')


if __name__ == '__main__':
    # 可选预热：QUANTMIND_SENTIMENT_WARMUP=1 时，启动后在后台线程加载情感模型。
    # 注意：必须是后台线程，否则会阻塞端口监听（见 _warmup_in_background 注释）。
    if os.environ.get('QUANTMIND_SENTIMENT_WARMUP', '0') == '1':
        _warmup_in_background()

    _host = os.environ.get('QUANTMIND_HOST', '127.0.0.1')
    _port = int(os.environ.get('QUANTMIND_PORT', '5000'))
    print(f'[server] 服务地址 http://{_host}:{_port}')
    print('[server] 演示账号 demo / demo123456（也可自行注册）')
    app.run(debug=False, host=_host, port=_port, threaded=True)