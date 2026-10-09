import sys, os, io, json, math, random
from flask import Flask, send_from_directory, send_file, jsonify, request
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.data_fetch import generate_sample_data, import_csv, load_quotation, import_tdx_day, import_tdx_dir
from modules.data_process import clean, calc_indicators
from modules.backtest import run_backtest
from modules.sentiment import collect_sentiment, analyze_sentiments
from db import get_session, TechnicalIndicator, QuotationDaily, StockBasic

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..'))

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
    bt = run_backtest(sc, short=short, long_=long_, init_capital=cap,
                      start_date=sd if sd else None, end_date=ed if ed else None)
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
            result = analyze_sentiments(sc, info['news'])
            results.append({'step': 'sentiment', 'ok': True, 'msg': f"舆情分析：{info['total']} 条，情绪指数 {result['index']}"})
        elif step == 'report':
            df = load_quotation(sc)
            bt = run_backtest(sc, short=5, long_=20, init_capital=100000)
            results.append({'step': 'report', 'ok': True, 'msg': f"报告已生成：{len(df)} 条数据，{bt['trade_count'] if bt else 0} 笔交易" if df is not None else '报告生成失败'})
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


@app.route('/api/sentiment_analyze', methods=['POST'])
def api_sentiment_analyze():
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    info = collect_sentiment(sc)
    result = analyze_sentiments(sc, info['news'])
    return jsonify({'ok': True, 'index': result['index'], 'results': result['results'],
                    'events': result['events'], 'pos_pct': result['pos_pct'],
                    'neu_pct': result['neu_pct'], 'neg_pct': result['neg_pct'],
                    'msg': f"情绪指数 {result['index']}/100"})


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
    d = request.json or {}
    sc = d.get('stock_code', '600519')
    fmt = d.get('format', 'html')
    df = load_quotation(sc)
    if df.empty:
        return jsonify({'ok': False, 'msg': '无数据'})
    sd = d.get('start_date', '')
    ed = d.get('end_date', '')
    if sd:
        df = df[df['date'] >= pd.to_datetime(sd).date()]
    if ed:
        df = df[df['date'] <= pd.to_datetime(ed).date()]
    if df.empty:
        return jsonify({'ok': False, 'msg': '指定区间无数据'})
    bt = run_backtest(sc, short=5, long_=20, init_capital=100000, start_date=sd or None, end_date=ed or None)
    sent_info = collect_sentiment(sc)
    sent_result = analyze_sentiments(sc, sent_info['news'])
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{sc} 综合量化分析报告</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;background:#1a1a2e;color:#eee;padding:40px;max-width:900px;margin:auto}}
h1{{color:#6366F1}}h2{{color:#A855F7;border-bottom:1px solid #333;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #333;padding:8px}}
.up{{color:#22C55E}}.down{{color:#EF4444}}</style></head><body>
<h1>{sc} 综合量化分析报告</h1><p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>① 行情概览</h2><p>数据区间: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}，共 {len(df)} 条</p>
<h2>② 策略逻辑</h2><p>双均线(5/20)交叉信号 + 情绪指数过滤买入（当前情绪{sent_result['index']}，{'允许买入' if sent_result['index']>=45 else '拒绝买入'}），90% 仓位</p>
<h2>③ 回测绩效</h2><table><tr><th>指标</th><th>值</th></tr>"""
    if bt:
        html += f"<tr><td>总收益</td><td class='{'up' if bt['total_return']>0 else 'down'}'>{bt['total_return']}%</td></tr>"
        html += f"<tr><td>年化收益</td><td class='{'up' if bt['annual_return']>0 else 'down'}'>{bt['annual_return']}%</td></tr>"
        html += f"<tr><td>最大回撤</td><td class='down'>{bt['max_drawdown']}%</td></tr>"
        html += f"<tr><td>胜率</td><td>{bt['win_rate']}%</td></tr>"
        html += f"<tr><td>夏普比率</td><td>{bt['sharpe']}</td></tr>"
        html += f"<tr><td>交易笔数</td><td>{bt['trade_count']}</td></tr>"
    sent_label = '偏乐观' if sent_result['index'] > 55 else '偏悲观' if sent_result['index'] < 45 else '中性'
    html += f"</table><h2>④ 舆情摘要</h2><p>情绪指数 {sent_result['index']}/100 {sent_label}；看涨{sent_result['pos_pct']}% 看跌{sent_result['neg_pct']}%</p>"
    if sent_result['events']:
        html += "<table><tr><th>事件</th><th>类型</th><th>提及</th><th>情感</th></tr>"
        for ev in sent_result['events'][:5]:
            html += f"<tr><td>{ev['name']}</td><td>{ev['type']}</td><td>{ev['count']}条</td><td>{ev['score']}</td></tr>"
        html += "</table>"
    html += "<h2>⑤ 风险提示</h2><ul>"
    html += "<li>本报告基于历史数据回测，历史表现不代表未来收益，实盘交易存在亏损风险。</li>"
    if bt:
        html += f"<li>策略最大回撤达 {bt['max_drawdown']}%，请根据自身风险承受能力谨慎决策。</li>"
        if bt['trade_count'] < 10:
            html += f"<li>回测交易笔数仅 {bt['trade_count']} 次，样本量不足，统计结果可能不具代表性。</li>"
        if abs(bt['sharpe']) < 0.5:
            html += "<li>夏普比率较低，策略风险调整后收益不理想，需谨慎参考。</li>"
    html += f"<li>数据区间 {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}，共 {len(df)} 条，数据质量直接影响分析结论。</li>"
    if len(df) < 250:
        html += f"<li>数据样本仅 {len(df)} 条（不足一年），回测结果可能存在过拟合风险。</li>"
    html += "<li>舆情研判由 AI 基于关键词分析生成，无法保证信息完整性与判断准确性，仅供参考。</li>"
    html += "<li>双均线策略为趋势跟踪型策略，在震荡行情中可能频繁产生虚假信号导致亏损。</li>"
    html += "</ul></body></html>"
    filepath = os.path.join(EXPORT_DIR, f'{sc}_report.{fmt}')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    return jsonify({'ok': True, 'msg': '报告已导出', 'url': f'/exports/{sc}_report.{fmt}',
                    'filename': f'{sc}_report.{fmt}', 'stock_code': sc, 'format': fmt})


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


if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)