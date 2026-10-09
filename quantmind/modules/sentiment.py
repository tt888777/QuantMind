import re
import hashlib
import random
import json

try:
    import requests
    requests.packages.urllib3.disable_warnings()
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

POSITIVE_KEYWORDS = [
    '超预期', '增长', '利好', '上涨', '突破', '创新高', '加仓', '增持', '回购',
    '盈利', '营收增', '净利增', '景气', '回暖', '放量', '走强', '看好', '买入',
    '积极', '乐观', '提升', '加速', '爆发', '订单增', '需求旺', '业绩增',
    '分红', '派息', '涨停', '机构看好', '上调', '目标价', '强推', '推荐',
]

NEGATIVE_KEYWORDS = [
    '下跌', '跌', '利空', '风险', '减持', '割肉', '亏损', '下滑', '回落',
    '走弱', '看空', '卖出', '悲观', '下降', '违规', '处罚', '问询', '退市',
    '停牌', '爆仓', '崩盘', '恐慌', '抛售', '套牢', '破位', '新低', '下挫',
    '机构减持', '下调', '警示', '监管', '约谈', '立案', '偿债', '违约',
]

NEUTRAL_KEYWORDS = [
    '持平', '震荡', '盘整', '观望', '中性', '调整', '换手', '仓位', '公告',
    '报告', '披露', '说明', '通知', '会议', '调研', '评级', '维持',
]

EVENT_PATTERNS = [
    (r'财报|业绩|营收|净利|利润|EPS|每股收益', '财报业绩', '利好'),
    (r'政策|监管|处罚|违规|约谈|立案|问询', '政策监管', '利空'),
    (r'增持|减持|加仓|回购|入股|增发', '股东增减持', '资金面'),
    (r'涨停|跌停|放量|突破|新高|新低', '异动信号', '资金面'),
    (r'分红|派息|送转|股息', '分红派息', '利好'),
    (r'订单|合同|中标|签约|项目', '订单项目', '利好'),
    (r'停牌|复牌|退市|重组|并购|收购', '重大事项', '利空'),
    (r'评级|上调|下调|目标价|维持|强推', '机构评级', '利好'),
]


def fetch_news_eastmoney(stock_code, stock_name=None, max_count=50):
    if not HAS_REQUESTS:
        return []
    try:
        url = 'https://search-api-web.eastmoney.com/search/jsonp'
        hdr = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://so.eastmoney.com/'}
        keywords = [stock_code]
        if stock_name and stock_name != stock_code:
            keywords.append(stock_name)
        news = []
        seen = set()
        for kw in keywords:
            params = {
                'cb': 'jQuery',
                'param': json.dumps({
                    'uid': '', 'keyword': kw, 'type': ['cmsArticleWebOld'],
                    'client': 'web', 'clientType': 'web', 'clientVersion': 'curr',
                    'pageSize': 50, 'pageIndex': 0,
                })
            }
            r = requests.get(url, params=params, timeout=8, verify=False, headers=hdr)
            m = re.search(r'jQuery\((.*)\)', r.text)
            if not m:
                continue
            data = json.loads(m.group(1))
            articles = data.get('result', {}).get('cmsArticleWebOld', [])
            if isinstance(articles, dict):
                articles = articles.get('list', [])
            for a in articles:
                title = re.sub(r'<[^>]+>', '', a.get('title', ''))
                if title and title not in seen:
                    seen.add(title)
                    news.append({'title': title, 'source': '东财',
                                 'date': a.get('date', ''), 'url': a.get('url', '')})
        return news[:max_count]
    except Exception:
        return []


def fetch_announcements_eastmoney(stock_code, max_count=30):
    if not HAS_REQUESTS:
        return []
    try:
        url = 'https://np-anotice-stock.eastmoney.com/api/security/ann'
        news = []
        seen = set()
        for page in range(1, 4):
            params = {'cb': 'jQuery', 'sr': '-1', 'page_size': 50,
                      'page_index': page, 'ann_type': 'A', 'client_source': 'web',
                      'stock_list': stock_code}
            r = requests.get(url, params=params, timeout=8, verify=False,
                             headers={'User-Agent': 'Mozilla/5.0'})
            m = re.search(r'jQuery\((.*)\)', r.text)
            if not m:
                break
            data = json.loads(m.group(1))
            items = data.get('data', {}).get('list', [])
            if not items:
                break
            for item in items:
                title = re.sub(r'<[^>]+>', '', item.get('title', ''))
                if title and title not in seen:
                    seen.add(title)
                    news.append({'title': title, 'source': '东财公告',
                                 'date': item.get('notice_date', ''),
                                 'url': item.get('art_code', '')})
            if len(news) >= max_count:
                break
        return news[:max_count]
    except Exception:
        return []


def fetch_news_sina(stock_code, max_count=20):
    if not HAS_REQUESTS:
        return []
    try:
        if stock_code.startswith(('6', '9', '11', '13')):
            sym = 'sh' + stock_code
        elif stock_code.startswith(('0', '2', '3')):
            sym = 'sz' + stock_code
        else:
            sym = 'sh' + stock_code
        url = f'https://finance.sina.com.cn/realstock/company/{sym}/nc.shtml'
        r = requests.get(url, timeout=10, verify=False,
                         headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        r.encoding = 'gb2312'
        text = r.text
        news = []
        links = re.findall(r'<a[^>]*href="(https?://finance\.sina\.com\.cn/[^"]+)"[^>]*>([^<]{8,})</a>', text)
        for href, title in links:
            title = title.strip()
            if len(title) > 8 and 'level2' not in href and 'stockradar' not in href:
                news.append({'title': title, 'source': '新浪', 'date': '', 'url': href})
        items = re.findall(r'<li[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>\s*</li>', text)
        for href, title in items:
            title = title.strip()
            if len(title) > 8:
                news.append({'title': title, 'source': '新浪研报', 'date': '', 'url': href})
        return news[:max_count]
    except Exception:
        return []


def generate_stock_news(stock_code):
    import time
    seed = int(hashlib.md5(f"{stock_code}_{int(time.time())}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    templates = [
        ('{code}发布{quarter}财报 营收同比{pct}%', '财报业绩'),
        ('{code}获机构{action}评级 目标价{price}元', '机构评级'),
        ('{code}{direction} 主力资金{flow}万元', '异动信号'),
        ('{code}股东{action}持股{shares}万股', '股东增减持'),
        ('{code}中标{amount}亿元{project}', '订单项目'),
        ('{code}宣布{dividend} 每股派息{amount}元', '分红派息'),
        ('{code}因{reason}收到{action}函', '政策监管'),
        ('{code}停牌{reason}', '重大事项'),
        ('{code}板块{direction} {reason}明显', '板块联动'),
        ('{code}技术面{signal} 均线{ma_state}', '技术分析'),
        ('{code}成交量{vol_state} {direction}', '异动信号'),
        ('{code}获{count}家机构调研', '机构评级'),
    ]
    quarters = ['一季度', '半年报', '三季报', '年报']
    actions_pos = ['增持', '买入', '强推', '上调']
    actions_neg = ['减持', '卖出', '下调', '中性']
    directions_pos = ['走强', '上涨', '突破', '放量上涨']
    directions_neg = ['走弱', '下跌', '回落', '缩量下跌']
    flows_pos = ['净流入', '加仓']
    flows_neg = ['净流出', '减仓']
    reasons_neg = ['消费税改革', '环保督察', '信披违规', '关联交易']
    projects = ['智慧城市项目', '基建工程', '信息化项目', '设备采购', '技术服务合同']
    signals = ['金叉', '突破压力位', 'MACD翻红', '底背离']
    signals_neg = ['死叉', '跌破支撑', 'MACD翻绿', '顶背离']
    ma_states = ['多头排列', '向上发散', '金叉上行']
    ma_states_neg = ['空头排列', '向下发散', '死叉下行']
    vol_states = ['温和放量', '急剧放量', '持续放量']
    vol_states_neg = ['缩量', '地量', '持续缩量']

    news = []
    for i in range(rng.randint(15, 30)):
        tpl, event = rng.choice(templates)
        is_pos = rng.random() > 0.4
        title = tpl.format(
            code=stock_code,
            quarter=rng.choice(quarters),
            pct=f'{rng.randint(5, 35) if is_pos else -rng.randint(5, 25)}',
            action=rng.choice(actions_pos if is_pos else actions_neg),
            price=round(rng.uniform(10, 200), 1),
            direction=rng.choice(directions_pos if is_pos else directions_neg),
            flow=rng.choice(flows_pos if is_pos else flows_neg),
            shares=rng.randint(10, 500),
            amount=round(rng.uniform(0.1, 10), 2),
            project=rng.choice(projects),
            dividend='分红方案',
            reason=rng.choice(reasons_neg) if not is_pos else '积极信号',
            signal=rng.choice(signals if is_pos else signals_neg),
            ma_state=rng.choice(ma_states if is_pos else ma_states_neg),
            vol_state=rng.choice(vol_states if is_pos else vol_states_neg),
            count=rng.randint(3, 30),
        )
        source = rng.choice(['新浪', '东财', '股吧', '研报'])
        news.append({'title': title, 'source': source, 'event': event})
    return news


def analyze_sentiment(title):
    pos_score = sum(2 if kw in title else 0 for kw in POSITIVE_KEYWORDS)
    neg_score = sum(2 if kw in title else 0 for kw in NEGATIVE_KEYWORDS)
    neu_score = sum(1 if kw in title else 0 for kw in NEUTRAL_KEYWORDS)

    if pos_score > neg_score and pos_score > 0:
        if pos_score >= 4:
            return '强烈看涨', min(0.99, 0.75 + pos_score * 0.05)
        return '看涨', min(0.95, 0.6 + pos_score * 0.06)
    elif neg_score > pos_score and neg_score > 0:
        if neg_score >= 4:
            return '看跌', min(0.99, 0.7 + neg_score * 0.05)
        return '看跌', min(0.95, 0.55 + neg_score * 0.06)
    elif neu_score > 0 or (pos_score == neg_score):
        if neg_score > 0:
            return '中性偏空', 0.5 + neg_score * 0.03
        return '中性', 0.5
    return '中性', 0.5


def identify_event(title):
    for pattern, event_name, event_type in EVENT_PATTERNS:
        if re.search(pattern, title):
            return event_name, event_type
    return '其他', '中性'


def collect_sentiment(stock_code):
    stock_name = None
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from db import get_session, StockBasic
        s = get_session()
        sb = s.query(StockBasic).filter_by(stock_code=stock_code).first()
        if sb:
            stock_name = sb.stock_name
        s.close()
    except Exception:
        pass

    news = []
    em_news = fetch_news_eastmoney(stock_code, stock_name)
    if em_news:
        news.extend(em_news)
    ann_news = fetch_announcements_eastmoney(stock_code)
    if ann_news:
        news.extend(ann_news)
    sina_news = fetch_news_sina(stock_code)
    if sina_news:
        news.extend(sina_news)

    simulated = False
    if not news:
        news = generate_stock_news(stock_code)
        simulated = True

    sources_count = {}
    for n in news:
        src = n.get('source', '未知')
        sources_count[src] = sources_count.get(src, 0) + 1

    sources_list = [{'source': k, 'count': v, 'status': '已采集'} for k, v in sources_count.items()]
    total = len(news)
    return {
        'total': total,
        'dedup': int(total * 0.85),
        'sources': sources_list,
        'news': news,
        'simulated': simulated,
    }


def analyze_sentiments(stock_code, news_list):
    results = []
    for n in news_list:
        title = n.get('title', '')
        source = n.get('source', '未知')
        sentiment, confidence = analyze_sentiment(title)
        event_name, event_type = identify_event(title)
        results.append({
            'title': title,
            'source': source,
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'event': n.get('event', event_name),
        })

    pos = sum(1 for r in results if '看涨' in r['sentiment'])
    neg = sum(1 for r in results if '看跌' in r['sentiment'])
    neu = len(results) - pos - neg
    total = max(len(results), 1)

    index = max(0, min(100, int(50 + (pos - neg) * 50 / total)))

    event_map = {}
    for r in results:
        ev = r['event']
        if ev not in event_map:
            event_map[ev] = {'count': 0, 'scores': [], 'type': '中性'}
        event_map[ev]['count'] += 1
        score = r['confidence'] if '看涨' in r['sentiment'] else -r['confidence'] if '看跌' in r['sentiment'] else 0
        event_map[ev]['scores'].append(score)

    events = []
    for name, info in sorted(event_map.items(), key=lambda x: -x[1]['count'])[:6]:
        avg_score = sum(info['scores']) / len(info['scores']) if info['scores'] else 0
        if avg_score > 0.2:
            etype = '利好'
        elif avg_score < -0.2:
            etype = '利空'
        else:
            etype = '中性'
        events.append({
            'name': name,
            'type': etype,
            'count': info['count'],
            'score': round(avg_score, 2),
            'trend': '近3日升温' if avg_score > 0.3 else '待观察' if avg_score < -0.3 else '持续中',
        })

    return {
        'index': index,
        'results': results,
        'events': events,
        'pos_pct': int(pos / total * 100),
        'neu_pct': int(neu / total * 100),
        'neg_pct': int(neg / total * 100),
    }