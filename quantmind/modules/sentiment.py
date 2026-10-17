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

# 可插拔的模型后端（torch/transformers 缺失或模型不存在时自动降级为关键词规则）
try:
    from .sentiment_model import get_backend
except ImportError:                                  # 直接以脚本方式运行本文件时
    try:
        from sentiment_model import get_backend
    except ImportError:
        def get_backend():
            class _Unavailable:
                available = False

                def status(self):
                    return {'available': False, 'engine': 'keyword',
                            'error': 'sentiment_model 模块不可用'}

                def predict(self, titles):
                    return None
            return _Unavailable()

POSITIVE_KEYWORDS = [
    '超预期', '增长', '利好', '上涨', '突破', '创新高', '加仓', '增持', '回购',
    '盈利', '营收增', '净利增', '景气', '回暖', '放量', '走强', '看好', '买入',
    '积极', '乐观', '提升', '加速', '爆发', '订单增', '需求旺', '业绩增',
    '分红', '派息', '涨停', '机构看好', '上调', '目标价', '强推', '推荐',
]

NEGATIVE_KEYWORDS = [
    '下跌', '跌', '利空', '减持', '割肉', '亏损', '下滑', '回落',
    '走弱', '看空', '卖出', '悲观', '下降', '违规', '处罚', '问询', '退市',
    '停牌', '爆仓', '崩盘', '恐慌', '抛售', '套牢', '破位', '新低', '下挫',
    '机构减持', '下调', '警示', '监管', '约谈', '立案', '偿债', '违约',
]
# ⚠️ 特意移除了「风险」二字：它在公告标题里常作为**中性合规词**出现
# （"关于……的风险评估报告"、"风险提示性公告"），把它当负面会误伤大量常规披露。
# 真正的风险事件由 EVENT_PATTERNS 里的"监管处罚 / 债务风险 / 业绩亏损"等具体规则捕获。

NEUTRAL_KEYWORDS = [
    '持平', '震荡', '盘整', '观望', '中性', '调整', '换手', '仓位', '公告',
    '报告', '披露', '说明', '通知', '会议', '调研', '评级', '维持',
]

# 事件识别规则表
# ---------------------------------------------------------------------------
# 每条：(正则, 事件名, 语义方向)。**按顺序匹配，先命中先返回** ——
# 所以"负面"和"具体业务"的规则要排在"中性治理"之前，
# 否则「关于股东会取消的公告」会被治理规则先捞走。
#
# 三个方向：
#   利好 / 利空  —— 对股价有明确方向性影响
#   中性         —— **确实发生了、但不指向涨跌**的事（如董事会决议、法律意见书）
#   资金面       —— 资金进出类信号，方向依上下文
#
# 中性也有价值：它让「其他」这个垃圾桶只留下真正无法归类的内容，
# 这样"其他占比"才是一个有意义的质量指标（而不是"我的规则表覆盖了没有"）。
EVENT_PATTERNS = [
    # ---- 1. 强负面：先判，优先级最高 ----
    (r'退市|暂停上市|终止上市|破产|清算', '退市风险', '利空'),
    (r'立案|调查|处罚|罚款|警示函|监管函|问询函|关注函|违规|违法|处罚决定', '监管处罚', '利空'),
    (r'亏损|预亏|首亏|增亏|业绩下滑|净利润下降|营业收入下降|商誉减值|计提减值', '业绩亏损', '利空'),
    (r'诉讼|仲裁|败诉|被执行|冻结|查封|失信', '法律诉讼', '利空'),
    (r'质押|平仓|爆仓|延期兑付|违约|偿债|资金链', '债务风险', '利空'),
    (r'辞职|离职|罢免|免职|被带走|留置', '人事变动', '利空'),
    (r'解禁|限售股上市流通', '限售解禁', '利空'),
    (r'取消|终止|中止|撤回|否决|未通过|失败', '事项终止', '利空'),

    # ---- 2. 强正面 ----
    (r'增持|回购|拟收购|要约收购|举牌', '股东增持回购', '利好'),
    (r'中标|签订.{0,8}合同|签署.{0,8}协议|获得.{0,6}订单|新签', '订单合同', '利好'),
    (r'超预期|预增|扭亏|业绩大增|净利增|营收增|利润增|同比增长', '业绩增长', '利好'),
    (r'分红|派息|送转|股息|权益分派|利润分配', '分红派息', '利好'),
    (r'涨停|创新高|突破|历史新高', '异动信号', '利好'),
    (r'复牌|重组|并购|借壳|资产注入', '重大事项', '利好'),

    # ---- 3. 财报与业绩（方向取决于内容，先归到财报） ----
    (r'年度报告|半年度报告|季度报告|第[一二三四]季度报告|年报|中报|半年报|季报|'
     r'财报|业绩预告|业绩快报|营收|净利|利润|EPS|每股收益|主要经营数据', '财报业绩', '利好'),
    (r'点评|跟踪报告|深度报告|研报|首次覆盖|投资建议|更新报告|调研', '机构研报', '中性'),

    # ---- 4. 运营与业务（有实质含义） ----
    (r'产能|扩产|投产|技改|产线', '产能扩张', '利好'),
    (r'提价|涨价|调价|出厂价', '产品提价', '利好'),
    (r'降价|促销|去库存|库存高', '价格库存', '利空'),
    (r'合作|战略协议|框架协议|联手|联合', '战略合作', '利好'),
    (r'新品|新产品|上市发布|推出|首发|动销|渠道|铺货|终端|营销|市场改革|市场化改革', '产品渠道', '中性'),
    (r'专利|研发|技术突破|创新|获奖|认证', '研发创新', '利好'),
    (r'减持|拟减持|股份转让', '股东减持', '利空'),
    (r'增发|定增|配股|可转债|融资|发行股份', '再融资', '中性'),

    # ---- 5. 舆情与媒体（非公告类） ----
    (r'论坛|峰会|发布会|展会|论坛丨|启幕|共探|酒文化|文化|品牌|专题|活动|品鉴|'
     r'走深|探访|登陆|上新|佳酿|好物|逛市集', '品牌活动', '中性'),
    (r'行业|板块|赛道|周期|供需|政策|新政|楼市|动销|旺季|淡季|供给侧|批价|'
     r'库存周期|渠道调研', '行业动态', '中性'),
    (r'[\u4e00-\u9fa5]{2,6}\(\d{6}\)\s*[:：]', '个股观点', '中性'),

    # ---- 6. 中性治理事务（公告里最常见的一类，原先 88% 都掉在这里） ----
    (r'董事会|监事会|股东会|股东大会|决议公告|会议决议|会议资料|会议通知', '公司治理', '中性'),
    (r'独立董事|提名人|候选人|声明与承诺|薪酬|考核|管理办法|工作制度|议事规则', '公司治理', '中性'),
    (r'法律意见书|律师事务所|审计|会计师事务所|履职情况|内部控制|内控|专项说明|核查意见', '中介机构报告', '中性'),
    (r'关联交易|关联方|商标许可|日常关联', '关联交易', '中性'),
    (r'聘任|辞任|选举|换届|变更|换届选举|任职', '人事变动', '中性'),
    (r'章程|修订|制度|规范', '制度修订', '中性'),
    (r'风险评估报告|评估报告|自查报告|整改报告', '专项报告', '中性'),
    (r'公告|通知|披露|提示性公告|进展|补充', '常规公告', '中性'),
]

# 兜底：什么都没命中时
FALLBACK_EVENT = ('其他', '中性')


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
    """识别事件类型。返回 (事件名, 方向)。

    按 ``EVENT_PATTERNS`` 顺序匹配，先命中先返回（规则已按"强负面 → 强正面 →
    业务 → 中性治理"排序）。全部不命中才归为「其他」。
    """
    t = title or ''
    for pattern, event_name, event_type in EVENT_PATTERNS:
        if re.search(pattern, t):
            return event_name, event_type
    return FALLBACK_EVENT


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


def classify_titles(titles, engine='keyword'):
    """批量判定标题情感。

    engine:
      * ``'keyword'`` —— 仅用关键词规则（原行为，默认）
      * ``'model'``   —— 用 BERT+LoRA 模型；**模型不可用时自动回退关键词**
      * ``'auto'``    —— 有模型就用模型，否则用关键词

    返回 ``(preds, backend)``：
      * ``preds``   —— ``[{'label': str, 'confidence': float, 'probs': dict|None}]``
      * ``backend`` —— 实际生效的引擎，``'model'`` 或 ``'keyword'``
    """
    titles = [str(t or '') for t in titles]

    if engine in ('model', 'auto'):
        backend = get_backend()
        if backend.available:
            preds = backend.predict(titles)
            if preds is not None and len(preds) == len(titles):
                return [
                    {'label': p['label'],
                     'confidence': round(float(p['confidence']), 4),
                     'probs': p.get('probs')}
                    for p in preds
                ], 'model'

    preds = []
    for t in titles:
        label, conf = analyze_sentiment(t)
        preds.append({'label': label, 'confidence': round(float(conf), 4), 'probs': None})
    return preds, 'keyword'


def analyze_sentiments(stock_code, news_list, engine='keyword'):
    """舆情聚合分析。

    ``engine`` 语义同 :func:`classify_titles`；返回结构保持向后兼容，
    仅额外增加 ``engine`` 与 ``model_status`` 两个字段。
    """
    titles = [n.get('title', '') for n in news_list]
    preds, backend = classify_titles(titles, engine=engine)

    results = []
    for n, p in zip(news_list, preds):
        title = n.get('title', '')
        event_name, event_type = identify_event(title)
        results.append({
            'title': title,
            'source': n.get('source', '未知'),
            'sentiment': p['label'],
            'confidence': round(float(p['confidence']), 2),
            'event': n.get('event', event_name),
            'event_type': n.get('event_type') or event_type,
            'engine': backend,
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
            # 事件方向来自规则表（r 里由 identify_event 给出），不要在这里写死"中性"
            event_map[ev] = {'count': 0, 'scores': [], 'type': r.get('event_type') or '中性'}
        event_map[ev]['count'] += 1
        score = r['confidence'] if '看涨' in r['sentiment'] else -r['confidence'] if '看跌' in r['sentiment'] else 0
        event_map[ev]['scores'].append(score)

    events = []
    # 不再截断到 6 条：前端只展示前几条，但「其他」占比 / 覆盖率必须基于**全部**
    # 事件统计，否则分母被缩小会把其他类占比算错。
    for name, info in sorted(event_map.items(), key=lambda x: -x[1]['count']):
        avg_score = sum(info['scores']) / len(info['scores']) if info['scores'] else 0
        # 方向以**规则表**为准：规则说"中性"就是中性（如董事会决议、法律意见书），
        # 不要被个别标题的情感分带偏。只有规则没给方向（None）时才用情感均值兜底。
        etype = info.get('type') or ('中性')
        if etype not in ('利好', '利空', '资金面', '中性'):
            etype = '利好' if avg_score > 0.2 else '利空' if avg_score < -0.2 else '中性'
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
        'engine': backend,
        'model_status': {**get_backend().status(probe=(backend == 'model')), 'engine': backend},
    }