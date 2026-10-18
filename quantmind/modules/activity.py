"""全站操作活动日志：落库 + 读取，供主页「最近活动」展示。

设计原则：**记日志绝不影响主流程**。所有写操作都吞异常，
最坏情况是少一条活动记录，绝不会让回测/采集/导出失败。
"""
from datetime import datetime, timedelta

from db import get_session, ActivityLog

# 与前端 badge 配色一一对应
STATUS_LABEL = {"ok": "完成", "fail": "失败", "info": "信息", "warn": "警告"}


def log(module, event, status="ok", stock_code=None, user_id=None):
    """写一条活动记录。返回是否写入成功，调用方不需要关心。"""
    s = get_session()
    try:
        s.add(ActivityLog(
            module=str(module)[:32],
            event=str(event)[:500],
            status=status if status in STATUS_LABEL else "info",
            stock_code=(str(stock_code)[:16] if stock_code else None),
            user_id=user_id,
            created_at=datetime.now(),
        ))
        s.commit()
        return True
    except Exception as e:                      # 日志失败不能冒泡
        s.rollback()
        print(f"[activity] 写入失败（已忽略）：{type(e).__name__}: {e}")
        return False
    finally:
        s.close()


def recent(limit=20, module=None, days=30):
    """最近活动，按时间倒序。默认只看 30 天内，避免历史无限堆积。"""
    s = get_session()
    try:
        q = s.query(ActivityLog)
        if days:
            q = q.filter(ActivityLog.created_at >= datetime.now() - timedelta(days=days))
        if module:
            q = q.filter(ActivityLog.module == module)
        rows = q.order_by(ActivityLog.created_at.desc(),
                          ActivityLog.id.desc()).limit(int(limit)).all()
        return [{
            "id": r.id,
            "time": r.created_at.strftime("%H:%M:%S") if r.created_at else "",
            "date": r.created_at.strftime("%Y-%m-%d") if r.created_at else "",
            "datetime": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
            "module": r.module,
            "event": r.event,
            "status": r.status,
            "status_label": STATUS_LABEL.get(r.status, "信息"),
            "stock_code": r.stock_code,
        } for r in rows]
    finally:
        s.close()


def summary(days=30):
    """按模块汇总条数，主页概览可用。"""
    s = get_session()
    try:
        since = datetime.now() - timedelta(days=days)
        rows = (s.query(ActivityLog.module, ActivityLog.status)
                .filter(ActivityLog.created_at >= since).all())
        agg = {}
        for mod, st in rows:
            d = agg.setdefault(mod, {"module": mod, "total": 0, "ok": 0, "fail": 0})
            d["total"] += 1
            if st == "ok":
                d["ok"] += 1
            elif st == "fail":
                d["fail"] += 1
        return sorted(agg.values(), key=lambda x: -x["total"])
    finally:
        s.close()


def purge(days=180):
    """清理过期活动，避免表无限增长。CLI 可调用。"""
    s = get_session()
    try:
        n = (s.query(ActivityLog)
             .filter(ActivityLog.created_at < datetime.now() - timedelta(days=days))
             .delete(synchronize_session=False))
        s.commit()
        return n
    except Exception:
        s.rollback()
        return 0
    finally:
        s.close()
