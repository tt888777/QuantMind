"""账号体系：注册 / 登录 / 会话 / 个人中心
========================================
设计上刻意不引入 Flask-Login 之类的额外依赖 —— 只用 Flask 自带的 session
和 Werkzeug 的密码哈希（scrypt），保证在一台全新电脑上装完 requirements.txt
就能直接跑通。

对外提供：
  - auth_bp               注册/登录/登出/资料/改密/登录记录 的蓝图
  - init_auth(app)        注册蓝图、注入密钥、安装全局登录守卫、播种演示账号
  - current_user()        取当前登录用户（dict，未登录返回 None）

会话只存一个 user id，用户资料每次从库里读，这样改昵称/改密码能立刻生效。
"""
from __future__ import annotations

import os
import re
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import quote

from flask import (Blueprint, g, jsonify, redirect, request,
                   session as flask_session, url_for)
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from db import Base, LoginLog, User, engine, get_session

# ---------------------------------------------------------------
# 常量
# ---------------------------------------------------------------
SESSION_KEY = "qm_uid"
SECRET_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".secret_key")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5-]{3,20}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

PASSWORD_MIN = 6
PASSWORD_MAX = 64
DISPLAY_MAX = 32

# 登录失败限流：同一用户名或同一 IP 5 分钟内失败 5 次即暂时拒绝
MAX_FAILS = 5
FAIL_WINDOW = 300
_FAILED: dict[str, list[float]] = {}

# 免登录即可访问的路径（登录页本身、登录/注册/探测会话的接口）
PUBLIC_PATHS = {
    "/login",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/me",
    "/api/auth/logout",     # 未登录时点退出也让它走通，不要报 401
    "/favicon.ico",
}


def auth_disabled() -> bool:
    """调试开关：QUANTMIND_AUTH_DISABLED=1 时跳过登录校验。"""
    return os.environ.get("QUANTMIND_AUTH_DISABLED", "0") == "1"


# ---------------------------------------------------------------
# 密钥
# ---------------------------------------------------------------
def load_secret_key() -> str:
    """优先用环境变量；否则在本机生成一个并落盘，避免重启后所有人掉线。

    注意：不要把密钥写进代码仓库 —— 该文件已在 .gitignore 中排除。
    """
    env = os.environ.get("QUANTMIND_SECRET_KEY", "").strip()
    if env:
        return env
    try:
        if os.path.isfile(SECRET_FILE):
            with open(SECRET_FILE, "r", encoding="utf-8") as f:
                key = f.read().strip()
            if len(key) >= 32:
                return key
        key = secrets.token_hex(32)
        with open(SECRET_FILE, "w", encoding="utf-8") as f:
            f.write(key)
        try:
            os.chmod(SECRET_FILE, 0o600)
        except OSError:
            pass
        print(f"[auth] 已生成本机会话密钥：{SECRET_FILE}")
        return key
    except OSError as e:
        # 磁盘不可写时退化为进程内随机值：重启会掉线，但功能不中断
        print(f"[auth] 会话密钥落盘失败（{e}），改用进程内临时密钥")
        return secrets.token_hex(32)


# ---------------------------------------------------------------
# 建表 / 演示账号
# ---------------------------------------------------------------
def ensure_schema() -> bool:
    """创建账号相关表（create_all 对已存在的表是空操作）。"""
    try:
        Base.metadata.create_all(engine)
        return True
    except Exception as e:  # 数据库没起来时不至于让整个应用 import 失败
        print(f"[auth] 建表失败：{type(e).__name__}: {e}")
        print("       若是 MySQL，请先执行 quantmind_schema.sql；"
              "本地演示可用 start_server_local.bat（SQLite）")
        return False


def ensure_demo_user() -> None:
    """首次运行时播种一个演示账号，方便直接体验登录流程。"""
    if os.environ.get("QUANTMIND_SEED_DEMO", "1") != "1":
        return
    s = get_session()
    try:
        if s.query(User).filter_by(username="demo").first():
            return
        s.add(User(
            username="demo",
            email="demo@quantmind.local",
            password_hash=generate_password_hash("demo123456"),
            display_name="演示研究员",
            role="admin",
            is_active=True,
            created_at=datetime.now(),
        ))
        s.commit()
        print("[auth] 已创建演示账号：demo / demo123456")
    except Exception as e:
        s.rollback()
        print(f"[auth] 演示账号创建失败：{type(e).__name__}: {e}")
    finally:
        s.close()


# ---------------------------------------------------------------
# 工具
# ---------------------------------------------------------------
def client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()[:64]
    return (request.remote_addr or "")[:64]


def _ua() -> str:
    return (request.headers.get("User-Agent") or "")[:255]


def _log(user_id, username, action, success=True) -> None:
    s = get_session()
    try:
        s.add(LoginLog(user_id=user_id, username=username, action=action,
                       success=success, ip=client_ip(), user_agent=_ua(),
                       created_at=datetime.now()))
        s.commit()
    except Exception:
        s.rollback()
    finally:
        s.close()


def _fmt(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else ""


def user_public(u: User) -> dict:
    """把 ORM 对象转成安全的字典 —— 绝不把 password_hash 带出去。"""
    name = (u.display_name or u.username or "?")
    return {
        "id": u.id,
        "username": u.username,
        "email": u.email or "",
        "display_name": name,
        "avatar": name[:1].upper(),
        "role": u.role or "user",
        "created_at": _fmt(u.created_at),
        "last_login_at": _fmt(u.last_login_at),
        "last_login_ip": u.last_login_ip or "",
    }


def current_user():
    """当前登录用户（dict）；未登录 / 已禁用返回 None。"""
    uid = flask_session.get(SESSION_KEY)
    if not uid:
        return None
    s = get_session()
    try:
        u = s.get(User, uid)
        if u is None or not u.is_active:
            flask_session.pop(SESSION_KEY, None)
            return None
        return user_public(u)
    except Exception:
        return None
    finally:
        s.close()


def login_required(fn):
    """接口用：未登录返回 401 JSON（前端据此跳登录页）。

    QUANTMIND_AUTH_DISABLED=1（调试模式）时注入一个匿名用户，
    让被装饰的接口仍能取到 g.qm_user 而不至于 500。
    """
    @wraps(fn)
    def _wrap(*a, **kw):
        if auth_disabled():
            g.qm_user = {"id": 0, "username": "anonymous", "email": "",
                         "display_name": "调试模式", "avatar": "?", "role": "user",
                         "created_at": "", "last_login_at": "", "last_login_ip": ""}
            return fn(*a, **kw)
        u = current_user()
        if u is None:
            return jsonify({"ok": False, "code": "UNAUTHORIZED",
                            "msg": "登录已过期，请重新登录"}), 401
        g.qm_user = u
        return fn(*a, **kw)
    return _wrap


def admin_required(fn):
    """接口用：在 login_required 基础上要求 role == 'admin'，否则 403 JSON。

    用于「修改全局配置」这类会影响到所有账号的操作（例如保存大模型 API Key）。
    """
    @login_required
    @wraps(fn)
    def _wrap(*a, **kw):
        u = getattr(g, "qm_user", None) or {}
        if (u.get("role") or "user") != "admin":
            return jsonify({"ok": False, "code": "FORBIDDEN",
                            "msg": "该操作仅管理员可用"}), 403
        return fn(*a, **kw)
    return _wrap


def _fail_key(username: str) -> str:
    return f"{username.lower()}|{client_ip()}"

def _too_many(key: str) -> bool:
    now = time.time()
    if len(_FAILED) > 2000:          # 简单清理，避免长期运行后无限增长
        for k in [k for k, v in _FAILED.items() if not v or now - v[-1] > FAIL_WINDOW]:
            _FAILED.pop(k, None)
    arr = [t for t in _FAILED.get(key, []) if now - t < FAIL_WINDOW]
    _FAILED[key] = arr
    return len(arr) >= MAX_FAILS


def _mark_fail(key: str) -> None:
    _FAILED.setdefault(key, []).append(time.time())


def _clear_fail(key: str) -> None:
    _FAILED.pop(key, None)


def _err(msg: str, code: str = "BAD_REQUEST", status: int = 400):
    return jsonify({"ok": False, "code": code, "msg": msg}), status


# ---------------------------------------------------------------
# 蓝图
# ---------------------------------------------------------------
auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/api/auth/me", methods=["GET"])
def api_me():
    """探测会话状态。未登录也返回 200（user=null），方便登录页直接调用。"""
    u = current_user()
    return jsonify({"ok": True, "user": u, "auth_required": not auth_disabled()})


@auth_bp.route("/api/auth/register", methods=["POST"])
def api_register():
    d = request.get_json(silent=True) or {}
    username = str(d.get("username", "")).strip()
    email = str(d.get("email", "")).strip().lower()
    password = str(d.get("password", ""))
    password2 = d.get("password2")
    display_name = str(d.get("display_name", "")).strip()

    if not USERNAME_RE.match(username):
        return _err("用户名需为 3~20 位的字母、数字、下划线、短横线或中文", "BAD_USERNAME")
    if len(password) < PASSWORD_MIN:
        return _err(f"密码至少 {PASSWORD_MIN} 位", "BAD_PASSWORD")
    if len(password) > PASSWORD_MAX:
        return _err(f"密码不能超过 {PASSWORD_MAX} 位", "BAD_PASSWORD")
    if password2 is not None and str(password2) != password:
        return _err("两次输入的密码不一致", "PASSWORD_MISMATCH")
    if email and not EMAIL_RE.match(email):
        return _err("邮箱格式不正确", "BAD_EMAIL")
    if display_name and len(display_name) > DISPLAY_MAX:
        return _err(f"昵称不能超过 {DISPLAY_MAX} 个字符", "BAD_DISPLAY")

    s = get_session()
    try:
        if s.query(User).filter_by(username=username).first():
            return _err("该用户名已被注册", "USERNAME_TAKEN", 409)
        if email and s.query(User).filter_by(email=email).first():
            return _err("该邮箱已被注册", "EMAIL_TAKEN", 409)

        u = User(
            username=username,
            email=email or None,
            password_hash=generate_password_hash(password),
            display_name=display_name or username,
            role="user",
            is_active=True,
            created_at=datetime.now(),
            last_login_at=datetime.now(),
            last_login_ip=client_ip(),
        )
        s.add(u)
        s.commit()
        uid, pub = u.id, user_public(u)
    except IntegrityError:
        # 并发注册同名时唯一索引会兜住，转成友好提示
        s.rollback()
        return _err("该用户名或邮箱已被注册", "USERNAME_TAKEN", 409)
    except Exception as e:
        s.rollback()
        return _err(f"注册失败：{type(e).__name__}", "DB_ERROR", 500)
    finally:
        s.close()

    flask_session.permanent = True
    flask_session[SESSION_KEY] = uid
    _log(uid, username, "register")
    return jsonify({"ok": True, "user": pub, "msg": "注册成功，已自动登录"})


@auth_bp.route("/api/auth/login", methods=["POST"])
def api_login():
    d = request.get_json(silent=True) or {}
    account = str(d.get("username", "")).strip()
    password = str(d.get("password", ""))
    remember = bool(d.get("remember", True))

    if not account or not password:
        return _err("请输入用户名和密码", "MISSING_FIELDS")

    key = _fail_key(account)
    if _too_many(key):
        return _err("尝试次数过多，请 5 分钟后再试", "TOO_MANY_ATTEMPTS", 429)

    s = get_session()
    try:
        u = s.query(User).filter_by(username=account).first()
        if u is None and "@" in account:
            u = s.query(User).filter_by(email=account.lower()).first()
        if u is None or not check_password_hash(u.password_hash, password):
            _mark_fail(key)
            _log(u.id if u else None, account, "login", success=False)
            return _err("用户名或密码错误", "BAD_CREDENTIALS", 401)
        if not u.is_active:
            _mark_fail(key)
            return _err("该账号已被停用，请联系管理员", "DISABLED", 403)

        u.last_login_at = datetime.now()
        u.last_login_ip = client_ip()
        s.commit()
        uid, pub = u.id, user_public(u)
    finally:
        s.close()

    _clear_fail(key)
    flask_session.permanent = remember
    flask_session[SESSION_KEY] = uid
    _log(uid, pub["username"], "login")
    return jsonify({"ok": True, "user": pub, "msg": f"欢迎回来，{pub['display_name']}"})


@auth_bp.route("/api/auth/logout", methods=["POST", "GET"])
def api_logout():
    uid = flask_session.get(SESSION_KEY)
    name = ""
    if uid:
        u = current_user()
        name = u["username"] if u else ""
        _log(uid, name, "logout")
    flask_session.clear()
    if request.method == "GET":
        return redirect(url_for("login_page"))
    return jsonify({"ok": True, "msg": "已退出登录"})


@auth_bp.route("/api/auth/profile", methods=["POST"])
@login_required
def api_profile():
    d = request.get_json(silent=True) or {}
    display_name = str(d.get("display_name", "")).strip()
    email = str(d.get("email", "")).strip().lower()

    if not display_name:
        return _err("昵称不能为空", "BAD_DISPLAY")
    if len(display_name) > DISPLAY_MAX:
        return _err(f"昵称不能超过 {DISPLAY_MAX} 个字符", "BAD_DISPLAY")
    if email and not EMAIL_RE.match(email):
        return _err("邮箱格式不正确", "BAD_EMAIL")

    s = get_session()
    try:
        u = s.get(User, g.qm_user["id"])
        if u is None:
            return _err("用户不存在", "NOT_FOUND", 404)
        if email and email != (u.email or ""):
            if s.query(User).filter(User.email == email, User.id != u.id).first():
                return _err("该邮箱已被其它账号使用", "EMAIL_TAKEN", 409)
        u.display_name = display_name
        u.email = email or None
        s.commit()
        pub = user_public(u)
    except Exception as e:
        s.rollback()
        return _err(f"保存失败：{type(e).__name__}", "DB_ERROR", 500)
    finally:
        s.close()

    _log(g.qm_user["id"], g.qm_user["username"], "profile")
    return jsonify({"ok": True, "user": pub, "msg": "个人资料已更新"})


@auth_bp.route("/api/auth/password", methods=["POST"])
@login_required
def api_password():
    d = request.get_json(silent=True) or {}
    old = str(d.get("old_password", ""))
    new = str(d.get("new_password", ""))
    new2 = d.get("new_password2")

    if len(new) < PASSWORD_MIN:
        return _err(f"新密码至少 {PASSWORD_MIN} 位", "BAD_PASSWORD")
    if len(new) > PASSWORD_MAX:
        return _err(f"新密码不能超过 {PASSWORD_MAX} 位", "BAD_PASSWORD")
    if new2 is not None and str(new2) != new:
        return _err("两次输入的新密码不一致", "PASSWORD_MISMATCH")

    s = get_session()
    try:
        u = s.get(User, g.qm_user["id"])
        if u is None:
            return _err("用户不存在", "NOT_FOUND", 404)
        if not check_password_hash(u.password_hash, old):
            return _err("当前密码不正确", "BAD_CREDENTIALS", 400)
        if check_password_hash(u.password_hash, new):
            return _err("新密码不能与当前密码相同", "SAME_PASSWORD")
        u.password_hash = generate_password_hash(new)
        s.commit()
    except Exception as e:
        s.rollback()
        return _err(f"修改失败：{type(e).__name__}", "DB_ERROR", 500)
    finally:
        s.close()

    _log(g.qm_user["id"], g.qm_user["username"], "password")
    return jsonify({"ok": True, "msg": "密码已修改，下次请用新密码登录"})


@auth_bp.route("/api/auth/logs", methods=["GET"])
@login_required
def api_logs():
    limit = min(int(request.args.get("limit", 12)), 50)
    label = {"register": "注册", "login": "登录", "logout": "退出",
             "password": "修改密码", "profile": "更新资料"}
    s = get_session()
    try:
        rows = (s.query(LoginLog)
                 .filter_by(user_id=g.qm_user["id"])
                 .order_by(LoginLog.id.desc())
                 .limit(limit).all())
        items = [{
            "time": _fmt(r.created_at),
            "action": label.get(r.action, r.action),
            "success": bool(r.success),
            "ip": r.ip or "-",
            "ua": (r.user_agent or "-")[:60],
        } for r in rows]
        return jsonify({"ok": True, "items": items, "count": len(items)})
    finally:
        s.close()


# ---------------------------------------------------------------
# 接线
# ---------------------------------------------------------------
def init_auth(app) -> None:
    app.secret_key = load_secret_key()
    app.config.update(
        SESSION_COOKIE_NAME="qm_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    )
    app.register_blueprint(auth_bp)

    if ensure_schema():
        ensure_demo_user()
    if auth_disabled():
        print("[auth] ⚠️ QUANTMIND_AUTH_DISABLED=1，已跳过登录校验（仅调试用）")

    @app.before_request
    def _guard():
        if auth_disabled():
            return None
        p = request.path
        if p in PUBLIC_PATHS or p.startswith("/static/"):
            return None
        if current_user() is not None:
            return None
        if p.startswith("/api/"):
            return jsonify({"ok": False, "code": "UNAUTHORIZED",
                            "msg": "请先登录"}), 401
        nxt = p if not request.query_string else f"{p}?{request.query_string.decode('utf-8', 'ignore')}"
        return redirect(f"/login?next={quote(nxt, safe='')}")
