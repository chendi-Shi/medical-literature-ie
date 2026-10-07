"""Small single-operator session boundary for the local literature workbench.

Production traffic must terminate TLS at a trusted reverse proxy. This module
does not implement user provisioning, SSO, or a multi-tenant authorization model.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from urllib.parse import urlsplit


SESSION_COOKIE = "medical_session"
SESSION_SECONDS = 12 * 60 * 60


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"生产模式缺少环境变量 {name}")
    return value


def _host(value: str) -> str:
    parsed = urlsplit("//" + value)
    hostname = parsed.hostname
    if not hostname or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("MEDICAL_ALLOWED_HOSTS 包含无效主机名")
    hostname = hostname.lower().rstrip(".")
    if not re.fullmatch(r"[a-z0-9.-]+|\[[0-9a-f:]+\]", hostname):
        raise ValueError("MEDICAL_ALLOWED_HOSTS 包含无效主机名")
    return hostname


@dataclass(frozen=True)
class ProductionSecurity:
    username: str
    password: str
    secret: bytes
    allowed_hosts: frozenset[str]
    public_origin: str

    @classmethod
    def from_environment(cls) -> "ProductionSecurity":
        username = _required("MEDICAL_APP_USER")
        password = _required("MEDICAL_APP_PASSWORD")
        secret = _required("MEDICAL_SESSION_SECRET").encode("utf-8")
        if not re.fullmatch(r"[A-Za-z0-9_.@+-]{1,128}", username):
            raise ValueError("MEDICAL_APP_USER 格式无效")
        if len(password) < 20 or len(password) > 1024:
            raise ValueError("MEDICAL_APP_PASSWORD 必须为 20–1024 个字符")
        if len(secret) < 32:
            raise ValueError("MEDICAL_SESSION_SECRET 至少需要 32 个字节")
        allowed_hosts = frozenset(_host(x.strip()) for x in _required("MEDICAL_ALLOWED_HOSTS").split(",") if x.strip())
        if not allowed_hosts:
            raise ValueError("MEDICAL_ALLOWED_HOSTS 不能为空")
        public_origin = _required("MEDICAL_PUBLIC_ORIGIN")
        parsed = urlsplit(public_origin)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query
                or parsed.fragment or parsed.username or parsed.password
                or _host(parsed.netloc) not in allowed_hosts):
            raise ValueError("MEDICAL_PUBLIC_ORIGIN 必须是允许主机上的 HTTPS origin")
        return cls(username, password, secret, allowed_hosts, public_origin.rstrip("/"))

    def issue(self) -> str:
        expires = int(time.time()) + SESSION_SECONDS
        payload = f"{self.username}\t{expires}\t{secrets.token_urlsafe(18)}".encode()
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=")
        signature = hmac.new(self.secret, encoded, hashlib.sha256).hexdigest().encode()
        return (encoded + b"." + signature).decode("ascii")

    def identity(self, token: str | None) -> str | None:
        if not token or len(token) > 2048 or token.count(".") != 1:
            return None
        encoded, signature = token.encode().split(b".", 1)
        expected = hmac.new(self.secret, encoded, hashlib.sha256).hexdigest().encode()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            payload = base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4)).decode()
            username, expires, nonce = payload.split("\t")
            if not nonce or int(expires) < int(time.time()):
                return None
        except (ValueError, UnicodeError):
            return None
        return username if hmac.compare_digest(username, self.username) else None

    def password_matches(self, username: str, password: str) -> bool:
        user_ok = hmac.compare_digest(username.encode("utf-8"), self.username.encode("utf-8"))
        password_ok = hmac.compare_digest(password.encode("utf-8"), self.password.encode("utf-8"))
        return user_ok and password_ok


class LoginThrottle:
    """Bounded per-client failure windows; production remains single-process."""

    def __init__(self, limit: int = 5, window_seconds: int = 900, max_clients: int = 4096):
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self.failures: OrderedDict[str, deque[float]] = OrderedDict()
        self.lock = threading.Lock()

    def retry_after(self, client: str) -> int:
        now = time.monotonic()
        with self.lock:
            recent = self.failures.get(client, deque())
            while recent and now - recent[0] >= self.window_seconds:
                recent.popleft()
            if not recent:
                self.failures.pop(client, None)
                return 0
            self.failures[client] = recent
            self.failures.move_to_end(client)
            if len(self.failures) > self.max_clients:
                self.failures.popitem(last=False)
            return max(0, self.window_seconds - int(now - recent[0])) if len(recent) >= self.limit else 0

    def record_failure(self, client: str) -> int:
        now = time.monotonic()
        with self.lock:
            recent = self.failures.setdefault(client, deque())
            while recent and now - recent[0] >= self.window_seconds:
                recent.popleft()
            recent.append(now)
            self.failures.move_to_end(client)
            if len(self.failures) > self.max_clients:
                self.failures.popitem(last=False)
            return max(1, self.window_seconds - int(now - recent[0])) if len(recent) >= self.limit else 0

    def clear(self, client: str) -> None:
        with self.lock:
            self.failures.pop(client, None)
