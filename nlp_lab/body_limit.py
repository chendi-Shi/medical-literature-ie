"""Bound HTTP request bodies before application middleware can buffer them."""
from __future__ import annotations

from collections import deque
from typing import Any, Awaitable, Callable

from fastapi.responses import JSONResponse


class RequestBodyLimitMiddleware:
    """Buffer at most ``max_bytes`` and replay a complete, bounded ASGI body.

    Counting the ASGI stream also covers requests without Content-Length. The
    proxy should enforce its own limit too, but the application remains safe
    when called directly or when a client sends a chunked body.
    """

    def __init__(self, app: Callable[..., Awaitable[Any]], max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_lengths = [value for name, value in scope.get("headers", ())
                           if name.lower() == b"content-length"]
        if len(content_lengths) == 1:
            try:
                declared = int(content_lengths[0])
            except ValueError:
                declared = None
            if declared is not None and declared > self.max_bytes:
                await self._reject(scope, receive, send)
                return

        messages = deque()
        size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                messages.append(message)
                break
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay_receive():
            if messages:
                return messages.popleft()
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, scope, receive, send):
        response = JSONResponse({"detail": "请求超过 2MB 限制"}, status_code=413)
        if scope.get("path", "").startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
            "object-src 'none'; form-action 'self'"
        )
        await response(scope, receive, send)
