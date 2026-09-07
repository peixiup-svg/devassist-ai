"""FastAPI application factory."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from devassist import __version__
from devassist.config import Settings
from devassist.models import (
    DiagnoseRequest,
    DiagnoseResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    SearchHit,
    SearchRequest,
)
from devassist.service import DevAssistService


class FixedWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int, max_keys: int = 10_000) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            if now - self._last_sweep >= self.window_seconds:
                expired = [
                    client_key
                    for client_key, client_events in self._events.items()
                    if not client_events or client_events[-1] < cutoff
                ]
                for client_key in expired:
                    self._events.pop(client_key, None)
                self._last_sweep = now
            if key not in self._events and len(self._events) >= self.max_keys:
                return False
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.service = await run_in_threadpool(DevAssistService, resolved_settings)
        application.state.rate_limiter = FixedWindowRateLimiter(
            resolved_settings.rate_limit_requests,
            resolved_settings.rate_limit_window_seconds,
        )
        yield

    application = FastAPI(
        title="DevAssist AI",
        version=__version__,
        description="版本感知、引用可验证、低置信度拒答的 Python/AI 技术支持系统。",
        lifespan=lifespan,
    )

    def service_from(request: Request) -> DevAssistService:
        service = getattr(request.app.state, "service", None)
        if not isinstance(service, DevAssistService) or not service.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="index not ready"
            )
        return service

    def check_rate_limit(request: Request) -> None:
        # Forwarded headers are attacker-controlled unless a trusted proxy strips
        # them. The safe local default keys only on the ASGI peer address.
        key = request.client.host if request.client else "unknown"
        limiter: FixedWindowRateLimiter = request.app.state.rate_limiter
        if not limiter.allow(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded"
            )

    @application.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(resolved_settings.static_dir / "index.html")

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health(request: Request) -> HealthResponse:
        service = service_from(request)
        return HealthResponse(
            status="ok",
            documents=service.document_count,
            index_generation=service.store.generation,
            version=__version__,
        )

    @application.post("/v1/search", response_model=list[SearchHit], tags=["retrieval"])
    async def search(request: Request, payload: SearchRequest) -> list[SearchHit]:
        check_rate_limit(request)
        if len(payload.query) > resolved_settings.max_query_chars:
            raise HTTPException(status_code=413, detail="query is too large")
        return await run_in_threadpool(service_from(request).search, payload)

    @application.post("/v1/diagnose", response_model=DiagnoseResponse, tags=["agent"])
    async def diagnose(request: Request, payload: DiagnoseRequest) -> DiagnoseResponse:
        check_rate_limit(request)
        if len(payload.query) > resolved_settings.max_query_chars:
            raise HTTPException(status_code=413, detail="query is too large")
        return await run_in_threadpool(service_from(request).diagnose, payload)

    @application.post("/v1/feedback", response_model=FeedbackResponse, tags=["feedback"])
    async def feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
        check_rate_limit(request)
        return await run_in_threadpool(service_from(request).save_feedback, payload)

    @application.get("/v1/metrics", response_model=MetricsResponse, tags=["operations"])
    async def metrics(request: Request) -> MetricsResponse:
        return service_from(request).metrics.snapshot()

    @application.post("/admin/reindex", response_model=HealthResponse, tags=["operations"])
    async def reindex(
        request: Request,
        x_admin_token: str | None = Header(default=None),
    ) -> HealthResponse:
        check_rate_limit(request)
        configured = resolved_settings.admin_token
        if not configured:
            raise HTTPException(status_code=503, detail="admin endpoint is disabled")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, configured):
            raise HTTPException(status_code=401, detail="invalid admin token")
        service = service_from(request)
        await run_in_threadpool(service.reindex)
        return HealthResponse(
            status="ok",
            documents=service.document_count,
            index_generation=service.store.generation,
            version=__version__,
        )

    return application


app = create_app()
