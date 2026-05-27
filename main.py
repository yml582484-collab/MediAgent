"""
MediAgent 智慧医疗助手 - FastAPI Web Application
Main entry point for the Medical AI API Server

安全加固、可观测性、SSE 流式支持
"""
import asyncio
import collections
import json
import os
import shutil
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables from .env file (must be before other imports)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path.name}")
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from uvicorn import Config, Server

from src.agent.core import Agent
from src.utils.config import get_config
from src.utils.logger import setup_logger, get_logger


# ==================== Configuration ====================

AUTH_API_KEYS: Optional[List[str]] = None
_auth_keys_str = os.getenv("AUTH_API_KEYS", "")
if _auth_keys_str.strip():
    AUTH_API_KEYS = [k.strip() for k in _auth_keys_str.split(",") if k.strip()]

RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "10"))
MAX_REQUEST_BODY_SIZE: int = int(os.getenv("MAX_REQUEST_BODY_SIZE", str(1024 * 1024)))  # 1MB
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "120"))  # seconds

ALLOWED_ORIGINS_OVERRIDE: Optional[List[str]] = None
_allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
if _allowed_origins_str.strip():
    ALLOWED_ORIGINS_OVERRIDE = [o.strip() for o in _allowed_origins_str.split(",") if o.strip()]


# ==================== Pydantic Models ====================


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    use_react: bool = Field(default=True, description="Use ReAct planning for complex tasks")
    stream: bool = Field(default=False, description="Enable streaming response")
    agent_mode: str = Field(default="react", description="Agent mode: direct, react, plan_and_execute")


class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str
    response: str
    success: bool
    reasoning_trace: Optional[list] = None
    token_usage: dict = {}
    metadata: dict = {}
    # 风险信息（从 metadata 中提取，便于前端直接访问）
    risk_level: Optional[str] = None
    risk_warning: Optional[str] = None
    source: Optional[str] = None
    cache_hit: Optional[bool] = None


class StatusResponse(BaseModel):
    """Agent status response"""
    status: dict
    sessions: list = []
    tools: list = []


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


class MedicalAnalyzeRequest(BaseModel):
    """Medical symptom analysis request model"""
    symptoms: str = Field(..., min_length=1, max_length=5000, description="症状描述，如：头痛、发热、咳嗽等")
    patient_age: Optional[int] = Field(default=None, ge=0, le=150, description="患者年龄")
    patient_gender: Optional[str] = Field(default=None, description="患者性别（男/女）")
    patient_history: Optional[str] = Field(default=None, max_length=3000, description="既往病史")
    duration: Optional[str] = Field(default=None, description="症状持续时间，如：3天、一周等")


class MedicalDisclaimerResponse(BaseModel):
    """Medical disclaimer response model"""
    disclaimer: str
    version: str
    last_updated: str


class SessionMemoryRequest(BaseModel):
    """Session memory request model"""
    session_id: str = Field(..., description="会话 ID")


# ==================== Rate Limiter ====================


class RateLimiter:
    """基于 IP 的内存速率限制器（令牌桶算法）"""

    def __init__(self, per_minute: int = 30, burst: int = 10):
        self.per_minute = per_minute
        self.burst = burst
        self._requests: Dict[str, collections.deque] = defaultdict(collections.deque)
        self._lock = asyncio.Lock()

    def _cleanup(self, ip: str, now: float):
        """清理过期的请求记录"""
        window = 60.0  # 1 分钟窗口
        while self._requests[ip] and now - self._requests[ip][0] > window:
            self._requests[ip].popleft()
        if not self._requests[ip]:
            del self._requests[ip]

    async def is_allowed(self, ip: str) -> bool:
        """检查是否允许请求"""
        async with self._lock:
            now = time.monotonic()
            self._cleanup(ip, now)
            current_count = len(self._requests[ip])
            if current_count >= self.per_minute:
                return False
            self._requests[ip].append(now)
            return True

    def get_remaining(self, ip: str) -> int:
        """获取剩余配额"""
        now = time.monotonic()
        self._cleanup(ip, now)
        return max(0, self.per_minute - len(self._requests[ip]))


rate_limiter = RateLimiter(per_minute=RATE_LIMIT_PER_MINUTE, burst=RATE_LIMIT_BURST)


# ==================== Metrics & Audit ====================


class MetricsCollector:
    """应用指标收集器"""

    def __init__(self):
        self.total_requests: int = 0
        self.success_count: int = 0
        self.failure_count: int = 0
        self.total_response_time: float = 0.0
        self.response_times: List[float] = []
        self.token_usage_totals: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.tool_call_counts: Dict[str, int] = defaultdict(int)
        self.active_sessions_count: int = 0
        self._lock = asyncio.Lock()

    async def record_request(self, success: bool, duration: float):
        async with self._lock:
            self.total_requests += 1
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            self.total_response_time += duration
            self.response_times.append(duration)
            # 只保留最近 1000 条响应时间记录
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-500:]

    async def record_token_usage(self, token_usage: dict):
        async with self._lock:
            if isinstance(token_usage, dict):
                self.token_usage_totals["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                self.token_usage_totals["completion_tokens"] += token_usage.get("completion_tokens", 0)
                self.token_usage_totals["total_tokens"] += token_usage.get("total_tokens", 0)

    async def record_tool_call(self, tool_name: str):
        async with self._lock:
            self.tool_call_counts[tool_name] += 1

    async def update_active_sessions(self, count: int):
        async with self._lock:
            self.active_sessions_count = count

    def get_metrics(self) -> dict:
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "average_response_time_ms": round(avg_response_time * 1000, 2),
            "token_usage": dict(self.token_usage_totals),
            "active_sessions_count": self.active_sessions_count,
            "tool_call_counts": dict(self.tool_call_counts),
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
            "rate_limit_burst": RATE_LIMIT_BURST,
        }


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._entries: collections.deque = collections.deque(maxlen=max_entries)
        self._lock = asyncio.Lock()

    async def log(
        self,
        timestamp: str,
        ip: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_id: str,
        user_agent: str = "",
        error: str = "",
    ):
        entry = {
            "timestamp": timestamp,
            "ip": ip,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "request_id": request_id,
            "user_agent": user_agent[:200] if user_agent else "",
            "error": error[:500] if error else "",
        }
        async with self._lock:
            self._entries.append(entry)

    def get_entries(self, limit: int = 100) -> list:
        return list(self._entries)[-limit:]


metrics_collector = MetricsCollector()
audit_logger = AuditLogger(max_entries=100)


# ==================== Global State ====================

agent_instance: Optional[Agent] = None
logger = get_logger(__name__)


# ==================== Lifespan ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler

    Manages startup and shutdown events.
    """
    global agent_instance

    # Startup
    logger.info("🚀 Starting MediAgent 智慧医疗助手 API Server...")

    try:
        # Initialize the agent
        agent_instance = Agent(auto_initialize=False)
        await agent_instance.initialize()

        logger.info("✅ Agent ready to serve requests!")

        yield  # Application is running

    except Exception as e:
        logger.error(f"❌ Failed to start agent: {e}")
        raise RuntimeError(f"Agent initialization failed: {e}")

    finally:
        # Shutdown
        logger.info("🛑 Shutting down agent...")

        if agent_instance:
            await agent_instance.close()

        logger.info("👋 Agent shut down complete")


# ==================== Create FastAPI Application ====================

app = FastAPI(
    title="MediAgent 智慧医疗助手 API",
    description="""
    🏥 MediAgent 智慧医疗健康助手，由 DeepSeek 驱动，具备以下核心能力：

    - **医疗知识记忆系统**: 短期和长期记忆，用于维护患者病史和医疗上下文
    - **医疗专用工具（症状分析、药品查询、知识库等）**: 内置医疗领域工具（症状分析器、药品查询、医疗知识库等）
    - **医疗推理循环**: 基于 ReAct 的高级医疗推理，支持多步骤诊断分析
    - **流式响应**: 实时逐 Token 输出，提升交互体验

    ## Quick Start

    发送 POST 请求到 `/api/chat` 进行医疗咨询，或使用 `/api/medical/analyze` 进行症状分析。
    """,
    version="2.1.0-medical",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ==================== Middleware ====================


# CORS middleware - restrict to configurable origins
server_config = get_config().config.server
cors_origins = ALLOWED_ORIGINS_OVERRIDE if ALLOWED_ORIGINS_OVERRIDE else server_config.cors_origins
if not cors_origins or cors_origins == ["*"]:
    cors_origins = ["*"]
    logger.warning("⚠️ CORS 配置为允许所有来源，生产环境请设置 ALLOWED_ORIGINS 环境变量")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_audit_middleware(request: Request, call_next):
    """
    安全与审计中间件：
    - API Key 认证
    - 请求 ID 注入
    - 请求体大小限制
    - 速率限制
    - 审计日志
    - 请求耗时追踪
    """
    start_time = time.monotonic()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # 获取客户端 IP
    client_ip = request.client.host if request.client else "unknown"

    # API Key 认证
    if AUTH_API_KEYS:
        api_key = request.headers.get("X-API-Key", "")
        if api_key not in AUTH_API_KEYS:
            duration_ms = (time.monotonic() - start_time) * 1000
            await audit_logger.log(
                timestamp=datetime.now().isoformat(),
                ip=client_ip,
                method=request.method,
                path=request.url.path,
                status_code=401,
                duration_ms=duration_ms,
                request_id=request_id,
                user_agent=request.headers.get("User-Agent", ""),
                error="无效的 API Key",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "未授权", "detail": "无效或缺失 X-API-Key 头"},
            )

    # 速率限制（排除健康检查和文档路径）
    skip_rate_limit_paths = {"/api/health", "/docs", "/redoc", "/openapi.json"}
    if request.url.path not in skip_rate_limit_paths:
        allowed = await rate_limiter.is_allowed(client_ip)
        if not allowed:
            duration_ms = (time.monotonic() - start_time) * 1000
            await audit_logger.log(
                timestamp=datetime.now().isoformat(),
                ip=client_ip,
                method=request.method,
                path=request.url.path,
                status_code=429,
                duration_ms=duration_ms,
                request_id=request_id,
                user_agent=request.headers.get("User-Agent", ""),
                error="速率限制",
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "请求过于频繁",
                    "detail": f"已超过速率限制 ({RATE_LIMIT_PER_MINUTE} 次/分钟)",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )

    # 请求体大小限制
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
        duration_ms = (time.monotonic() - start_time) * 1000
        await audit_logger.log(
            timestamp=datetime.now().isoformat(),
            ip=client_ip,
            method=request.method,
            path=request.url.path,
            status_code=413,
            duration_ms=duration_ms,
            request_id=request_id,
            user_agent=request.headers.get("User-Agent", ""),
            error=f"请求体过大 ({content_length} > {MAX_REQUEST_BODY_SIZE})",
        )
        return JSONResponse(
            status_code=413,
            content={
                "error": "请求体过大",
                "detail": f"最大允许 {MAX_REQUEST_BODY_SIZE // 1024} KB",
            },
        )

    # 处理请求
    try:
        response: Response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.error(f"请求处理异常: {exc}", exc_info=True)
        await audit_logger.log(
            timestamp=datetime.now().isoformat(),
            ip=client_ip,
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
            request_id=request_id,
            user_agent=request.headers.get("User-Agent", ""),
            error=str(exc)[:500],
        )
        return JSONResponse(
            status_code=500,
            content={"error": "内部服务器错误", "detail": "请求处理过程中发生异常"},
        )

    # 注入响应头
    duration_ms = (time.monotonic() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

    # 记录审计日志
    success = 200 <= response.status_code < 400
    await audit_logger.log(
        timestamp=datetime.now().isoformat(),
        ip=client_ip,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
        request_id=request_id,
        user_agent=request.headers.get("User-Agent", ""),
    )

    # 记录指标
    await metrics_collector.record_request(success=success, duration=duration_ms / 1000.0)

    return response


# ==================== Mount Static Files ====================

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")
    print(f"✅ Static files mounted from {frontend_path / 'static'}")

    @app.get("/", tags=["Root"])
    async def root():
        """Serve b1t-AI frontend"""
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {
            "name": "MediAgent",
            "version": "2.1.0-medical",
            "status": "running" if agent_instance else "initializing",
            "docs": "/docs",
        }


# ==================== API Routes ====================


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    Send a message to the agent and get a response

    - **message**: Your message or question (required)
    - **session_id**: Existing session ID (optional, creates new if not provided)
    - **use_react**: Enable ReAct planning for complex tasks (default: true)
    - **stream**: Enable streaming (use /api/chat/stream instead for true SSE)
    - **agent_mode**: Agent mode - direct, react, or plan_and_execute (default: react)
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # 根据 agent_mode 选择处理方式
        if request.agent_mode == "direct" or not request.use_react:
            result = await agent_instance.chat(
                message=request.message,
                session_id=request.session_id,
            )
        else:
            # 使用 process() 并指定 mode
            result = await agent_instance.process(
                input_text=request.message,
                session_id=request.session_id,
                mode=request.agent_mode,
            )

        # 记录 token 使用量
        if hasattr(result, "token_usage") and result.token_usage:
            await metrics_collector.record_token_usage(result.token_usage)

        # 更新活跃会话数
        if agent_instance and hasattr(agent_instance, "_sessions"):
            await metrics_collector.update_active_sessions(len(agent_instance._sessions))

        return ChatResponse(
            session_id=result.session_id,
            response=result.response,
            success=result.success,
            reasoning_trace=result.reasoning_trace,
            token_usage=result.token_usage,
            metadata=result.metadata,
            # 从 metadata 提取风险信息
            risk_level=result.metadata.get("risk_level"),
            risk_warning=result.metadata.get("risk_warning"),
            source=result.metadata.get("source"),
            cache_hit=result.metadata.get("cache_hit"),
        )

    except asyncio.TimeoutError:
        logger.error(f"Chat endpoint timeout: request_id={getattr(req.state, 'request_id', 'N/A')}")
        raise HTTPException(status_code=504, detail="请求超时，请稍后重试")
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(request: ChatRequest, req: Request):
    """
    Stream chat response using Server-Sent Events (SSE)

    支持 ReAct 模式流式输出，事件类型包括：
    - `thought`: 推理思考过程
    - `action`: 工具调用动作
    - `observation`: 工具调用结果
    - `final`: 最终回复
    - `error`: 错误信息

    Use this for better user experience in chat interfaces.
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))

    async def generate():
        try:
            if request.agent_mode == "direct" or not request.use_react:
                # 直接模式 - 简单流式输出
                async for chunk in agent_instance.chat_stream(
                    message=request.message,
                    session_id=request.session_id,
                ):
                    data = json.dumps({"content": chunk, "request_id": request_id}, ensure_ascii=False)
                    yield f"event: final\ndata: {data}\n\n"

                yield "event: done\ndata: {}\n\n"
            else:
                # ReAct 模式 - 结构化 SSE 事件流
                # 发送思考事件
                thought_data = json.dumps({
                    "content": "正在分析您的问题...",
                    "request_id": request_id,
                }, ensure_ascii=False)
                yield f"event: thought\ndata: {thought_data}\n\n"

                # 执行 ReAct 处理
                result = await asyncio.wait_for(
                    agent_instance.process(
                        input_text=request.message,
                        session_id=request.session_id,
                        mode=request.agent_mode,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )

                # 发送推理轨迹
                if result.reasoning_trace:
              