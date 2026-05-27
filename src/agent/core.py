"""
Agent Core Engine - MediAgent 智慧医疗助手
Main orchestrator that integrates all components: LLM, Memory, Tools, and Planner
卫宁健康风格智慧医疗AI助手核心引擎

Major Rewrite:
- Per-session isolation with SessionContext
- Agent mode support (react, plan_and_execute, direct)
- Request ID tracking and audit logging
- Fixed chat_stream() buffering
- Fixed process() double message bug
- Long-term memory with try/except fallback
- process_stream() for SSE streaming
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .planner import ReActPlanner, PlanAndExecutePlanner, PlanExecutionResult, PlannerState
from ..llm.provider import DeepSeekProvider, Message, LLMResponse
from ..llm.prompts import PromptTemplates
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import LongTermMemory
from ..memory.working_memory import WorkingMemory
from ..tools.base import tool_registry, BaseTool
from ..utils.config import get_config, ConfigManager
from ..utils.logger import get_logger, log_execution_time
from ..utils.compliance import MedicalComplianceChecker

logger = get_logger(__name__)


class AgentMode(Enum):
    """Agent 运行模式"""
    REACT = "react"                  # ReAct 推理循环
    PLAN_AND_EXECUTE = "plan_and_execute"  # 先规划后执行
    DIRECT = "direct"                # 简单对话


@dataclass
class AgentResponse:
    """Agent 标准化响应"""
    session_id: str
    response: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: Optional[List[Dict]] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 API 响应）"""
        return {
            "session_id": self.session_id,
            "response": self.response,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "reasoning_trace": self.reasoning_trace,
            "token_usage": self.token_usage,
            "request_id": self.request_id,
        }


@dataclass
class SessionContext:
    """
    会话上下文 - 每个会话独立的记忆和状态

    实现会话隔离：每个 session 拥有独立的短期记忆和工作记忆，
    确保不同会话之间的数据不会互相干扰。
    """
    session_id: str
    short_memory: ShortTermMemory
    working_memory: WorkingMemory
    created_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    mode: AgentMode = AgentMode.REACT

    def update_activity(self) -> None:
        """更新会话活动时间"""
        self.message_count += 1
        self.last_activity = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id[:8] + "...",
            "created_at": self.created_at.isoformat(),
            "message_count": self.message_count,
            "last_activity": self.last_activity.isoformat(),
            "mode": self.mode.value,
            "memory_size": self.short_memory.size,
        }


@dataclass
class AuditLog:
    """审计日志条目"""
    request_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    mode: str = ""
    input_length: int = 0
    output_length: int = 0
    success: bool = True
    token_usage: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    is_emergency: bool = False
    error: Optional[str] = None


class Agent:
    """
    MediAgent 智慧医疗助手 - 核心引擎（卫宁健康风格）

    This is the main class that integrates all subsystems:
    - **LLM Provider**: DeepSeek API integration with streaming support
    - **Memory System**: Short-term (conversation), Long-term (vector DB), Working (task state)
    - **Tool System**: Extensible tool registry with built-in tools + medical tools
    - **ReAct Planner**: Reasoning loop for complex task execution
    - **Plan-and-Execute Planner**: Plan first, then execute step by step
    - **Compliance System**: Medical compliance checking and PHI sanitization

    Features:
    - Per-session isolation (each session has independent memory)
    - Multi-mode operation: ReAct, Plan-and-Execute, Direct chat
    - Request ID tracking for distributed tracing
    - Audit logging for all requests/responses
    - Multi-turn conversations with context management
    - Semantic memory retrieval and storage
    - Tool calling via DeepSeek native Function Calling
    - Streaming responses with SSE events
    - Session management
    - Medical compliance and safety checks
    - PHI (Protected Health Information) sanitization
    - Automatic medical disclaimer attachment

    Usage:
        # 初始化 agent
        agent = Agent()
        await agent.initialize()

        # 简单医疗对话
        response = await agent.chat("我最近头痛怎么办？")
        print(response.response)

        # ReAct 模式处理复杂任务
        response = await agent.process("我发热3天，咳嗽有痰，请帮我分析一下")
        print(response.response)
        print(response.reasoning_trace)

        # Plan-and-Execute 模式
        response = await agent.process(
            "帮我全面分析一下这个症状",
            mode="plan_and_execute"
        )

        # 流式响应
        async for chunk in agent.chat_stream("布洛芬有什么副作用？"):
            print(chunk, end="", flush=True)

        # 流式 ReAct
        async for event in agent.process_stream("分析一下我的症状"):
            print(event)

        # 清理
        await agent.close()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        auto_initialize: bool = False,
    ):
        """
        初始化 Agent

        Args:
            config_path: 配置文件路径（可选）
            auto_initialize: 是否自动初始化（默认 False，需手动调用 initialize()）
        """
        # 配置
        self.config_manager = get_config(config_path)
        self.config = self.config_manager.config

        # 组件占位符（在 initialize() 中初始化）
        self._llm: Optional[DeepSeekProvider] = None
        self._long_memory: Optional[LongTermMemory] = None

        # 医疗合规检查器
        compliance_mode = getattr(self.config.agent, 'compliance_mode', True)
        auto_disclaimer = getattr(self.config.agent, 'medical_disclaimer', True)
        self._compliance_checker = MedicalComplianceChecker(
            compliance_mode=compliance_mode,
            auto_disclaimer=auto_disclaimer,
        )

        # 会话管理（每个 session 独立的 SessionContext）
        self._sessions: Dict[str, SessionContext] = {}
        self._current_session_id: Optional[str] = None

        # 审计日志
        self._audit_logs: List[AuditLog] = []
        self._max_audit_logs: int = 1000

        # 状态追踪
        self._initialized: bool = False
        self._total_requests: int = 0
        self._start_time: datetime = datetime.now()

        logger.info("MediAgent 实例已创建")

    async def initialize(self) -> None:
        """
        初始化所有 Agent 组件

        必须在使用 Agent 之前调用此方法。
        """
        if self._initialized:
            logger.warning("Agent 已初始化，跳过重复初始化")
            return

        logger.info("=" * 60)
        logger.info("正在初始化 MediAgent 智慧医疗助手...")
        logger.info("=" * 60)

        try:
            # 1. 初始化 LLM Provider
            logger.info("[1/5] 初始化 LLM Provider...")
            self._llm = DeepSeekProvider(self.config.llm)

            # 2. 初始化长期记忆（带 try/except 降级）
            logger.info("[2/5] 初始化记忆系统...")
            self._init_long_term_memory()

            # 3. 注册内置工具
            logger.info("[3/5] 注册工具...")
            self._register_builtin_tools()

            # 4. 创建默认会话
            logger.info("[4/5] 设置会话管理...")
            self._current_session_id = self._create_session()

            # 5. 完成初始化
            logger.info("[5/5] 初始化完成")
            self._initialized = True

            logger.info("=" * 60)
            logger.info("Agent 初始化成功!")
            logger.info(f"   模型: {self.config.llm.model}")
            logger.info(f"   已注册工具: {tool_registry.count}")
            logger.info(f"   长期记忆: {'已启用' if self._long_memory else '未启用（ChromaDB 不可用）'}")
            logger.info(f"   最大 ReAct 迭代: {self.config.agent.max_iterations}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Agent 初始化失败: {e}", exc_info=True)
            raise RuntimeError(f"Agent 初始化失败: {e}")

    def _init_long_term_memory(self) -> None:
        """
        初始化长期记忆系统

        使用 try/except 包裹，ChromaDB 不可用时优雅降级。
        """
        try:
            from ..memory.long_term import LongTermMemory

            self._long_memory = LongTermMemory()
            # 异步初始化在后台完成
            # 注意：LongTermMemory 的 initialize() 是异步的，
            # 但这里我们不能 await，所以延迟到第一次使用时初始化
            self._long_memory_initialized = False
            logger.info("长期记忆系统已创建（将在首次使用时初始化）")
        except ImportError:
            self._long_memory = None
            self._long_memory_initialized = False
            logger.warning(
                "ChromaDB 不可用，长期记忆功能已禁用。"
                "如需启用，请安装: pip install chromadb"
            )
        except Exception as e:
            self._long_memory = None
            self._long_memory_initialized = False
            logger.warning(f"长期记忆初始化失败，已禁用: {e}")

    async def _ensure_long_memory_initialized(self) -> None:
        """确保长期记忆已异步初始化"""
        if self._long_memory and not getattr(self, '_long_memory_initialized', False):
            try:
                await self._long_memory.initialize()
                self._long_memory_initialized = True
                logger.info("长期记忆系统异步初始化完成")
            except Exception as e:
                logger.warning(f"长期记忆异步初始化失败: {e}")
                self._long_memory = None
                self._long_memory_initialized = False

    def _register_builtin_tools(self) -> None:
        """注册所有内置工具（包括医疗工具）"""
        # 导入通用工具类以触发自动注册
        from ..tools.web_search import WebSearchTool
        from ..tools.calculator import CalculatorTool
        from ..tools.code_executor import CodeExecutorTool
        from ..tools.file_manager import FileManagerTool

        # 导入医疗工具类以触发自动注册
        from ..tools.medical_knowledge import MedicalKnowledgeTool
        from ..tools.drug_query import DrugQueryTool
        from ..tools.drug_api import DrugApiTool  # 新增：外部药品API工具
        from ..tools.symptom_analyzer import SymptomAnalyzerTool

        # 根据配置启用/禁用工具
        enabled_tools = set(self.config.tools.enabled)

        all_tools = {
            "web_search": WebSearchTool,
            "calculator": CalculatorTool,
            "code_executor": CodeExecutorTool,
            "file_manager": FileManagerTool,
            "medical_knowledge": MedicalKnowledgeTool,
            "drug_query": DrugQueryTool,
            "drug_api": DrugApiTool,  # 新增
            "symptom_analyzer": SymptomAnalyzerTool,
        }

        for tool_name, tool_class in all_tools.items():
            if tool_name in enabled_tools:
                tool_registry.enable_tool(tool_name)
                logger.debug(f"已启用工具: {tool_name}")
            else:
                tool_registry.disable_tool(tool_name)
                logger.debug(f"已禁用工具: {tool_name}")

    # ==================== 会话管理 ====================

    def _create_session(
        self,
        mode: AgentMode = AgentMode.REACT,
    ) -> str:
        """
        创建新的会话（每个会话拥有独立的记忆）

        Args:
            mode: 会话默认模式

        Returns:
            session_id
        """
        session_id = str(uuid.uuid4())

        # 每个会话独立的短期记忆和工作记忆
        short_memory = ShortTermMemory(
            window_size=self.config.memory.short_term.window_size,
            max_tokens=self.config.memory.short_term.max_tokens,
        )
        working_memory = WorkingMemory()

        session_ctx = SessionContext(
            session_id=session_id,
            short_memory=short_memory,
            working_memory=working_memory,
            mode=mode,
        )

        self._sessions[session_id] = session_ctx
        logger.debug(f"创建会话: {session_id[:8]} (模式: {mode.value})")
        return session_id

    def _get_session(self, session_id: Optional[str] = None) -> SessionContext:
        """获取会话上下文，如果不存在则创建"""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        # 自动创建新会话
        new_id = self._create_session()
        self._current_session_id = new_id
        return self._sessions[new_id]

    def get_session_memory(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取指定会话的记忆状态

        Args:
            session_id: 会话 ID（默认当前会话）

        Returns:
            包含记忆信息的字典，或 None
        """
        session = self._sessions.get(session_id or self._current_session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "short_term": session.short_memory.to_dict(),
            "working_memory": session.working_memory.to_dict(),
            "message_count": session.message_count,
            "mode": session.mode.value,
        }

    # ==================== 请求追踪 ====================

    def _generate_request_id(self) -> str:
        """生成唯一的请求 ID"""
        return f"req_{uuid.uuid4().hex[:12]}"

    def _log_audit(self, audit: AuditLog) -> None:
        """记录审计日志"""
        self._audit_logs.append(audit)
        # 限制日志数量
        if len(self._audit_logs) > self._max_audit_logs:
            self._audit_logs = self._audit_logs[-self._max_audit_logs:]

        logger.info(
            f"[审计] request_id={audit.request_id} "
            f"session={audit.session_id[:8]} "
            f"mode={audit.mode} "
            f"success={audit.success} "
            f"tokens={audit.token_usage.get('total_tokens', 0)} "
            f"duration={audit.duration_seconds:.2f}s"
        )

    # ==================== 核心方法 ====================

    @log_execution_time(logger, "agent_chat")
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        简单对话模式 - 发送消息并获取回复

        适用于不需要工具调用的简单对话场景。
        使用直接 LLM 调用配合记忆上下文。

        Args:
            message: 用户消息
            session_id: 会话 ID（默认使用当前会话）

        Returns:
            AgentResponse
        """
        await self._ensure_initialized()
        await self._ensure_long_memory_initialized()

        request_id = self._generate_request_id()
        self._total_requests += 1

        session = self._get_session(session_id)
        sid = session.session_id

        logger.info(f"\n{'─'*60}")
        logger.info(f"Chat 请求 [Session: {sid[:8]}] [Request: {request_id}]")
        logger.info(f"用户: {message[:100]}...")
        logger.info(f"{'─'*60}\n")

        start_time = datetime.now()

        try:
            # 合规检查：输入脱敏
            sanitized_message, compliance_info = self._compliance_checker.check_input(message)
            if compliance_info.get("sanitized"):
                logger.info(f"输入已脱敏: {compliance_info['detected_types']}")

            # 紧急医疗情况检测
            if compliance_info.get("is_emergency"):
                emergency_response = self._compliance_checker.get_emergency_response()
                session.short_memory.add_assistant_message(emergency_response)
                session.update_activity()

                audit = AuditLog(
                    request_id=request_id,
                    session_id=sid,
                    mode="chat",
                    input_length=len(message),
                    output_length=len(emergency_response),
                    success=True,
                    is_emergency=True,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
                self._log_audit(audit)

                return AgentResponse(
                    session_id=sid,
                    response=emergency_response,
                    success=True,
                    request_id=request_id,
                    metadata={
                        "mode": "chat",
                        "is_emergency": True,
                        "model": self.config.llm.model,
                    },
                )

            # 存储用户消息到会话的短期记忆
            session.short_memory.add_user_message(sanitized_message)

            # 构建消息列表
            system_prompt = PromptTemplates.get_system_prompt(
                agent_name=self.config.agent.name,
                tools=[{"name": t.name, "description": t.description}
                       for t in tool_registry.get_all_tools()],
            )

            messages = [
                Message(role="system", content=system_prompt),
                *session.short_memory.get_context(),
            ]

            # 获取 LLM 响应
            response = await self._llm.chat_with_retry(messages)

            # 合规检查：输出免责声明
            checked_response, output_info = self._compliance_checker.check_output(response.content)

            # 存储助手响应到会话的短期记忆
            session.short_memory.add_assistant_message(checked_response)

            # 更新会话活动
            session.update_activity()

            # 定期提取记忆（每 5 次请求）
            if self._total_requests % 5 == 0:
                asyncio.create_task(self._extract_and_store_memories(session))

            duration = (datetime.now() - start_time).total_seconds()

            audit = AuditLog(
                request_id=request_id,
                session_id=sid,
                mode="chat",
                input_length=len(message),
                output_length=len(checked_response),
                success=True,
                token_usage=response.usage,
                duration_seconds=duration,
            )
            self._log_audit(audit)

            agent_response = AgentResponse(
                session_id=sid,
                response=checked_response,
                success=True,
                request_id=request_id,
                token_usage=response.usage,
                metadata={
                    "mode": "chat",
                    "model": self.config.llm.model,
                    "message_length": len(message),
                    "input_sanitized": compliance_info.get("sanitized", False),
                    "disclaimer_added": output_info.get("disclaimer_added", False),
                    "has_diagnosis": output_info.get("has_diagnosis", False),
                    "duration_seconds": duration,
                },
            )

            logger.info(f"\n响应生成完成 ({len(checked_response)} 字符)")
            return agent_response

        except Exception as e:
            logger.error(f"Chat 失败: {e}", exc_info=True)

            duration = (datetime.now() - start_time).total_seconds()

            audit = AuditLog(
                request_id=request_id,
                session_id=sid,
                mode="chat",
                input_length=len(message),
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            self._log_audit(audit)

            return AgentResponse(
                session_id=sid,
                response=f"很抱歉，处理您的请求时出现了错误：{str(e)}。如有健康相关问题，建议前往正规医疗机构咨询。",
                success=False,
                request_id=request_id,
                metadata={"error": str(e)},
            )

    @log_execution_time(logger, "agent_process")
    async def process(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
        stream_callback: Optional[Callable] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        处理复杂请求（支持多种模式）

        Args:
            input_text: 用户请求
            session_id: 会话 ID
            mode: 运行模式 ("react", "plan_and_execute", "direct")
                   默认使用会话的当前模式
            stream_callback: 中间结果回调函数

        Returns:
            AgentResponse 包含最终答案和推理链
        """
        await self._ensure_initialized()
        await self._ensure_long_memory_initialized()

        request_id = self._generate_request_id()
        self._total_requests += 1

        session = self._get_session(session_id)
        sid = session.session_id

        # 确定运行模式
        agent_mode = AgentMode(mode) if mode else session.mode

        logger.info(f"\n{'='*60}")
        logger.info(f"Process 请求 [Session: {sid[:8]}] [Request: {request_id}]")
        logger.info(f"输入: {input_text[:100]}...")
        logger.info(f"模式: {agent_mode.value}")
        logger.info(f"{'='*60}\n")

        start_time = datetime.now()

        try:
            # 合规检查：输入脱敏
            sanitized_input, compliance_info = self._compliance_checker.check_input(input_text)
            if compliance_info.get("sanitized"):
                logger.info(f"输入已脱敏: {compliance_info['detected_types']}")

            # 紧急医疗情况检测
            if compliance_info.get("is_emergency"):
                emergency_response = self._compliance_checker.get_emergency_response()
                session.short_memory.add_user_message(input_text)
                session.short_memory.add_assistant_message(emergency_response)
                session.update_activity()

                audit = AuditLog(
                    request_id=request_id,
                    session_id=sid,
                    mode=agent_mode.value,
                    input_length=len(input_text),
                    output_length=len(emergency_response),
                    success=True,
                    is_emergency=True,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
                self._log_audit(audit)

                return AgentResponse(
                    session_id=sid,
                    response=emergency_response,
                    success=True,
                    request_id=request_id,
                    metadata={
                        "mode": agent_mode.value,
                        "is_emergency": True,
                        "model": self.config.llm.model,
                    },
                )

            # 存储用户消息到会话的短期记忆（只添加一次）
            session.short_memory.add_user_message(sanitized_input)

            if agent_mode == AgentMode.REACT:
                response = await self._process_react(
                    session=session,
                    sanitized_input=sanitized_input,
                    request_id=request_id,
                    stream_callback=stream_callback,
                    compliance_info=compliance_info,
                    start_time=start_time,
                )

            elif agent_mode == AgentMode.PLAN_AND_EXECUTE:
                response = await self._process_plan_and_execute(
                    session=session,
                    sanitized_input=sanitized_input,
                    request_id=request_id,
                    stream_callback=stream_callback,
                    compliance_info=compliance_info,
                    start_time=start_time,
                )

            else:  # AgentMode.DIRECT
                # 直接模式：不重复添加用户消息，直接调用 LLM
                response = await self._process_direct(
                    session=session,
                    sanitized_input=sanitized_input,
                    request_id=request_id,
                    compliance_info=compliance_info,
                    start_time=start_time,
                )

            # 更新会话活动
            session.update_activity()

            # 定期提取记忆
            if self._total_requests % 5 == 0:
                asyncio.create_task(self._extract_and_store_memories(session))

            return response

        except Exception as e:
            logger.error(f"Process 失败: {e}", exc_info=True)

            duration = (datetime.now() - start_time).total_seconds()

            audit = AuditLog(
                request_id=request_id,
                session_id=sid,
                mode=agent_mode.value,
                input_length=len(input_text),
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            self._log_audit(audit)

            return AgentResponse(
                session_id=sid,
                response=f"处理您的请求时出现了错误：{str(e)}。如有健康相关问题，建议前往正规医疗机构咨询。",
                success=False,
                request_id=request_id,
                metadata={"error": str(e)},
            )

    async def _process_react(
        self,
        session: SessionContext,
        sanitized_input: str,
        request_id: str,
        stream_callback: Optional[Callable],
        compliance_info: Dict[str, Any],
        start_time: datetime,
    ) -> AgentResponse:
        """ReAct 模式处理"""
        # 创建 ReAct 规划器（使用会话的短期记忆）
        planner = ReActPlanner(
            llm_provider=self._llm,
            short_term_memory=session.short_memory,
            long_term_memory=self._long_memory,
            max_iterations=self.config.agent.max_iterations,
            verbose=self.config.agent.thinking_verbose,
        )

        result = await planner.plan_and_execute(
            user_input=sanitized_input,
            conversation_history=session.short_memory.get_context(),
            stream_callback=stream_callback,
        )

        # 合规检查：输出免责声明
        checked_answer, output_info = self._compliance_checker.check_output(result.final_answer)

        # 存储助手响应到会话的短期记忆
        session.short_memory.add_assistant_message(checked_answer)

        duration = (datetime.now() - start_time).total_seconds()

        audit = AuditLog(
            request_id=request_id,
            session_id=session.session_id,
            mode="react",
            input_length=len(sanitized_input),
            output_length=len(checked_answer),
            success=result.success,
            token_usage=result.token_usage,
            duration_seconds=duration,
        )
        self._log_audit(audit)

        return AgentResponse(
            session_id=session.session_id,
            response=checked_answer,
            success=result.success,
            request_id=request_id,
            reasoning_trace=[
                {
                    "step": i + 1,
                    "thought": t.content[:200],
                    "action": result.actions[i].tool_name if i < len(result.actions) else None,
                    "observation_success": result.observations[i].success if i < len(result.observations) else None,
                }
                for i, t in enumerate(result.thoughts)
            ],
            token_usage=result.token_usage,
            metadata={
                "mode": "react",
                "steps_completed": result.steps_completed,
                "state": result.state.value,
                "duration_seconds": result.duration_seconds,
                "input_sanitized": compliance_info.get("sanitized", False),
                "disclaimer_added": output_info.get("disclaimer_added", False),
                "has_diagnosis": output_info.get("has_diagnosis", False),
                # 合并工具执行的风险信息
                "risk_level": result.metadata.get("risk_level", "none"),
                "risk_warning": result.metadata.get("risk_warning", ""),
                "source": result.metadata.get("source", ""),
                "cache_hit": result.metadata.get("cache_hit", False),
            },
        )

    async def _process_plan_and_execute(
        self,
        session: SessionContext,
        sanitized_input: str,
        request_id: str,
        stream_callback: Optional[Callable],
        compliance_info: Dict[str, Any],
        start_time: datetime,
    ) -> AgentResponse:
        """Plan-and-Execute 模式处理"""
        planner = PlanAndExecutePlanner(
            llm_provider=self._llm,
            short_term_memory=session.short_memory,
            long_term_memory=self._long_memory,
            verbose=self.config.agent.thinking_verbose,
        )

        result = await planner.plan_and_execute(
            user_input=sanitized_input,
            conversation_history=session.short_memory.get_context(),
            stream_callback=stream_callback,
        )

        # 合规检查
        checked_answer, output_info = self._compliance_checker.check_output(result.final_answer)

        # 存储助手响应
        session.short_memory.add_assistant_message(checked_answer)

        duration = (datetime.now() - start_time).total_seconds()

        audit = AuditLog(
            request_id=request_id,
            session_id=session.session_id,
            mode="plan_and_execute",
            input_length=len(sanitized_input),
            output_length=len(checked_answer),
            success=result.success,
            token_usage=result.token_usage,
            duration_seconds=duration,
        )
        self._log_audit(audit)

        return AgentResponse(
            session_id=session.session_id,
            response=checked_answer,
            success=result.success,
            request_id=request_id,
            reasoning_trace=[
                {
                    "step": i + 1,
                    "thought": t.content[:200],
                    "action": result.actions[i].tool_name if i < len(result.actions) else None,
                    "observation_success": result.observations[i].success if i < len(result.observations) else None,
                }
                for i, t in enumerate(result.thoughts)
            ],
            token_usage=result.token_usage,
            metadata={
                "mode": "plan_and_execute",
                "steps_completed": result.steps_completed,
                "state": result.state.value,
                "duration_seconds": result.duration_seconds,
                "plan_steps": [s.to_dict() for s in result.plan_steps] if result.plan_steps else None,
                "input_sanitized": compliance_info.get("sanitized", False),
                "disclaimer_added": output_info.get("disclaimer_added", False),
                "has_diagnosis": output_info.get("has_diagnosis", False),
            },
        )

    async def _process_direct(
        self,
        session: SessionContext,
        sanitized_input: str,
        request_id: str,
        compliance_info: Dict[str, Any],
        start_time: datetime,
    ) -> AgentResponse:
        """
        Direct 模式处理

        注意：用户消息已经在 process() 中添加到短期记忆了，
        这里不要再重复添加，直接构建 messages 并调用 LLM。
        """
        system_prompt = PromptTemplates.get_system_prompt(
            agent_name=self.config.agent.name,
            tools=[{"name": t.name, "description": t.description}
                   for t in tool_registry.get_all_tools()],
        )

        messages = [
            Message(role="system", content=system_prompt),
            *session.short_memory.get_context(),
        ]

        response = await self._llm.chat_with_retry(messages)

        # 合规检查
        checked_response, output_info = self._compliance_checker.check_output(response.content)

        # 存储助手响应
        session.short_memory.add_assistant_message(checked_response)

        duration = (datetime.now() - start_time).total_seconds()

        audit = AuditLog(
            request_id=request_id,
            session_id=session.session_id,
            mode="direct",
            input_length=len(sanitized_input),
            output_length=len(checked_response),
            success=True,
            token_usage=response.usage,
            duration_seconds=duration,
        )
        self._log_audit(audit)

        return AgentResponse(
            session_id=session.session_id,
            response=checked_response,
            success=True,
            request_id=request_id,
            token_usage=response.usage,
            metadata={
                "mode": "direct",
                "model": self.config.llm.model,
                "input_sanitized": compliance_info.get("sanitized", False),
                "disclaimer_added": output_info.get("disclaimer_added", False),
                "has_diagnosis": output_info.get("has_diagnosis", False),
                "duration_seconds": duration,
            },
        )

    # ==================== 流式方法 ====================

    async def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        流式聊天响应 - 逐 token 输出

        修复：缓冲完整响应并在流式结束后存储到短期记忆。

        Yields:
            字符串片段

        Usage:
            async for chunk in agent.chat_stream("Hello!"):
                print(chunk, end="", flush=True)
        """
        await self._ensure_initialized()

        session = self._get_session(session_id)
        sid = session.session_id

        # 存储用户消息
        session.short_memory.add_user_message(message)

        system_prompt = PromptTemplates.get_system_prompt(
            agent_name=self.config.agent.name,
        )

        messages = [
            Message(role="system", content=system_prompt),
            *session.short_memory.get_context(),
        ]

        # 缓冲完整响应
        full_response = []

        try:
            async for chunk in self._llm.chat_stream(messages):
                if chunk.content:
                    full_response.append(chunk.content)
                    yield chunk.content
        finally:
            # 流式结束后，将完整响应存储到短期记忆
            complete_response = "".join(full_response)
            if complete_response:
                # 合规检查
                checked_response, _ = self._compliance_checker.check_output(complete_response)
                session.short_memory.add_assistant_message(checked_response)
                session.update_activity()
                logger.debug(
                    f"流式响应已存储到短期记忆 ({len(checked_response)} 字符)"
                )

    async def process_stream(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式 ReAct 处理 - 逐步 yield SSE 事件

        Yields:
            字典事件，包含 type, step, content 等字段
            type 可以是: thought, action, observation, final, error

        Usage:
            async for event in agent.process_stream("分析一下我的症状"):
                if event["type"] == "thought":
                    print(f"思考: {event['content']}")
                elif event["type"] == "action":
                    print(f"调用工具: {event['tool_name']}")
                elif event["type"] == "observation":
                    print(f"结果: {event['result']}")
                elif event["type"] == "final":
                    print(f"最终答案: {event['final_answer']}")
        """
        await self._ensure_initialized()
        await self._ensure_long_memory_initialized()

        request_id = self._generate_request_id()
        self._total_requests += 1

        session = self._get_session(session_id)
        sid = session.session_id
        agent_mode = AgentMode(mode) if mode else session.mode

        # 合规检查
        sanitized_input, compliance_info = self._compliance_checker.check_input(input_text)

        # 紧急情况检测
        if compliance_info.get("is_emergency"):
            emergency_response = self._compliance_checker.get_emergency_response()
            session.short_memory.add_user_message(input_text)
            session.short_memory.add_assistant_message(emergency_response)
            session.update_activity()
            yield {
                "type": "final",
                "success": True,
                "final_answer": emergency_response,
                "is_emergency": True,
                "request_id": request_id,
                "session_id": sid,
            }
            return

        # 存储用户消息
        session.short_memory.add_user_message(sanitized_input)

        # yield request_id
        yield {
            "type": "start",
            "request_id": request_id,
            "session_id": sid,
            "mode": agent_mode.value,
        }

        try:
            if agent_mode == AgentMode.REACT:
                planner = ReActPlanner(
                    llm_provider=self._llm,
                    short_term_memory=session.short_memory,
                    long_term_memory=self._long_memory,
                    max_iterations=self.config.agent.max_iterations,
                    verbose=self.config.agent.thinking_verbose,
                )

                async for event in planner.plan_and_execute_stream(
                    user_input=sanitized_input,
                    conversation_history=session.short_memory.get_context(),
                ):
                    event["request_id"] = request_id
                    event["session_id"] = sid
                    yield event

                    # 如果是最终结果，存储到记忆
                    if event.get("type") == "final":
                        final_answer = event.get("final_answer", "")
                        if final_answer:
                            checked, _ = self._compliance_checker.check_output(final_answer)
                            session.short_memory.add_assistant_message(checked)
                            session.update_activity()

            elif agent_mode == AgentMode.PLAN_AND_EXECUTE:
                planner = PlanAndExecutePlanner(
                    llm_provider=self._llm,
                    short_term_memory=session.short_memory,
                    long_term_memory=self._long_memory,
                    verbose=self.config.agent.thinking_verbose,
                )

                # Plan-and-Execute 也使用 stream_callback 桥接
                import asyncio
                queue: asyncio.Queue = asyncio.Queue()

                async def _cb(event: Dict[str, Any]):
                    await queue.put(event)

                task = asyncio.create_task(
                    planner.plan_and_execute(
                        user_input=sanitized_input,
                        conversation_history=session.short_memory.get_context(),
                        stream_callback=_cb,
                    )
                )

                while not task.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        event["request_id"] = request_id
                        event["session_id"] = sid
                        yield event
                    except asyncio.TimeoutError:
                        continue

                result = task.result()
                if result.final_answer:
                    checked, _ = self._compliance_checker.check_output(result.final_answer)
                    session.short_memory.add_assistant_message(checked)
                    session.update_activity()

                yield {
                    "type": "final",
                    "success": result.success,
                    "final_answer": result.final_answer,
                    "state": result.state.value,
                    "steps_completed": result.steps_completed,
                    "duration_seconds": result.duration_seconds,
                    "token_usage": result.token_usage,
                    "request_id": request_id,
                    "session_id": sid,
                }

            else:  # DIRECT
                # 直接模式使用 chat_stream
                async for chunk in self.chat_stream(sanitized_input, session_id=sid):
                    yield {
                        "type": "chunk",
                        "content": chunk,
                        "request_id": request_id,
                        "session_id": sid,
                    }

                yield {
                    "type": "final",
                    "success": True,
                    "request_id": request_id,
                    "session_id": sid,
                }

        except Exception as e:
            logger.error(f"process_stream 失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "request_id": request_id,
                "session_id": sid,
            }

    # ==================== 记忆管理 ====================

    async def _extract_and_store_memories(self, session: SessionContext) -> None:
        """
        从最近的对话中提取重要信息并存储到长期记忆

        修复：使用 Message 对象而非 dict 传递给 LLM。
        """
        if not self._long_memory:
            return

        try:
            conversation = session.short_memory.get_conversation_history()

            if not conversation or len(conversation) < 50:
                return

            from ..llm.prompts import PromptTemplates

            prompt_text = PromptTemplates.get_memory_extraction_prompt(conversation)

            # 使用 Message 对象（修复：原来传 dict）
            messages = [
                Message(role="system", content=prompt_text),
                Message(role="user", content="请从以上对话中提取值得保存的医疗信息。"),
            ]

            response = await self._llm.chat_with_retry(messages)

            # 解析 JSON 响应
            import json
            extracted = json.loads(response.content)
            memories_data = extracted.get("memories", [])

            if memories_data:
                await self._long_memory.add_memories_batch(memories_data)
                logger.info(f"提取并存储了 {len(memories_data)} 条记忆")

        except Exception as e:
            logger.warning(f"记忆提取失败: {e}")

    # ==================== 会话操作 ====================

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有活跃会话"""
        return [s.to_dict() for s in self._sessions.values()]

    def clear_session(self, session_id: Optional[str] = None) -> bool:
        """清除指定会话或当前会话"""
        target_id = session_id or self._current_session_id

        if target_id and target_id in self._sessions:
            del self._sessions[target_id]

            if target_id == self._current_session_id:
                self._current_session_id = self._create_session()

            logger.info(f"已清除会话: {target_id[:8]}")
            return True

        return False

    def set_session_mode(
        self,
        mode: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        设置会话的运行模式

        Args:
            mode: "react", "plan_and_execute", "direct"
            session_id: 会话 ID

        Returns:
            是否设置成功
        """
        try:
            agent_mode = AgentMode(mode)
        except ValueError:
            logger.error(f"无效的模式: {mode}")
            return False

        session = self._sessions.get(session_id or self._current_session_id)
        if session:
            session.mode = agent_mode
            logger.info(f"会话 {session.session_id[:8]} 模式已设置为 {mode}")
            return True

        return False

    def get_audit_logs(
        self,
        limit: int = 50,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取审计日志

        Args:
            limit: 返回条数
            session_id: 按会话过滤

        Returns:
            审计日志列表
        """
        logs = self._audit_logs
        if session_id:
            logs = [l for l in logs if l.session_id == session_id]

        return [
            {
                "request_id": l.request_id,
                "session_id": l.session_id[:8] + "...",
                "timestamp": l.timestamp.isoformat(),
                "mode": l.mode,
                "success": l.success,
                "token_usage": l.token_usage,
                "duration_seconds": l.duration_seconds,
                "is_emergency": l.is_emergency,
                "error": l.error,
            }
            for l in logs[-limit:]
        ]

    # ==================== 状态和生命周期 ====================

    async def _ensure_initialized(self) -> None:
        """确保 Agent 已初始化"""
        if not self._initialized:
            logger.info("自动初始化 Agent...")
            await self.initialize()

    def get_status(self) -> Dict[str, Any]:
        """
        获取 Agent 综合状态信息

        Returns:
            状态详情字典
        """
        uptime = (datetime.now() - self._start_time).total_seconds()

        # 汇总所有会话的记忆信息
        total_short_term = sum(
            s.short_memory.size for s in self._sessions.values()
        )

        return {
            "initialized": self._initialized,
            "uptime_seconds": round(uptime, 2),
            "config": {
                "model": self.config.llm.model,
                "provider": self.config.llm.provider,
                "agent_name": self.config.agent.name,
            },
            "statistics": {
                "total_requests": self._total_requests,
                "active_sessions": len(self._sessions),
                "tools_available": tool_registry.count,
                "audit_log_entries": len(self._audit_logs),
            },
            "memory": {
                "short_term_total_conversations": total_short_term,
                "long_term_memories": self._long_memory.count if self._long_memory else 0,
                "long_term_enabled": self._long_memory is not None,
            },
            "llm_stats": self._llm.stats if self._llm else {},
            "sessions": [
                {
                    "session_id": s.session_id[:8] + "...",
                    "mode": s.mode.value,
                    "messages": s.message_count,
                }
                for s in self._sessions.values()
            ],
        }

    async def reset(self) -> None:
        """重置 Agent 到初始状态"""
        logger.warning("正在重置 Agent...")

        # 清除所有会话
        for session in self._sessions.values():
            session.short_memory.clear()
            session.working_memory.clear()

        self._sessions.clear()
        self._current_session_id = self._create_session()
        self._total_requests = 0
        self._audit_logs.clear()

        if self._llm:
            self._llm.reset_stats()

        logger.info("Agent 重置完成")

    async def close(self) -> None:
        """
        关闭 Agent 并释放资源

        使用完毕后务必调用此方法。
        """
        logger.info("正在关闭 Agent...")

        try:
            if self._llm:
                await self._llm.close()

            if self._long_memory:
                await self._long_memory.close()

            self._initialized = False

            logger.info("Agent 已成功关闭")

        except Exception as e:
            logger.error(f"关闭 Agent 时出错: {e}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    def __repr__(self) -> str:
        status = "已初始化" if self._initialized else "未初始化"
        return (
            f"MediAgent({status}, "
            f"model={self.config.llm.model}, "
            f"requests={self._total_requests}, "
            f"sessions={len(self._sessions)})"
        )


# 便捷工厂函数
async def create_agent(
    config_path: Optional[str] = None,
) -> Agent:
    """
    快速创建并初始化 Agent

    Usage:
        agent = await create_agent()
        response = await agent.chat("Hello!")
        await agent.close()
    """
    agent = Agent(config_path=config_path)
    await agent.initialize()
    return agent
