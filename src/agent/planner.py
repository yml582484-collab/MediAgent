"""
ReAct (Reason-Act-Observe) Task Planner & Plan-and-Execute Planner
实现基于 DeepSeek 原生 Function Calling 的推理引擎
支持 ReAct 模式和 Plan-and-Execute 模式
"""
import json
import re
import asyncio
import hashlib
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..llm.provider import DeepSeekProvider, Message, LLMResponse
from ..llm.prompts import PromptTemplates
from ..tools.base import BaseTool, tool_registry
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import LongTermMemory
from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PlannerState(Enum):
    """ReAct 规划循环中的状态"""
    INITIALIZING = "initializing"
    REASONING = "reasoning"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    REPLAN = "replan"  # 重新规划


@dataclass
class Thought:
    """ReAct 循环中的单步推理"""
    step: int
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Action:
    """Agent 执行的动作"""
    step: int
    tool_name: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Observation:
    """动作执行后的观察结果"""
    step: int
    result: Any
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 工具返回的元数据（风险信息等）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "result": str(self.result)[:500] if self.result else None,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class PlanStep:
    """Plan-and-Execute 模式中的计划步骤"""
    step_number: int
    description: str
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "description": self.description,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class PlanExecutionResult:
    """计划执行的完整结果"""
    final_answer: str
    success: bool
    steps_completed: int
    thoughts: List[Thought] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    state: PlannerState = PlannerState.COMPLETED
    token_usage: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    plan_steps: Optional[List[PlanStep]] = None  # Plan-and-Execute 模式的计划步骤
    metadata: Dict[str, Any] = field(default_factory=dict)  # 工具执行返回的元数据（风险信息等）

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "final_answer": self.final_answer,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "state": self.state.value,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "actions": [a.to_dict() for a in self.actions],
            "observations": [o.to_dict() for o in self.observations],
            "token_usage": self.token_usage,
            "duration_seconds": round(self.duration_seconds, 2),
            "metadata": self.metadata,
        }
        if self.plan_steps is not None:
            result["plan_steps"] = [s.to_dict() for s in self.plan_steps]
        return result


def _params_hash(params: Dict[str, Any]) -> str:
    """计算参数的哈希值，用于工具结果缓存"""
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()


class ReActPlanner:
    """
    ReAct (Reason-Act-Observe) 任务规划器

    使用 DeepSeek 原生 Function Calling API 实现工具调用，
    同时保留正则解析作为降级方案。

    认知循环:
    1. **Thought**: 分析当前情况，决定下一步
    2. **Action**: 通过 Function Calling 或正则解析执行工具
    3. **Observe**: 分析工具执行结果
    4. **Repeat**: 直到任务完成

    Features:
    - DeepSeek 原生 Function Calling（主要策略）
    - 正则解析降级方案（兼容模式）
    - 工具结果缓存，避免重复调用
    - 流式中间结果输出
    - 最大迭代次数安全限制
    - 详细的执行追踪日志
    """

    def __init__(
        self,
        llm_provider: DeepSeekProvider,
        short_term_memory: ShortTermMemory,
        long_term_memory: Optional[LongTermMemory] = None,
        max_iterations: Optional[int] = None,
        verbose: bool = True,
        use_function_calling: bool = True,
    ):
        """
        初始化 ReAct 规划器

        Args:
            llm_provider: LLM 提供者
            short_term_memory: 短期记忆
            long_term_memory: 长期记忆（可选）
            max_iterations: 最大迭代次数
            verbose: 是否输出详细日志
            use_function_calling: 是否使用原生 Function Calling（False 则仅用正则）
        """
        config = get_config().config.agent

        self.llm = llm_provider
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.max_iterations = max_iterations or config.max_iterations
        self.verbose = verbose
        self.use_function_calling = use_function_calling

        # 工具 schemas（OpenAI 格式）
        self._tool_schemas: List[Dict[str, Any]] = []
        self._refresh_tool_schemas()

        # 工具结果缓存: (tool_name, params_hash) -> result_str
        self._tool_result_cache: Dict[Tuple[str, str], str] = {}

        self._state = PlannerState.INITIALIZING
        self._current_step = 0

        logger.info(
            f"ReActPlanner 初始化完成 "
            f"(max_iterations={self.max_iterations}, "
            f"verbose={self.verbose}, "
            f"function_calling={self.use_function_calling}, "
            f"tools={len(self._tool_schemas)})"
        )

    def _refresh_tool_schemas(self) -> None:
        """刷新工具 schemas 列表"""
        try:
            self._tool_schemas = tool_registry.get_all_schemas()
        except Exception as e:
            logger.warning(f"获取工具 schemas 失败: {e}")
            self._tool_schemas = []

    async def plan_and_execute(
        self,
        user_input: str,
        conversation_history: Optional[List[Message]] = None,
        stream_callback: Optional[Callable] = None,
    ) -> PlanExecutionResult:
        """
        执行完整的 ReAct 规划循环

        Args:
            user_input: 用户输入
            conversation_history: 对话历史（可选）
            stream_callback: 流式回调函数

        Returns:
            PlanExecutionResult 包含最终答案和完整推理链
        """
        start_time = datetime.now()
        max_execution_time = 120  # 最大执行时间 120 秒

        result = PlanExecutionResult(
            final_answer="",
            success=False,
            steps_completed=0,
        )

        # 错误追踪
        consecutive_failures = 0
        max_consecutive_failures = 3

        try:
            self._state = PlannerState.REASONING
            self._current_step = 0

            # 获取相关上下文
            context = await self._gather_context(user_input)

            # 构建 ReAct prompt（用于正则降级模式）
            react_prompt = PromptTemplates.get_react_prompt(
                user_input=user_input,
                tools_description=tool_registry.get_tools_description(),
                conversation_history=self._format_history(conversation_history),
                relevant_memories=context,
            )

            # 构建 messages 列表
            messages: List[Message] = [
                Message(role="system", content=react_prompt),
                Message(role="user", content=user_input),
            ]

            # 主 ReAct 循环
            while self._current_step < self.max_iterations:
                # 检查超时
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > max_execution_time:
                    logger.warning(f"ReAct 执行超时 ({elapsed:.1f}s)")
                    result.state = PlannerState.MAX_ITERATIONS_REACHED
                    result.final_answer = (
                        f"任务执行时间过长（{elapsed:.1f}秒），已自动终止。\n\n"
                        f"当前进度：\n"
                        + self._summarize_progress(result)
                    )
                    break

                self._current_step += 1

                if self.verbose:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"ReAct 步骤 {self._current_step}/{self.max_iterations}")
                    logger.info(f"{'='*60}\n")

                # ===== 核心推理步骤 =====
                try:
                    if self.use_function_calling and self._tool_schemas:
                        # 使用 DeepSeek 原生 Function Calling
                        step_result = await self._reason_with_function_calling(
                            messages, user_input, stream_callback
                        )
                    else:
                        # 降级到正则解析模式
                        step_result = await self._reason_with_regex(
                            messages, user_input, stream_callback
                        )
                except Exception as e:
                    logger.error(f"推理步骤 {self._current_step} 失败: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        result.final_answer = (
                            f"推理过程连续失败 {consecutive_failures} 次，无法完成任务。"
                            f"错误：{str(e)}"
                        )
                        result.state = PlannerState.FAILED
                        break

                    # Function Calling 失败时降级到正则模式
                    if self.use_function_calling:
                        logger.warning("Function Calling 失败，降级到正则解析模式")
                        try:
                            step_result = await self._reason_with_regex(
                                messages, user_input, stream_callback
                            )
                        except Exception as regex_e:
                            logger.error(f"正则解析也失败: {regex_e}")
                            continue
                    else:
                        continue

                # 处理推理结果
                thought_content = step_result.get("thought", "")
                tool_calls = step_result.get("tool_calls", None)
                final_answer = step_result.get("final_answer", None)

                # 记录 thought
                thought = Thought(
                    step=self._current_step,
                    content=thought_content,
                )
                result.thoughts.append(thought)

                if stream_callback:
                    await stream_callback({
                        "type": "thought",
                        "step": self._current_step,
                        "content": thought_content,
                    })

                # 检查是否有最终答案
                if final_answer:
                    result.final_answer = final_answer
                    result.success = True
                    result.state = PlannerState.COMPLETED
                    break

                # 检查 thought 中是否包含最终答案标记（正则模式）
                if self._is_final_answer(thought_content):
                    result.final_answer = self._extract_final_answer(thought_content)
                    result.success = True
                    result.state = PlannerState.COMPLETED
                    break

                # 如果有 tool_calls，执行工具
                if tool_calls:
                    # 将 assistant 的完整响应（包含 tool_calls）添加到 messages
                    assistant_msg = step_result.get("assistant_message", None)
                    if assistant_msg:
                        messages.append(assistant_msg)

                    for tc in tool_calls:
                        tool_name = tc.get("function", {}).get("name", "")
                        try:
                            params = json.loads(tc.get("function", {}).get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            params = {}

                        action = Action(
                            step=self._current_step,
                            tool_name=tool_name,
                            parameters=params,
                        )
                        result.actions.append(action)

                        if stream_callback:
                            await stream_callback({
                                "type": "action",
                                "step": self._current_step,
                                "tool_name": tool_name,
                                "parameters": params,
                            })

                        # 执行工具
                        observation = await self._execute_action(action)
                        result.observations.append(observation)

                        if stream_callback:
                            await stream_callback({
                                "type": "observation",
                                "step": self._current_step,
                                "result": observation.result,
                                "success": observation.success,
                            })

                        # 添加 tool result message 到 messages
                        tool_call_id = tc.get("id", f"call_{self._current_step}")
                        tool_result_msg = Message(
                            role="tool",
                            content=observation.result if observation.success else f"错误: {observation.error}",
                            name=tool_name,
                            tool_call_id=tool_call_id,
                        )
                        messages.append(tool_result_msg)

                        if not observation.success:
                            consecutive_failures += 1
                            logger.warning(
                                f"工具 {tool_name} 执行失败: {observation.error}"
                            )
                            if consecutive_failures >= max_consecutive_failures:
                                result.final_answer = (
                                    f"工具调用连续失败 {consecutive_failures} 次。\n\n"
                                    f"最后错误：{observation.error}\n\n"
                                    f"已完成步骤：\n"
                                    + self._summarize_progress(result)
                                )
                                result.state = PlannerState.FAILED
                                break
                        else:
                            consecutive_failures = 0
                else:
                    # 没有工具调用也没有最终答案
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        result.final_answer = (
                            f"无法确定下一步操作（连续 {consecutive_failures} 次）。\n\n"
                            f"最后的思考：\n{thought_content[:500]}\n\n"
                            f"建议：请尝试更具体的指令，或者简化任务要求。"
                        )
                        result.state = PlannerState.COMPLETED
                        break

                    if self._current_step > 2 and len(thought_content) > 50:
                        logger.info(
                            f"步骤 {self._current_step} 无动作，"
                            f"尝试将当前思考作为潜在答案"
                        )
                        continue

                    logger.warning(
                        f"步骤 {self._current_step} 未解析到动作，重试..."
                    )
                    # 将 assistant 的回复添加到 messages，让 LLM 知道需要继续
                    messages.append(Message(role="assistant", content=thought_content))
                    continue

                # 如果因连续失败而 break 了内部循环，需要 break 外部循环
                if result.state in (PlannerState.FAILED, PlannerState.COMPLETED):
                    break

            else:
                # 达到最大迭代次数
                result.state = PlannerState.MAX_ITERATIONS_REACHED
                result.final_answer = (
                    f"已完成 {self.max_iterations} 步推理，任务处理中...\n\n"
                    + self._summarize_progress(result)
                    + "\n\n如需继续，请提供更明确的指令。"
                )

            result.steps_completed = self._current_step

            # 收集所有 observation 的元数据（风险信息等）
            collected_metadata = {}
            for obs in result.observations:
                if obs.metadata:
                    # 合并元数据，保留最高风险级别
                    if "risk_level" in obs.metadata:
                        current_risk = collected_metadata.get("risk_level", "none")
                        new_risk = obs.metadata.get("risk_level", "none")
                        # 风险级别优先级: high > medium > low > none
                        risk_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
                        if risk_order.get(new_risk, 0) > risk_order.get(current_risk, 0):
                            collected_metadata["risk_level"] = new_risk
                            collected_metadata["risk_warning"] = obs.metadata.get("risk_warning", "")
                            collected_metadata["source"] = obs.metadata.get("source", "")
                    # 其他元数据直接合并
                    for key in ["cache_hit", "fallback_source"]:
                        if key in obs.metadata and key not in collected_metadata:
                            collected_metadata[key] = obs.metadata[key]

            result.metadata = collected_metadata

        except Exception as e:
            logger.error(f"ReAct 规划执行失败: {e}", exc_info=True)
            result.state = PlannerState.FAILED
            result.final_answer = f"执行过程中发生错误：{str(e)}"

        finally:
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            result.token_usage = self.llm.token_usage.to_dict()

            # 重置状态
            self._current_step = 0
            self._state = PlannerState.INITIALIZING

            logger.info(
                f"\n{'='*60}"
                f"\nReAct 执行完成:"
                f"\n   状态: {result.state.value}"
                f"\n   步骤: {result.steps_completed}/{self.max_iterations}"
                f"\n   耗时: {result.duration_seconds:.2f}s"
                f"\n   Token: {result.token_usage.get('total_tokens', 0)}"
                f"\n{'='*60}\n"
            )

        return result

    async def plan_and_execute_stream(
        self,
        user_input: str,
        conversation_history: Optional[List[Message]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式执行 ReAct 规划循环，逐步 yield 中间结果

        Yields:
            字典，包含 type, step, content 等字段
        """
        async def _stream_cb(event: Dict[str, Any]):
            yield event  # type: ignore

        # 由于不能在 async generator 中直接使用 callback，
        # 这里用 queue 来桥接
        import asyncio
        queue: asyncio.Queue = asyncio.Queue()

        async def _callback(event: Dict[str, Any]):
            await queue.put(event)

        # 启动 plan_and_execute
        task = asyncio.create_task(
            self.plan_and_execute(
                user_input=user_input,
                conversation_history=conversation_history,
                stream_callback=_callback,
            )
        )

        # 从 queue 中读取事件并 yield
        while not task.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

        # yield 最终结果
        result = task.result()
        yield {
            "type": "final",
            "success": result.success,
            "final_answer": result.final_answer,
            "state": result.state.value,
            "steps_completed": result.steps_completed,
            "duration_seconds": result.duration_seconds,
            "token_usage": result.token_usage,
        }

    async def _reason_with_function_calling(
        self,
        messages: List[Message],
        user_input: str,
        stream_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        使用 DeepSeek 原生 Function Calling 进行推理

        Returns:
            字典，包含:
            - thought: 思考内容
            - tool_calls: 工具调用列表（如果有）
            - final_answer: 最终答案（如果有）
            - assistant_message: 完整的 assistant Message（用于追加到 messages）
        """
        self._state = PlannerState.REASONING

        response = await self.llm.chat(
            messages=messages,
            tools=self._tool_schemas if self._tool_schemas else None,
            tool_choice="auto",
        )

        thought_content = response.content or ""
        tool_calls_raw = response.tool_calls
        finish_reason = response.finish_reason

        if self.verbose:
            logger.info(f"\n[Function Calling] 响应:")
            logger.info(f"  finish_reason: {finish_reason}")
            logger.info(f"  content: {thought_content[:200]}...")
            if tool_calls_raw:
                logger.info(f"  tool_calls: {len(tool_calls_raw)} 个调用")

        # 构建 assistant message（包含 tool_calls 信息）
        assistant_message = Message(
            role="assistant",
            content=thought_content or None,
            tool_calls=tool_calls_raw,
        )

        result: Dict[str, Any] = {
            "thought": thought_content,
            "tool_calls": None,
            "final_answer": None,
            "assistant_message": assistant_message,
        }

        # 处理 tool_calls
        if tool_calls_raw and finish_reason == "tool_calls":
            # 将 OpenAI 格式的 tool_calls 转换为可执行格式
            tool_calls = []
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    },
                })
            result["tool_calls"] = tool_calls

        elif finish_reason == "stop" and thought_content:
            # LLM 停止生成，检查是否为最终答案
            # 如果 content 中包含最终答案标记
            if self._is_final_answer(thought_content):
                result["final_answer"] = self._extract_final_answer(thought_content)
            elif not tool_calls_raw:
                # 没有 tool_calls 且没有最终答案标记，
                # 可能是 LLM 直接回答了问题
                # 判断内容长度：如果内容较长，视为直接回答
                if len(thought_content.strip()) > 100:
                    result["final_answer"] = thought_content
                else:
                    # 内容太短，可能是中间状态，继续循环
                    pass

        return result

    async def _reason_with_regex(
        self,
        messages: List[Message],
        user_input: str,
        stream_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        使用正则解析模式进行推理（降级方案）

        Returns:
            同 _reason_with_function_calling
        """
        self._state = PlannerState.REASONING

        # 不传 tools 参数，让 LLM 纯文本输出
        response = await self.llm.chat_with_retry(messages)

        thought_content = response.content

        if self.verbose:
            logger.info(f"\n[正则模式] 思考内容:")
            logger.info(thought_content[:500])

        result: Dict[str, Any] = {
            "thought": thought_content,
            "tool_calls": None,
            "final_answer": None,
            "assistant_message": Message(role="assistant", content=thought_content),
        }

        # 检查是否为最终答案
        if self._is_final_answer(thought_content):
            result["final_answer"] = self._extract_final_answer(thought_content)
            return result

        # 尝试解析动作
        action_data = self._parse_action_from_thought(thought_content)

        if action_data:
            tool_name = action_data["tool"]
            params = action_data["params"]

            # 构造类似 Function Calling 的格式
            tool_calls = [{
                "id": f"call_regex_{self._current_step}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(params, ensure_ascii=False),
                },
            }]
            result["tool_calls"] = tool_calls

            # 更新 assistant_message，加入思考内容
            result["assistant_message"] = Message(
                role="assistant",
                content=thought_content,
            )

        return result

    async def _gather_context(self, query: str) -> str:
        """
        从记忆系统中收集相关上下文

        Args:
            query: 用户查询

        Returns:
            格式化的上下文字符串
        """
        context_parts = []

        if self.long_term_memory:
            try:
                memories = await self.long_term_memory.search(query, n_results=3)
                if memories:
                    context_parts.append("**相关记忆:**")
                    for mem in memories[:3]:
                        context_parts.append(f"- {mem.content}")
            except Exception as e:
                logger.warning(f"长期记忆搜索失败: {e}")

        return "\n".join(context_parts) if context_parts else ""

    async def _execute_action(self, action: Action) -> Observation:
        """
        执行工具调用

        Args:
            action: 要执行的动作

        Returns:
            Observation 包含执行结果
        """
        self._state = PlannerState.OBSERVING

        # 检查缓存
        cache_key = (action.tool_name, _params_hash(action.parameters))
        if cache_key in self._tool_result_cache:
            cached_result = self._tool_result_cache[cache_key]
            if self.verbose:
                logger.info(f"\n[缓存命中] {action.tool_name}")
            return Observation(
                step=action.step,
                result=cached_result,
                success=True,
            )

        try:
            tool = tool_registry.get_tool(action.tool_name)

            if not tool:
                return Observation(
                    step=action.step,
                    result=f"工具 '{action.tool_name}' 未找到或未启用",
                    success=False,
                    error="Tool not found",
                )

            # 验证参数
            try:
                tool.validate_parameters(action.parameters)
            except (ValueError, TypeError) as e:
                return Observation(
                    step=action.step,
                    result=f"参数验证失败: {str(e)}",
                    success=False,
                    error=f"Parameter validation failed: {e}",
                )

            # 执行工具
            raw_result = await tool.execute(**action.parameters)

            # 提取元数据（风险信息等）
            tool_metadata = {}
            if isinstance(raw_result, dict):
                # 提取风险相关字段
                for key in ["risk_level", "risk_warning", "source", "cache_hit", "fallback_source"]:
                    if key in raw_result:
                        tool_metadata[key] = raw_result[key]

            # 格式化结果
            formatted_result = tool.format_result(raw_result)

            # 缓存结果
            self._tool_result_cache[cache_key] = formatted_result

            observation = Observation(
                step=action.step,
                result=formatted_result,
                success=True,
                metadata=tool_metadata,  # 传递元数据
            )

            if self.verbose:
                logger.info(f"\n[观察] {action.tool_name}:")
                logger.info(formatted_result[:500])

            return observation

        except Exception as e:
            logger.error(f"工具 {action.tool_name} 执行失败: {e}")
            return Observation(
                step=action.step,
                result=f"执行 {action.tool_name} 时出错: {str(e)}",
                success=False,
                error=str(e),
            )

    # ==================== 正则解析（降级方案）====================

    def _parse_action_from_thought(self, thought: str) -> Optional[Dict]:
        """
        从 LLM 思考内容中解析动作信息（正则降级方案）

        支持多种格式:
        1. 函数调用格式: tool_name(params)
        2. 结构化格式: **Action:** tool\nParameters: {json}
        3. 宽松搜索: 在文本中查找工具名并推断参数
        """
        # ===== 策略1: 函数调用格式 =====
        # 匹配所有已注册的工具名（不再硬编码 4 个）
        available_tools = [t.name for t in tool_registry.get_all_tools()]
        if available_tools:
            tools_pattern = "|".join(re.escape(t) for t in available_tools)
            func_patterns = [
                rf'({tools_pattern})\s*[\(（]\s*(.+?)\s*[\)）]',
                rf'({tools_pattern})\s*[\(（]([^)]+)[\)）]',
            ]

            for pattern in func_patterns:
                match = re.search(pattern, thought, re.IGNORECASE | re.DOTALL)
                if match:
                    tool_name = match.group(1).lower().strip()
                    params_raw = match.group(2).strip()

                    logger.debug(
                        f"函数模式匹配: {tool_name}({params_raw[:100]}...)"
                    )

                    params = self._extract_function_params(tool_name, params_raw)
                    if params:
                        return {"tool": tool_name, "params": params}

        # ===== 策略2: 结构化格式 (JSON/键值对) =====
        structured_patterns = [
            (r'\*\*Action:\*\*\s*(\w+)', r'Parameters?:\s*(\{{[^}}]+\}})'),
            (r'[\u884c\u52a8][\uff1a:]\s*(\w+)', r'[\u53c2\u6570][\uff1a:]\s*(\{{[^}}]+\}})'),
            (r'\*\*\u5de5\u5177\u540d\u79f0\*\*[：:]*\s*[`"]?(\w+)[`"]?',
             r'\*\*\u53c2\u6570\*\*[：:]*\s*(`[^`]+`|\{{[^}}]+\}})'),
        ]

        for tool_pattern, param_pattern in structured_patterns:
            tool_match = re.search(tool_pattern, thought, re.IGNORECASE)
            if tool_match:
                tool_name = tool_match.group(1).lower().strip()
                param_match = re.search(param_pattern, thought, re.IGNORECASE | re.DOTALL)

                if param_match:
                    params_str = param_match.group(1).strip().strip('`')
                    params = self._try_parse_json(params_str)
                    if params:
                        return {"tool": tool_name, "params": params}

        # ===== 策略3: 宽松搜索 (Fallback) =====
        return self._fallback_action_parse(thought)

    def _extract_function_params(self, tool_name: str, params_raw: str) -> Optional[Dict]:
        """从函数式参数字符串中提取参数"""
        # 尝试直接解析为 JSON
        params = self._try_parse_json(params_raw)
        if params:
            return params

        # 尝试解析 key=value 格�