"""
Comprehensive Tests for Agent Core
Tests agent initialization, chat, process (react/direct/plan_and_execute),
session management, emergency detection, compliance, streaming, audit logging.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime
import asyncio

from conftest import SRC_DIR


# ==================== Agent Initialization Tests ====================


class TestAgentInitialization:
    """Agent 初始化相关测试"""

    @pytest.mark.asyncio
    async def test_agent_creation_without_init(self):
        """未初始化的 Agent 实例应处于未初始化状态"""
        from src.agent.core import Agent

        agent = Agent(auto_initialize=False)

        assert agent is not None
        assert agent._initialized is False
        assert agent._total_requests == 0
        assert len(agent._sessions) == 0

    @pytest.mark.asyncio
    async def test_agent_initialization_with_mocks(self, mock_llm_provider):
        """使用 mock LLM 初始化 Agent"""
        from src.agent.core import Agent

        with patch("src.agent.core.DeepSeekProvider", return_value=mock_llm_provider), \
             patch("src.agent.core.LongTermMemory") as MockMemory, \
             patch("src.agent.core.tool_registry") as mock_registry:

            mock_memory_instance = AsyncMock()
            mock_memory_instance.initialize.return_value = None
            mock_memory_instance.count = 0
            MockMemory.return_value = mock_memory_instance

            mock_registry.get_all_tools.return_value = []
            mock_registry.get_all_schemas.return_value = []
            mock_registry.get_tools_description.return_value = ""
            mock_registry.count = 0

            agent = Agent(auto_initialize=False)
            await agent.initialize()

            assert agent._initialized is True
            assert agent._llm is not None
            assert agent._current_session_id is not None
            assert len(agent._sessions) == 1

            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_double_initialization(self, mock_llm_provider):
        """重复初始化应被跳过"""
        from src.agent.core import Agent

        with patch("src.agent.core.DeepSeekProvider", return_value=mock_llm_provider), \
             patch("src.agent.core.LongTermMemory") as MockMemory, \
             patch("src.agent.core.tool_registry") as mock_registry:

            MockMemory.return_value = AsyncMock(initialize=AsyncMock(), count=0)
            mock_registry.count = 0

            agent = Agent(auto_initialize=False)
            await agent.initialize()
            await agent.initialize()  # 第二次调用应被跳过

            assert agent._initialized is True
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_async_context_manager(self, mock_llm_provider):
        """测试 async with 上下文管理器"""
        from src.agent.core import Agent

        with patch("src.agent.core.DeepSeekProvider", return_value=mock_llm_provider), \
             patch("src.agent.core.LongTermMemory") as MockMemory, \
             patch("src.agent.core.tool_registry") as mock_registry:

            MockMemory.return_value = AsyncMock(initialize=AsyncMock(), count=0)
            mock_registry.count = 0

            async with Agent(auto_initialize=False) as agent:
                assert agent._initialized is True
                status = agent.get_status()
                assert status["initialized"] is True

            assert agent._initialized is False


# ==================== Agent Chat Tests ====================


class TestAgentChat:
    """Agent.chat() 简单对话模式测试"""

    @pytest.mark.asyncio
    async def test_chat_returns_response(self, initialized_agent):
        """chat() 应返回 AgentResponse 对象"""
        response = await initialized_agent.chat("你好")

        assert response.success is True
        assert isinstance(response.response, str)
        assert response.session_id is not None
        assert response.request_id is not None
        assert response.token_usage is not None

    @pytest.mark.asyncio
    async def test_chat_increments_request_count(self, initialized_agent):
        """每次 chat 应增加请求计数"""
        count_before = initialized_agent._total_requests
        await initialized_agent.chat("测试消息")
        assert initialized_agent._total_requests == count_before + 1

    @pytest.mark.asyncio
    async def test_chat_stores_messages_in_memory(self, initialized_agent):
        """chat 应将消息存储到会话短期记忆"""
        await initialized_agent.chat("你好")
        session = initialized_agent._sessions[initialized_agent._current_session_id]
        assert session.short_memory.size >= 1

    @pytest.mark.asyncio
    async def test_chat_with_specific_session(self, initialized_agent):
        """使用指定 session_id 进行 chat"""
        session_id = initialized_agent._create_session()
        response = await initialized_agent.chat("你好", session_id=session_id)
        assert response.session_id == session_id

    @pytest.mark.asyncio
    async def test_chat_emergency_detection(self, initialized_agent):
        """chat 应检测紧急医疗情况"""
        response = await initialized_agent.chat("我胸痛呼吸困难")
        assert response.success is True
        assert response.metadata.get("is_emergency") is True
        assert "紧急" in response.response or "120" in response.response

    @pytest.mark.asyncio
    async def test_chat_compliance_check(self, initialized_agent):
        """chat 应进行合规检查"""
        response = await initialized_agent.chat("我身份证号是110101199001011234，头痛")
        assert response.success is True
        # 输入应被脱敏，不含完整身份证号
        assert "110101199001011234" not in response.response

    @pytest.mark.asyncio
    async def test_chat_error_handling(self, initialized_agent):
        """chat 应处理 LLM 调用异常"""
        initialized_agent._llm.chat_with_retry.side_effect = Exception("LLM 调用失败")
        response = await initialized_agent.chat("测试错误")
        assert response.success is False
        assert "错误" in response.response


# ==================== Agent Process Tests ====================


class TestAgentProcess:
    """Agent.process() 多模式处理测试"""

    @pytest.mark.asyncio
    async def test_process_react_mode(self, initialized_agent, mock_llm_provider):
        """process() react 模式应调用 ReActPlanner"""
        from src.agent.planner import PlanExecutionResult, PlannerState

        mock_result = PlanExecutionResult(
            final_answer="根据分析，您可能患有感冒。",
            success=True,
            state=PlannerState.COMPLETED,
            steps_completed=2,
            duration_seconds=1.0,
            token_usage={"total_tokens": 50},
            thoughts=[MagicMock(content="思考过程")],
            actions=[MagicMock(tool_name="symptom_analyzer")],
            observations=[MagicMock(success=True)],
            plan_steps=None,
        )

        with patch("src.agent.core.ReActPlanner") as MockPlanner:
            mock_planner_instance = AsyncMock()
            mock_planner_instance.plan_and_execute.return_value = mock_result
            MockPlanner.return_value = mock_planner_instance

            response = await initialized_agent.process(
                "我头痛发热",
                mode="react",
            )

            assert response.success is True
            assert response.metadata.get("mode") == "react"
            mock_planner_instance.plan_and_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_direct_mode(self, initialized_agent):
        """process() direct 模式应直接调用 LLM"""
        response = await initialized_agent.process(
            "你好",
            mode="direct",
        )

        assert response.success is True
        assert response.metadata.get("mode") == "direct"

    @pytest.mark.asyncio
    async def test_process_plan_and_execute_mode(self, initialized_agent, mock_llm_provider):
        """process() plan_and_execute 模式应调用 PlanAndExecutePlanner"""
        from src.agent.planner import PlanExecutionResult, PlannerState

        mock_result = PlanExecutionResult(
            final_answer="分析完成。",
            success=True,
            state=PlannerState.COMPLETED,
            steps_completed=3,
            duration_seconds=2.0,
            token_usage={"total_tokens": 100},
            thoughts=[MagicMock(content="规划思考")],
            actions=[MagicMock(tool_name="medical_knowledge")],
            observations=[MagicMock(success=True)],
            plan_steps=[MagicMock(description="步骤1", status="completed")],
        )

        with patch("src.agent.core.PlanAndExecutePlanner") as MockPlanner:
            mock_planner_instance = AsyncMock()
            mock_planner_instance.plan_and_execute.return_value = mock_result
            MockPlanner.return_value = mock_planner_instance

            response = await initialized_agent.process(
                "全面分析我的症状",
                mode="plan_and_execute",
            )

            assert response.success is True
            assert response.metadata.get("mode") == "plan_and_execute"

    @pytest.mark.asyncio
    async def test_process_emergency_detection(self, initialized_agent):
        """process() 应检测紧急医疗情况"""
        response = await initialized_agent.process("我剧烈头痛，意识丧失")
        assert response.success is True
        assert response.metadata.get("is_emergency") is True

    @pytest.mark.asyncio
    async def test_process_invalid_mode(self, initialized_agent):
        """process() 无效模式应抛出异常"""
        with pytest.raises(ValueError):
            await initialized_agent.process("测试", mode="invalid_mode")


# ==================== Session Management Tests ====================


class TestSessionManagement:
    """会话管理测试"""

    @pytest.mark.asyncio
    async def test_create_session(self, initialized_agent):
        """创建新会话"""
        session_id = initialized_agent._create_session()
        assert session_id is not None
        assert session_id in initialized_agent._sessions

    @pytest.mark.asyncio
    async def test_list_sessions(self, initialized_agent):
        """列出所有会话"""
        initialized_agent._create_session()
        initialized_agent._create_session()
        sessions = initialized_agent.list_sessions()
        assert len(sessions) >= 3  # 默认会话 + 2 个新会话

    @pytest.mark.asyncio
    async def test_clear_session(self, initialized_agent):
        """清除指定会话"""
        session_id = initialized_agent._create_session()
        success = initialized_agent.clear_session(session_id)
        assert success is True
        assert session_id not in initialized_agent._sessions

    @pytest.mark.asyncio
    async def test_clear_nonexistent_session(self, initialized_agent):
        """清除不存在的会话应返回 False"""
        success = initialized_agent.clear_session("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_session_isolation(self, initialized_agent, mock_llm_provider):
        """两个会话不应共享记忆"""
        session_a = initialized_agent._create_session()
        session_b = initialized_agent._create_session()

        # 在 session_a 中添加消息
        await initialized_agent.chat("这是会话A的消息", session_id=session_a)

        # session_b 应为空
        assert initialized_agent._sessions[session_b].short_memory.size == 0

    @pytest.mark.asyncio
    async def test_set_session_mode(self, initialized_agent):
        """设置会话模式"""
        session_id = initialized_agent._create_session()
        success = initialized_agent.set_session_mode("direct", session_id=session_id)
        assert success is True
        assert initialized_agent._sessions[session_id].mode.value == "direct"

    @pytest.mark.asyncio
    async def test_set_session_mode_invalid(self, initialized_agent):
        """设置无效模式应返回 False"""
        session_id = initialized_agent._create_session()
        success = initialized_agent.set_session_mode("invalid", session_id=session_id)
        assert success is False


# ==================== Agent Lifecycle Tests ====================


class TestAgentLifecycle:
    """Agent 生命周期测试"""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, initialized_agent):
        """reset 应清除所有状态"""
        initialized_agent._total_requests = 10
        initialized_agent._create_session()
        initialized_agent._create_session()

        await initialized_agent.reset()

        assert initialized_agent._total_requests == 0
        assert len(initialized_agent._sessions) == 1  # 一个新的默认会话
        assert len(initialized_agent._audit_logs) == 0

    @pytest.mark.asyncio
    async def test_close_sets_initialized_false(self, initialized_agent):
        """close 应将 _initialized 设为 False"""
        assert initialized_agent._initialized is True
        await initialized_agent.close()
        assert initialized_agent._initialized is False

    @pytest.mark.asyncio
    async def test_get_status(self, initialized_agent):
        """get_status 应返回完整的状态信息"""
        status = initialized_agent.get_status()

        assert "initialized" in status
        assert "uptime_seconds" in status
        assert "config" in status
        assert "statistics" in status
        assert "memory" in status
        assert "llm_stats" in status
        assert status["initialized"] is True

    @pytest.mark.asyncio
    async def test_get_session_memory(self, initialized_agent):
        """get_session_memory 应返回会话记忆信息"""
        await initialized_agent.chat("测试消息")
        memory = initialized_agent.get_session_memory()

        assert memory is not None
        assert "session_id" in memory
        assert "short_term" in memory
        assert "working_memory" in memory
        assert "message_count" in memory

    @pytest.mark.asyncio
    async def test_get_session_memory_nonexistent(self, initialized_agent):
        """获取不存在的会话记忆应返回 None"""
        memory = initialized_agent.get_session_memory("nonexistent-id")
        assert memory is None


# ==================== Chat Stream Tests ====================


class TestAgentChatStream:
    """Agent.chat_stream() 流式响应测试"""

    @pytest.mark.asyncio
    async def test_chat_stream_yields_content(self, initialized_agent, mock_llm_provider):
        """chat_stream 应 yield 内容片段"""
        from src.llm.provider import LLMResponse

        # 配置 mock 流式响应
        async def mock_stream(messages):
            chunks = [
                LLMResponse(content="你", model="deepseek-chat", finish_reason=None, usage={}),
                LLMResponse(content="好", model="deepseek-chat", finish_reason=None, usage={}),
                LLMResponse(content="！", model="deepseek-chat", finish_reason="stop", usage={}),
            ]
            for chunk in chunks:
                yield chunk

        mock_llm_provider.chat_stream = mock_stream

        collected = []
        async for chunk in initialized_agent.chat_stream("你好"):
            collected.append(chunk)

        assert len(collected) == 3
        assert "".join(collected) == "你好！"

    @pytest.mark.asyncio
    async def test_chat_stream_stores_response(self, initialized_agent, mock_llm_provider):
        """chat_stream 应在流结束后存储完整响应到短期记忆"""
        from src.llm.provider import LLMResponse

        async def mock_stream(messages):
            yield LLMResponse(content="完整回复", model="deepseek-chat", finish_reason="stop", usage={})

        mock_llm_provider.chat_stream = mock_stream

        async for _ in initialized_agent.chat_stream("测试"):
            pass

        session = initialized_agent._sessions[initialized_agent._current_session_id]
        # 应有用户消息 + 助手回复
        assert session.short_memory.size >= 1


# ==================== Audit Logging Tests ====================


class TestAuditLogging:
    """审计日志测试"""

    @pytest.mark.asyncio
    async def test_chat_creates_audit_log(self, initialized_agent):
        """chat 应创建审计日志"""
        await initialized_agent.chat("测试审计日志")
        assert len(initialized_agent._audit_logs) >= 1

        log = initialized_agent._audit_logs[0]
        assert log.request_id is not None
        assert log.session_id is not None
        assert log.mode == "chat"
        assert log.success is True

    @pytest.mark.asyncio
    async def test_get_audit_logs(self, initialized_agent):
        """get_audit_logs 应返回审计日志列表"""
        await initialized_agent.chat("测试1")
        await initialized_agent.chat("测试2")

        logs = initialized_agent.get_audit_logs(limit=10)
        assert len(logs) >= 2
        assert all("request_id" in log for log in logs)

    @pytest.mark.asyncio
    async def test_audit_log_max_entries(self, initialized_agent):
        """审计日志不应超过最大条数"""
        original_max = initialized_agent._max_audit_logs
        initialized_agent._max_audit_logs = 5

        for i in range(10):
            await initialized_agent.chat(f"消息 {i}")

        assert len(initialized_agent._audit_logs) <= 5
        initialized_agent._max_audit_logs = original_max


# ==================== Request ID Tests ====================


class TestRequestId:
    """请求 ID 生成测试"""

    @pytest.mark.asyncio
    async def test_request_id_format(self, initialized_agent):
        """request_id 应以 'req_' 开头"""
        response = await initialized_agent.chat("测试")
        assert response.request_id is not None
        assert response.request_id.startswith("req_")

    @pytest.mark.asyncio
    async def test_request_id_unique(self, initialized_agent):
        """每次请求应生成唯一的 request_id"""
        ids = set()
        for _ in range(10):
            response = await initialized_agent.chat("测试唯一性")
            ids.add(response.request_id)

        assert len(ids) == 10


# ==================== Agent Repr Tests ====================


class TestAgentRepr:
    """Agent __repr__ 测试"""

    @pytest.mark.asyncio
    async def test_repr_uninitialized(self):
        """未初始化 Agent 的 repr"""
        from src.agent.core import Agent
        agent = Agent(auto_initialize=False)
        repr_str = repr(agent)
        assert "未初始化" in repr_str

    @pytest.mark.asyncio
    async def test_repr_initialized(self, initialized_agent):
        """已初始化 Agent 的 repr"""
        repr_str = repr(initialized_agent)
        assert "已初始化" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
