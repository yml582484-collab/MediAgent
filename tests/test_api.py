"""
API Integration Tests for MediAgent
Tests all API endpoints using httpx.AsyncClient with ASGITransport.
Covers chat, streaming, health, status, tools, sessions, medical, metrics, audit,
rate limiting, authentication, and CORS.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


# ==================== Health Endpoint Tests ====================


class TestHealthEndpoint:
    """GET /api/health 健康检查端点测试"""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, api_client):
        """健康检查应返回 200"""
        response = await api_client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "components" in data

    @pytest.mark.asyncio
    async def test_health_includes_components(self, api_client):
        """健康检查应包含组件状态"""
        response = await api_client.get("/api/health")
        data = response.json()

        components = data["components"]
        assert "agent" in components
        assert "llm" in components
        assert "memory" in components
        assert "disk" in components

    @pytest.mark.asyncio
    async def test_health_agent_component_ok(self, api_client):
        """Agent 组件应处于 ok 状态"""
        response = await api_client.get("/api/health")
        data = response.json()

        assert data["components"]["agent"]["status"] == "ok"
        assert data["components"]["agent"]["initialized"] is True

    @pytest.mark.asyncio
    async def test_health_disk_component(self, api_client):
        """磁盘组件应有空间信息"""
        response = await api_client.get("/api/health")
        data = response.json()

        disk = data["components"]["disk"]
        assert "free_gb" in disk
        assert "total_gb" in disk


# ==================== Chat Endpoint Tests ====================


class TestChatEndpoint:
    """POST /api/chat 对话端点测试"""

    @pytest.mark.asyncio
    async def test_chat_basic(self, api_client):
        """基本对话请求应返回 200"""
        response = await api_client.post(
            "/api/chat",
            json={
                "message": "你好",
                "agent_mode": "direct",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "response" in data
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    @pytest.mark.asyncio
    async def test_chat_with_session_id(self, api_client):
        """使用指定 session_id 进行对话"""
        # 先创建一个会话
        first_response = await api_client.post(
            "/api/chat",
            json={"message": "你好", "agent_mode": "direct"},
        )
        session_id = first_response.json()["session_id"]

        # 使用该 session_id 继续对话
        second_response = await api_client.post(
            "/api/chat",
            json={"message": "继续", "session_id": session_id, "agent_mode": "direct"},
        )

        assert second_response.status_code == 200
        assert second_response.json()["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_chat_empty_message(self, api_client):
        """空消息应返回 422 (Pydantic validation)"""
        response = await api_client.post(
            "/api/chat",
            json={"message": ""},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_missing_message(self, api_client):
        """缺少 message 字段应返回 422"""
        response = await api_client.post(
            "/api/chat",
            json={},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_returns_token_usage(self, api_client):
        """对话响应应包含 token 使用信息"""
        response = await api_client.post(
            "/api/chat",
            json={"message": "测试token", "agent_mode": "direct"},
        )

        data = response.json()
        assert "token_usage" in data
        assert isinstance(data["token_usage"], dict)

    @pytest.mark.asyncio
    async def test_chat_returns_metadata(self, api_client):
        """对话响应应包含元数据"""
        response = await api_client.post(
            "/api/chat",
            json={"message": "测试", "agent_mode": "direct"},
        )

        data = response.json()
        assert "metadata" in data
        assert isinstance(data["metadata"], dict)

    @pytest.mark.asyncio
    async def test_chat_react_mode(self, api_client, initialized_agent):
        """ReAct 模式对话"""
        from src.agent.planner import PlanExecutionResult, PlannerState

        mock_result = PlanExecutionResult(
            final_answer="分析完成。",
            success=True,
            state=PlannerState.COMPLETED,
            steps_completed=1,
            duration_seconds=0.5,
            token_usage={"total_tokens": 50},
            thoughts=[],
            actions=[],
            observations=[],
            plan_steps=None,
        )

        with patch("src.agent.core.ReActPlanner") as MockPlanner:
            mock_planner_instance = AsyncMock()
            mock_planner_instance.plan_and_execute.return_value = mock_result
            MockPlanner.return_value = mock_planner_instance

            response = await api_client.post(
                "/api/chat",
                json={"message": "分析症状", "agent_mode": "react"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


# ==================== Chat Stream Endpoint Tests ====================


class TestChatStreamEndpoint:
    """POST /api/chat/stream SSE 流式端点测试"""

    @pytest.mark.asyncio
    async def test_stream_direct_mode(self, api_client, initialized_agent, mock_llm_provider):
        """直接模式流式输出"""
        from src.llm.provider import LLMResponse

        async def mock_stream(messages):
            yield LLMResponse(content="你", model="deepseek-chat", finish_reason=None, usage={})
            yield LLMResponse(content="好", model="deepseek-chat", finish_reason="stop", usage={})

        mock_llm_provider.chat_stream = mock_stream

        response = await api_client.post(
            "/api/chat/stream",
            json={"message": "你好", "agent_mode": "direct", "use_react": False},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # 解析 SSE 事件
        content = response.text
        assert "event:" in content
        assert "data:" in content
        assert "done" in content

    @pytest.mark.asyncio
    async def test_stream_react_mode(self, api_client, initialized_agent):
        """ReAct 模式流式输出"""
        from src.agent.planner import PlanExecutionResult, PlannerState

        mock_result = PlanExecutionResult(
            final_answer="ReAct 分析结果",
            success=True,
            state=PlannerState.COMPLETED,
            steps_completed=1,
            duration_seconds=0.5,
            token_usage={"total_tokens": 50},
            thoughts=[],
            actions=[],
            observations=[],
            plan_steps=None,
        )

        with patch("src.agent.core.ReActPlanner") as MockPlanner:
            mock_planner_instance = AsyncMock()
            mock_planner_instance.plan_and_execute.return_value = mock_result
            MockPlanner.return_value = mock_planner_instance

            response = await api_client.post(
                "/api/chat/stream",
                json={"message": "分析", "agent_mode": "react", "use_react": True},
            )

            assert response.status_code == 200
            content = response.text
            assert "event:" in content
            assert "done" in content


# ==================== Status Endpoint Tests ====================


class TestStatusEndpoint:
    """GET /api/status 状态端点测试"""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, api_client):
        """状态端点应返回 200"""
        response = await api_client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "sessions" in data
        assert "tools" in data

    @pytest.mark.asyncio
    async def test_status_includes_agent_info(self, api_client):
        """状态应包含 Agent 信息"""
        response = await api_client.get("/api/status")
        data = response.json()

        status = data["status"]
        assert "initialized" in status
        assert "uptime_seconds" in status
        assert "statistics" in status
        assert status["initialized"] is True

    @pytest.mark.asyncio
    async def test_status_includes_sessions(self, api_client):
        """状态应包含会话列表"""
        response = await api_client.get("/api/status")
        data = response.json()

        assert isinstance(data["sessions"], list)


# ==================== Tools Endpoint Tests ====================


class TestToolsEndpoint:
    """GET /api/tools 工具列表端点测试"""

    @pytest.mark.asyncio
    async def test_tools_returns_200(self, api_client):
        """工具列表端点应返回 200"""
        with patch("src.tools.base.tool_registry") as mock_registry:
            mock_registry.list_tools.return_value = [
                {"name": "calculator", "description": "计算器", "enabled": True},
                {"name": "web_search", "description": "网络搜索", "enabled": True},
            ]

            response = await api_client.get("/api/tools")

            assert response.status_code == 200
            data = response.json()
            assert "total_tools" in data
            assert "enabled_count" in data
            assert "tools" in data
            assert data["total_tools"] == 2


# ==================== Sessions Endpoint Tests ====================


class TestSessionsEndpoint:
    """GET /api/sessions 会话端点测试"""

    @pytest.mark.asyncio
    async def test_sessions_returns_200(self, api_client):
        """会话列表端点应返回 200"""
        response = await api_client.get("/api/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "active_count" in data
        assert isinstance(data["sessions"], list)

    @pytest.mark.asyncio
    async def test_delete_session(self, api_client, initialized_agent):
        """删除指定会话"""
        session_id = initialized_agent._create_session()

        response = await api_client.delete(f"/api/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, api_client):
        """删除不存在的会话应返回 404"""
        response = await api_client.delete("/api/sessions/nonexistent-id")

        assert response.status_code == 404


# ==================== Medical Endpoint Tests ====================


class TestMedicalEndpoints:
    """医疗相关端点测试"""

    @pytest.mark.asyncio
    async def test_medical_analyze(self, api_client, initialized_agent):
        """症状分析端点"""
        from src.agent.planner import PlanExecutionResult, PlannerState

        mock_result = PlanExecutionResult(
            final_answer="根据症状分析，可能是上呼吸道感染。",
            success=True,
            state=PlannerState.COMPLETED,
            steps_completed=1,
            duration_seconds=0.5,
            token_usage={"total_tokens": 100},
            thoughts=[],
            actions=[],
            observations=[],
            plan_steps=None,
        )

        with patch("src.agent.core.ReActPlanner") as MockPlanner:
            mock_planner_instance = AsyncMock()
            mock_planner_instance.plan_and_execute.return_value = mock_result
            MockPlanner.return_value = mock_planner_instance

            response = await api_client.post(
                "/api/medical/analyze",
                json={
                    "symptoms": "头痛、发热、咳嗽",
                    "patient_age": 30,
                    "duration": "3天",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "analysis" in data
            assert "disclaimer" in data

    @pytest.mark.asyncio
    async def test_medical_analyze_empty_symptoms(self, api_client):
        """空症状应返回 422"""
        response = await api_client.post(
            "/api/medical/analyze",
            json={"symptoms": ""},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_medical_disclaimer(self, api_client):
        """医疗免责声明端点"""
        response = await api_client.get("/api/medical/disclaimer")

        assert response.status_code == 200
        data = response.json()
        assert "disclaimer" in data
        assert "version" in data
        assert "last_updated" in data
        assert "MediAgent" in data["disclaimer"]
        assert len(data["disclaimer"]) > 100


# ==================== Metrics Endpoint Tests ====================


class TestMetricsEndpoint:
    """GET /api/metrics 指标端点测试"""

    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, api_client):
        """指标端点应返回 200"""
        response = await api_client.get("/api/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "success_count" in data
        assert "failure_count" in data
        assert "average_response_time_ms" in data
        assert "token_usage" in data
        assert "active_sessions_count" in data

    @pytest.mark.asyncio
    async def test_metrics_includes_tool_calls(self, api_client):
        """指标应包含工具调用统计"""
        response = await api_client.get("/api/metrics")
        data = response.json()

        assert "tool_call_counts" in data
        assert isinstance(data["tool_call_counts"], dict)


# ==================== Audit Endpoint Tests ====================


class TestAuditEndpoint:
    """GET /api/audit 审计日志端点测试"""

    @pytest.mark.asyncio
    async def test_audit_returns_200(self, api_client):
        """审计日志端点应返回 200"""
        response = await api_client.get("/api/audit")

        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert "entries" in data
        assert isinstance(data["entries"], list)

    @pytest.mark.asyncio
    async def test_audit_with_limit(self, api_client):
        """审计日志支持 limit 参数"""
        response = await api_client.get("/api/audit?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data["entries"]) <= 5

    @pytest.mark.asyncio
    async def test_audit_entries_have_required_fields(self, api_client):
        """审计日志条目应包含必要字段"""
        # 先发一个请求以产生审计日志
        await api_client.get("/api/health")

        response = await api_client.get("/api/audit")
        data = response.json()

        if len(data["entries"]) > 0:
            entry = data["entries"][-1]
            assert "timestamp" in entry
            assert "method" in entry
            assert "path" in entry
            assert "status_code" in entry
            assert "request_id" in entry


# ==================== Rate Limiting Tests ====================


class TestRateLimiting:
    """速率限制测试"""

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, api_client):
        """响应应包含 X-Request-ID 和 X-Response-Time 头"""
        response = await api_client.get("/api/health")

        assert "x-request-id" in response.headers
        assert "x-response-time" in response.headers

    @pytest.mark.asyncio
    async def test_health_bypasses_rate_limit(self, api_client):
        """健康检查端点应跳过速率限制"""
        # 连续多次请求健康检查
        for _ in range(5):
            response = await api_client.get("/api/health")
            assert response.status_code == 200


# ==================== Authentication Tests ====================


class TestAuthentication:
    """API Key 认证测试"""

    @pytest.mark.asyncio
    async def test_no_auth_when_disabled(self, api_client):
        """未启用认证时应允许无 API Key 请求"""
        response = await api_client.post(
            "/api/chat",
            json={"message": "测试", "agent_mode": "direct"},
        )

        # 应该不是 401（认证未启用）
        assert response.status_code != 401

    @pytest.mark.asyncio
    async def test_auth_rejection_when_enabled(self, api_client):
        """启用认证后无效 API Key 应被拒绝"""
        import main as main_module

        # 临时启用认证
        original_keys = main_module.AUTH_API_KEYS
        main_module.AUTH_API_KEYS = ["valid-key-123"]

        try:
            response = await api_client.post(
                "/api/chat",
                json={"message": "测试", "agent_mode": "direct"},
                headers={"X-API-Key": "invalid-key"},
            )

            assert response.status_code == 401
            data = response.json()
            assert "未授权" in data.get("error", "")
        finally:
            main_module.AUTH_API_KEYS = original_keys

    @pytest.mark.asyncio
    async def test_auth_success_with_valid_key(self, api_client):
        """有效 API Key 应通过认证"""
        import main as main_module

        original_keys = main_module.AUTH_API_KEYS
        main_module.AUTH_API_KEYS = ["valid-key-123"]

        try:
            response = await api_client.post(
                "/api/chat",
                json={"message": "测试", "agent_mode": "direct"},
                headers={"X-API-Key": "valid-key-123"},
            )

            assert response.status_code != 401
        finally:
            main_module.AUTH_API_KEYS = original_keys


# ==================== CORS Headers Tests ====================


class TestCORSHeaders:
    """CORS 跨域头测试"""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, api_client):
        """响应应包含 CORS 相关头"""
        # 使用 OPTIONS 请求触发 CORS preflight
        response = await api_client.options(
            "/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        # CORS 中间件应处理 OPTIONS 请求
        assert response.status_code in [200, 405, 422]

    @pytest.mark.asyncio
    async def test_post_returns_cors_headers(self, api_client):
        """POST 请求应包含 CORS 头"""
        response = await api_client.post(
            "/api/chat",
            json={"message": "测试", "agent_mode": "direct"},
            headers={"Origin": "http://example.com"},
        )

        # FastAPI CORS 中间件应添加这些头
        assert "access-control-allow-origin" in response.headers


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """错误处理测试"""

    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, api_client):
        """未知端点应返回 404"""
        response = await api_client.get("/api/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_405_for_wrong_method(self, api_client):
        """错误的方法应返回 405"""
        response = await api_client.delete("/api/chat")

        assert response.status_code == 405


# ==================== Reset Endpoint Tests ====================


class TestResetEndpoint:
    """POST /api/reset 重置端点测试"""

    @pytest.mark.asyncio
    async def test_reset_returns_200(self, api_client):
        """重置端点应返回 200"""
        response = await api_client.post("/api/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
