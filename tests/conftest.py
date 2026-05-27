# Test Configuration and Fixtures
# MediAgent 智慧医疗助手 - 测试配置与公共 fixtures
#
# 使用 pytest-asyncio 的现代方式:
#   - scope="session" 的 event_loop fixture
#   - 所有异步测试使用 @pytest.mark.asyncio 装饰器
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
import yaml

# 确保 src 目录在 Python 路径中
SRC_DIR = str(Path(__file__).parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ==================== Event Loop (pytest-asyncio 现代写法) ====================

@pytest_asyncio.fixture(scope="session", loop_scope="session")
def event_loop():
    """
    创建 session 级别的 event loop，供所有异步测试共享。

    注意: pytest-asyncio >= 0.21 推荐使用 scope 参数而非旧的 event_loop fixture。
    这里保留以兼容旧版本，同时使用 loop_scope 参数适配新版本。
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==================== 配置相关 Fixtures ====================

@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """返回测试用的配置字典（不写入文件）"""
    return {
        "llm": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_base": "https://api.deepseek.com/v1",
            "api_key": "test-api-key-for-testing",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
            "retry": {"max_retries": 2, "retry_delay": 0.5},
        },
        "memory": {
            "short_term": {"window_size": 5, "max_tokens": 1000},
            "long_term": {
                "vector_db": "chromadb",
                "persist_directory": "./data/test_chromadb",
                "collection_name": "test_memories",
                "embedding_model": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.7,
                "max_results": 5,
            },
        },
        "tools": {
            "enabled": [
                "web_search", "calculator", "code_executor", "file_manager",
                "medical_knowledge", "drug_query", "symptom_analyzer",
            ],
            "web_search": {"engine": "duckduckgo", "max_results": 3},
            "code_executor": {"timeout": 10, "allowed_languages": ["python"]},
            "file_manager": {
                "base_path": "./workspace",
                "allowed_extensions": [".txt", ".md", ".py", ".json"],
            },
        },
        "agent": {
            "name": "Test MediAgent",
            "max_iterations": 3,
            "thinking_verbose": False,
            "safe_mode": True,
            "compliance_mode": True,
            "medical_disclaimer": True,
        },
        "server": {"host": "127.0.0.1", "port": 8888, "debug": False, "cors_origins": ["*"]},
        "logging": {"level": "WARNING", "format": "%(message)s", "file": "/dev/null"},
        "security": {
            "auth_api_keys": [],
            "rate_limit_per_minute": 100,
            "rate_limit_burst": 50,
            "allowed_origins": ["*"],
        },
        "observability": {
            "enable_metrics": True,
            "enable_audit_log": True,
            "audit_log_max_entries": 100,
        },
    }


@pytest.fixture
def test_config_yaml(tmp_path, sample_config_dict) -> str:
    """
    创建临时 config.yaml 文件并返回其路径。

    测试结束后由 tmp_path 自动清理。
    """
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_config_dict, f, allow_unicode=True)
    return str(config_file)


# ==================== LLM Provider Mock Fixtures ====================

@pytest.fixture
def mock_llm_provider():
    """
    创建模拟的 LLM Provider。

    返回 AsyncMock 实例，已配置:
    - chat_with_retry: 返回标准 LLMResponse
    - chat: 返回标准 LLMResponse（含 tool_calls 支持）
    - chat_stream: 异步生成器，yield 多个 LLMResponse
    - token_usage.to_dict: 返回 token 使用统计
    - stats: 返回调用统计
    - reset_stats: 空操作
    - close: 空操作
    """
    from src.llm.provider import LLMResponse

    provider = AsyncMock()

    # 默认 chat_with_retry 响应
    default_response = LLMResponse(
        content="这是一个测试回复。",
        model="deepseek-chat",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    provider.chat_with_retry.return_value = default_response

    # 默认 chat 响应（用于 ReAct planner 的 function calling）
    provider.chat.return_value = LLMResponse(
        content="",
        model="deepseek-chat",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        tool_calls=None,
    )

    # token_usage mock
    token_usage_mock = MagicMock()
    token_usage_mock.to_dict.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    provider.token_usage = token_usage_mock

    # stats
    provider.stats = {
        "total_calls": 0,
        "total_errors": 0,
        "success_rate": 100.0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    provider.reset_stats = MagicMock()
    provider.close = AsyncMock()

    return provider


@pytest.fixture
def mock_llm_response_factory():
    """
    工厂 fixture: 生成 LLMResponse 的辅助函数。

    用法:
        response = factory("回复内容", finish_reason="stop")
    """
    def _factory(
        content: str = "测试回复",
        finish_reason: str = "stop",
        tool_calls: list = None,
        usage: dict = None,
    ):
        from src.llm.provider import LLMResponse
        return LLMResponse(
            content=content,
            model="deepseek-chat",
            finish_reason=finish_reason,
            usage=usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            tool_calls=tool_calls,
        )
    return _factory


# ==================== Tool Registry Mock Fixtures ====================

@pytest.fixture
def mock_tool_registry():
    """
    创建模拟的 ToolRegistry。

    返回 MagicMock 实例，已配置:
    - get_all_tools: 返回空列表
    - get_all_schemas: 返回空列表
    - get_tools_description: 返回描述字符串
    - count: 返回 0
    """
    from unittest.mock import MagicMock

    registry = MagicMock()
    registry.get_all_tools.return_value = []
    registry.get_all_schemas.return_value = []
    registry.get_tools_description.return_value = "No tools available"
    registry.count = 0
    registry.enable_tool.return_value = True
    registry.disable_tool.return_value = True
    return registry


# ==================== 消息 Fixtures ====================

@pytest.fixture
def sample_messages():
    """
    返回一组标准的测试消息列表。

    包含 system / user / assistant 三种角色。
    """
    from src.llm.provider import Message

    return [
        Message(role="system", content="你是一个医疗健康AI助手。"),
        Message(role="user", content="我最近头痛怎么办？"),
        Message(role="assistant", content="头痛的原因有很多，建议您先休息观察。"),
        Message(role="user", content="已经持续三天了，还有点发热。"),
    ]


# ==================== Agent Fixtures ====================

@pytest.fixture
async def initialized_agent(mock_llm_provider, tmp_path):
    """
    创建并初始化一个测试 Agent（使用 mock LLM）。

    自动 patch DeepSeekProvider 和 LongTermMemory，
    返回已初始化的 Agent 实例。
    测试结束后自动关闭。
    """
    from src.agent.core import Agent

    with patch("src.agent.core.DeepSeekProvider", return_value=mock_llm_provider), \
         patch("src.agent.core.LongTermMemory") as MockMemory, \
         patch("src.agent.core.tool_registry") as mock_registry:

        # 配置 mock memory
        mock_memory_instance = AsyncMock()
        mock_memory_instance.initialize.return_value = None
        mock_memory_instance.count = 0
        MockMemory.return_value = mock_memory_instance

        # 配置 mock registry
        mock_registry.get_all_tools.return_value = []
        mock_registry.get_all_schemas.return_value = []
        mock_registry.get_tools_description.return_value = ""
        mock_registry.count = 0

        agent = Agent(auto_initialize=False)
        await agent.initialize()

        yield agent

        if agent._initialized:
            await agent.close()


# ==================== FastAPI Test Client Fixture ====================

@pytest.fixture
async def api_client(initialized_agent):
    """
    创建 FastAPI 测试客户端 (httpx.AsyncClient + ASGITransport)。

    使用已初始化的 Agent 实例，自动 patch main.py 中的 agent_instance。
    """
    from fastapi.testclient import TestClient
    import httpx
    from main import app

    # 注入已初始化的 agent
    import main as main_module
    original_agent = main_module.agent_instance
    main_module.agent_instance = initialized_agent

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    # 恢复原始 agent
    main_module.agent_instance = original_agent


# ==================== Patch 辅助 ====================

from unittest.mock import patch as _patch


@pytest.fixture
def patch_module():
    """返回 patch 函数，方便在测试中使用。"""
    return _patch


# ==================== Compliance Checker Fixture ====================

@pytest.fixture
def compliance_checker():
    """创建一个合规检查器实例（使用默认配置）。"""
    from src.utils.compliance import MedicalComplianceChecker
    return MedicalComplianceChecker(compliance_mode=True, auto_disclaimer=True)
