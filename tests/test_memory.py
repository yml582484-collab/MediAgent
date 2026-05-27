"""
Comprehensive Tests for Memory System
Tests ShortTermMemory, WorkingMemory, LongTermMemory, and per-session isolation.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# ==================== ShortTermMemory Tests ====================


class TestShortTermMemory:
    """ShortTermMemory 短期记忆测试"""

    def test_initialization(self):
        """初始化时应设置正确的参数"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10, max_tokens=5000)

        assert memory.window_size == 10
        assert memory.max_tokens == 5000
        assert memory.is_empty is True
        assert memory.size == 0

    def test_add_user_message(self):
        """添加用户消息"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_user_message("你好！")

        assert memory.is_empty is False
        assert memory.size == 1

    def test_add_assistant_message(self):
        """添加助手消息"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_user_message("你好")
        memory.add_assistant_message("你好！有什么可以帮您？")

        assert memory.size == 1  # 一个完整的 turn

    def test_add_assistant_without_user(self):
        """在没有用户消息时添加助手消息应自动创建 turn"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_assistant_message("自动回复")

        assert memory.size == 1

    def test_get_context(self):
        """获取对话上下文"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_user_message("什么是感冒？")
        memory.add_assistant_message("感冒是一种常见的呼吸道感染。")

        context = memory.get_context()

        assert len(context) == 2
        assert context[0].role == "user"
        assert context[0].content == "什么是感冒？"
        assert context[1].role == "assistant"
        assert "呼吸道感染" in context[1].content

    def test_get_recent(self):
        """获取最近 N 轮对话"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)

        for i in range(5):
            memory.add_user_message(f"问题{i}")
            memory.add_assistant_message(f"回答{i}")

        recent = memory.get_recent(n_turns=2)

        # 最近 2 轮 = 4 条消息
        assert len(recent) == 4
        assert recent[-1].content == "回答4"

    def test_sliding_window(self):
        """滑动窗口应丢弃最旧的对话"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=3)

        for i in range(5):
            memory.add_user_message(f"Q{i}")
            memory.add_assistant_message(f"A{i}")

        # 窗口大小为 3，不应超过 3 轮
        assert memory.size <= 3

    def test_clear(self):
        """清除所有记忆"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_user_message("测试")
        memory.add_assistant_message("回复")

        memory.clear()

        assert memory.is_empty is True
        assert memory.size == 0
        assert memory.token_count == 0

    def test_token_count(self):
        """token 计数应随消息增加"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10, max_tokens=100000)
        initial_count = memory.token_count

        memory.add_user_message("这是一条测试消息")
        memory.add_assistant_message("这是一条回复消息")

        assert memory.token_count > initial_count

    def test_token_compression(self):
        """当 token 超过阈值时应触发压缩"""
        from src.memory.short_term import ShortTermMemory

        # 设置很小的 max_tokens 以触发压缩
        memory = ShortTermMemory(window_size=100, max_tokens=50)

        # 添加大量消息
        for i in range(20):
            memory.add_user_message(f"这是一条很长的测试消息，用于测试token压缩功能，序号{i}")
            memory.add_assistant_message(f"这是一条很长的回复消息，用于测试token压缩功能，序号{i}")

        # 压缩后 token 数应减少
        assert memory.token_count <= memory.max_tokens or memory.size < 20

    def test_get_conversation_history(self):
        """获取格式化的对话历史字符串"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_user_message("你好")
        memory.add_assistant_message("你好！")

        history = memory.get_conversation_history()

        assert "用户" in history
        assert "助手" in history
        assert "你好" in history

    def test_get_conversation_history_empty(self):
        """空记忆的对话历史应返回默认文本"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        history = memory.get_conversation_history()

        assert "无历史记录" in history

    def test_to_dict(self):
        """to_dict 应返回正确的结构"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_user_message("测试")

        d = memory.to_dict()

        assert "window_size" in d
        assert "max_tokens" in d
        assert "current_size" in d
        assert "token_count" in d
        assert "turns" in d
        assert d["current_size"] == 1

    def test_multiple_messages_same_turn(self):
        """同一轮中连续添加两条用户消息应更新当前 turn"""
        from src.memory.short_term import ShortTermMemory

        memory = ShortTermMemory(window_size=10)
        memory.add_user_message("第一条")
        memory.add_user_message("第二条（更新）")

        # 应该只有一个 turn（用户消息被更新）
        context = memory.get_context()
        user_msgs = [m for m in context if m.role == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "第二条（更新）"


# ==================== WorkingMemory Tests ====================


class TestWorkingMemory:
    """WorkingMemory 工作记忆测试"""

    def test_initialization(self):
        """初始化时应为空"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()

        assert working.has_active_task is False
        assert working.current_task is None
        assert len(working.task_history) == 0

    def test_start_task(self):
        """启动任务"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        task = working.start_task("task_1", "分析数据", total_steps=3)

        assert working.has_active_task is True
        assert task.task_id == "task_1"
        assert task.description == "分析数据"
        assert task.status == "in_progress"
        assert task.total_steps == 3

    def test_complete_task(self):
        """完成任务"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "测试任务")
        completed = working.complete_task(success=True)

        assert completed.status == "completed"
        assert working.has_active_task is False
        assert len(working.task_history) == 1

    def test_complete_task_failure(self):
        """标记任务失败"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "失败任务")
        completed = working.complete_task(success=False)

        assert completed.status == "failed"

    def test_task_chaining(self):
        """任务链：完成一个任务后启动下一个"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()

        working.start_task("task_1", "第一个任务")
        working.complete_task(success=True)

        working.start_task("task_2", "第二个任务")
        assert working.current_task.task_id == "task_2"
        assert len(working.task_history) == 1

    def test_variable_storage(self):
        """变量存储和读取"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "测试")

        working.set_variable("result", 42)
        working.set_variable("name", "test")
        working.set_variable("data", {"key": "value"})

        assert working.get_variable("result") == 42
        assert working.get_variable("name") == "test"
        assert working.get_variable("data") == {"key": "value"}
        assert working.get_variable("nonexistent") is None

    def test_variable_without_task(self):
        """没有活跃任务时 get_variable 应返回 None 或全局变量"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        assert working.get_variable("any") is None

    def test_global_variable(self):
        """全局变量应跨任务持久化"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()

        working.start_task("task_1", "第一个任务")
        working.set_global_variable("config", "value1")
        working.complete_task(success=True)

        working.start_task("task_2", "第二个任务")
        assert working.get_variable("config") == "value1"

    def test_progress_tracking(self):
        """进度追踪"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "处理", total_steps=5)

        working.update_progress(step=2, total=5)

        assert working.current_task.current_step == 2
        assert working.current_task.total_steps == 5

    def test_error_recording(self):
        """错误记录"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "测试")

        working.add_error("Connection timeout")
        working.add_error("Invalid response format")

        assert len(working.current_task.errors) == 2
        assert "timeout" in working.current_task.errors[0]
        assert "Invalid" in working.current_task.errors[1]

    def test_add_result(self):
        """添加中间结果"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "测试")

        working.add_result({"analysis": "complete"})
        working.add_result({"score": 0.95})

        assert len(working.current_task.intermediate_results) == 2
        assert working.current_task.intermediate_results[0]["data"]["analysis"] == "complete"

    def test_clear(self):
        """清除工作记忆"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "测试")
        working.set_variable("x", 1)
        working.set_global_variable("g", 2)

        working.clear()

        assert working.has_active_task is False
        assert working.current_task is None
        assert len(working.task_history) == 0
        assert working.get_variable("g") is None

    def test_to_dict(self):
        """to_dict 应返回正确的结构"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "测试任务")

        d = working.to_dict()

        assert "current_task" in d
        assert "global_variables" in d
        assert "completed_tasks" in d
        assert d["current_task"]["task_id"] == "task_1"

    def test_repr_with_task(self):
        """有活跃任务时的 repr"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        working.start_task("task_1", "测试")

        repr_str = repr(working)
        assert "task_1" in repr_str
        assert "in_progress" in repr_str

    def test_repr_without_task(self):
        """无活跃任务时的 repr"""
        from src.memory.working_memory import WorkingMemory

        working = WorkingMemory()
        repr_str = repr(working)
        assert "no active task" in repr_str


# ==================== LongTermMemory Tests ====================


class TestLongTermMemory:
    """LongTermMemory 长期记忆测试（使用 mock ChromaDB）"""

    @pytest.mark.asyncio
    async def test_initialization_with_mock(self):
        """使用 mock 初始化长期记忆"""
        from src.memory.long_term import LongTermMemory

        with patch("src.memory.long_term.CHROMADB_AVAILABLE", True), \
             patch("src.memory.long_term.chromadb") as mock_chromadb:

            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            memory = LongTermMemory(
                persist_directory="./test_data",
                collection_name="test_collection",
            )
            await memory.initialize()

            assert memory.is_initialized is True
            assert memory.count == 0

            await memory.close()

    @pytest.mark.asyncio
    async def test_add_memory(self):
        """添加记忆"""
        from src.memory.long_term import LongTermMemory

        with patch("src.memory.long_term.CHROMADB_AVAILABLE", True), \
             patch("src.memory.long_term.chromadb") as mock_chromadb:

            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 1
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            memory = LongTermMemory()
            await memory.initialize()

            mem = await memory.add_memory(
                content="用户喜欢 Python 编程",
                memory_type="preference",
                importance="high",
            )

            assert mem.id is not None
            assert mem.content == "用户喜欢 Python 编程"
            assert mem.memory_type == "preference"
            mock_collection.upsert.assert_called_once()

            await memory.close()

    @pytest.mark.asyncio
    async def test_search_memory(self):
        """搜索记忆"""
        from src.memory.long_term import LongTermMemory

        with patch("src.memory.long_term.CHROMADB_AVAILABLE", True), \
             patch("src.memory.long_term.chromadb") as mock_chromadb:

            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 1
            mock_collection.query.return_value = {
                "documents": [["用户喜欢 Python 编程"]],
                "metadatas": [[{
                    "id": "abc123",
                    "memory_type": "preference",
                    "importance": "high",
                    "source": "conversation",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "metadata": {},
                }]],
                "distances": [[0.2]],  # similarity = 0.8
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            memory = LongTermMemory()
            await memory.initialize()

            results = await memory.search("编程语言")

            assert isinstance(results, list)
            assert len(results) >= 1
            assert results[0].content == "用户喜欢 Python 编程"

            await memory.close()

    @pytest.mark.asyncio
    async def test_delete_memory(self):
        """删除记忆"""
        from src.memory.long_term import LongTermMemory

        with patch("src.memory.long_term.CHROMADB_AVAILABLE", True), \
             patch("src.memory.long_term.chromadb") as mock_chromadb:

            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            memory = LongTermMemory()
            await memory.initialize()

            success = await memory.delete_memory("test_id")
            assert success is True
            mock_collection.delete.assert_called_once_with(ids=["test_id"])

            await memory.close()

    @pytest.mark.asyncio
    async def test_add_memories_batch(self):
        """批量添加记忆"""
        from src.memory.long_term import LongTermMemory

        with patch("src.memory.long_term.CHROMADB_AVAILABLE", True), \
             patch("src.memory.long_term.chromadb") as mock_chromadb:

            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 2
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            memory = LongTermMemory()
            await memory.initialize()

            memories_data = [
                {"content": "记忆1", "memory_type": "fact", "importance": "high"},
                {"content": "记忆2", "memory_type": "context", "importance": "low"},
            ]

            created = await memory.add_memories_batch(memories_data)

            assert len(created) == 2
            mock_collection.upsert.assert_called_once()

            await memory.close()

    @pytest.mark.asyncio
    async def test_close(self):
        """关闭记忆系统"""
        from src.memory.long_term import LongTermMemory

        with patch("src.memory.long_term.CHROMADB_AVAILABLE", True), \
             patch("src.memory.long_term.chromadb") as mock_chromadb:

            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            memory = LongTermMemory()
            await memory.initialize()
            assert memory.is_initialized is True

            await memory.close()
            assert memory.is_initialized is False
            assert memory._client is None

    @pytest.mark.asyncio
    async def test_chromadb_not_available(self):
        """ChromaDB 不可用时应抛出 ImportError"""
        from src.memory.long_term import LongTermMemory

        with patch("src.memory.long_term.CHROMADB_AVAILABLE", False):
            with pytest.raises(ImportError):
                LongTermMemory()


# ==================== Per-Session Memory Isolation Tests ====================


class TestPerSessionMemoryIsolation:
    """会话间记忆隔离测试"""

    def test_short_term_memory_isolation(self):
        """不同会话的短期记忆应完全隔离"""
        from src.memory.short_term import ShortTermMemory

        memory_a = ShortTermMemory(window_size=10)
        memory_b = ShortTermMemory(window_size=10)

        memory_a.add_user_message("会话A的消息")
        memory_b.add_user_message("会话B的消息")

        assert memory_a.size == 1
        assert memory_b.size == 1

        context_a = memory_a.get_context()
        context_b = memory_b.get_context()

        assert context_a[0].content == "会话A的消息"
        assert context_b[0].content == "会话B的消息"

    def test_working_memory_isolation(self):
        """不同会话的工作记忆应完全隔离"""
        from src.memory.working_memory import WorkingMemory

        working_a = WorkingMemory()
        working_b = WorkingMemory()

        working_a.start_task("task_a", "任务A")
        working_a.set_variable("x", 100)

        working_b.start_task("task_b", "任务B")
        working_b.set_variable("x", 200)

        assert working_a.get_variable("x") == 100
        assert working_b.get_variable("x") == 200

    def test_clear_one_session_does_not_affect_other(self):
        """清除一个会话不应影响另一个会话"""
        from src.memory.short_term import ShortTermMemory

        memory_a = ShortTermMemory(window_size=10)
        memory_b = ShortTermMemory(window_size=10)

        memory_a.add_user_message("消息A")
        memory_b.add_user_message("消息B")

        memory_a.clear()

        assert memory_a.is_empty is True
        assert memory_b.is_empty is False
        assert memory_b.size == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
