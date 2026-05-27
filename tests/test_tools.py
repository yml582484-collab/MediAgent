"""
Comprehensive Tests for Tools System
Tests ToolRegistry, CalculatorTool, CodeExecutorTool, FileManagerTool,
WebSearchTool, MedicalKnowledgeTool, DrugQueryTool, SymptomAnalyzerTool,
and tool parameter validation.
"""
import os
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


# ==================== ToolRegistry Tests ====================


class TestToolRegistry:
    """ToolRegistry 工具注册中心测试"""

    def test_register_tool(self):
        """注册工具"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "测试工具"
        mock_tool.enabled = True

        registry.register(mock_tool)

        assert registry.count == 1
        assert "test_tool" in registry._tools

    def test_register_duplicate_tool(self):
        """注册重复工具应抛出异常"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "测试工具"
        mock_tool.enabled = True

        registry.register(mock_tool)

        with pytest.raises(ValueError, match="已注册"):
            registry.register(mock_tool)

    def test_enable_tool(self):
        """启用工具"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "测试工具"
        mock_tool.enabled = False

        registry.register(mock_tool)
        registry.enable_tool("test_tool")

        assert mock_tool.enabled is True

    def test_disable_tool(self):
        """禁用工具"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "测试工具"
        mock_tool.enabled = True

        registry.register(mock_tool)
        registry.disable_tool("test_tool")

        assert mock_tool.enabled is False

    def test_get_all_schemas(self):
        """获取所有工具的 JSON Schema"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "测试工具"
        mock_tool.enabled = True
        mock_tool.to_openai_schema.return_value = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "测试工具",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        registry.register(mock_tool)
        schemas = registry.get_all_schemas()

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "test_tool"

    def test_get_all_schemas_excludes_disabled(self):
        """禁用的工具不应出现在 schema 列表中"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()

        enabled_tool = MagicMock(spec=BaseTool)
        enabled_tool.name = "enabled_tool"
        enabled_tool.description = "启用"
        enabled_tool.enabled = True
        enabled_tool.to_openai_schema.return_value = {
            "type": "function",
            "function": {"name": "enabled_tool", "description": "启用", "parameters": {}},
        }

        disabled_tool = MagicMock(spec=BaseTool)
        disabled_tool.name = "disabled_tool"
        disabled_tool.description = "禁用"
        disabled_tool.enabled = False
        disabled_tool.to_openai_schema.return_value = {
            "type": "function",
            "function": {"name": "disabled_tool", "description": "禁用", "parameters": {}},
        }

        registry.register(enabled_tool)
        registry.register(disabled_tool)

        schemas = registry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "enabled_tool"

    def test_get_tools_description(self):
        """获取工具描述文本"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "calc"
        mock_tool.description = "计算器工具"
        mock_tool.enabled = True
        mock_tool.to_openai_schema.return_value = {
            "type": "function",
            "function": {"name": "calc", "description": "计算器工具", "parameters": {}},
        }

        registry.register(mock_tool)
        desc = registry.get_tools_description()

        assert "calc" in desc
        assert "计算器工具" in desc

    def test_get_tool(self):
        """获取指定工具"""
        from src.tools.base import ToolRegistry, BaseTool

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "my_tool"
        mock_tool.description = "我的工具"
        mock_tool.enabled = True

        registry.register(mock_tool)
        retrieved = registry.get_tool("my_tool")

        assert retrieved is mock_tool

    def test_get_nonexistent_tool(self):
        """获取不存在的工具应返回 None"""
        from src.tools.base import ToolRegistry

        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None


# ==================== CalculatorTool Tests ====================


class TestCalculatorTool:
    """CalculatorTool 计算器工具测试"""

    def test_basic_addition(self):
        """基本加法运算"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3")

        assert result.success is True
        assert result.data["result"] == 5.0

    def test_basic_subtraction(self):
        """基本减法运算"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="10 - 4")

        assert result.success is True
        assert result.data["result"] == 6.0

    def test_basic_multiplication(self):
        """基本乘法运算"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="6 * 7")

        assert result.success is True
        assert result.data["result"] == 42.0

    def test_basic_division(self):
        """基本除法运算"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="20 / 4")

        assert result.success is True
        assert result.data["result"] == 5.0

    def test_division_by_zero(self):
        """除以零应返回错误"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="10 / 0")

        assert result.success is False
        assert "除以零" in result.error or "error" in str(result.error).lower() or result.error is not None

    def test_complex_expression(self):
        """复杂表达式"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="(2 + 3) * 4 - 1")

        assert result.success is True
        assert result.data["result"] == 19.0

    def test_power_operation(self):
        """幂运算"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="2 ** 10")

        assert result.success is True
        assert result.data["result"] == 1024.0

    def test_percentage(self):
        """百分比运算"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="200 * 0.15")

        assert result.success is True
        assert result.data["result"] == 30.0

    def test_empty_expression(self):
        """空表达式应返回错误"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="")

        assert result.success is False

    def test_invalid_expression(self):
        """无效表达式应返回错误"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="abc + xyz")

        assert result.success is False

    def test_tool_schema(self):
        """工具应返回有效的 JSON Schema"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert "expression" in schema["function"]["parameters"]["properties"]


# ==================== CodeExecutorTool Tests ====================


class TestCodeExecutorTool:
    """CodeExecutorTool 代码执行工具测试"""

    def test_python_execution(self):
        """Python 代码执行"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="result = sum(range(1, 101))\nprint(result)",
            language="python",
        )

        assert result.success is True
        assert "5050" in result.data.get("stdout", "")

    def test_python_return_value(self):
        """Python 代码返回值"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="x = 2 + 2\nresult = x * 3",
            language="python",
        )

        assert result.success is True

    def test_python_syntax_error(self):
        """Python 语法错误"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="def broken(",
            language="python",
        )

        assert result.success is False
        assert result.error is not None

    def test_python_runtime_error(self):
        """Python 运行时错误"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="x = 1 / 0",
            language="python",
        )

        assert result.success is False
        assert result.error is not None

    def test_timeout(self):
        """执行超时"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=1)
        result = tool.execute(
            code="import time\ntime.sleep(10)",
            language="python",
        )

        assert result.success is False
        assert "超时" in result.error or "timeout" in str(result.error).lower()

    def test_security_violation_import_os(self):
        """安全违规: 导入 os 模块"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="import os\nos.system('echo hack')",
            language="python",
        )

        assert result.success is False
        assert result.error is not None

    def test_security_violation_import_subprocess(self):
        """安全违规: 导入 subprocess 模块"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="import subprocess\nsubprocess.run(['ls'])",
            language="python",
        )

        assert result.success is False

    def test_security_violation_exec(self):
        """安全违规: 使用 exec/eval"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="exec('print(1)')",
            language="python",
        )

        assert result.success is False

    def test_unsupported_language(self):
        """不支持的语言应返回错误"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(
            code="console.log('hello')",
            language="javascript",
        )

        assert result.success is False

    def test_tool_schema(self):
        """工具应返回有效的 JSON Schema"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "code_executor"
        assert "code" in schema["function"]["parameters"]["properties"]


# ==================== FileManagerTool Tests ====================


class TestFileManagerTool:
    """FileManagerTool 文件管理工具测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于文件操作测试"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def file_manager(self, temp_dir):
        """创建使用临时目录的 FileManagerTool"""
        from src.tools.file_manager import FileManagerTool

        tool = FileManagerTool(
            base_path=temp_dir,
            allowed_extensions=[".txt", ".md", ".py", ".json", ".csv"],
        )
        return tool

    def test_write_file(self, file_manager, temp_dir):
        """写入文件"""
        result = file_manager.execute(
            operation="write",
            filename="test.txt",
            content="Hello, MediAgent!",
        )

        assert result.success is True
        file_path = Path(temp_dir) / "test.txt"
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == "Hello, MediAgent!"

    def test_read_file(self, file_manager, temp_dir):
        """读取文件"""
        # 先写入
        test_file = Path(temp_dir) / "read_test.txt"
        test_file.write_text("测试内容", encoding="utf-8")

        result = file_manager.execute(
            operation="read",
            filename="read_test.txt",
        )

        assert result.success is True
        assert "测试内容" in result.data.get("content", "")

    def test_read_nonexistent_file(self, file_manager):
        """读取不存在的文件应返回错误"""
        result = file_manager.execute(
            operation="read",
            filename="nonexistent.txt",
        )

        assert result.success is False

    def test_list_files(self, file_manager, temp_dir):
        """列出文件"""
        # 创建一些文件
        (Path(temp_dir) / "a.txt").write_text("a", encoding="utf-8")
        (Path(temp_dir) / "b.md").write_text("b", encoding="utf-8")
        (Path(temp_dir) / "c.py").write_text("c", encoding="utf-8")

        result = file_manager.execute(operation="list")

        assert result.success is True
        files = result.data.get("files", [])
        assert len(files) >= 3

    def test_delete_file(self, file_manager, temp_dir):
        """删除文件"""
        test_file = Path(temp_dir) / "to_delete.txt"
        test_file.write_text("delete me", encoding="utf-8")

        result = file_manager.execute(
            operation="delete",
            filename="to_delete.txt",
        )

        assert result.success is True
        assert not test_file.exists()

    def test_disallowed_extension(self, file_manager):
        """不允许的文件扩展名应被拒绝"""
        result = file_manager.execute(
            operation="write",
            filename="malicious.exe",
            content="virus",
        )

        assert result.success is False

    def test_path_traversal_prevention(self, file_manager):
        """路径遍历攻击应被阻止"""
        result = file_manager.execute(
            operation="read",
            filename="../../etc/passwd",
        )

        assert result.success is False

    def test_tool_schema(self):
        """工具应返回有效的 JSON Schema"""
        from src.tools.file_manager import FileManagerTool

        tool = FileManagerTool(base_path="./workspace")
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "file_manager"
        assert "operation" in schema["function"]["parameters"]["properties"]


# ==================== WebSearchTool Tests ====================


class TestWebSearchTool:
    """WebSearchTool 网络搜索工具测试"""

    @pytest.mark.asyncio
    async def test_search_with_mock(self):
        """使用 mock 测试网络搜索"""
        from src.tools.web_search import WebSearchTool

        tool = WebSearchTool(engine="duckduckgo", max_results=3)

        with patch.object(tool, "_do_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {"title": "测试结果1", "url": "https://example.com/1", "snippet": "内容1"},
                {"title": "测试结果2", "url": "https://example.com/2", "snippet": "内容2"},
            ]

            result = await tool.execute_async(query="Python 教程")

            assert result.success is True
            assert len(result.data.get("results", [])) >= 1
            assert result.data["results"][0]["title"] == "测试结果1"

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """空搜索查询应返回错误"""
        from src.tools.web_search import WebSearchTool

        tool = WebSearchTool()

        result = await tool.execute_async(query="")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """无搜索结果"""
        from src.tools.web_search import WebSearchTool

        tool = WebSearchTool(engine="duckduckgo", max_results=3)

        with patch.object(tool, "_do_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            result = await tool.execute_async(query="xyznonexistent12345")

            assert result.success is True
            assert len(result.data.get("results", [])) == 0

    def test_tool_schema(self):
        """工具应返回有效的 JSON Schema"""
        from src.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "web_search"
        assert "query" in schema["function"]["parameters"]["properties"]


# ==================== MedicalKnowledgeTool Tests ====================


class TestMedicalKnowledgeTool:
    """MedicalKnowledgeTool 医学知识库工具测试"""

    @pytest.mark.asyncio
    async def test_search_medical_knowledge(self):
        """搜索医学知识"""
        from src.tools.medical_knowledge import MedicalKnowledgeTool

        tool = MedicalKnowledgeTool()

        with patch.object(tool, "_search_knowledge_base", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "title": "感冒",
                    "content": "感冒是一种常见的上呼吸道感染疾病。",
                    "source": "medical_db",
                    "relevance": 0.95,
                },
            ]

            result = await tool.execute_async(
                query="感冒的症状和治疗方法",
                category="disease",
            )

            assert result.success is True
            assert len(result.data.get("results", [])) >= 1
            assert "感冒" in result.data["results"][0]["title"]

    @pytest.mark.asyncio
    async def test_search_by_category(self):
        """按分类搜索"""
        from src.tools.medical_knowledge import MedicalKnowledgeTool

        tool = MedicalKnowledgeTool()

        with patch.object(tool, "_search_knowledge_base", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {"title": "阿司匹林", "content": "非甾体抗炎药", "source": "drug_db", "relevance": 0.9},
            ]

            result = await tool.execute_async(
                query="阿司匹林",
                category="drug",
            )

            assert result.success is True
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """空查询应返回错误"""
        from src.tools.medical_knowledge import MedicalKnowledgeTool

        tool = MedicalKnowledgeTool()
        result = await tool.execute_async(query="")

        assert result.success is False

    def test_tool_schema(self):
        """工具应返回有效的 JSON Schema"""
        from src.tools.medical_knowledge import MedicalKnowledgeTool

        tool = MedicalKnowledgeTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "medical_knowledge"
        assert "query" in schema["function"]["parameters"]["properties"]


# ==================== DrugQueryTool Tests ====================


class TestDrugQueryTool:
    """DrugQueryTool 药物查询工具测试"""

    @pytest.mark.asyncio
    async def test_drug_search(self):
        """搜索药物信息"""
        from src.tools.drug_query import DrugQueryTool

        tool = DrugQueryTool()

        with patch.object(tool, "_query_drug_database", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {
                "name": "布洛芬",
                "generic_name": "Ibuprofen",
                "category": "非甾体抗炎药",
                "indications": ["发热", "疼痛", "炎症"],
                "dosage": "成人每次200-400mg",
                "side_effects": ["胃肠道不适", "头晕"],
                "contraindications": ["消化性溃疡", "严重肝肾功能不全"],
                "interactions": ["华法林", "阿司匹林"],
            }

            result = await tool.execute_async(drug_name="布洛芬")

            assert result.success is True
            assert result.data["name"] == "布洛芬"
            assert "发热" in result.data["indications"]

    @pytest.mark.asyncio
    async def test_drug_not_found(self):
        """药物未找到"""
        from src.tools.drug_query import DrugQueryTool

        tool = DrugQueryTool()

        with patch.object(tool, "_query_drug_database", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = None

            result = await tool.execute_async(drug_name="不存在的药物xyz")

            assert result.success is False

    @pytest.mark.asyncio
    async def test_empty_drug_name(self):
        """空药物名应返回错误"""
        from src.tools.drug_query import DrugQueryTool

        tool = DrugQueryTool()
        result = await tool.execute_async(drug_name="")

        assert result.success is False

    def test_tool_schema(self):
        """工具应返回有效的 JSON Schema"""
        from src.tools.drug_query import DrugQueryTool

        tool = DrugQueryTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "drug_query"
        assert "drug_name" in schema["function"]["parameters"]["properties"]


# ==================== SymptomAnalyzerTool Tests ====================


class TestSymptomAnalyzerTool:
    """SymptomAnalyzerTool 症状分析工具测试"""

    @pytest.mark.asyncio
    async def test_symptom_analysis(self):
        """基本症状分析"""
        from src.tools.symptom_analyzer import SymptomAnalyzerTool

        tool = SymptomAnalyzerTool()

        with patch.object(tool, "_analyze_symptoms", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "possible_conditions": ["上呼吸道感染", "普通感冒"],
                "severity": "mild",
                "recommendations": [
                    "多休息，保证充足睡眠",
                    "多饮水",
                    "如症状持续超过一周，请就医",
                ],
                "emergency": False,
                "confidence": 0.75,
            }

            result = await tool.execute_async(
                symptoms="头痛, 流鼻涕, 轻微发热",
                duration="2天",
            )

            assert result.success is True
            assert "possible_conditions" in result.data
            assert result.data["emergency"] is False

    @pytest.mark.asyncio
    async def test_emergency_detection(self):
        """紧急症状检测"""
        from src.tools.symptom_analyzer import SymptomAnalyzerTool

        tool = SymptomAnalyzerTool()

        with patch.object(tool, "_analyze_symptoms", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "possible_conditions": ["心肌梗死"],
                "severity": "critical",
                "recommendations": ["立即拨打120急救电话"],
                "emergency": True,
                "confidence": 0.9,
            }

            result = await tool.execute_async(
                symptoms="剧烈胸痛, 呼吸困难, 冷汗",
                duration="突然发作",
            )

            assert result.success is True
            assert result.data["emergency"] is True
            assert result.data["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_empty_symptoms(self):
        """空症状应返回错误"""
        from src.tools.symptom_analyzer import SymptomAnalyzerTool

        tool = SymptomAnalyzerTool()
        result = await tool.execute_async(symptoms="")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_symptom_with_body_part(self):
        """带身体部位的症状分析"""
        from src.tools.symptom_analyzer import SymptomAnalyzerTool

        tool = SymptomAnalyzerTool()

        with patch.object(tool, "_analyze_symptoms", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "possible_conditions": ["腰椎间盘突出"],
                "severity": "moderate",
                "recommendations": ["建议进行MRI检查", "避免重体力劳动"],
                "emergency": False,
                "confidence": 0.6,
                "body_parts": ["腰部"],
            }

            result = await tool.execute_async(
                symptoms="腰痛, 下肢麻木",
                body_part="腰部",
                duration="1周",
            )

            assert result.success is True
            assert "body_parts" in result.data

    def test_tool_schema(self):
        """工具应返回有效的 JSON Schema"""
        from src.tools.symptom_analyzer import SymptomAnalyzerTool

        tool = SymptomAnalyzerTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "symptom_analyzer"
        assert "symptoms" in schema["function"]["parameters"]["properties"]


# ==================== Tool Parameter Validation Tests ====================


class TestToolParameterValidation:
    """工具参数验证测试"""

    def test_calculator_missing_expression(self):
        """计算器缺少 expression 参数"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression=None)

        assert result.success is False

    def test_code_executor_missing_code(self):
        """代码执行器缺少 code 参数"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(code=None, language="python")

        assert result.success is False

    def test_code_executor_missing_language(self):
        """代码执行器缺少 language 参数"""
        from src.tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool(timeout=10)
        result = tool.execute(code="print('hello')", language=None)

        assert result.success is False

    def test_file_manager_missing_operation(self):
        """文件管理器缺少 operation 参数"""
        from src.tools.file_manager import FileManagerTool

        tool = FileManagerTool(base_path="./workspace")
        result = tool.execute(operation=None, filename="test.txt")

        assert result.success is False

    def test_file_manager_invalid_operation(self):
        """文件管理器无效的 operation"""
        from src.tools.file_manager import FileManagerTool

        tool = FileManagerTool(base_path="./workspace")
        result = tool.execute(operation="hack", filename="test.txt")

        assert result.success is False


# ==================== BaseTool Tests ====================


class TestBaseTool:
    """BaseTool 基础工具类测试"""

    def test_base_tool_properties(self):
        """BaseTool 基本属性"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()

        assert tool.name == "calculator"
        assert tool.description is not None
        assert isinstance(tool.description, str)
        assert tool.enabled is True

    def test_base_tool_enable_disable(self):
        """工具启用/禁用"""
        from src.tools.calculator import CalculatorTool

        tool = CalculatorTool()
        assert tool.enabled is True

        tool.enabled = False
        assert tool.enabled is False

        tool.enabled = True
        assert tool.enabled is True

    def test_tool_result_structure(self):
        """ToolResult 数据结构"""
        from src.tools.base import ToolResult

        success_result = ToolResult(success=True, data={"key": "value"})
        assert success_result.success is True
        assert success_result.data == {"key": "value"}
        assert success_result.error is None

        error_result = ToolResult(success=False, error="出错了")
        assert error_result.success is False
        assert error_result.error == "出错了"
        assert error_result.data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
