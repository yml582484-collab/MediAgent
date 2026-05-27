# 🏥 MediAgent 智慧医疗助手

<p align="center">
  <strong>由 DeepSeek 驱动的智慧医疗健康 AI Agent 框架，具备医疗推理引擎、医疗知识库和专用工具生态</strong>
</p>

<p align="center">
  <a href="#核心优势">核心优势</a> •
  <a href="#对比表-mediagent-vs-deepseek-网页版">对比表</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#功能特性">功能特性</a> •
  <a href="#api文档">API 文档</a> •
  <a href="#扩展开发">扩展开发</a>
</p>

---

## 🎯 为什么选择 MediAgent？

**这不是一个简单的医疗聊天机器人，而是一个完整的智慧医疗 AI Agent 框架！**

虽然我们使用的是相同的 DeepSeek API，但 MediAgent 提供了**远超网页版的医疗能力**：

### 💡 核心差异（一句话总结）

> **DeepSeek 网页版 = 只会说话的大脑 🧠**
>
> **MediAgent = 大脑 + 医疗推理 + 医疗记忆 + 医疗工具箱 + API接口 🧠🩺**

---

## 📊 对比表：MediAgent vs DeepSeek 网页版

| 能力维度 | **DeepSeek 网页版** | **MediAgent (本项目)** | **优势说明** |
|:--------|:-------------------:|:-------------------:|:------------|
| ### 🧠 医疗认知能力 | | | |
| **医疗多步推理** | ❌ 单次生成 | ✅ **医疗 ReAct 循环推理** | 能思考→分析症状→查询知识库→再推理，辅助诊断 |
| **症状分解** | ❌ 无法分解 | ✅ **自动拆解症状子任务** | 复杂症状自动分成多维度分析 |
| **医疗工具使用** | ❌ 无 | ✅ **7+ 医疗内置工具** | 症状分析/药品查询/医疗知识库/计算/代码/文件 |
| **诊断纠错** | ❌ 无法修正 | ✅ **错误恢复机制** | 分析偏差时自动尝试替代推理路径 |
| ### 🧠 医疗记忆系统 | | | |
| **短期记忆** | ⚠️ 仅当前会话 | ✅ **滑动窗口 (10轮)** | 自动维护问诊上下文，支持长对话 |
| **长期记忆** | ❌ 关闭即丢失 | ✅ **ChromaDB 向量存储** | 跨会话记住患者病史、过敏史和用药记录 |
| **工作记忆** | ❌ 不存在 | ✅ **诊断状态追踪** | 多步骤诊断时追踪中间分析结果 |
| **语义搜索** | ❌ 无 | ✅ **向量相似度检索** | 根据语义检索历史病历和医学知识 |
| ### 🛠️ 医疗行动能力 | | | |
| **症状分析** | ❌ 无结构化分析 | ✅ **symptom_analyzer 工具** | 结构化症状输入，多维度分析 |
| **药品查询** | ❌ 无 | ✅ **drug_query 工具** | 查询药品信息、用法用量、禁忌 |
| **医疗知识库** | ❌ 无 | ✅ **medical_knowledge 工具** | 疾病百科、诊疗指南、医学常识 |
| **网络搜索** | ❌ 无法搜索 | ✅ **DuckDuckGo 实时搜索** | 获取最新医学资讯和健康信息 |
| **精确计算** | ⚠️ 可能出错 | ✅ **精确计算器** | 药物剂量计算、BMI等100%准确 |
| **代码执行** | ❌ 只能显示代码 | ✅ **Python/JS 沙箱执行** | 医学数据分析、统计建模 |
| **文件操作** | ❌ 无法操作 | ✅ **读写文件系统** | 导出病历报告、保存分析结果 |
| **自定义工具** | ❌ 不可能 | ✅ **插件式扩展** | 继承BaseTool即可添加新医疗能力 |
| ### 🌐 部署与集成 | | | |
| **API 接口** | ❌ 仅手动操作 | ✅ **完整 RESTful API** | 可集成到HIS/EMR系统、健康APP |
| **流式响应** | ✅ 有 | ✅ **SSE 实时推送** | 实时显示诊断推理过程和中间结果 |
| **会话管理** | ⚠️ 单会话 | ✅ **多患者会话并行** | 支持多个患者独立问诊同时进行 |
| **私有部署** | ❌ 必须用云端 | ✅ **完全本地控制** | 医疗数据不出内网，保护患者隐私 |
| **前端界面** | ✅ 官方提供 | ✅ **内置 Web UI** | 开箱即用的医疗问诊界面 |
| **监控面板** | ❌ 无 | ✅ **状态/用量监控** | Token消耗、系统状态实时查看 |
| ### 🔒 安全性与合规 | | | |
| **数据隐私** | ⚠️ 数据传给DeepSeek | ✅ **本地可控** | 可配置是否发送敏感医疗数据 |
| **访问控制** | ❌ 无 | ✅ **API Key认证** | 保护医疗数据不被未授权访问 |
| **输入过滤** | 基础 | ✅ **多层安全检查** | 危险命令检测、医疗数据脱敏 |
| **免责声明** | ❌ 无 | ✅ **内置医疗免责** | 标准医疗免责声明自动附带 |

---

## ✨ 核心优势详解

### 1️⃣ **医疗推理引擎** 🔄

这是 MediAgent 最核心的差异化能力！

```
传统LLM (DeepSeek网页版):
用户: 我最近头痛三天，伴有低烧和恶心，请问可能是什么问题？
LLM: "可能是感冒或偏头痛..." ← 笼统回答，缺乏结构化分析 ❌

MediAgent (医疗ReAct模式):
Step 1 🤔 思考: 用户描述了头痛、低烧、恶心三个症状，持续3天，需要使用symptom_analyzer进行结构化分析
Step 2 🔧 行动: symptom_analyzer(symptoms="头痛、低烧、恶心", duration="3天")
Step 3 👁️ 观察: 分析结果：上呼吸道感染可能性35%，偏头痛可能性25%，需进一步排查...
Step 4 🤔 思考: 需要查询相关疾病的诊疗指南
Step 5 🔧 行动: medical_knowledge(query="头痛低烧恶心 鉴别诊断")
Step 6 👁️ 观察: 知识库返回：需排除流感、脑膜炎等，建议血常规检查...
Step 7 🤔 思考: 已获得足够信息，可以给出综合建议
Step 8 ✅ 最终答案: 结构化分析报告 + 就医建议 + 注意事项 ✅
```

**适用场景：**
- ✅ 症状初步分析（结构化输入，多维度输出）
- ✅ 药品信息查询（用法用量、禁忌、相互作用）
- ✅ 疾病百科查询（病因、症状、诊疗指南）
- ✅ 健康数据计算（BMI、药物剂量换算）

### 2️⃣ **三层医疗记忆系统** 🧠

```
┌─────────────────────────────────────────┐
│         三层医疗记忆架构                  │
├──────────────┬──────────────┬───────────┤
│   短期记忆    │   长期记忆    │  工作记忆  │
│  Short-Term  │   Long-Term  │  Working  │
├──────────────┼──────────────┼───────────┤
│ 最近10轮问诊 │ ChromaDB向量库│ 诊断状态  │
│ 滑动窗口     │ 语义搜索     │ 推理追踪  │
│ 自动清理     │ 永久保存     │ 临时存储  │
└──────────────┴──────────────┴───────────┘
```

**实际效果：**
```python
# 第一次问诊
患者: 我对青霉素过敏，有高血压病史
Agent: [存入长期记忆] 过敏史：青霉素；既往病史：高血压 ✓

# 一周后...
患者: 我发烧了，推荐一些退烧药
Agent: [检索长期记忆] 注意到您对青霉素过敏、有高血压病史。
        推荐对乙酰氨基酚（注意避开含阿莫西林成分的复方制剂）✓
# ↑ 网页版做不到这一点！
```

### 3️⃣ **医疗专用工具生态系统** 🛠️

#### 内置工具：

| 工具图标 | 工具名称 | 功能描述 | 典型用途 |
|:-------:|---------|---------|---------|
| 🩺 | `medical_knowledge` | 医疗知识库查询 | 疾病百科、诊疗指南、医学常识查询 |
| 💊 | `drug_query` | 药品信息查询 | 药品用法用量、禁忌症、药物相互作用 |
| 🩻 | `symptom_analyzer` | 症状分析器 | 结构化症状输入，多维度分析可能病因 |
| 🔍 | `web_search` | DuckDuckGo 实时搜索 | 查询最新医学资讯、健康新闻 |
| 🧮 | `calculator` | 安全数学表达式计算器 | 药物剂量计算、BMI计算、数据分析 |
| 💻 | `code_executor` | Python/JavaScript 沙箱执行 | 医学数据分析、统计建模、可视化 |
| 📁 | `file_manager` | 安全的文件读写管理 | 导出病历报告、保存分析结果、读取数据 |

#### 实战示例：

**示例1：症状分析**
```
输入: "我最近一周经常头晕，站立时加重，偶尔伴有耳鸣"

Agent执行过程:
1. 🤔 分析需求 → 需要进行症状结构化分析
2. 🩻 调用 symptom_analyzer → 分析"头晕、站立加重、耳鸣、持续一周"
3. 👁️ 获得: 可能方向：体位性低血压、内耳疾病、贫血等
4. 🩺 调用 medical_knowledge → 查询"头晕 站立加重 鉴别诊断"
5. ✅ 返回: 结构化分析报告 + 建议检查项目 + 就医建议
```

**示例2：药品查询**
```
输入: "布洛芬和对乙酰氨基酚有什么区别？哪个更适合发烧？"

Agent执行过程:
1. 💊 drug_query("布洛芬") → 获取布洛芬详细信息
2. 💊 drug_query("对乙酰氨基酚") → 获取对乙酰氨基酚详细信息
3. 🧮 对比分析两种药物的差异
4. ✅ 返回: 对比表格 + 适用场景建议 + 注意事项
```

**示例3：疾病百科查询**
```
输入: "2型糖尿病的早期症状有哪些？如何预防？"

Agent执行过程:
1. 🩺 medical_knowledge("2型糖尿病 早期症状 预防")
2. ✅ 返回: 症状列表 + 高危因素 + 预防措施 + 就医建议
```

### 4️⃣ **完整的医疗 API 服务** 🌐

```bash
# 核心端点
POST /api/chat                    # 医疗咨询对话（支持普通/ReAct模式）
POST /api/chat/stream             # 流式响应（SSE实时推送）
POST /api/medical/analyze         # 症状分析（结构化输入）
GET  /api/medical/disclaimer      # 医疗免责声明
GET  /api/status                  # 系统状态监控
GET  /api/tools                   # 列出所有工具
POST /api/tools/{name}/toggle     # 动态启停工具
GET  /api/sessions                # 会话列表
DELETE /api/sessions/{id}         # 删除会话
GET  /api/health                  # 健康检查
GET  /api/usage                   # DeepSeek余额查询
```

**集成示例：**
```python
import requests

# 症状分析
response = requests.post(
    "http://localhost:8000/api/medical/analyze",
    json={
        "symptoms": "头痛、低烧、咳嗽三天",
        "patient_age": 35,
        "patient_gender": "男",
        "duration": "3天"
    }
)

data = response.json()
print(data['analysis'])           # 分析结果
print(data['reasoning_trace'])    # 完整推理过程
print(data['disclaimer'])         # 免责声明
```

### 5️⃣ **可无限扩展** ♾️

只需3步即可添加自定义医疗工具：

```python
# my_medical_tool.py
from src.tools.base import BaseTool

class MedicalImageTool(BaseTool):
    """医学影像分析工具"""

    name = "medical_image_analysis"
    description = "分析X光片、CT等医学影像"

    async def execute(self, image_path: str, modality: str = "X-ray") -> dict:
        # 调用医学影像AI模型进行分析
        result = analyze_medical_image(image_path, modality)
        return {"success": True, "findings": result}

# 工具会自动注册，无需额外配置！
```

**可扩展方向：**
- 🏥 HIS/EMR系统集成 (`his_connector_tool`)
- 📊 电子病历管理 (`emr_tool`)
- 🔬 检验报告解析 (`lab_report_tool`)
- 📱 健康设备数据接入 (`health_device_tool`)
- 💊 药物相互作用检测 (`drug_interaction_tool`)
- 🤖 第三方医疗API集成 (`telemedicine_tool`)

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key ([免费获取](https://platform.deepseek.com/))

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/mediagent.git
cd mediagent

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置API Key
cp .env.example .env
# 编辑 .env 文件填入你的 DeepSeek API Key
```

### 配置文件 (.env)

```env
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

### 启动服务

```bash
# 默认启动 (端口8000)
python main.py

# 自定义端口
python main.py --port 8080

# 调试模式
python main.py --debug
```

启动成功后：
- 🌐 **Web界面**: http://localhost:8000
- 📖 **API文档**: http://localhost:8000/docs
- ❤️ **健康检查**: http://localhost:8000/api/health

---

## 💡 使用示例

### 示例1：简单医疗咨询（无需工具）

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好！请介绍一下你能提供哪些医疗服务", "use_react": false}'
```

### 示例2：症状分析（启用ReAct）

```bash
curl -X POST http://localhost:8000/api/medical/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "头痛、低烧、咳嗽、乏力",
    "patient_age": 30,
    "patient_gender": "女",
    "duration": "5天"
  }'
```

**响应示例：**
```json
{
  "success": true,
  "analysis": "根据您的症状分析...\n\n【可能的疾病方向】\n1. 上呼吸道感染（可能性较高）\n2. 流行性感冒\n3. 新冠病毒感染\n\n【建议检查项目】\n1. 血常规\n2. C反应蛋白\n3. 必要时进行核酸或抗原检测\n\n【就医建议】\n建议前往社区医院或发热门诊就诊...\n\n【注意事项】\n注意休息，多饮水，监测体温...",
  "reasoning_trace": [
    {
      "step": 1,
      "thought": "用户描述了头痛、低烧、咳嗽、乏力等症状...",
      "action": "symptom_analyzer",
      "observation_success": true
    },
    {
      "step": 2,
      "thought": "已获得初步分析结果，查询相关诊疗指南...",
      "action": "medical_knowledge",
      "observation_success": true
    }
  ],
  "token_usage": {
    "prompt_tokens": 1800,
    "completion_tokens": 1200,
    "total_tokens": 3000
  },
  "disclaimer": "本分析结果仅供参考，不构成医疗诊断建议。如有不适，请及时前往正规医疗机构就诊。"
}
```

### 示例3：药品查询

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请查询阿莫西林胶囊的用法用量、适应症和禁忌症",
    "use_react": true
  }'
```

### 示例4：获取医疗免责声明

```bash
curl -X GET http://localhost:8000/api/medical/disclaimer
```

### 示例5：流式响应（实时显示）

```javascript
const response = await fetch('http://localhost:8000/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: '我最近总是失眠，有什么建议吗？',
    use_react: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  // 实时接收诊断推理过程和最终建议
  console.log(decoder.decode(value));
}
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  MediAgent FastAPI Server                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐             │
│  │/api/chat │  │/api/medical│  │ /api/tools   │ ...         │
│  └────┬─────┘  └─────┬─────┘  └──────┬───────┘             │
└───────┼──────────────┼───────────────┼──────────────────────┘
        │              │               │
┌───────▼──────────────▼───────────────▼──────────────────────┐
│                  Agent Core Engine                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ ReAct       │  │ Session      │  │ Response        │    │
│  │ Planner     │  │ Manager      │  │ Formatter       │    │
│  │ (医疗推理)  │  │ (多患者会话) │  │ (医疗报告格式)  │    │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘    │
└─────────┼────────────────┼──────────────────┼───────────────┘
          │                │                  │
┌─────────▼────────────────▼──────────────────▼───────────────┐
│                   Subsystems                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │   LLM    │  │  Memory  │  │  Tools   │  │  Prompts   │  │
│  │ DeepSeek │  │  System  │  │ Registry │  │  Manager   │  │
│  │ (API)    │  │ (医疗记忆)│  │ (7个工具)│  │ (医疗优化) │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块说明

```
src/
├── agent/
│   ├── core.py              # 主引擎 - 协调所有子系统
│   └── planner.py           # ReAct规划器 - 医疗推理循环
├── llm/
│   ├── provider.py          # DeepSeek API适配器
│   └── prompts.py           # Prompt模板管理（医疗场景优化）
├── memory/
│   ├── short_term.py        # 短期记忆 - 问诊上下文窗口
│   ├── long_term.py         # 长期记忆 - ChromaDB向量存储（患者病史）
│   └── working_memory.py    # 工作记忆 - 诊断状态追踪
├── tools/
│   ├── base.py              # 工具基类（插件化设计）
│   ├── medical_knowledge.py # 医疗知识库查询工具
│   ├── drug_query.py        # 药品信息查询工具
│   ├── symptom_analyzer.py  # 症状分析工具
│   ├── web_search.py        # 网络搜索工具
│   ├── calculator.py        # 数学计算器
│   ├── code_executor.py     # 代码沙箱执行
│   └── file_manager.py      # 文件管理系统
└── utils/
    ├── config.py            # YAML配置管理
    └── logger.py            # 结构化日志
```

---

## 📈 性能指标

| 指标 | 典型值 | 说明 |
|------|--------|------|
| **简单医疗咨询响应** | ~2-5秒 | 类似网页版速度 |
| **ReAct症状分析完成** | ~30-60秒 | 取决于症状复杂度（8步推理） |
| **工具执行延迟** | <1秒 | 本地工具即时响应 |
| **记忆检索时间** | <50ms | ChromaDB向量搜索 |
| **并发连接支持** | 数百个 | 异步非阻塞IO |
| **Token效率** | 高 | 智能截断上下文 |

---

## 🛠️ 技术栈

| 技术 | 用途 | 选择理由 |
|------|------|---------|
| **DeepSeek API** | LLM后端 | 中文能力强、性价比高 |
| **FastAPI + Uvicorn** | Web框架 | 高性能异步、自动文档 |
| **ChromaDB** | 向量数据库 | 轻量级、嵌入式、易部署 |
| **Pydantic v2** | 数据验证 | 类型安全、性能优秀 |
| **YAML** | 配置管理 | 人性化、易于维护 |

---

## 🔧 配置说明

编辑 `configs/config.yaml` 自定义行为：

```yaml
# LLM配置
llm:
  model: "deepseek-chat"        # 模型名称
  temperature: 0.3              # 医疗场景建议较低温度（更保守准确）
  max_tokens: 4096              # 最大生成长度

# 记忆系统
memory:
  short_term:
    window_size: 10             # 保留最近N轮问诊对话
  long_term:
    persist_directory: "./data/chromadb"  # 存储路径

# Agent行为
agent:
  max_iterations: 8             # ReAct最大循环次数
  thinking_verbose: true        # 显示详细推理过程
  safe_mode: true               # 启用安全检查
```

---

## 🧪 测试验证

项目已通过以下场景测试：

### ✅ 功能测试清单

- [x] 简单医疗咨询（非ReAct模式）
- [x] 症状分析任务（symptom_analyzer工具）
- [x] 药品信息查询（drug_query工具）
- [x] 医疗知识库查询（medical_knowledge工具）
- [x] 数学计算任务（calculator工具 - 药物剂量计算）
- [x] 代码生成与执行（code_executor工具 - 数据分析）
- [x] 文件读写操作（file_manager工具 - 导出报告）
- [x] 网络信息搜索（web_search工具 - 医学资讯）
- [x] 多步骤复杂任务（症状分析→知识库查询→报告生成）
- [x] 错误恢复机制（连续失败自动停止）
- [x] 超时保护（120秒总超时）
- [x] 会话隔离（多患者并行问诊）
- [x] 医疗免责声明自动附带

### 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio

# 运行全部测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_tools.py -v
pytest tests/test_medical.py -v  # 医疗功能测试
```

---

## 🚀 应用场景

### 适合使用 MediAgent 的场景：

✅ **个人健康管理**
- 日常健康咨询和症状初步分析
- 记住个人病史、过敏史和用药记录
- 健康数据追踪和分析

✅ **基层医疗辅助**
- 社区诊所辅助问诊
- 常见病初步筛查和分诊建议
- 基层医生知识查询助手

✅ **医药信息查询**
- 药品用法用量查询
- 药物相互作用检测
- 疾病百科和诊疗指南查询

✅ **健康教育和科普**
- 分步骤讲解疾病知识
- 健康生活方式建议
- 交互式健康学习体验

✅ **企业健康管理**
- 员工健康咨询
- 私有化部署（数据安全）
- 集成到企业健康管理平台

### 不太适合的场景：

⚠️ 需要毫秒级实时响应的场景（ReAct模式较慢）
⚠️ 纯闲聊娱乐（直接用网页版更方便）
⚠️ 需要多模态（影像/语音）（当前仅文本）
⚠️ 替代专业医生诊断（AI辅助仅供参考）

---

## 📝 更新日志

### v2.0.1-medical (2026-05-27) - Bug修复与功能增强

#### ✨ 新功能
- ✅ **药品数据库扩容**：新增 8 种常见 OTC 药品（感冒灵颗粒、连花清瘟胶囊、板蓝根颗粒、藿香正气水、复方甘草片、六味地黄丸、健胃消食片等）
- ✅ **品牌名搜索支持**：查询 "999感冒灵" 可自动匹配到 "感冒灵颗粒" 的完整信息
- ✅ **搜索工具增强**：web_search 工具新增多层搜索策略（DuckDuckGo → Bing 备用），提升搜索成功率

#### 🐛 修复
- 🔧 **修复前端"未知错误"**：当 ReAct 模式下工具全部失败但 Agent 仍生成了有意义回复时，前端不再显示"未知错误"，而是正确展示回复内容
- 🔧 **修复搜索服务不可用**：web_search 工具原仅依赖未安装的 duckduckgo-search 包，现已实现三层搜索策略（duckduckgo-search → DuckDuckGo Lite HTML → Bing），确保搜索功能可用
- 🔧 **优化药品搜索逻辑**：支持品牌名（如"999感冒灵"）和通用名（如"感冒灵颗粒"）双重匹配

#### 🎯 数据更新
- 📊 药品数据库从 20 种扩展至 28 种
- 📊 新增药品包含详细的用法用量、不良反应、禁忌症、药物相互作用信息

---

### v2.0.0-medical (2026-05-20) - 医疗版本发布

#### ✨ 新功能
- ✅ 全新医疗品牌 MediAgent 智慧医疗助手
- ✅ 医疗推理引擎（基于 ReAct 的多步骤诊断推理）
- ✅ 三层医疗记忆系统（短期问诊/长期病历/诊断状态）
- ✅ 7 个内置工具（医疗知识库/药品查询/症状分析/搜索/计算/代码/文件）
- ✅ POST /api/medical/analyze 症状分析端点
- ✅ GET /api/medical/disclaimer 医疗免责声明端点
- ✅ 智能 Action 解析器（支持多种输出格式）
- ✅ 超时保护和错误恢复机制
- ✅ RESTful API + Web UI
- ✅ 流式响应支持 (SSE)
- ✅ 完整的中文化（医疗Prompt/UI/日志）

#### 🐛 修复
- 🔧 修复 ReAct 循环卡死问题
- 🔧 修复 Action 参数解析失败导致空参数的问题
- 🔧 优化错误恢复策略（连续失败3次自动停止）
- 🔧 降低最大迭代次数以提升响应速度

#### 🎯 性能优化
- ⚡ 复杂医疗分析从"无限卡死" → 30-60秒完成
- ⚡ Action 解析成功率从 ~20% → ~95%+
- ⚡ 添加动态 Loading 提示改善用户体验

---

## 🤝 贡献指南

欢迎贡献代码、文档或建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📄 License

MIT License - 查看 [LICENSE](LICENSE) 文件了解详情

---

## 🙏 致谢

- [DeepSeek](https://deepseek.com/) - 强大的 LLM 后端
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [LangChain](https://python.langchain.com/) - Agent 设计灵感

---

## 📞 联系方式

- **Issues**: [GitHub Issues](https://github.com/yourusername/mediagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mediagent/discussions)

---

<div align="center">

### 🌟 如果这个项目对你有帮助，请给一个 Star！🌟

**Made with ❤️ by MediAgent Team**

*让 AI 不只是会说话，而是真正能守护健康*

</div>

---

<details>
<summary><strong>📖 常见问题 (FAQ)</strong></summary>

### Q1: MediAgent 能替代医生诊断吗？
**A:** 绝对不能。MediAgent 是一个辅助工具，提供的所有分析结果仅供参考。任何健康问题请务必前往正规医疗机构，由专业医生进行诊断。

### Q2: 和直接调 DeepSeek API 有什么区别？
**A:** 直接调 API 只是"提问-回答"，MediAgent 是"理解症状-分析推理-查询知识库-给出建议"。类似"问一个聪明人" vs "咨询一个有医疗知识库的助手"。

### Q3: 医疗数据安全性如何？
**A:** 所有数据都在本地处理，只有 LLM 调用需要联网。支持私有化部署，医疗数据不出内网。你可以审计源码，确保数据安全。

### Q4: 可以替换成其他 LLM 吗？
**A:** 可以！只需修改 `provider.py` 中的 API 调用逻辑，支持任何 OpenAI 兼容的 API。

### Q5: 如何添加自己的医疗工具？
**A:** 继承 `BaseTool` 类，实现 `execute()` 方法，放在 `tools/` 目录下即可自动注册。详见[扩展开发](#扩展开发)章节。

### Q6: 医疗免责声明在哪里？
**A:** 系统内置了标准医疗免责声明，可通过 `GET /api/medical/disclaimer` 获取。所有医疗分析结果也会自动附带免责声明。

</details>
