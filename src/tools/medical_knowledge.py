"""
医疗知识库查询工具 - MediAgent 智慧医疗助手
提供疾病百科、检验检查、临床指南、医保政策等医疗知识查询功能
"""
from typing import Any, Dict, List, Optional
from ..tools.base import BaseTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MedicalKnowledgeTool(BaseTool):
    """
    医疗知识库查询工具

    功能：
    - 查询疾病百科信息
    - 查询检验检查项目说明
    - 查询临床指南摘要
    - 查询医保政策信息

    内置本地医疗知识字典，支持关键词搜索和分类筛选。
    """

    name = "medical_knowledge"
    description = "查询医疗知识库，包括疾病百科、检验检查、临床指南、医保政策等"

    # 内置医疗知识字典
    _knowledge_base: Dict[str, List[Dict[str, Any]]] = {
        "疾病百科": [
            {
                "keywords": ["高血压", "血压高", "原发性高血压"],
                "title": "高血压（Hypertension）",
                "category": "疾病百科",
                "overview": "高血压是一种以动脉血压持续升高为特征的慢性疾病，是最常见的心血管疾病之一。正常成人血压收缩压<120mmHg，舒张压<80mmHg。",
                "symptoms": "多数患者早期无明显症状，部分患者可有头痛、头晕、耳鸣、颈项僵硬、疲劳等。长期高血压可导致心、脑、肾等靶器官损害。",
                "diagnosis": "在非同日3次测量血压，收缩压>=140mmHg和/或舒张压>=90mmHg可诊断为高血压。需进一步完善血常规、尿常规、肾功能、心电图等检查。",
                "treatment": "生活方式干预（低盐饮食、适量运动、控制体重、戒烟限酒）为基础，根据血压水平选择合适的降压药物（如ACEI、ARB、CCB、利尿剂等）。",
                "prevention": "保持健康生活方式，定期监测血压，低盐低脂饮食，适量运动，控制体重，戒烟限酒，保持心理平衡。",
                "department": "心血管内科",
            },
            {
                "keywords": ["糖尿病", "血糖高", "高血糖"],
                "title": "糖尿病（Diabetes Mellitus）",
                "category": "疾病百科",
                "overview": "糖尿病是一组以高血糖为特征的代谢性疾病，由胰岛素分泌缺陷或胰岛素作用障碍引起。主要分为1型糖尿病、2型糖尿病和妊娠期糖尿病。",
                "symptoms": "典型症状为'三多一少'：多饮、多食、多尿、体重减轻。也可有乏力、视力模糊、皮肤瘙痒、伤口愈合慢等症状。",
                "diagnosis": "空腹血糖>=7.0mmol/L，或餐后2小时血糖>=11.1mmol/L，或随机血糖>=11.1mmol/L，伴典型症状可诊断。糖化血红蛋白(HbA1c)>=6.5%也可作为诊断标准。",
                "treatment": "1型糖尿病需胰岛素治疗；2型糖尿病首选二甲双胍，根据情况联合其他口服药或胰岛素。配合饮食控制、运动疗法和血糖监测。",
                "prevention": "健康饮食，控制体重，规律运动，定期检测血糖，避免久坐，限制含糖饮料摄入。",
                "department": "内分泌科",
            },
            {
                "keywords": ["感冒", "上呼吸道感染", "鼻塞", "流鼻涕"],
                "title": "上呼吸道感染（普通感冒）",
                "category": "疾病百科",
                "overview": "上呼吸道感染是由病毒引起的鼻腔、咽或喉部急性炎症，是最常见的呼吸道感染性疾病。全年均可发病，以冬春季节多见。",
                "symptoms": "鼻塞、流涕、打喷嚏、咽痛、咳嗽、低热、头痛、全身不适等。一般病程5-7天，可自愈。",
                "diagnosis": "根据临床症状和体征即可诊断。一般不需要特殊检查，必要时可做血常规、C反应蛋白等排除细菌感染。",
                "treatment": "以对症治疗为主：发热头痛可用解热镇痛药（如对乙酰氨基酚、布洛芬）；鼻塞可用减充血剂；咳嗽可用止咳药。注意休息、多饮水。",
                "prevention": "勤洗手，避免接触感染者，保持室内通风，增强体质，规律作息，均衡营养。",
                "department": "内科/呼吸内科",
            },
            {
                "keywords": ["冠心病", "冠状动脉粥样硬化性心脏病", "心绞痛"],
                "title": "冠心病（冠状动脉粥样硬化性心脏病）",
                "category": "疾病百科",
                "overview": "冠心病是由于冠状动脉粥样硬化导致管腔狭窄或阻塞，引起心肌缺血缺氧的心脏病。是威胁人类健康的主要疾病之一。",
                "symptoms": "典型表现为劳力性胸骨后压榨样疼痛，可放射至左肩、左臂、颈部。可伴有胸闷、气短、心悸。不稳定型心绞痛可在休息时发作。",
                "diagnosis": "心电图（静息、动态、负荷）、心脏超声、冠脉CT血管成像、冠脉造影（金标准）、心肌酶谱检查等。",
                "treatment": "药物治疗（抗血小板药、他汀类、硝酸酯类、β受体阻滞剂等）；介入治疗（PCI支架植入）；外科治疗（CABG搭桥手术）。",
                "prevention": "控制危险因素：戒烟、控制血压血糖血脂、健康饮食、适量运动、控制体重、心理平衡。",
                "department": "心血管内科",
            },
            {
                "keywords": ["胃炎", "胃痛", "胃不舒服", "胃酸"],
                "title": "慢性胃炎（Chronic Gastritis）",
                "category": "疾病百科",
                "overview": "慢性胃炎是指各种原因引起的胃黏膜慢性炎症，是消化系统最常见的疾病之一。幽门螺杆菌（Hp）感染是最主要的病因。",
                "symptoms": "上腹部不适、饱胀、疼痛、反酸、嗳气、食欲减退、恶心等。部分患者可无明显症状。",
                "diagnosis": "胃镜检查及胃黏膜活检是确诊的金标准。可检测幽门螺杆菌（呼气试验、快速尿素酶试验等）。",
                "treatment": "根除幽门螺杆菌（四联疗法：PPI+铋剂+两种抗生素）；抑酸护胃（PPI或H2受体阻滞剂）；胃黏膜保护剂；促动力药。对症治疗。",
                "prevention": "规律饮食，避免暴饮暴食，少吃辛辣刺激食物，戒烟限酒，避免长期使用NSAIDs，及时根除幽门螺杆菌。",
                "department": "消化内科",
            },
            {
                "keywords": ["肺炎", "肺部感染", "发烧咳嗽"],
                "title": "肺炎（Pneumonia）",
                "category": "疾病百科",
                "overview": "肺炎是指终末气道、肺泡和肺间质的炎症，可由细菌、病毒、真菌等病原体引起。社区获得性肺炎是最常见的类型。",
                "symptoms": "发热、咳嗽、咳痰（可为脓性痰）、胸痛、呼吸困难、寒战等。老年患者症状可能不典型。",
                "diagnosis": "胸部X线或CT检查可见肺部浸润影。血常规、C反应蛋白、降钙素原、痰培养、血培养等有助于病原学诊断。",
                "treatment": "细菌性肺炎使用抗生素治疗（根据病原体选择）；病毒性肺炎使用抗病毒药物；对症支持治疗（退热、补液、氧疗等）。重症需住院治疗。",
                "prevention": "接种肺炎球菌疫苗和流感疫苗，戒烟，增强体质，注意手卫生，避免接触呼吸道感染患者。",
                "department": "呼吸内科",
            },
        ],
        "检验检查": [
            {
                "keywords": ["血常规", "血液检查", "CBC"],
                "title": "血常规检查（CBC）",
                "category": "检验检查",
                "overview": "血常规是最基本的血液检查，通过检测血液中各类细胞的数量和形态，辅助诊断多种疾病。",
                "items": [
                    {"name": "白细胞计数(WBC)", "正常值": "4.0-10.0×10^9/L", "意义": "升高提示感染、炎症；降低提示免疫力低下"},
                    {"name": "红细胞计数(RBC)", "正常值": "男4.0-5.5×10^12/L，女3.5-5.0×10^12/L", "意义": "降低提示贫血"},
                    {"name": "血红蛋白(Hb)", "正常值": "男120-160g/L，女110-150g/L", "意义": "降低提示贫血"},
                    {"name": "血小板计数(PLT)", "正常值": "100-300×10^9/L", "意义": "异常提示出血或凝血功能障碍"},
                ],
                "note": "空腹或非空腹均可采血，一般采指尖血或静脉血。检查前无需特殊准备。",
            },
            {
                "keywords": ["肝功能", "肝功", "转氨酶"],
                "title": "肝功能检查",
                "category": "检验检查",
                "overview": "肝功能检查是通过生化方法检测肝脏代谢功能的各项指标，用于评估肝脏健康状况。",
                "items": [
                    {"name": "谷丙转氨酶(ALT)", "正常值": "0-40U/L", "意义": "升高提示肝细胞损伤"},
                    {"name": "谷草转氨酶(AST)", "正常值": "0-40U/L", "意义": "升高提示肝细胞或心肌损伤"},
                    {"name": "总胆红素(TBIL)", "正常值": "3.4-17.1μmol/L", "意义": "升高提示黄疸"},
                    {"name": "白蛋白(ALB)", "正常值": "35-55g/L", "意义": "降低提示肝功能减退或营养不良"},
                ],
                "note": "需空腹8-12小时采血。检查前避免饮酒和剧烈运动。",
            },
            {
                "keywords": ["血糖", "空腹血糖", "葡萄糖"],
                "title": "血糖检测",
                "category": "检验检查",
                "overview": "血糖检测是评估糖代谢状态的基本检查，用于糖尿病的诊断和血糖管理。",
                "items": [
                    {"name": "空腹血糖(FPG)", "正常值": "3.9-6.1mmol/L", "意义": ">=7.0mmol/L提示糖尿病"},
                    {"name": "餐后2小时血糖", "正常值": "<7.8mmol/L", "意义": ">=11.1mmol/L提示糖尿病"},
                    {"name": "糖化血红蛋白(HbA1c)", "正常值": "<6.0%", "意义": "反映近2-3个月平均血糖水平"},
                ],
                "note": "空腹血糖需禁食8-12小时。糖化血红蛋白不受饮食影响，随时可查。",
            },
            {
                "keywords": ["尿常规", "尿液检查"],
                "title": "尿常规检查",
                "category": "检验检查",
                "overview": "尿常规检查是通过检测尿液的各种成分，辅助诊断泌尿系统疾病、代谢性疾病等。",
                "items": [
                    {"name": "尿蛋白", "正常值": "阴性", "意义": "阳性提示肾脏损伤"},
                    {"name": "尿糖", "正常值": "阴性", "意义": "阳性提示血糖过高或肾糖阈降低"},
                    {"name": "尿潜血", "正常值": "阴性", "意义": "阳性提示泌尿系统出血"},
                    {"name": "白细胞", "正常值": "阴性/少量", "意义": "增多提示泌尿系统感染"},
                ],
                "note": "建议留取晨尿中段尿，女性需避开月经期。检查前避免大量饮水。",
            },
        ],
        "临床指南": [
            {
                "keywords": ["高血压指南", "血压管理", "降压"],
                "title": "中国高血压防治指南（2023修订版）要点",
                "category": "临床指南",
                "overview": "中国高血压防治指南是指导国内高血压诊疗的权威文件，2023年进行了修订。",
                "key_points": [
                    "高血压诊断标准：收缩压>=140mmHg和/或舒张压>=90mmHg",
                    "正常血压标准：收缩压<120mmHg且舒张压<80mmHg",
                    "正常高值：收缩压120-139mmHg和/或舒张压80-89mmHg",
                    "降压目标：一般患者<140/90mmHg，合并糖尿病/冠心病者<130/80mmHg",
                    "生活方式干预是降压治疗的基础",
                    "常用降压药：ACEI/ARB、CCB、利尿剂、β受体阻滞剂",
                ],
                "source": "中国高血压防治指南修订委员会",
            },
            {
                "keywords": ["糖尿病指南", "血糖管理"],
                "title": "中国2型糖尿病防治指南（2022版）要点",
                "category": "临床指南",
                "overview": "中国2型糖尿病防治指南是指导国内糖尿病诊疗的权威文件。",
                "key_points": [
                    "2型糖尿病诊断标准：空腹血糖>=7.0mmol/L或餐后2h血糖>=11.1mmol/L",
                    "HbA1c>=6.5%可作为诊断标准",
                    "血糖控制目标：一般患者HbA1c<7.0%",
                    "生活方式干预是糖尿病治疗的基础",
                    "二甲双胍是2型糖尿病一线用药",
                    "合并心血管疾病者推荐使用SGLT2i或GLP-1RA",
                    "定期筛查并发症：视网膜病变、肾病、神经病变、足病等",
                ],
                "source": "中华医学会糖尿病学分会",
            },
        ],
        "医保政策": [
            {
                "keywords": ["医保", "医疗保险", "医保报销"],
                "title": "基本医疗保险政策概述",
                "category": "医保政策",
                "overview": "中国基本医疗保险包括城镇职工基本医疗保险和城乡居民基本医疗保险两大类。",
                "key_points": [
                    "城镇职工医保：由单位和个人共同缴纳，缴费比例为工资总额的8%-10%",
                    "城乡居民医保：个人缴费+政府补贴，年度缴费",
                    "起付标准（门槛费）：超过起付线才能报销",
                    "报销比例：因医院等级不同而异，一般在50%-90%之间",
                    "封顶线：年度报销有最高限额",
                    "门诊慢特病：部分慢性病可申请门诊报销",
                    "异地就医：需提前备案，可通过国家医保服务平台办理",
                ],
                "note": "具体政策以当地医保部门规定为准，建议咨询当地医保中心或拨打12333。",
            },
            {
                "keywords": ["异地就医", "跨省就医", "外地看病"],
                "title": "异地就医直接结算政策",
                "category": "医保政策",
                "overview": "异地就医直接结算是指参保人员在参保地以外的定点医疗机构就医时，可直接刷卡结算医疗费用。",
                "key_points": [
                    "备案条件：异地安置退休人员、异地长期居住人员、常驻异地工作人员、异地转诊人员",
                    "备案方式：线上（国家医保服务平台APP/小程序）、线下（医保经办机构）",
                    "备案材料：身份证、社保卡、转诊证明（转诊人员）",
                    "结算范围：住院费用可直接结算，门诊费用逐步推进",
                    "报销比例：执行就医地目录，参保地政策",
                    "备案有效期：长期备案有效，临时备案一般6个月",
                ],
                "note": "建议就医前先完成备案，可拨打参保地医保热线咨询具体流程。",
            },
        ],
    }

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如疾病名称、检查项目名称、指南关键词等",
                },
                "category": {
                    "type": "string",
                    "description": "知识分类",
                    "enum": ["疾病百科", "检验检查", "临床指南", "医保政策"],
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行医疗知识库查询

        Args:
            query: 搜索关键词
            category: 可选分类筛选

        Returns:
            查询结果字典
        """
        query = kwargs.get("query", "").strip()
        category = kwargs.get("category", "").strip()

        if not query:
            return {
                "success": False,
                "error": "请提供搜索关键词",
                "results": [],
            }

        logger.info(f"医疗知识库查询: query='{query}', category='{category}'")

        results = []

        # 确定搜索范围
        if category and category in self._knowledge_base:
            search_categories = {category: self._knowledge_base[category]}
        else:
            search_categories = self._knowledge_base

        # 在知识库中搜索
        for cat_name, items in search_categories.items():
            for item in items:
                # 关键词匹配
                keywords = item.get("keywords", [])
                title = item.get("title", "")
                relevance = self._calculate_relevance(query, keywords, title)

                if relevance > 0:
                    results.append({
                        "title": item.get("title", ""),
                        "category": item.get("category", cat_name),
                        "relevance": relevance,
                        "data": item,
                    })

        # 按相关度排序
        results.sort(key=lambda x: x["relevance"], reverse=True)

        # 格式化返回结果
        formatted_results = []
        for r in results[:5]:  # 最多返回5条
            data = r["data"]
            formatted = {
                "title": data.get("title", ""),
                "category": data.get("category", ""),
                "overview": data.get("overview", ""),
            }

            # 根据分类添加特定字段
            if data.get("category") == "疾病百科":
                formatted["symptoms"] = data.get("symptoms", "")
                formatted["diagnosis"] = data.get("diagnosis", "")
                formatted["treatment"] = data.get("treatment", "")
                formatted["prevention"] = data.get("prevention", "")
                formatted["department"] = data.get("department", "")
            elif data.get("category") == "检验检查":
                formatted["items"] = data.get("items", [])
                formatted["note"] = data.get("note", "")
            elif data.get("category") == "临床指南":
                formatted["key_points"] = data.get("key_points", [])
                formatted["source"] = data.get("source", "")
            elif data.get("category") == "医保政策":
                formatted["key_points"] = data.get("key_points", [])
                formatted["note"] = data.get("note", "")

            formatted_results.append(formatted)

        if not formatted_results:
            return {
                "success": True,
                "message": f"未找到与'{query}'相关的医疗知识信息。建议您前往正规医疗机构咨询专业医生。",
                "results": [],
            }

        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }

    def _calculate_relevance(self, query: str, keywords: List[str], title: str) -> float:
        """
        计算查询与知识条目的相关度

        Args:
            query: 用户查询
            keywords: 条目关键词列表
            title: 条目标题

        Returns:
            相关度分数 (0-1)
        """
        score = 0.0
        query_lower = query.lower()

        # 标题完全匹配
        if query_lower in title.lower():
            score += 0.5

        # 标题部分匹配
        for char in query_lower:
            if char in title.lower():
                score += 0.02

        # 关键词匹配
        for keyword in keywords:
            if query_lower in keyword.lower() or keyword.lower() in query_lower:
                score += 0.3
            # 部分匹配
            for char in query_lower:
                if char in keyword.lower():
                    score += 0.01

        return min(score, 1.0)
