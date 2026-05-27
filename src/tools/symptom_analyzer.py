"""
症状分析工具 - MediAgent 智慧医疗助手
根据患者描述的症状进行初步分析，提供可能的疾病方向和就医建议
"""
from typing import Any, Dict, List, Optional
from ..tools.base import BaseTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SymptomAnalyzerTool(BaseTool):
    """
    症状分析工具

    功能：
    - 根据患者症状进行初步分析
    - 提供可能的疾病方向
    - 推荐就诊科室
    - 评估紧急程度
    - 给出就医建议

    注意：本工具仅提供参考信息，不能替代专业医生的诊断。
    """

    name = "symptom_analyzer"
    description = "根据患者描述的症状进行初步分析，提供可能的疾病方向和就医建议"

    # 紧急症状关键词
    _emergency_keywords = [
        "胸痛", "呼吸困难", "大出血", "意识丧失", "剧烈头痛",
        "严重过敏", "抽搐", "昏迷", "窒息", "心跳骤停",
        "偏瘫", "吐血", "剧烈腹痛", "高热惊厥", "自杀",
    ]

    # 症状-疾病关联字典
    _symptom_disease_mapping: Dict[str, List[Dict[str, Any]]] = {
        "头痛": [
            {
                "disease": "偏头痛",
                "probability": "中",
                "department": "神经内科",
                "description": "反复发作的搏动性头痛，常伴恶心、呕吐、畏光、畏声",
                "urgency": "低",
            },
            {
                "disease": "紧张型头痛",
                "probability": "高",
                "department": "神经内科",
                "description": "双侧压迫感或紧缩感，轻至中度，不因活动加重",
                "urgency": "低",
            },
            {
                "disease": "高血压",
                "probability": "中",
                "department": "心血管内科",
                "description": "血压升高引起的头痛，常伴头晕、耳鸣",
                "urgency": "中",
            },
            {
                "disease": "上呼吸道感染",
                "probability": "中",
                "department": "内科",
                "description": "感冒引起的头痛，常伴鼻塞、流涕、发热",
                "urgency": "低",
            },
        ],
        "发热": [
            {
                "disease": "上呼吸道感染",
                "probability": "高",
                "department": "内科/呼吸内科",
                "description": "病毒或细菌引起的呼吸道感染，常伴咳嗽、鼻塞、咽痛",
                "urgency": "低",
            },
            {
                "disease": "流感",
                "probability": "中",
                "department": "呼吸内科/发热门诊",
                "description": "流行性感冒，高热伴全身酸痛、乏力",
                "urgency": "中",
            },
            {
                "disease": "肺炎",
                "probability": "中",
                "department": "呼吸内科",
                "description": "肺部感染，发热伴咳嗽、咳痰、胸痛",
                "urgency": "中",
            },
            {
                "disease": "尿路感染",
                "probability": "低",
                "department": "泌尿外科/肾内科",
                "description": "发热伴尿频、尿急、尿痛",
                "urgency": "中",
            },
        ],
        "咳嗽": [
            {
                "disease": "上呼吸道感染",
                "probability": "高",
                "department": "内科/呼吸内科",
                "description": "感冒后咳嗽，可伴鼻塞、咽痛",
                "urgency": "低",
            },
            {
                "disease": "急性支气管炎",
                "probability": "中",
                "department": "呼吸内科",
                "description": "咳嗽伴咳痰，可有发热",
                "urgency": "低",
            },
            {
                "disease": "肺炎",
                "probability": "中",
                "department": "呼吸内科",
                "description": "咳嗽伴发热、咳痰、胸痛、呼吸困难",
                "urgency": "中",
            },
            {
                "disease": "支气管哮喘",
                "probability": "低",
                "department": "呼吸内科",
                "description": "反复发作的喘息、咳嗽、胸闷",
                "urgency": "中",
            },
        ],
        "腹痛": [
            {
                "disease": "急性胃肠炎",
                "probability": "高",
                "department": "消化内科/急诊",
                "description": "腹痛伴腹泻、恶心、呕吐，多有不洁饮食史",
                "urgency": "中",
            },
            {
                "disease": "消化性溃疡",
                "probability": "中",
                "department": "消化内科",
                "description": "上腹部规律性疼痛，与进食相关",
                "urgency": "中",
            },
            {
                "disease": "胆囊炎",
                "probability": "中",
                "department": "肝胆外科/消化内科",
                "description": "右上腹疼痛，可放射至右肩，常在油腻饮食后发作",
                "urgency": "中",
            },
            {
                "disease": "急性阑尾炎",
                "probability": "中",
                "department": "普外科/急诊",
                "description": "转移性右下腹痛，初为脐周痛后转移至右下腹",
                "urgency": "高",
            },
        ],
        "腹泻": [
            {
                "disease": "急性胃肠炎",
                "probability": "高",
                "department": "消化内科",
                "description": "腹泻伴腹痛、恶心、呕吐，多有不洁饮食史",
                "urgency": "低",
            },
            {
                "disease": "肠易激综合征",
                "probability": "中",
                "department": "消化内科",
                "description": "反复发作的腹泻或便秘，与情绪、饮食相关",
                "urgency": "低",
            },
            {
                "disease": "食物中毒",
                "probability": "中",
                "department": "急诊/消化内科",
                "description": "进食后出现腹泻、呕吐、腹痛",
                "urgency": "中",
            },
        ],
        "胸闷": [
            {
                "disease": "冠心病",
                "probability": "中",
                "department": "心血管内科",
                "description": "胸闷伴胸痛，活动后加重，休息后缓解",
                "urgency": "高",
            },
            {
                "disease": "心律失常",
                "probability": "中",
                "department": "心血管内科",
                "description": "胸闷伴心悸，可有心跳不规则感",
                "urgency": "中",
            },
            {
                "disease": "焦虑症",
                "probability": "低",
                "department": "精神科/心理科",
                "description": "胸闷伴焦虑、紧张、呼吸急促",
                "urgency": "低",
            },
            {
                "disease": "支气管哮喘",
                "probability": "低",
                "department": "呼吸内科",
                "description": "胸闷伴喘息、咳嗽，可有呼吸困难",
                "urgency": "中",
            },
        ],
        "头晕": [
            {
                "disease": "高血压",
                "probability": "中",
                "department": "心血管内科",
                "description": "血压升高引起的头晕，可伴头痛、耳鸣",
                "urgency": "中",
            },
            {
                "disease": "颈椎病",
                "probability": "中",
                "department": "骨科/神经内科",
                "description": "颈部活动时头晕加重，可伴颈部疼痛、手麻",
                "urgency": "低",
            },
            {
                "disease": "眩晕症",
                "probability": "中",
                "department": "耳鼻喉科/神经内科",
                "description": "天旋地转感，可伴恶心、呕吐",
                "urgency": "低",
            },
            {
                "disease": "低血糖",
                "probability": "低",
                "department": "内分泌科",
                "description": "饥饿时头晕、出冷汗、心慌、手抖",
                "urgency": "中",
            },
        ],
        "恶心呕吐": [
            {
                "disease": "急性胃肠炎",
                "probability": "高",
                "department": "消化内科",
                "description": "恶心呕吐伴腹痛、腹泻",
                "urgency": "中",
            },
            {
                "disease": "食物中毒",
                "probability": "中",
                "department": "急诊",
                "description": "进食后出现恶心呕吐、腹痛",
                "urgency": "中",
            },
            {
                "disease": "晕动症",
                "probability": "低",
                "department": "内科",
                "description": "乘车、船、飞机时出现恶心呕吐",
                "urgency": "低",
            },
        ],
        "心悸": [
            {
                "disease": "心律失常",
                "probability": "高",
                "department": "心血管内科",
                "description": "心跳异常感，可有心跳过快、过慢或不规则",
                "urgency": "中",
            },
            {
                "disease": "焦虑症",
                "probability": "中",
                "department": "精神科/心理科",
                "description": "紧张、焦虑时出现心悸",
                "urgency": "低",
            },
            {
                "disease": "甲亢",
                "probability": "低",
                "department": "内分泌科",
                "description": "心悸伴多汗、消瘦、手抖、情绪激动",
                "urgency": "低",
            },
        ],
        "皮疹": [
            {
                "disease": "荨麻疹",
                "probability": "高",
                "department": "皮肤科",
                "description": "风团样皮疹，瘙痒明显，时起时消",
                "urgency": "低",
            },
            {
                "disease": "湿疹",
                "probability": "中",
                "department": "皮肤科",
                "description": "红斑、丘疹、水疱，瘙痒，慢性反复发作",
                "urgency": "低",
            },
            {
                "disease": "药物过敏",
                "probability": "中",
                "department": "皮肤科/急诊",
                "description": "用药后出现皮疹，可伴发热、瘙痒",
                "urgency": "中",
            },
        ],
        "关节痛": [
            {
                "disease": "骨关节炎",
                "probability": "高",
                "department": "骨科/风湿免疫科",
                "description": "关节疼痛、僵硬，活动后加重，休息后缓解，多见于中老年",
                "urgency": "低",
            },
            {
                "disease": "类风湿关节炎",
                "probability": "中",
                "department": "风湿免疫科",
                "description": "对称性小关节肿痛，晨僵>1小时",
                "urgency": "低",
            },
            {
                "disease": "痛风",
                "probability": "中",
                "department": "风湿免疫科",
                "description": "急性关节红肿热痛，多见于大脚趾，夜间发作",
                "urgency": "中",
            },
        ],
        "尿频尿急": [
            {
                "disease": "尿路感染",
                "probability": "高",
                "department": "泌尿外科/肾内科",
                "description": "尿频、尿急、尿痛，可伴发热",
                "urgency": "中",
            },
            {
                "disease": "前列腺增生",
                "probability": "中",
                "department": "泌尿外科",
                "description": "尿频、排尿困难，多见于中老年男性",
                "urgency": "低",
            },
            {
                "disease": "膀胱过度活动症",
                "probability": "中",
                "department": "泌尿外科",
                "description": "尿急、尿频，可有急迫性尿失禁",
                "urgency": "低",
            },
        ],
        "失眠": [
            {
                "disease": "失眠症",
                "probability": "高",
                "department": "精神科/睡眠科",
                "description": "入睡困难、早醒或睡眠质量差，影响日间功能",
                "urgency": "低",
            },
            {
                "disease": "焦虑症",
                "probability": "中",
                "department": "精神科/心理科",
                "description": "焦虑不安伴失眠，思虑过多",
                "urgency": "低",
            },
            {
                "disease": "抑郁症",
                "probability": "中",
                "department": "精神科/心理科",
                "description": "失眠伴情绪低落、兴趣减退、乏力",
                "urgency": "中",
            },
        ],
    }

    # 症状组合关联（多症状联合分析）
    _symptom_combinations: Dict[str, Dict[str, Any]] = {
        "发热+咳嗽+咳痰": {
            "disease": "肺炎/下呼吸道感染",
            "department": "呼吸内科",
            "urgency": "中",
            "advice": "建议做胸部影像学检查和血常规，注意休息和补充水分",
        },
        "头痛+发热+鼻塞": {
            "disease": "上呼吸道感染/感冒",
            "department": "内科",
            "urgency": "低",
            "advice": "多休息、多饮水，可对症使用解热镇痛药",
        },
        "腹痛+腹泻+呕吐": {
            "disease": "急性胃肠炎",
            "department": "消化内科/急诊",
            "urgency": "中",
            "advice": "注意补充水分和电解质，避免油腻食物，必要时就医",
        },
        "胸闷+胸痛+气短": {
            "disease": "冠心病/心绞痛",
            "department": "心血管内科/急诊",
            "urgency": "高",
            "advice": "建议立即就医，做心电图和心肌酶谱检查",
        },
        "头晕+头痛+血压升高": {
            "disease": "高血压",
            "department": "心血管内科",
            "urgency": "中",
            "advice": "建议监测血压，必要时就医调整降压方案",
        },
    }

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "症状列表，如 ['头痛', '发热', '咳嗽']",
                },
                "duration": {
                    "type": "string",
                    "description": "症状持续时间，如 '2天'、'1周'、'反复发作3个月'",
                },
                "patient_info": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "description": "患者年龄"},
                        "gender": {"type": "string", "description": "患者性别：男/女"},
                    },
                    "description": "患者基本信息（年龄、性别）",
                },
            },
            "required": ["symptoms"],
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行症状分析

        Args:
            symptoms: 症状列表
            duration: 持续时间
            patient_info: 患者基本信息

        Returns:
            症状分析结果
        """
        symptoms = kwargs.get("symptoms", [])
        duration = kwargs.get("duration", "未提供")
        patient_info = kwargs.get("patient_info", {})

        if not symptoms:
            return {
                "success": False,
                "error": "请提供症状信息",
            }

        # 标准化症状列表
        if isinstance(symptoms, str):
            symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
        else:
            symptoms = [str(s).strip() for s in symptoms if str(s).strip()]

        logger.info(f"症状分析: symptoms={symptoms}, duration={duration}, patient_info={patient_info}")

        # 第一步：检查是否为紧急情况
        emergency_check = self._check_emergency(symptoms)
        if emergency_check["is_emergency"]:
            return {
                "success": True,
                "is_emergency": True,
                "emergency_warning": emergency_check["warning"],
                "immediate_action": "请立即拨打急救电话 120 或前往最近的急诊科就诊！",
                "disclaimer": self._get_disclaimer(),
            }

        # 第二步：分析症状组合
        combination_result = self._analyze_combination(symptoms)

        # 第三步：逐个症状分析
        individual_results = []
        for symptom in symptoms:
            analysis = self._analyze_single_symptom(symptom)
            if analysis:
                individual_results.append(analysis)

        # 第四步：综合分析
        possible_diseases = self._aggregate_results(individual_results, combination_result)

        # 第五步：确定紧急程度
        overall_urgency = self._determine_urgency(possible_diseases, duration)

        # 第六步：推荐就诊科室
        recommended_departments = self._recommend_departments(possible_diseases)

        # 第七步：生成就医建议
        medical_advice = self._generate_advice(
            symptoms, duration, patient_info, overall_urgency, recommended_departments
        )

        return {
            "success": True,
            "is_emergency": False,
            "symptoms": symptoms,
            "duration": duration,
            "patient_info": patient_info,
            "possible_diseases": possible_diseases[:5],  # 最多返回5个可能疾病
            "recommended_departments": recommended_departments[:3],
            "urgency_level": overall_urgency,
            "medical_advice": medical_advice,
            "disclaimer": self._get_disclaimer(),
        }

    def _check_emergency(self, symptoms: List[str]) -> Dict[str, Any]:
        """检查是否为紧急医疗情况"""
        for symptom in symptoms:
            for keyword in self._emergency_keywords:
                if keyword in symptom:
                    return {
                        "is_emergency": True,
                        "warning": f"检测到紧急症状：'{symptom}'，可能属于紧急医疗情况！",
                    }
        return {"is_emergency": False}

    def _analyze_combination(self, symptoms: List[str]) -> Optional[Dict[str, Any]]:
        """分析症状组合"""
        symptom_key = "+".join(sorted(symptoms))

        # 精确匹配
        if symptom_key in self._symptom_combinations:
            return self._symptom_combinations[symptom_key]

        # 部分匹配
        for combo_key, combo_data in self._symptom_combinations.items():
            combo_symptoms = combo_key.split("+")
            match_count = sum(1 for s in symptoms if s in combo_symptoms)
            if match_count >= 2:
                return combo_data

        return None

    def _analyze_single_symptom(self, symptom: str) -> Optional[Dict[str, Any]]:
        """分析单个症状"""
        # 精确匹配
        if symptom in self._symptom_disease_mapping:
            return {
                "symptom": symptom,
                "possible_diseases": self._symptom_disease_mapping[symptom],
            }

        # 模糊匹配
        for key, diseases in self._symptom_disease_mapping.items():
            if symptom in key or key in symptom:
                return {
                    "symptom": symptom,
                    "matched_symptom": key,
                    "possible_diseases": diseases,
                }

        return None

    def _aggregate_results(
        self,
        individual_results: List[Dict],
        combination_result: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        """综合分析结果"""
        disease_scores: Dict[str, Dict[str, Any]] = {}

        # 添加症状组合结果（权重更高）
        if combination_result:
            disease_name = combination_result.get("disease", "")
            if disease_name:
                disease_scores[disease_name] = {
                    "disease": disease_name,
                    "probability": "高",
                    "department": combination_result.get("department", ""),
                    "description": combination_result.get("advice", ""),
                    "urgency": combination_result.get("urgency", "中"),
                    "score": 3.0,
                }

        # 添加单个症状分析结果
        for result in individual_results:
            for disease in result.get("possible_diseases", []):
                name = disease.get("disease", "")
                if name in disease_scores:
                    disease_scores[name]["score"] += 1.0
                    # 提升概率等级
                    if disease_scores[name]["score"] >= 2.0:
                        disease_scores[name]["probability"] = "高"
                else:
                    disease_scores[name] = {
                        "disease": disease.get("disease", ""),
                        "probability": disease.get("probability", "低"),
                        "department": disease.get("department", ""),
                        "description": disease.get("description", ""),
                        "urgency": disease.get("urgency", "低"),
                        "score": 1.0,
                    }

        # 按分数排序
        sorted_diseases = sorted(
            disease_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        # 移除 score 字段，返回干净的结果
        for d in sorted_diseases:
            d.pop("score", None)

        return sorted_diseases

    def _determine_urgency(
        self,
        possible_diseases: List[Dict],
        duration: str,
    ) -> str:
        """确定紧急程度"""
        urgency_levels = {"高": 3, "中": 2, "低": 1}
        max_urgency = "低"

        for disease in possible_diseases:
            u = disease.get("urgency", "低")
            if urgency_levels.get(u, 0) > urgency_levels.get(max_urgency, 0):
                max_urgency = u

        # 持续时间较长也提升紧急程度
        duration_lower = duration.lower()
        if any(kw in duration_lower for kw in ["月", "周", "反复", "持续"]):
            if max_urgency == "低":
                max_urgency = "中"

        return max_urgency

    def _recommend_departments(self, possible_diseases: List[Dict]) -> List[str]:
        """推荐就诊科室"""
        departments: Dict[str, int] = {}

        for disease in possible_diseases:
            dept = disease.get("department", "")
            if dept:
                # 可能有多个科室
                for d in dept.replace("/", "、").split("、"):
                    d = d.strip()
                    if d:
                        departments[d] = departments.get(d, 0) + 1

        # 按出现频率排序
        sorted_depts = sorted(departments.items(), key=lambda x: x[1], reverse=True)
        return [d[0] for d in sorted_depts]

    def _generate_advice(
        self,
        symptoms: List[str],
        duration: str,
        patient_info: Dict,
        urgency: str,
        departments: List[str],
    ) -> List[str]:
        """生成就医建议"""
        advice = []

        # 基础建议
        advice.append("以上分析仅供参考，建议前往正规医疗机构就诊以获得准确诊断。")

        if urgency == "高":
            advice.append("您的症状较为紧急，建议尽快前往医院急诊科就诊。")
        elif urgency == "中":
            advice.append("建议您在近期内安排就医，进行详细检查。")
        else:
            advice.append("如症状持续不缓解或加重，请及时就医。")

        # 科室建议
        if departments:
            advice.append(f"建议就诊科室：{'、'.join(departments[:3])}")

        # 患者特殊建议
        age = patient_info.get("age")
        gender = patient_info.get("gender")

        if age is not None:
            if age >= 65:
                advice.append("老年患者建议尽早就诊，注意监测生命体征。")
            elif age <= 6:
                advice.append("儿童患者建议及时就诊儿科，注意观察病情变化。")

        if gender == "女":
            advice.append("如涉及生育相关问题，建议告知医生月经史和生育史。")

        # 持续时间建议
        if duration and "天" in str(duration):
            try:
                days = int(''.join(filter(str.isdigit, str(duration))))
                if days >= 7:
                    advice.append(f"症状已持续{days}天，建议尽快就医检查。")
            except ValueError:
                pass

        return advice

    def _get_disclaimer(self) -> str:
        """获取医疗免责声明"""
        return (
            "【医疗免责声明】本症状分析结果仅供参考，不能替代专业医生的诊断和治疗建议。"
            "疾病的诊断需要结合详细的病史、体格检查和辅助检查综合判断。"
            "如有任何健康问题，请及时前往正规医疗机构就诊。"
            "紧急情况请立即拨打急救电话 120。"
        )
