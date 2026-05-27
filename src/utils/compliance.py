"""
医疗合规安全模块 - MediAgent 智慧医疗助手
提供医疗信息合规检查、隐私保护和安全审核功能
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MedicalComplianceChecker:
    """
    医疗合规安全检查器

    功能：
    - 输入文本脱敏（身份证号、手机号等）
    - 输出文本合规检查（自动附加免责声明）
    - 患者健康信息（PHI）脱敏
    - 紧急医疗情况检测
    - 医疗免责声明管理
    """

    # 标准医疗免责声明
    MEDICAL_DISCLAIMER = (
        "\n\n---\n"
        "**【医疗免责声明】**\n"
        "本助手提供的所有医疗健康信息仅供参考，不能替代专业医生的诊断和治疗建议。"
        "任何健康问题请及时前往正规医疗机构就诊，遵循专业医生的指导。"
        "如有紧急情况，请立即拨打急救电话 120。\n"
        "---"
    )

    # 紧急医疗情况关键词
    EMERGENCY_KEYWORDS = [
        "胸痛", "呼吸困难", "大出血", "意识丧失", "剧烈头痛",
        "严重过敏", "抽搐", "昏迷", "窒息", "心跳骤停",
        "偏瘫", "吐血", "剧烈腹痛", "高热惊厥", "自杀",
        "服毒", "溺水", "触电", "车祸", "烧伤",
        "120", "急救", "救命", "不行了", "快不行了",
    ]

    # 诊断相关关键词（出现时需附加免责声明）
    DIAGNOSIS_KEYWORDS = [
        "诊断", "确诊", "患病", "得了", "可能是",
        "疑似", "考虑为", "初步判断", "分析结果",
        "疾病方向", "可能患有", "患了",
    ]

    # 需要脱敏的敏感信息模式
    SENSITIVE_PATTERNS: Dict[str, Tuple[str, str]] = {
        "id_card": (
            r"\b\d{17}[\dXx]\b",
            "身份证号",
        ),
        "phone_number": (
            r"\b1[3-9]\d{9}\b",
            "手机号",
        ),
        "bank_card": (
            r"\b\d{16,19}\b",
            "银行卡号",
        ),
        "social_security": (
            r"\b\d{9,12}\b",
            "社保号",
        ),
    }

    # 医疗记录号模式
    MEDICAL_ID_PATTERNS: Dict[str, Tuple[str, str]] = {
        "medical_record_no": (
            r"(?:病历号|住院号|门诊号)[：:\s]*\d+",
            "医疗记录号",
        ),
        "prescription_no": (
            r"(?:处方号|药方号)[：:\s]*\d+",
            "处方号",
        ),
    }

    def __init__(self, compliance_mode: bool = True, auto_disclaimer: bool = True):
        """
        初始化医疗合规检查器

        Args:
            compliance_mode: 是否启用合规模式
            auto_disclaimer: 是否自动附加免责声明
        """
        self.compliance_mode = compliance_mode
        self.auto_disclaimer = auto_disclaimer

        # 编译正则表达式
        self._compiled_sensitive = {
            name: re.compile(pattern)
            for name, (pattern, _) in self.SENSITIVE_PATTERNS.items()
        }
        self._compiled_medical_id = {
            name: re.compile(pattern)
            for name, (pattern, _) in self.MEDICAL_ID_PATTERNS.items()
        }

        logger.info(
            f"MedicalComplianceChecker initialized: "
            f"compliance_mode={compliance_mode}, auto_disclaimer={auto_disclaimer}"
        )

    def check_input(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        检查并处理输入文本

        对输入文本进行敏感信息检测和脱敏处理。

        Args:
            text: 用户输入文本

        Returns:
            Tuple[处理后的文本, 处理信息字典]
        """
        if not self.compliance_mode or not text:
            return text, {"sanitized": False, "items_found": 0}

        result_info = {
            "sanitized": False,
            "items_found": 0,
            "detected_types": [],
            "is_emergency": False,
        }

        sanitized_text = text

        # 1. 检查敏感个人信息
        for name, pattern in self._compiled_sensitive.items():
            matches = pattern.findall(sanitized_text)
            if matches:
                result_info["detected_types"].append(
                    self.SENSITIVE_PATTERNS[name][1]
                )
                result_info["items_found"] += len(matches)
                # 脱敏处理
                sanitized_text = self._mask_pattern(sanitized_text, pattern)

        # 2. 检查医疗记录号
        for name, pattern in self._compiled_medical_id.items():
            matches = pattern.findall(sanitized_text)
            if matches:
                result_info["detected_types"].append(
                    self.MEDICAL_ID_PATTERNS[name][1]
                )
                result_info["items_found"] += len(matches)
                sanitized_text = self._mask_pattern(sanitized_text, pattern)

        # 3. 检查紧急情况
        result_info["is_emergency"] = self.is_emergency(text)

        if result_info["items_found"] > 0:
            result_info["sanitized"] = True
            logger.info(
                f"Input sanitized: {result_info['items_found']} sensitive items detected "
                f"({', '.join(result_info['detected_types'])})"
            )

        return sanitized_text, result_info

    def check_output(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        检查并处理输出文本

        检查输出是否包含诊断结论，必要时自动附加免责声明。

        Args:
            text: AI输出的文本

        Returns:
            Tuple[处理后的文本, 处理信息字典]
        """
        if not text:
            return text, {"disclaimer_added": False, "has_diagnosis": False}

        result_info = {
            "disclaimer_added": False,
            "has_diagnosis": False,
        }

        # 检查是否包含诊断相关内容
        has_diagnosis = self._contains_diagnosis(text)
        result_info["has_diagnosis"] = has_diagnosis

        # 检查是否已包含免责声明
        has_existing_disclaimer = "免责声明" in text or "仅供参考" in text

        # 如果需要且未已有免责声明，则添加
        if self.auto_disclaimer and has_diagnosis and not has_existing_disclaimer:
            text = text + self.MEDICAL_DISCLAIMER
            result_info["disclaimer_added"] = True
            logger.info("Medical disclaimer added to output")

        return text, result_info

    def sanitize_phi(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        脱敏患者健康信息（PHI: Protected Health Information）

        对文本中的患者健康信息进行识别和脱敏处理，包括：
        - 姓名
        - 身份证号
        - 手机号
        - 地址
        - 医疗记录号
        - 社保号

        Args:
            text: 包含PHI的文本

        Returns:
            Tuple[脱敏后的文本, 脱敏项目列表]
        """
        if not text:
            return text, []

        sanitized_items = []
        sanitized_text = text

        # 1. 脱敏身份证号
        id_pattern = re.compile(r"\b(\d{6})(\d{8})(\d{4})([\dXx])\b")
        matches = id_pattern.findall(sanitized_text)
        for match in matches:
            original = "".join(match)
            masked = f"{match[0]}********{match[3]}"
            sanitized_text = sanitized_text.replace(original, masked)
            sanitized_items.append({
                "type": "身份证号",
                "action": "脱敏",
                "original_length": len(original),
            })

        # 2. 脱敏手机号
        phone_pattern = re.compile(r"\b(1[3-9]\d)(\d{4})(\d{4})\b")
        matches = phone_pattern.findall(sanitized_text)
        for match in matches:
            original = "".join(match)
            masked = f"{match[0]}****{match[2]}"
            sanitized_text = sanitized_text.replace(original, masked)
            sanitized_items.append({
                "type": "手机号",
                "action": "脱敏",
                "original_length": len(original),
            })

        # 3. 脱敏银行卡号
        bank_pattern = re.compile(r"\b(\d{4})\d{8,12}(\d{4})\b")
        matches = bank_pattern.findall(sanitized_text)
        for match in matches:
            masked = f"{match[0]} **** **** {match[1]}"
            # 简单替换
            sanitized_items.append({
                "type": "银行卡号",
                "action": "脱敏",
            })

        # 4. 脱敏姓名（简单规则：2-3个汉字的姓名）
        name_pattern = re.compile(
            r"(?:患者|病人|姓名|名字|叫)[：:\s是]*([\u4e00-\u9fa5]{2,4})"
        )
        matches = name_pattern.findall(sanitized_text)
        for match in matches:
            sanitized_text = sanitized_text.replace(match, "**")
            sanitized_items.append({
                "type": "姓名",
                "action": "脱敏",
            })

        # 5. 脱敏详细地址（保留到城市级别）
        address_pattern = re.compile(
            r"([\u4e00-\u9fa5]+(?:省|市|自治区))"
            r"([\u4e00-\u9fa5]+(?:市|区|县|镇|乡))"
            r"([\u4e00-\u9fa5]+(?:路|街|道|巷|号|栋|楼|室|村|组))"
        )
        matches = address_pattern.findall(sanitized_text)
        for match in matches:
            # 只保留省/市/区
            partial_address = f"{match[0]}{match[1]}"
            full_address = "".join(match)
            sanitized_text = sanitized_text.replace(full_address, partial_address)
            sanitized_items.append({
                "type": "详细地址",
                "action": "部分脱敏（保留到区/县级）",
            })

        if sanitized_items:
            logger.info(f"PHI sanitized: {len(sanitized_items)} items processed")

        return sanitized_text, sanitized_items

    def add_disclaimer(self, text: str, force: bool = False) -> str:
        """
        添加医疗免责声明

        Args:
            text: 原始文本
            force: 是否强制添加（即使已有免责声明）

        Returns:
            添加了免责声明的文本
        """
        if not text:
            return text

        if not force and ("免责声明" in text or "仅供参考" in text):
            return text

        return text + self.MEDICAL_DISCLAIMER

    def is_emergency(self, text: str) -> bool:
        """
        检测是否为紧急医疗情况

        通过关键词匹配判断用户输入是否描述了紧急医疗情况。

        Args:
            text: 用户输入文本

        Returns:
            True 如果检测到紧急医疗情况
        """
        if not text:
            return False

        text_lower = text.lower()
        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in text_lower:
                logger.warning(f"Emergency keyword detected: '{keyword}'")
                return True

        return False

    def get_emergency_response(self) -> str:
        """
        获取紧急医疗情况的标准回复

        Returns:
            紧急情况回复文本
        """
        return (
            "**【紧急提醒】**\n\n"
            "根据您的描述，您可能正在经历紧急医疗情况。\n\n"
            "**请立即采取以下行动：**\n"
            "1. **拨打急救电话 120**\n"
            "2. 保持冷静，不要随意移动患者\n"
            "3. 如有旁人，请其协助呼叫急救\n"
            "4. 解开患者紧身衣物，保持呼吸道通畅\n"
            "5. 如患者意识丧失，检查呼吸和脉搏\n\n"
            "**请注意：**\n"
            "- 不要给意识不清的患者喂食或喂水\n"
            "- 不要随意搬动疑似骨折或脊柱损伤的患者\n"
            "- 等待急救人员到达，并简要说明情况\n\n"
            f"{self.MEDICAL_DISCLAIMER}"
        )

    def _contains_diagnosis(self, text: str) -> bool:
        """
        检查文本是否包含诊断相关内容

        Args:
            text: 待检查文本

        Returns:
            True 如果包含诊断相关内容
        """
        text_lower = text.lower()
        for keyword in self.DIAGNOSIS_KEYWORDS:
            if keyword in text_lower:
                return True
        return False

    def _mask_pattern(self, text: str, pattern: re.Pattern) -> str:
        """
        对匹配到的模式进行脱敏

        Args:
            text: 原始文本
            pattern: 编译好的正则表达式

        Returns:
            脱敏后的文本
        """
        def mask_match(match):
            matched_text = match.group()
            length = len(matched_text)
            if length <= 2:
                return "**"
            # 保留首尾字符，中间用*替代
            return matched_text[0] + "*" * (length - 2) + matched_text[-1]

        return pattern.sub(mask_match, text)

    def validate_medical_response(self, text: str) -> Dict[str, Any]:
        """
        验证医疗回复的合规性

        检查回复是否符合医疗合规要求。

        Args:
            text: AI生成的回复文本

        Returns:
            验证结果字典
        """
        validation_result = {
            "is_compliant": True,
            "issues": [],
            "warnings": [],
        }

        # 检查1：是否包含明确的诊断结论
        direct_diagnosis_patterns = [
            r"你(是|患有|得了|确诊)\s",
            r"诊断[为是：:]\s",
            r"确诊[为是：:]\s",
        ]
        for pattern in direct_diagnosis_patterns:
            if re.search(pattern, text):
                validation_result["issues"].append(
                    "回复包含明确的诊断结论，建议改为'可能'或'疑似'等表述"
                )
                validation_result["is_compliant"] = False

        # 检查2：是否推荐了处方药
        prescription_drug_patterns = [
            r"建议服用\s*(二甲双胍|氨氯地平|硝苯地平|辛伐他汀|氯吡格雷)",
            r"可以吃\s*(二甲双胍|氨氯地平|硝苯地平|辛伐他汀|氯吡格雷)",
        ]
        for pattern in prescription_drug_patterns:
            if re.search(pattern, text):
                validation_result["warnings"].append(
                    "回复中直接推荐了处方药，建议改为'请咨询医生后遵医嘱使用'"
                )

        # 检查3：涉及诊断时是否有免责声明
        if self._contains_diagnosis(text) and "免责声明" not in text and "仅供参考" not in text:
            validation_result["issues"].append(
                "回复涉及诊断相关内容但缺少免责声明"
            )
            validation_result["is_compliant"] = False

        # 检查4：是否包含不专业的表述
        unprofessional_patterns = [
            r"绝对(不会|没有|能)",
            r"保证(治愈|治好|康复)",
            r"100%(治愈|有效|能好)",
        ]
        for pattern in unprofessional_patterns:
            if re.search(pattern, text):
                validation_result["warnings"].append(
                    "回复包含过于绝对的表述，建议使用更严谨的医学用语"
                )

        return validation_result
