"""
Compliance Tests for MediAgent
Tests input sanitization, PHI sanitization, emergency detection,
output compliance checking, disclaimer addition, and medical response validation.
"""
import pytest
from src.utils.compliance import MedicalComplianceChecker


# ==================== Input Sanitization Tests ====================


class TestInputSanitization:
    """输入文本脱敏测试"""

    def test_id_card_sanitization(self):
        """身份证号应被脱敏"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "我身份证号是110101199001011234"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert info["items_found"] >= 1
        assert "身份证号" in info["detected_types"]
        # 原始身份证号不应出现在脱敏后的文本中
        assert "110101199001011234" not in sanitized

    def test_phone_number_sanitization(self):
        """手机号应被脱敏"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "我的手机号是13812345678"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "手机号" in info["detected_types"]
        assert "13812345678" not in sanitized

    def test_bank_card_sanitization(self):
        """银行卡号应被脱敏"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "银行卡号是6222021234567890123"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "银行卡号" in info["detected_types"]

    def test_multiple_sensitive_types(self):
        """同时包含多种敏感信息应全部被检测"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "身份证110101199001011234，手机13812345678"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert info["items_found"] >= 2
        assert len(info["detected_types"]) >= 2

    def test_no_sensitive_info(self):
        """不包含敏感信息的文本不应被修改"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "我最近头痛，有点发热"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is False
        assert info["items_found"] == 0
        assert sanitized == text

    def test_compliance_mode_disabled(self):
        """关闭合规模式时不应脱敏"""
        checker = MedicalComplianceChecker(compliance_mode=False)
        text = "身份证号是110101199001011234"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is False
        assert sanitized == text

    def test_empty_input(self):
        """空输入应返回空文本"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        sanitized, info = checker.check_input("")

        assert sanitized == ""
        assert info["sanitized"] is False

    def test_none_input(self):
        """None 输入应返回 None"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        sanitized, info = checker.check_input(None)

        assert sanitized is None
        assert info["sanitized"] is False

    def test_social_security_number(self):
        """社保号应被脱敏"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "我的社保号是123456789"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "社保号" in info["detected_types"]

    def test_id_card_with_x_suffix(self):
        """末尾为 X 的身份证号也应被脱敏"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "身份证号11010119900101123X"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "身份证号" in info["detected_types"]

    def test_id_card_with_lowercase_x(self):
        """末尾为小写 x 的身份证号也应被脱敏"""
        checker = MedicalComplianceChecker(compliance_mode=True)
        text = "身份证号11010119900101123x"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "身份证号" in info["detected_types"]


# ==================== PHI Sanitization Tests ====================


class TestPHISanitization:
    """患者健康信息（PHI）脱敏测试"""

    def test_phi_id_card(self):
        """PHI 脱敏应处理身份证号"""
        checker = MedicalComplianceChecker()
        text = "患者身份证号110101199001011234"
        sanitized, items = checker.sanitize_phi(text)

        assert len(items) >= 1
        id_items = [i for i in items if i["type"] == "身份证号"]
        assert len(id_items) >= 1
        assert "110101199001011234" not in sanitized

    def test_phi_phone_number(self):
        """PHI 脱敏应处理手机号"""
        checker = MedicalComplianceChecker()
        text = "联系电话13812345678"
        sanitized, items = checker.sanitize_phi(text)

        assert len(items) >= 1
        phone_items = [i for i in items if i["type"] == "手机号"]
        assert len(phone_items) >= 1
        assert "13812345678" not in sanitized

    def test_phi_name(self):
        """PHI 脱敏应处理姓名"""
        checker = MedicalComplianceChecker()
        text = "患者叫张三"
        sanitized, items = checker.sanitize_phi(text)

        assert len(items) >= 1
        name_items = [i for i in items if i["type"] == "姓名"]
        assert len(name_items) >= 1
        assert "张三" not in sanitized

    def test_phi_name_with_label(self):
        """不同标签的姓名也应被脱敏"""
        checker = MedicalComplianceChecker()
        text = "病人姓名：李四"
        sanitized, items = checker.sanitize_phi(text)

        name_items = [i for i in items if i["type"] == "姓名"]
        assert len(name_items) >= 1

    def test_phi_address(self):
        """PHI 脱敏应处理详细地址"""
        checker = MedicalComplianceChecker()
        text = "地址：广东省深圳市南山区科技路100号"
        sanitized, items = checker.sanitize_phi(text)

        address_items = [i for i in items if i["type"] == "详细地址"]
        assert len(address_items) >= 1
        # 详细地址应被截断
        assert "科技路100号" not in sanitized

    def test_phi_empty_text(self):
        """空文本不应产生脱敏项"""
        checker = MedicalComplianceChecker()
        sanitized, items = checker.sanitize_phi("")

        assert len(items) == 0

    def test_phi_no_phi_in_text(self):
        """不含 PHI 的文本不应被修改"""
        checker = MedicalComplianceChecker()
        text = "我最近感觉不太好，头痛乏力"
        sanitized, items = checker.sanitize_phi(text)

        assert len(items) == 0
        assert sanitized == text

    def test_phi_medical_record_number(self):
        """病历号应被脱敏"""
        checker = MedicalComplianceChecker()
        text = "病历号：20240001"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "医疗记录号" in info["detected_types"]

    def test_phi_prescription_number(self):
        """处方号应被脱敏"""
        checker = MedicalComplianceChecker()
        text = "处方号：RX20240001"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "处方号" in info["detected_types"]


# ==================== Emergency Detection Tests ====================


class TestEmergencyDetection:
    """紧急医疗情况检测测试"""

    @pytest.fixture
    def checker(self):
        return MedicalComplianceChecker()

    def test_chest_pain(self, checker):
        """胸痛应被检测为紧急情况"""
        assert checker.is_emergency("我胸痛") is True

    def test_difficulty_breathing(self, checker):
        """呼吸困难应被检测为紧急情况"""
        assert checker.is_emergency("呼吸困难") is True

    def test_heavy_bleeding(self, checker):
        """大出血应被检测为紧急情况"""
        assert checker.is_emergency("大出血") is True

    def test_loss_of_consciousness(self, checker):
        """意识丧失应被检测为紧急情况"""
        assert checker.is_emergency("意识丧失") is True

    def test_severe_allergy(self, checker):
        """严重过敏应被检测为紧急情况"""
        assert checker.is_emergency("严重过敏") is True

    def test_seizure(self, checker):
        """抽搐应被检测为紧急情况"""
        assert checker.is_emergency("抽搐") is True

    def test_coma(self, checker):
        """昏迷应被检测为紧急情况"""
        assert checker.is_emergency("昏迷") is True

    def test_call_120(self, checker):
        """提到120应被检测为紧急情况"""
        assert checker.is_emergency("请帮我打120") is True

    def test_suicide_keyword(self, checker):
        """自杀关键词应被检测为紧急情况"""
        assert checker.is_emergency("我想自杀") is True

    def test_non_emergency(self, checker):
        """普通头痛不应被检测为紧急情况"""
        assert checker.is_emergency("我有点头痛") is False

    def test_mild_cold(self, checker):
        """普通感冒不应被检测为紧急情况"""
        assert checker.is_emergency("我感冒了，流鼻涕") is False

    def test_empty_text(self, checker):
        """空文本不应被检测为紧急情况"""
        assert checker.is_emergency("") is False

    def test_none_text(self, checker):
        """None 文本不应被检测为紧急情况"""
        assert checker.is_emergency(None) is False

    def test_emergency_in_input_check(self, checker):
        """check_input 应同时检测紧急情况和敏感信息"""
        text = "我胸痛，身份证号110101199001011234"
        sanitized, info = checker.check_input(text)

        assert info["is_emergency"] is True
        assert info["sanitized"] is True

    def test_get_emergency_response(self, checker):
        """紧急回复应包含关键信息"""
        response = checker.get_emergency_response()

        assert "120" in response
        assert "急救" in response
        assert "免责声明" in response
        assert len(response) > 100


# ==================== Output Compliance Tests ====================


class TestOutputCompliance:
    """输出合规检查测试"""

    def test_output_with_diagnosis_adds_disclaimer(self):
        """包含诊断内容的输出应自动附加免责声明"""
        checker = MedicalComplianceChecker(auto_disclaimer=True)
        text = "根据分析，您可能患有上呼吸道感染。"
        result, info = checker.check_output(text)

        assert info["has_diagnosis"] is True
        assert info["disclaimer_added"] is True
        assert "免责声明" in result

    def test_output_without_diagnosis_no_disclaimer(self):
        """不包含诊断内容的输出不应附加免责声明"""
        checker = MedicalComplianceChecker(auto_disclaimer=True)
        text = "建议您多喝水，注意休息。"
        result, info = checker.check_output(text)

        assert info["has_diagnosis"] is False
        assert info["disclaimer_added"] is False
        assert "免责声明" not in result

    def test_output_existing_disclaimer_not_duplicated(self):
        """已有免责声明的输出不应重复添加"""
        checker = MedicalComplianceChecker(auto_disclaimer=True)
        text = "您可能患有感冒，仅供参考。"
        result, info = checker.check_output(text)

        assert info["has_diagnosis"] is True
        assert info["disclaimer_added"] is False
        # 不应有两个免责声明
        assert result.count("免责声明") <= 1

    def test_output_with_diagnosis_keyword(self):
        """各种诊断关键词应触发免责声明"""
        checker = MedicalComplianceChecker(auto_disclaimer=True)

        diagnosis_texts = [
            "初步判断您患有高血压。",
            "疑似上呼吸道感染。",
            "分析结果显示可能是感冒。",
            "考虑为过敏性鼻炎。",
        ]

        for text in diagnosis_texts:
            result, info = checker.check_output(text)
            assert info["has_diagnosis"] is True, f"应检测到诊断关键词: {text}"

    def test_output_empty_text(self):
        """空输出不应附加免责声明"""
        checker = MedicalComplianceChecker(auto_disclaimer=True)
        result, info = checker.check_output("")

        assert info["disclaimer_added"] is False
        assert result == ""

    def test_auto_disclaimer_disabled(self):
        """禁用自动免责声明时不应添加"""
        checker = MedicalComplianceChecker(auto_disclaimer=False)
        text = "根据分析，您可能患有感冒。"
        result, info = checker.check_output(text)

        assert info["has_diagnosis"] is True
        assert info["disclaimer_added"] is False
        assert "免责声明" not in result


# ==================== Disclaimer Tests ====================


class TestDisclaimer:
    """免责声明测试"""

    def test_add_disclaimer_basic(self):
        """基本免责声明添加"""
        checker = MedicalComplianceChecker()
        text = "这是一条医疗建议。"
        result = checker.add_disclaimer(text)

        assert "免责声明" in result
        assert "仅供参考" in result
        assert "120" in result

    def test_add_disclaimer_force(self):
        """强制添加免责声明"""
        checker = MedicalComplianceChecker()
        text = "已有免责声明的文本，仅供参考。"
        result = checker.add_disclaimer(text, force=True)

        # 强制添加应产生两个免责声明
        assert result.count("免责声明") >= 2

    def test_add_disclaimer_no_duplicate(self):
        """已有免责声明时不应重复添加"""
        checker = MedicalComplianceChecker()
        text = "这是一条建议，仅供参考。"
        result = checker.add_disclaimer(text)

        assert result.count("免责声明") == 1

    def test_add_disclaimer_empty_text(self):
        """空文本添加免责声明应返回免责声明"""
        checker = MedicalComplianceChecker()
        result = checker.add_disclaimer("")

        assert result == ""

    def test_disclaimer_content(self):
        """免责声明应包含关键内容"""
        checker = MedicalComplianceChecker()
        text = "测试"
        result = checker.add_disclaimer(text)

        assert "医疗健康信息" in result
        assert "专业医生" in result
        assert "正规医疗机构" in result


# ==================== Medical Response Validation Tests ====================


class TestMedicalResponseValidation:
    """医疗回复合规性验证测试"""

    def test_compliant_response(self):
        """合规的医疗回复应通过验证"""
        checker = MedicalComplianceChecker()
        text = "根据您的症状，可能是普通感冒。建议多休息，多饮水。如症状持续，请及时就医。"

        result = checker.validate_medical_response(text)

        assert result["is_compliant"] is True
        assert len(result["issues"]) == 0

    def test_direct_diagnosis_not_compliant(self):
        """直接诊断结论应标记为不合规"""
        checker = MedicalComplianceChecker()
        text = "你患有高血压。"

        result = checker.validate_medical_response(text)

        assert result["is_compliant"] is False
        assert len(result["issues"]) >= 1
        assert any("诊断结论" in issue for issue in result["issues"])

    def test_prescription_drug_recommendation_warning(self):
        """直接推荐处方药应产生警告"""
        checker = MedicalComplianceChecker()
        text = "建议服用二甲双胍来控制血糖。"

        result = checker.validate_medical_response(text)

        assert len(result["warnings"]) >= 1
        assert any("处方药" in w for w in result["warnings"])

    def test_missing_disclaimer_with_diagnosis(self):
        """包含诊断但缺少免责声明应标记为不合规"""
        checker = MedicalComplianceChecker()
        text = "初步判断您可能患有糖尿病。"

        result = checker.validate_medical_response(text)

        assert result["is_compliant"] is False
        assert any("免责声明" in issue for issue in result["issues"])

    def test_absolute_claims_warning(self):
        """过于绝对的表述应产生警告"""
        checker = MedicalComplianceChecker()
        text = "这个药绝对能治愈你的病。"

        result = checker.validate_medical_response(text)

        assert len(result["warnings"]) >= 1
        assert any("绝对" in w for w in result["warnings"])

    def test_guarantee_claims_warning(self):
        """保证治愈的表述应产生警告"""
        checker = MedicalComplianceChecker()
        text = "我们保证治愈您的疾病。"

        result = checker.validate_medical_response(text)

        assert len(result["warnings"]) >= 1

    def test_100_percent_claims_warning(self):
        """100% 有效的表述应产生警告"""
        checker = MedicalComplianceChecker()
        text = "这个治疗方案100%有效。"

        result = checker.validate_medical_response(text)

        assert len(result["warnings"]) >= 1

    def test_comprehensive_validation(self):
        """综合验证：多个问题同时存在"""
        checker = MedicalComplianceChecker()
        text = "诊断：你患有高血压。建议服用氨氯地平。"

        result = checker.validate_medical_response(text)

        # 至少应有诊断结论问题和处方药警告
        assert result["is_compliant"] is False
        assert len(result["issues"]) >= 1 or len(result["warnings"]) >= 1

    def test_validation_result_structure(self):
        """验证结果应包含正确的结构"""
        checker = MedicalComplianceChecker()
        result = checker.validate_medical_response("普通回复")

        assert "is_compliant" in result
        assert "issues" in result
        assert "warnings" in result
        assert isinstance(result["issues"], list)
        assert isinstance(result["warnings"], list)


# ==================== Edge Cases ====================


class TestComplianceEdgeCases:
    """合规检查边界情况测试"""

    def test_very_long_text(self):
        """超长文本的合规检查"""
        checker = MedicalComplianceChecker()
        long_text = "测试内容。" * 10000 + "身份证号110101199001011234"

        sanitized, info = checker.check_input(long_text)

        assert info["sanitized"] is True
        assert "110101199001011234" not in sanitized

    def test_unicode_text(self):
        """Unicode 文本的合规检查"""
        checker = MedicalComplianceChecker()
        text = "患者姓名：山田太郎，電話番号：13812345678"

        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True

    def test_mixed_content(self):
        """混合内容的合规检查"""
        checker = MedicalComplianceChecker()
        text = """
        患者信息：
        姓名：张三
        身份证号：110101199001011234
        手机号：13812345678
        症状：胸痛、呼吸困难
        """

        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert info["is_emergency"] is True
        assert info["items_found"] >= 2

    def test_special_characters(self):
        """特殊字符不应影响脱敏"""
        checker = MedicalComplianceChecker()
        text = "身份证号：110101199001011234！@#￥%……&*（）"
        sanitized, info = checker.check_input(text)

        assert info["sanitized"] is True
        assert "110101199001011234" not in sanitized

    def test_consecutive_sensitive_values(self):
        """连续的敏感信息应全部被脱敏"""
        checker = MedicalComplianceChecker()
        text = "13812345678 13987654321"
        sanitized, info = checker.check_input(text)

        assert info["items_found"] >= 2
        assert "13812345678" not in sanitized
        assert "13987654321" not in sanitized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
