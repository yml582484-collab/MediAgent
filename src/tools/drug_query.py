"""
药品信息查询工具 - MediAgent 智慧医疗助手
提供药品名称、适应症、用法用量、不良反应、禁忌症等药品信息查询功能
支持内置数据库查询 + 外部API抓取 + 网络搜索三层Fallback
包含内容验证、结果缓存、并发执行等增强功能
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from hashlib import md5

from .base import BaseTool, tool_registry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DrugQueryTool(BaseTool):
    """
    药品信息查询工具（增强版 v2.1）

    功能：
    - 查询药品基本信息（通用名、商品名、分类等）
    - 查询药品用法用量
    - 查询药品不良反应
    - 查询药物相互作用

    查询策略（三层Fallback + 并发优化）：
    1. 内置药品数据库（30种常见药品）- 本地查询，毫秒级
    2. 外部药品API抓取（丁香园、百度百科等）- 并发执行
    3. 网络搜索（DuckDuckGo、Bing等）- 并发执行

    增强功能：
    - 内容验证：过滤空/低质量结果
    - 结果缓存：TTL 24小时，避免重复查询
    - 并发执行：第二层+第三层并发，提升性能
    - 风险标记：外部数据源结果添加风险提示

    当内置数据库未找到时，自动并发尝试外部数据源和网络搜索。
    """

    name = "drug_query"
    description = "查询药品信息，包括药品名称、适应症、用法用量、不良反应、禁忌症等"

    # 结果缓存（TTL 24小时）
    _result_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
    _cache_ttl_hours = 24

    # 内容验证的关键字段（至少包含一项）
    _key_fields = ["indications", "dosage", "adverse_reactions", "contraindications", "summary", "generic_name"]

    # 内置常见药品信息字典（20种以上常见药品）
    _drug_database: Dict[str, Dict[str, Any]] = {
        "布洛芬": {
            "generic_name": "布洛芬（Ibuprofen）",
            "category": "非甾体抗炎药（NSAIDs）",
            "prescription": "OTC（非处方药，部分剂型为处方药）",
            "indications": "用于缓解轻至中度疼痛，如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经。也用于普通感冒或流行性感冒引起的发热。",
            "dosage": {
                "成人常规": "口服，一次200-400mg，每4-6小时一次，24小时不超过1200mg（OTC）或2400mg（处方）",
                "儿童": "口服，每次5-10mg/kg，每6-8小时一次",
            },
            "adverse_reactions": [
                "胃肠道：恶心、呕吐、腹痛、消化不良、胃肠道溃疡",
                "神经系统：头晕、头痛、嗜睡",
                "过敏反应：皮疹、瘙痒，罕见过敏性休克",
                "心血管：长期使用可能增加心血管事件风险",
                "肾脏：长期大量使用可能导致肾功能损害",
            ],
            "contraindications": [
                "对布洛芬或其他NSAIDs过敏者禁用",
                "活动性消化性溃疡患者禁用",
                "严重肝肾功能不全者禁用",
                "孕晚期妇女禁用",
                "严重心力衰竭者禁用",
            ],
            "interactions": [
                "与阿司匹林合用可能降低阿司匹林的心血管保护作用",
                "与抗凝药（如华法林）合用增加出血风险",
                "与糖皮质激素合用增加胃肠道溃疡风险",
                "与利尿剂合用可能降低利尿效果",
                "避免与其他NSAIDs合用",
            ],
            "note": "饭后服用可减少胃肠道不适。不宜长期使用，如症状持续应就医。",
        },
        "对乙酰氨基酚": {
            "generic_name": "对乙酰氨基酚（Acetaminophen/Paracetamol）",
            "category": "解热镇痛药",
            "prescription": "OTC（非处方药）",
            "indications": "用于普通感冒或流行性感冒引起的发热，也用于缓解轻至中度疼痛，如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经。",
            "dosage": {
                "成人常规": "口服，一次300-600mg，每4-6小时一次，24小时不超过2000mg（推荐）或4000mg（最大）",
                "儿童": "口服，每次10-15mg/kg，每4-6小时一次，24小时不超过4次",
            },
            "adverse_reactions": [
                "常规剂量下不良反应较少",
                "过量使用可引起严重肝损伤",
                "罕见：过敏反应（皮疹、瘙痒）",
                "罕见：血小板减少、粒细胞减少",
            ],
            "contraindications": [
                "严重肝肾功能不全者禁用",
                "对本品过敏者禁用",
                "酒精中毒者禁用",
            ],
            "interactions": [
                "长期与华法林合用可能增加出血风险",
                "与酒精合用增加肝损伤风险",
                "与肝酶诱导剂（如苯巴比妥）合用增加肝毒性风险",
                "避免同时使用多种含对乙酰氨基酚的复方制剂",
            ],
            "note": "严格按剂量使用，避免超量。许多复方感冒药中含有对乙酰氨基酚，注意不要重复用药。",
        },
        "阿莫西林": {
            "generic_name": "阿莫西林（Amoxicillin）",
            "category": "青霉素类抗生素",
            "prescription": "处方药",
            "indications": "用于敏感菌所致的感染：上呼吸道感染、泌尿生殖道感染、皮肤软组织感染、消化道感染等。",
            "dosage": {
                "成人常规": "口服，一次0.5g，每6-8小时一次，一日剂量不超过4g",
                "儿童": "口服，每次20-40mg/kg/日，分3次服用",
            },
            "adverse_reactions": [
                "胃肠道：恶心、呕吐、腹泻",
                "过敏反应：皮疹、瘙痒，严重者可发生过敏性休克",
                "罕见：假膜性肠炎、血液系统异常",
            ],
            "contraindications": [
                "对青霉素类抗生素过敏者禁用",
                "传染性单核细胞增多症患者禁用",
            ],
            "interactions": [
                "与丙磺舒合用可升高血药浓度",
                "与避孕药合用可能降低避孕效果",
                "与甲氨蝶呤合用增加甲氨蝶呤毒性",
            ],
            "note": "处方药，必须遵医嘱使用。青霉素过敏者禁用。完成整个疗程，不要因症状好转而自行停药。",
        },
        "二甲双胍": {
            "generic_name": "二甲双胍（Metformin）",
            "category": "双胍类口服降糖药",
            "prescription": "处方药",
            "indications": "用于2型糖尿病患者，特别是肥胖的2型糖尿病患者。可作为一线用药单独使用或与其他口服降糖药/胰岛素联合使用。",
            "dosage": {
                "成人常规": "起始剂量500mg，每日2-3次，随餐服用。可逐渐增加至最大剂量2550mg/日",
                "老年人": "起始剂量宜小，缓慢加量，定期监测肾功能",
            },
            "adverse_reactions": [
                "胃肠道：恶心、呕吐、腹泻、腹胀、食欲减退（最常见）",
                "长期使用可能影响维生素B12吸收",
                "罕见：乳酸酸中毒（严重不良反应）",
            ],
            "contraindications": [
                "1型糖尿病患者禁用",
                "糖尿病酮症酸中毒者禁用",
                "严重肝肾功能不全者禁用",
                "严重感染、缺氧状态者禁用",
                "酗酒者禁用",
                "碘造影检查前后48小时暂停使用",
            ],
            "interactions": [
                "与酒精合用增加乳酸酸中毒风险",
                "与碘造影剂合用增加乳酸酸中毒风险",
                "某些药物（如西咪替丁）可能影响其排泄",
            ],
            "note": "处方药，必须遵医嘱使用。建议随餐服用以减少胃肠道反应。定期监测肾功能。",
        },
        "阿司匹林": {
            "generic_name": "阿司匹林（Aspirin）",
            "category": "解热镇痛药/抗血小板药",
            "prescription": "OTC（低剂量）/处方药（高剂量）",
            "indications": "低剂量（75-100mg/日）：心脑血管疾病二级预防。高剂量：解热镇痛、抗炎抗风湿。",
            "dosage": {
                "抗血小板（低剂量）": "口服，75-100mg/日，长期服用",
                "解热镇痛": "口服，一次300-600mg，需要时服用",
            },
            "adverse_reactions": [
                "胃肠道：恶心、腹痛，可引起胃肠道溃疡和出血",
                "出血风险增加：消化道出血、牙龈出血等",
                "过敏反应：皮疹、哮喘（阿司匹林哮喘）",
                "罕见：Reye综合征（儿童和青少年）",
            ],
            "contraindications": [
                "活动性消化道出血者禁用",
                "血友病或出血倾向者禁用",
                "对阿司匹林或其他水杨酸类过敏者禁用",
                "儿童和青少年伴病毒感染时禁用（Reye综合征风险）",
                "孕晚期妇女禁用",
            ],
            "interactions": [
                "与抗凝药合用增加出血风险",
                "与布洛芬合用可能降低阿司匹林心血管保护作用",
                "与糖皮质激素合用增加胃肠道出血风险",
                "与甲氨蝶呤合用增加其毒性",
            ],
            "note": "低剂量阿司匹林用于心血管保护需遵医嘱。肠溶片应空腹服用。",
        },
        "奥美拉唑": {
            "generic_name": "奥美拉唑（Omeprazole）",
            "category": "质子泵抑制剂（PPI）",
            "prescription": "处方药",
            "indications": "用于胃溃疡、十二指肠溃疡、胃食管反流病（GERD）、幽门螺杆菌感染（联合疗法）、卓-艾综合征等。",
            "dosage": {
                "成人常规": "口服，一次20mg，每日1-2次，早餐前服用",
                "幽门螺杆菌根除": "20mg，每日2次，联合抗生素使用10-14天",
            },
            "adverse_reactions": [
                "头痛、头晕",
                "胃肠道：腹泻、便秘、恶心、腹痛",
                "长期使用：骨质疏松、骨折风险增加",
                "长期使用：低镁血症、维生素B12缺乏",
                "罕见：间质性肾炎",
            ],
            "contraindications": [
                "对本品过敏者禁用",
            ],
            "interactions": [
                "影响多种药物的吸收（如氯吡格雷，降低其抗血小板效果）",
                "与地西泮、苯妥英钠等合用可能升高其血药浓度",
                "与铁剂合用可能降低铁剂吸收",
                "长期使用可能影响钙、镁、维生素B12吸收",
            ],
            "note": "处方药，必须遵医嘱使用。长期使用需定期评估是否需要继续用药。",
        },
        "硝苯地平": {
            "generic_name": "硝苯地平（Nifedipine）",
            "category": "钙通道阻滞剂（CCB）",
            "prescription": "处方药",
            "indications": "用于高血压、心绞痛（稳定型和不稳定型）。",
            "dosage": {
                "控释片（高血压）": "口服，30mg/日，可增至60mg/日",
                "普通片（高血压）": "口服，起始10mg/次，每日3次",
            },
            "adverse_reactions": [
                "面部潮红、头痛、心悸",
                "下肢水肿",
                "牙龈增生",
                "低血压（尤其起始用药时）",
                "便秘",
            ],
            "contraindications": [
                "心源性休克者禁用",
                "严重主动脉瓣狭窄者禁用",
                "不稳定型心绞痛患者禁用（普通片）",
                "对本品过敏者禁用",
            ],
            "interactions": [
                "与β受体阻滞剂合用可能加重低血压和心衰",
                "与地高辛合用可能升高地高辛血药浓度",
                "与西柚汁合用可升高血药浓度",
                "避免与其他CCB合用",
            ],
            "note": "处方药，必须遵医嘱使用。控释片应整片吞服，不可掰开或咀嚼。",
        },
        "氨氯地平": {
            "generic_name": "氨氯地平（Amlodipine）",
            "category": "钙通道阻滞剂（CCB）",
            "prescription": "处方药",
            "indications": "用于高血压、慢性稳定性心绞痛和变异型心绞痛。",
            "dosage": {
                "成人常规": "口服，起始5mg/日，最大剂量10mg/日",
                "老年人/肝功能不全": "起始2.5mg/日",
            },
            "adverse_reactions": [
                "下肢水肿（最常见）",
                "头痛、面部潮红",
                "心悸、头晕",
                "疲劳、恶心",
            ],
            "contraindications": [
                "严重主动脉瓣狭窄者禁用",
                "对本品过敏者禁用",
                "不稳定型心绞痛者慎用",
            ],
            "interactions": [
                "与CYP3A4抑制剂（如酮康唑）合用可升高血药浓度",
                "与地高辛合用需监测地高辛血药浓度",
            ],
            "note": "处方药，必须遵医嘱使用。半衰期长，每日一次即可。不受食物影响。",
        },
        "头孢克洛": {
            "generic_name": "头孢克洛（Cefaclor）",
            "category": "第二代头孢菌素类抗生素",
            "prescription": "处方药",
            "indications": "用于敏感菌所致的呼吸道感染、中耳炎、尿路感染、皮肤软组织感染等。",
            "dosage": {
                "成人常规": "口服，一次250mg，每8小时一次",
                "儿童": "口服，每次20mg/kg/日，分3次服用",
            },
            "adverse_reactions": [
                "胃肠道：腹泻、恶心、呕吐",
                "过敏反应：皮疹、瘙痒",
                "罕见：假膜性肠炎、血清病样反应",
            ],
            "contraindications": [
                "对头孢菌素类抗生素过敏者禁用",
                "青霉素过敏者慎用（交叉过敏）",
            ],
            "interactions": [
                "与丙磺舒合用可升高血药浓度",
                "与强利尿剂合用可能增加肾毒性",
                "与抗凝药合用可能增强抗凝效果",
            ],
            "note": "处方药，必须遵医嘱使用。完成整个疗程。用药期间及停药后一周内避免饮酒。",
        },
        "氯雷他定": {
            "generic_name": "氯雷他定（Loratadine）",
            "category": "第二代抗组胺药",
            "prescription": "OTC（非处方药）",
            "indications": "用于缓解过敏性鼻炎的有关症状（喷嚏、流涕、鼻痒、鼻塞），也用于缓解慢性荨麻疹及其他过敏性皮肤病症状。",
            "dosage": {
                "成人及12岁以上儿童": "口服，10mg/日",
                "2-12岁儿童": "体重>30kg：10mg/日；体重<=30kg：5mg/日",
            },
            "adverse_reactions": [
                "不良反应较少且轻微",
                "偶见口干、头痛、嗜睡",
                "罕见：肝功能异常、心动过速",
            ],
            "contraindications": [
                "对本品过敏者禁用",
                "严重肝功能不全者需减量",
            ],
            "interactions": [
                "与酮康唑、红霉素等CYP3A4抑制剂合用可升高血药浓度",
                "与其他中枢抑制药合用需注意",
            ],
            "note": "OTC药物。第二代抗组胺药，嗜睡副作用较小。建议在医生指导下使用。",
        },
        "蒙脱石散": {
            "generic_name": "蒙脱石散（Montmorillonite Powder）",
            "category": "肠道吸附剂/止泻药",
            "prescription": "OTC（非处方药）",
            "indications": "用于成人及儿童急慢性腹泻，也用于食道、胃、十二指肠相关疾病的辅助治疗。",
            "dosage": {
                "成人": "口服，一次1袋（3g），每日3次",
                "儿童": "1岁以下：每日1袋；1-2岁：每日1-2袋；2岁以上：每日2-3袋，均分3次服用",
            },
            "adverse_reactions": [
                "不良反应少见",
                "偶见便秘",
                "极少数人可能出现轻度过敏",
            ],
            "contraindications": [
                "对本品过敏者禁用",
            ],
            "interactions": [
                "可能影响其他药物的吸收，建议与其他药物间隔1-2小时服用",
            ],
            "note": "OTC药物。将药物倒入50ml温水中搅匀后服用。急性腹泻时注意补充水分和电解质。",
        },
        "辛伐他汀": {
            "generic_name": "辛伐他汀（Simvastatin）",
            "category": "他汀类调脂药",
            "prescription": "处方药",
            "indications": "用于高脂血症（高胆固醇血症、混合型高脂血症），也用于冠心病和脑卒中的二级预防。",
            "dosage": {
                "成人常规": "口服，起始10-20mg/日，晚间顿服，最大剂量40mg/日",
            },
            "adverse_reactions": [
                "肌肉相关：肌痛、肌无力，罕见横纹肌溶解",
                "肝脏：转氨酶升高",
                "胃肠道：腹痛、便秘、消化不良",
                "罕见：记忆障碍、周围神经病变",
            ],
            "contraindications": [
                "活动性肝脏疾病或不明原因转氨酶升高者禁用",
                "孕妇及哺乳期妇女禁用",
                "对他汀类药物过敏者禁用",
            ],
            "interactions": [
                "与CYP3A4强抑制剂（如酮康唑、红霉素、克拉霉素）合用增加肌病风险",
                "与贝特类降脂药合用增加横纹肌溶解风险",
                "与胺碘酮合用时剂量不超过20mg/日",
                "避免大量饮用西柚汁",
            ],
            "note": "处方药，必须遵医嘱使用。晚间服用效果更好。定期监测肝功能和肌酸激酶（CK）。",
        },
        "盐酸曲马多": {
            "generic_name": "盐酸曲马多（Tramadol）",
            "category": "中枢性镇痛药",
            "prescription": "处方药（管制药品）",
            "indications": "用于中度至重度急性或慢性疼痛，如术后疼痛、创伤疼痛、癌性疼痛等。",
            "dosage": {
                "成人常规": "口服，起始50-100mg/次，必要时每4-6小时一次，24小时不超过400mg",
            },
            "adverse_reactions": [
                "恶心、呕吐、便秘",
                "头晕、嗜睡",
                "出汗、口干",
                "有成瘾性和依赖性风险",
                "过量可致呼吸抑制",
            ],
            "contraindications": [
                "对曲马多过敏者禁用",
                "严重呼吸抑制者禁用",
                "癫痫患者慎用",
                "与MAO抑制剂合用者禁用",
                "酒精或药物依赖者慎用",
            ],
            "interactions": [
                "与CNS抑制剂（酒精、镇静药）合用增强中枢抑制",
                "与MAO抑制剂合用可致5-羟色胺综合征",
                "与卡马西平合用降低曲马多血药浓度",
                "与华法林合用可能增加INR值",
            ],
            "note": "管制处方药，严格遵医嘱使用。有成瘾风险，不宜长期使用。",
        },
        "氯吡格雷": {
            "generic_name": "氯吡格雷（Clopidogrel）",
            "category": "抗血小板药",
            "prescription": "处方药",
            "indications": "用于预防动脉粥样硬化血栓形成事件（如心肌梗死、缺血性卒中、外周动脉疾病），也用于冠脉支架术后抗血小板治疗。",
            "dosage": {
                "成人常规": "口服，75mg/日",
                "急性冠脉综合征（负荷剂量）": "300mg负荷剂量，随后75mg/日",
            },
            "adverse_reactions": [
                "出血：消化道出血、皮下出血、牙龈出血等",
                "胃肠道：腹痛、消化不良、腹泻",
                "罕见：血栓性血小板减少性紫癜（TTP）",
                "罕见：中性粒细胞减少",
            ],
            "contraindications": [
                "活动性病理性出血者禁用",
                "严重肝功能损害者禁用",
                "对本品过敏者禁用",
            ],
            "interactions": [
                "与奥美拉唑/埃索美拉唑合用可能降低抗血小板效果",
                "与阿司匹林合用（双联抗血小板）增加出血风险",
                "与华法林合用增加出血风险",
                "与NSAIDs合用增加出血风险",
            ],
            "note": "处方药，必须遵医嘱使用。支架术后通常需要双联抗血小板治疗。如需使用PPI，建议选用泮托拉唑。",
        },
        "左氧氟沙星": {
            "generic_name": "左氧氟沙星（Levofloxacin）",
            "category": "氟喹诺酮类抗生素",
            "prescription": "处方药",
            "indications": "用于敏感菌所致的呼吸道感染、泌尿生殖系统感染、皮肤软组织感染、肠道感染等。",
            "dosage": {
                "成人常规": "口服，一次500mg，每日1次",
                "重症感染": "一次500mg，每日2次",
            },
            "adverse_reactions": [
                "胃肠道：恶心、呕吐、腹泻",
                "神经系统：头晕、头痛、失眠",
                "肌腱炎和肌腱断裂（黑框警告）",
                "QT间期延长",
                "光敏反应",
                "罕见：周围神经病变",
            ],
            "contraindications": [
                "对喹诺酮类药物过敏者禁用",
                "18岁以下未成年人禁用",
                "孕妇及哺乳期妇女禁用",
                "癫痫患者禁用",
            ],
            "interactions": [
                "与含铝、镁的抗酸剂合用降低吸收",
                "与华法林合用增加出血风险",
                "与NSAIDs合用增加癫痫发作风险",
                "避免与延长QT间期的药物合用",
            ],
            "note": "处方药，必须遵医嘱使用。用药期间避免阳光暴晒。注意肌腱炎风险，出现关节痛应立即就医。",
        },
        "甲氧氯普胺": {
            "generic_name": "甲氧氯普胺（Metoclopramide）",
            "category": "促胃动力药/止吐药",
            "prescription": "处方药",
            "indications": "用于功能性消化不良、胃排空迟缓、恶心呕吐（如化疗后、术后）、反流性食管炎等。",
            "dosage": {
                "成人常规": "口服，一次5-10mg，每日3次，餐前30分钟服用",
            },
            "adverse_reactions": [
                "中枢神经系统：嗜睡、乏力、头晕",
                "锥体外系反应：急性肌张力障碍（尤其在年轻患者中）",
                "内分泌：高催乳素血症（溢乳、月经紊乱）",
                "罕见：迟发性运动障碍（长期使用）",
            ],
            "contraindications": [
                "胃肠道出血、机械性梗阻或穿孔者禁用",
                "嗜铬细胞瘤患者禁用",
                "接受单胺氧化酶抑制剂治疗者禁用",
            ],
            "interactions": [
                "与中枢抑制药合用增强中枢抑制",
                "与抗胆碱药合用可能相互拮抗",
                "与吩噻嗪类合用增加锥体外系反应风险",
            ],
            "note": "处方药，必须遵医嘱使用。不宜长期使用。年轻患者更易出现锥体外系反应。",
        },
        "地塞米松": {
            "generic_name": "地塞米松（Dexamethasone）",
            "category": "糖皮质激素",
            "prescription": "处方药",
            "indications": "用于过敏性疾病、自身免疫性疾病、炎症性疾病（如严重哮喘、过敏性休克辅助治疗）、某些肿瘤的辅助治疗、脑水肿等。",
            "dosage": {
                "成人常规": "剂量因适应症而异，从0.75mg/日到数mg/kg不等",
                "抗炎/免疫抑制": "口服，起始0.75-9mg/日",
            },
            "adverse_reactions": [
                "长期使用：医源性库欣综合征（满月脸、水牛背、向心性肥胖）",
                "血糖升高、血糖控制恶化",
                "骨质疏松、股骨头坏死",
                "免疫力下降，易感染",
                "消化道溃疡",
                "精神症状：失眠、情绪改变",
                "肾上腺皮质功能抑制",
            ],
            "contraindications": [
                "全身性真菌感染者禁用",
                "活动性消化道溃疡者慎用",
                "严重精神病史者慎用",
                "糖尿病、高血压患者慎用",
            ],
            "interactions": [
                "与NSAIDs合用增加消化道溃疡风险",
                "与利尿剂合用增加低钾血症风险",
                "与苯巴比妥、苯妥英钠合用降低其疗效",
                "与疫苗合用可能降低疫苗效果",
            ],
            "note": "处方药，严格遵医嘱使用。不可突然停药，需逐渐减量。长期使用需监测血糖、骨密度等。",
        },
        "硝苯地平控释片": {
            "generic_name": "硝苯地平控释片（Nifedipine Controlled-release）",
            "category": "钙通道阻滞剂（CCB）",
            "prescription": "处方药",
            "indications": "用于高血压、慢性稳定性心绞痛。",
            "dosage": {
                "成人常规": "口服，30mg/日，可增至60mg/日，整片吞服",
            },
            "adverse_reactions": [
                "下肢水肿、面部潮红",
                "头痛、心悸、头晕",
                "牙龈增生",
                "便秘",
            ],
            "contraindications": [
                "心源性休克者禁用",
                "严重主动脉瓣狭窄者禁用",
                "对本品过敏者禁用",
            ],
            "interactions": [
                "与β受体阻滞剂合用注意低血压和心衰风险",
                "与地高辛合用需监测地高辛血药浓度",
                "与西柚汁合用可升高血药浓度",
            ],
            "note": "处方药，必须遵医嘱使用。控释片应整片吞服，不可掰开或咀嚼。粪便中可能出现药片外壳，属正常现象。",
        },
        "氨溴索": {
            "generic_name": "氨溴索（Ambroxol）",
            "category": "祛痰药/黏液溶解剂",
            "prescription": "OTC（部分剂型）/处方药",
            "indications": "用于急慢性支气管炎、支气管哮喘、支气管扩张、肺炎等引起的痰液黏稠、咳痰困难。",
            "dosage": {
                "成人常规": "口服，一次30mg，每日3次，餐后服用",
                "儿童": "口服，每次1.2-1.6mg/kg/日，分2-3次服用",
            },
            "adverse_reactions": [
                "不良反应较少",
                "偶见轻度胃肠道不适（恶心、胃痛）",
                "罕见：过敏反应（皮疹）",
            ],
            "contraindications": [
                "对本品过敏者禁用",
                "消化性溃疡患者慎用",
            ],
            "interactions": [
                "与抗生素（如阿莫西林、头孢克洛）合用可提高抗生素在肺组织中的浓度",
                "与止咳药合用可能因抑制咳嗽反射导致痰液潴留",
            ],
            "note": "建议在医生指导下使用。服药后多饮水有助于痰液稀释。",
        },
        "碳酸钙D3": {
            "generic_name": "碳酸钙D3（Calcium Carbonate + Vitamin D3）",
            "category": "钙剂+维生素D补充剂",
            "prescription": "OTC（非处方药）",
            "indications": "用于钙和维生素D缺乏的补充，如骨质疏松症的预防和辅助治疗、孕期和哺乳期补钙等。",
            "dosage": {
                "成人常规": "口服，一次1片，每日1-2次",
                "儿童": "遵医嘱或按说明书年龄分段使用",
            },
            "adverse_reactions": [
                "偶见便秘",
                "过量使用可致高钙血症",
                "嗳气、腹胀",
            ],
            "contraindications": [
                "高钙血症患者禁用",
                "高钙尿症患者禁用",
                "肾结石患者慎用",
                "甲状旁腺功能亢进者禁用",
            ],
            "interactions": [
                "与噻嗪类利尿剂合用增加高钙血症风险",
                "与四环素类抗生素合用影响抗生素吸收（间隔2小时以上）",
                "与铁剂合用可能相互影响吸收",
            ],
            "note": "OTC药物。建议随餐服用以提高钙吸收率。注意不要超量补充。",
        },
        "沙丁胺醇": {
            "generic_name": "沙丁胺醇（Salbutamol/Albuterol）",
            "category": "短效β2受体激动剂（SABA）",
            "prescription": "处方药",
            "indications": "用于缓解支气管哮喘、慢性阻塞性肺疾病（COPD）等的支气管痉挛。气雾剂用于急性发作的快速缓解。",
            "dosage": {
                "气雾剂（急性发作）": "吸入100-200μg（1-2喷），必要时每4-6小时重复",
                "口服片剂": "成人一次2-4mg，每日3次",
            },
            "adverse_reactions": [
                "骨骼肌震颤（手抖）",
                "心悸、心动过速",
                "头痛、头晕",
                "低钾血症（过量使用时）",
                "罕见：过敏反应、支气管痉挛反常加重",
            ],
            "contraindications": [
                "对本品过敏者禁用",
                "嗜铬细胞瘤患者禁用",
                "甲亢患者慎用",
                "心血管功能不全者慎用",
            ],
            "interactions": [
                "与β受体阻滞剂合用可相互拮抗",
                "与茶碱类合用增加心律失常风险",
                "与洋地黄类合用增加心律失常风险",
                "与单胺氧化酶抑制剂合用增加心血管不良反应",
            ],
            "note": "处方药，必须遵医嘱使用。气雾剂为急救用药，不宜过度依赖。频繁使用提示哮喘控制不佳，需调整长期治疗方案。",
        },
        "感冒灵颗粒": {
            "generic_name": "感冒灵颗粒（999感冒灵颗粒）",
            "category": "中西药复方制剂 / 解热镇痛药",
            "prescription": "OTC（非处方药，甲类）",
            "indications": "用于感冒引起的头痛、发热、鼻塞、流涕、咽痛等症状。",
            "dosage": {
                "成人常规": "口服，一次1袋（10g），一日3次，开水冲服",
                "儿童": "儿童应在医师指导下服用，酌减或遵医嘱",
            },
            "adverse_reactions": [
                "偶见皮疹、荨麻疹、药热及粒细胞减少",
                "长期大量用药会导致肝肾功能异常",
                "可见困倦、嗜睡、口渴、虚弱感",
            ],
            "contraindications": [
                "严重肝肾功能不全者禁用",
                "对本品及成分过敏者禁用",
                "孕妇及哺乳期妇女慎用",
                "对阿司匹林过敏者慎用（含对乙酰氨基酚和马来酸氯苯那敏）",
            ],
            "interactions": [
                "与其他解热镇痛药同用可增加肾毒性风险",
                "不宜与氯霉素、巴比妥类等合用",
                "与酒精及中枢抑制药同用可增强嗜睡等不良反应",
                "如正在使用其他药品，使用前请咨询医师或药师",
            ],
            "note": "本品含对乙酰氨基酚、马来酸氯苯那敏、咖啡因。服用期间不得饮酒或含有酒精的饮料；不能同时服用与本品成份相似的其他抗感冒药。服药后不得驾驶机、车、船、从事高空作业、机械作业及操作精密仪器。疗程3-7天，症状未缓解请就医。",
            "brand_names": ["999感冒灵", "三九感冒灵", "999感冒灵颗粒"],
        },
        "连花清瘟胶囊": {
            "generic_name": "连花清瘟胶囊/颗粒",
            "category": "中成药 / 清热解毒药",
            "prescription": "OTC（非处方药）",
            "indications": "用于治疗流行性感冒属热毒袭肺证，症见：发热或高热、恶寒、肌肉酸痛、鼻塞流涕、咳嗽、头痛、咽干咽痛、舌偏红、苔黄或黄腻等。",
            "dosage": {
                "胶囊（成人）": "口服，一次4粒，一日3次",
                "颗粒（成人）": "口服，一次1袋，一日3次，开水冲服",
                "儿童": "儿童酌减或遵医嘱",
            },
            "adverse_reactions": [
                "偶见胃肠道不适（恶心、腹泻、腹胀）",
                "个别患者出现皮疹",
            ],
            "contraindications": [
                "对本品及成分过敏者禁用",
                "风寒感冒者不适用（表现为恶寒重、发热轻、无汗、头痛身痛、鼻流清涕）",
                "孕妇慎用",
                "高血压、心脏病患者慎用",
            ],
            "interactions": [
                "不宜在服药期间同时服用滋补性中药",
                "如正在使用其他药品，使用前请咨询医师或药师",
            ],
            "note": "OTC中成药。服药后若体温超过38.5℃或症状加重应及时就医。糖尿病患者及有高血压、心脏病、肝病、糖尿病、肾病等慢性病严重者应在医师指导下服用。",
        },
        "板蓝根颗粒": {
            "generic_name": "板蓝根颗粒",
            "category": "中成药 / 清热解毒药",
            "prescription": "OTC（非处方药）",
            "indications": "用于病毒性感冒、咽喉肿痛、急性扁桃体炎、腮腺炎等。",
            "dosage": {
                "成人常规": "口服，一次1-2袋（5-10g），一日3-4次，开水冲服",
                "儿童": "酌减或遵医嘱",
            },
            "adverse_reactions": [
                "不良反应少见",
                "偶见胃肠道不适",
                "个别患者出现过敏反应（皮疹等）",
            ],
            "contraindications": [
                "对本品过敏者禁用",
                "风寒感冒者不适用",
                "孕妇慎用",
            ],
            "interactions": [
                "不宜在服药期间同时服用滋补性中药",
            ],
            "note": "OTC中成药。症状较重或服药3天后症状无改善者应去医院就诊。糖尿病患者慎用（含蔗糖型）。",
        },
        "藿香正气水": {
            "generic_name": "藿香正气水/口服液/胶囊",
            "category": "中成药 / 理气和中药",
            "prescription": "OTC（非处方药）",
            "indications": "用于暑湿感冒、头痛身重胸闷、恶寒发热、脘腹胀痛、呕吐泄泻、胃肠型感冒。也用于中暑。",
            "dosage": {
                "水剂（成人）": "口服，一次半支-1支（5-10ml），一日2次，用时摇匀",
                "口服液（成人）": "口服，一次1支（10ml），一日2次",
                "胶囊（成人）": "口服，一次2-4粒，一日2次",
                "儿童": "酌减或遵医嘱",
            },
            "adverse_reactions": [
                "偶有过敏反应（皮疹、过敏性休克）",
                "藿香正气水含酒精，服用后不得驾驶车辆等",
                "个别出现胃肠道不适",
            ],
            "contraindications": [
                "对本品及成分过敏者禁用",
                "酒精过敏者禁用藿香正气水（含40%-50%酒精），可选用口服液或胶囊",
                "孕妇慎用",
            ],
            "interactions": [
                "藿香正气水含酒精，不可与头孢类抗生素合用（双硫仑样反应）",
                "不可与甲硝唑、呋喃唑酮等合用",
                "不宜与滋补性中药同服",
            ],
            "note": "OTC中成药。注意：藿香正气水含酒精，与头孢类抗生素同服可引起双硫仑样反应（面部潮红、头痛、恶心、呕吐等，严重可危及生命），务必避免。如需与头孢类同用，请选用不含酒精的藿香正气口服液或胶囊。",
        },
        "复方甘草片": {
            "generic_name": "复方甘草片（复方甘草口服片）",
            "category": "镇咳祛痰药",
            "prescription": "处方药（含阿片粉，已列为管制药品）",
            "indications": "用于镇咳祛痰。",
            "dosage": {
                "成人常规": "口服，一次3-4片，一日3次，含服或吞服",
            },
            "adverse_reactions": [
                "偶见便秘、恶心、口干",
                "长期使用可产生依赖性（含阿片粉）",
                "过量可引起呼吸抑制",
            ],
            "contraindications": [
                "对本品成分过敏者禁用",
                "孕妇及哺乳期妇女禁用",
                "呼吸抑制者禁用",
                "1岁以下婴儿禁用",
            ],
            "interactions": [
                "与其他中枢抑制药合用可增强呼吸抑制",
                "与单胺氧化酶抑制剂合用可引起高血压危象",
            ],
            "note": "处方药，含阿片粉，属管制药品。严格遵医嘱使用，不宜长期服用。已从部分药店下架，需凭处方购买。",
        },
        "六味地黄丸": {
            "generic_name": "六味地黄丸",
            "category": "中成药 / 补益剂",
            "prescription": "OTC（非处方药）",
            "indications": "用于肾阴亏损、头晕耳鸣、腰膝酸软、骨蒸潮热、盗汗遗精、消渴。",
            "dosage": {
                "水蜜丸（成人）": "口服，一次30粒（6g），一日2次",
                "大蜜丸（成人）": "口服，一次1丸，一日2次",
                "浓缩丸（成人）": "口服，一次8丸，一日3次",
            },
            "adverse_reactions": [
                "偶见胃肠道不适",
                "长期服用可能出现消化功能下降",
            ],
            "contraindications": [
                "对本品过敏者禁用",
                "感冒发热患者不宜服用",
                "脾虚泄泻者慎用",
                "孕妇慎用",
            ],
            "interactions": [
                "不宜在服药期间同时服用感冒药",
                "不宜与藜芦或其制剂同用（十八反）",
            ],
            "note": "OTC中成药。服药期间忌油腻食物。感冒患者暂停服用。服药2周后症状未改善应就医。",
        },
        "健胃消食片": {
            "generic_name": "健胃消食片",
            "category": "中成药 / 消食药",
            "prescription": "OTC（非处方药）",
            "indications": "用于脾胃虚弱所致的食积胀满、不思饮食、嗳腐酸臭、脘腹胀痛、大便失调。",
            "dosage": {
                "成人": "口服，一次3-4片，一日3次，咀嚼后咽下",
                "儿童": "口服，一次2-3片，一日3次，咀嚼后咽下",
            },
            "adverse_reactions": [
                "不良反应少见",
                "偶见胃部不适",
            ],
            "contraindications": [
                "对本品过敏者禁用",
                "胃阴虚者（表现为口干舌燥、大便干结）不宜使用",
            ],
            "interactions": [
                "如正在使用其他药品，使用前请咨询医师或药师",
            ],
            "note": "OTC中成药。服药期间忌食生冷辛辣油腻食物。服药3天症状无改善应就医。",
        },
    }

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "药品名称（通用名或商品名），如'布洛芬'、'阿莫西林'",
                },
                "query_type": {
                    "type": "string",
                    "description": "查询类型",
                    "enum": ["基本信息", "用法用量", "不良反应", "药物相互作用"],
                },
            },
            "required": ["drug_name"],
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行药品信息查询（三层Fallback + 并发优化 + 缓存）

        Args:
            drug_name: 药品名称
            query_type: 查询类型
            enable_fallback: 是否启用外部fallback（默认True）
            enable_cache: 是否启用缓存（默认True）
            concurrent_fallback: 是否并发执行fallback（默认True）

        Returns:
            药品信息查询结果
        """
        drug_name = kwargs.get("drug_name", "").strip()
        query_type = kwargs.get("query_type", "基本信息").strip()
        enable_fallback = kwargs.get("enable_fallback", True)
        enable_cache = kwargs.get("enable_cache", True)
        concurrent_fallback = kwargs.get("concurrent_fallback", True)

        if not drug_name:
            return {
                "success": False,
                "error": "请提供药品名称",
            }

        logger.info(f"药品查询: drug_name='{drug_name}', query_type='{query_type}', enable_fallback={enable_fallback}")

        # Step 1: 检查缓存
        cache_key = self._get_cache_key(drug_name, query_type)
        if enable_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"命中缓存: {drug_name}")
                cached_result["cache_hit"] = True
                return cached_result

        # Step 2: 内置药品数据库搜索（本地，毫秒级）
        drug_info = self._search_drug(drug_name)
        if not drug_info:
            drug_info = self._fuzzy_search_drug(drug_name)

        if drug_info:
            result = self._format_result(drug_info, drug_name, query_type, source="内置数据库")
            result["risk_level"] = "low"  # 内置数据库风险低
            self._store_to_cache(cache_key, result)
            return result

        # Step 3: 外部数据源查询（并发或串行）
        if enable_fallback:
            if concurrent_fallback:
                # 并发执行第二层和第三层，取最先成功的
                result = await self._concurrent_fallback_search(drug_name, query_type)
            else:
                # 串行执行（原有逻辑）
                result = await self._serial_fallback_search(drug_name, query_type)

            if result and self._validate_result(result):
                # 添加风险标记
                result["risk_level"] = "medium" if result.get("source") == "外部药品API" else "high"
                result["risk_warning"] = self._get_risk_warning(result.get("source", ""))
                self._store_to_cache(cache_key, result)
                return result

        # 所有数据源都未找到或验证失败
        return {
            "success": True,
            "message": f"未找到药品'{drug_name}'的有效信息。",
            "suggestion": "请确认药品名称是否正确，或前往正规医疗机构咨询药师。",
            "tried_sources": ["内置数据库", "外部API", "网络搜索"] if enable_fallback else ["内置数据库"],
            "risk_level": "none",
        }

    # ==================== 缓存管理 ====================

    def _get_cache_key(self, drug_name: str, query_type: str) -> str:
        """生成缓存键"""
        return md5(f"{drug_name}:{query_type}".encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存获取结果"""
        if cache_key not in self._result_cache:
            return None
        result, expire_time = self._result_cache[cache_key]
        if datetime.now() > expire_time:
            # 缓存过期，删除
            del self._result_cache[cache_key]
            return None
        return result.copy()

    def _store_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """存储结果到缓存"""
        expire_time = datetime.now() + timedelta(hours=self._cache_ttl_hours)
        self._result_cache[cache_key] = (result.copy(), expire_time)
        logger.debug(f"缓存存储: key={cache_key[:8]}, expire={expire_time}")

    def clear_cache(self) -> int:
        """清除所有缓存"""
        count = len(self._result_cache)
        self._result_cache.clear()
        logger.info(f"清除缓存: {count} 条")
        return count

    # ==================== 内容验证 ====================

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """
        验证药品信息结果是否有效

        检查：
        1. 必须有 drug_name
        2. 至少包含一项关键信息（indications/dosage/adverse_reactions等）
        3. 不能是"未找到"类型的空结果

        Args:
            result: 查询结果

        Returns:
            是否有效
        """
        if not result:
            return False

        if not result.get("success"):
            return False

        # 必须有药品名称
        drug_name = result.get("drug_name")
        if not drug_name or drug_name.strip() == "":
            return False

        # 检查是否是"未找到"类型的结果
        message = result.get("message", "")
        if "未找到" in message or "无结果" in message:
            return False

        # 至少包含一项关键信息
        has_key_info = False
        for field in self._key_fields:
            value = result.get(field)
            if value:
                if isinstance(value, str) and value.strip():
                    has_key_info = True
                    break
                elif isinstance(value, (list, dict)) and len(value) > 0:
                    has_key_info = True
                    break

        # 网络搜索结果特殊处理：有 search_results 就算有效
        if result.get("search_results"):
            has_key_info = True

        return has_key_info

    # ==================== 风险提示 ====================

    def _get_risk_warning(self, source: str) -> str:
        """根据数据来源生成风险提示"""
        warnings = {
            "内置数据库": "",
            "丁香园用药助手": "⚠️ 信息来自丁香园用药助手，仅供参考，请以药品说明书为准。",
            "百度百科": "⚠️ 信息来自百度百科，可能不够准确或完整，请核实后使用。",
            "DuckDuckGo/Bing": "⚠️ 信息来自网络搜索，未经审核，可能存在错误或过期信息，请务必核实！",
            "外部药品API": "⚠️ 信息来自外部数据源，仅供参考，请以药品说明书为准。",
            "网络搜索": "⚠️ 信息来自网络搜索，未经审核，可能存在错误或过期信息，请务必核实！",
        }
        return warnings.get(source, "⚠️ 信息来自外部数据源，请核实后使用。")

    # ==================== 并发查询 ====================

    async def _concurrent_fallback_search(self, drug_name: str, query_type: str) -> Optional[Dict[str, Any]]:
        """
        并发执行第二层和第三层fallback

        同时发起外部API和网络搜索请求，取最先成功的有效结果
        设置超时限制，避免长时间等待
        """
        logger.info(f"并发fallback查询: {drug_name}")

        # 创建并发任务
        api_task = asyncio.create_task(
            self._fallback_to_drug_api(drug_name, query_type),
            name="drug_api_search"
        )
        web_task = asyncio.create_task(
            self._fallback_to_web_search(drug_name, query_type),
            name="web_search"
        )

        # 并发等待，设置超时（最多8秒）
        try:
            done, pending = await asyncio.wait(
                [api_task, web_task],
                timeout=8.0,
                return_when=asyncio.FIRST_COMPLETED
            )

            # 检查已完成的任务
            for task in done:
                result = task.result()
                if result and self._validate_result(result):
                    # 取消其他未完成的任务
                    for p_task in pending:
                        p_task.cancel()
                    logger.info(f"并发查询成功: source={result.get('source')}")
                    return result

            # 如果第一个完成的任务无效，等待其他任务
            for task in pending:
                try:
                    result = await asyncio.wait_for(task, timeout=3.0)
                    if result and self._validate_result(result):
                        logger.info(f"并发查询成功（延迟）: source={result.get('source')}")
                        return result
                except asyncio.TimeoutError:
                    continue

        except asyncio.TimeoutError:
            logger.warning(f"并发查询超时: {drug_name}")
        except Exception as e:
            logger.error(f"并发查询异常: {e}")

        return None

    async def _serial_fallback_search(self, drug_name: str, query_type: str) -> Optional[Dict[str, Any]]:
        """
        串行执行fallback（原有逻辑，作为备用）
        """
        # 第二层：外部药品API抓取
        api_result = await self._fallback_to_drug_api(drug_name, query_type)
        if api_result and self._validate_result(api_result):
            return api_result

        # 第三层：网络搜索
        web_result = await self._fallback_to_web_search(drug_name, query_type)
        if web_result and self._validate_result(web_result):
            return web_result

        return None

    async def _fallback_to_drug_api(self, drug_name: str, query_type: str) -> Optional[Dict[str, Any]]:
        """
        Fallback到外部药品API抓取工具

        尝试从丁香园、百度百科等公开网站获取药品信息
        """
        try:
            # 获取 drug_api 工具实例
            drug_api_tool = tool_registry.get_tool("drug_api")
            if drug_api_tool is None:
                logger.warning("drug_api 工具未注册，跳过外部API fallback")
                return None

            # 调用外部API工具
            result = await drug_api_tool.execute(
                drug_name=drug_name,
                query_type=query_type,
                source="auto",
            )

            if result.get("success"):
                result["fallback_source"] = "外部药品API"
                logger.info(f"外部药品API成功: source={result.get('source')}")
                return result

            logger.warning(f"外部药品API失败: {result.get('error')}")
            return None

        except Exception as e:
            logger.error(f"外部药品API调用异常: {e}")
            return None

    async def _fallback_to_web_search(self, drug_name: str, query_type: str) -> Optional[Dict[str, Any]]:
        """
        Fallback到网络搜索工具

        使用 DuckDuckGo/Bing 搜索药品说明书信息
        """
        try:
            # 获取 web_search 工具实例
            web_search_tool = tool_registry.get_tool("web_search")
            if web_search_tool is None:
                logger.warning("web_search 工具未注册，跳过网络搜索 fallback")
                return None

            # 构建搜索查询
            search_queries = [
                f"{drug_name} 药品说明书",
                f"{drug_name} 用法用量 不良反应",
                f"{drug_name} 适应症 禁忌",
            ]

            # 执行搜索
            search_result = await web_search_tool.execute(
                query=search_queries[0],  # 使用第一个查询
                max_results=5,
            )

            if not search_result.get("success"):
                logger.warning(f"网络搜索失败: {search_result.get('error')}")
                return None

            # 解析搜索结果
            results = search_result.get("results", [])
            if not results:
                return None

            # 格式化搜索结果为药品信息格式
            formatted_result = {
                "success": True,
                "drug_name": drug_name,
                "query_type": query_type,
                "fallback_source": "网络搜索",
                "source": "DuckDuckGo/Bing",
                "search_results": results,
                "summary": self._summarize_web_results(results, query_type),
                "disclaimer": "以上信息来自网络搜索，仅供参考，不能替代专业医生的诊断和治疗建议。请前往正规医疗机构或药店咨询。",
            }

            logger.info(f"网络搜索成功: 找到 {len(results)} 条结果")
            return formatted_result

        except Exception as e:
            logger.error(f"网络搜索调用异常: {e}")
            return None

    def _summarize_web_results(self, results: List[Dict], query_type: str) -> str:
        """
        从网络搜索结果中提取摘要信息

        Args:
            results: 搜索结果列表
            query_type: 查询类型

        Returns:
            摘要文本
        """
        if not results:
            return "未找到相关信息"

        # 提取标题和摘要
        summaries = []
        for r in results[:3]:  # 只取前3条
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            if title or snippet:
                summaries.append(f"【{title}】\n{snippet}\n来源: {url}")

        return "\n\n---\n\n".join(summaries) if summaries else "未找到相关信息"

    def _format_result(self, drug_info: Dict, drug_name: str, query_type: str, source: str) -> Dict[str, Any]:
        """
        格式化药品信息查询结果

        Args:
            drug_info: 药品信息字典
            drug_name: 药品名称
            query_type: 查询类型
            source: 数据来源

        Returns:
            格式化的查询结果
        """
        result = {
            "success": True,
            "drug_name": drug_info.get("generic_name", drug_name),
            "query_type": query_type,
            "source": source,
        }

        if query_type == "基本信息":
            result.update({
                "generic_name": drug_info.get("generic_name", ""),
                "category": drug_info.get("category", ""),
                "prescription": drug_info.get("prescription", ""),
                "indications": drug_info.get("indications", ""),
                "contraindications": drug_info.get("contraindications", []),
                "note": drug_info.get("note", ""),
            })
        elif query_type == "用法用量":
            result.update({
                "generic_name": drug_info.get("generic_name", ""),
                "dosage": drug_info.get("dosage", {}),
                "note": drug_info.get("note", ""),
            })
        elif query_type == "不良反应":
            result.update({
                "generic_name": drug_info.get("generic_name", ""),
                "adverse_reactions": drug_info.get("adverse_reactions", []),
                "note": drug_info.get("note", ""),
            })
        elif query_type == "药物相互作用":
            result.update({
                "generic_name": drug_info.get("generic_name", ""),
                "interactions": drug_info.get("interactions", []),
                "note": drug_info.get("note", ""),
            })

        # 添加处方药提醒
        if drug_info.get("prescription", "") == "处方药" or "处方药" in drug_info.get("prescription", ""):
            result["prescription_reminder"] = "此药品为处方药，请务必在医生指导下使用，切勿自行用药。"

        return result

    def _search_drug(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """精确搜索药品（支持品牌名匹配）"""
        # 直接匹配
        if drug_name in self._drug_database:
            return self._drug_database[drug_name]

        # 在通用名中搜索
        for name, info in self._drug_database.items():
            generic_name = info.get("generic_name", "")
            if drug_name in generic_name or generic_name in drug_name:
                return info

        # 在品牌名列表中搜索
        for name, info in self._drug_database.items():
            brand_names = info.get("brand_names", [])
            for brand in brand_names:
                if drug_name in brand or brand in drug_name:
                    return info

        return None

    def _fuzzy_search_drug(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """模糊搜索药品"""
        best_match = None
        best_score = 0

        for name, info in self._drug_database.items():
            score = 0
            # 计算字符重叠度
            for char in drug_name:
                if char in name:
                    score += 1
            # 计算在通用名中的匹配
            generic_name = info.get("generic_name", "")
            for char in drug_name:
                if char in generic_name:
                    score += 0.5

            if score > best_score:
                best_score = score
                best_match = info

        # 设置最低匹配阈值
        if best_score >= 1:
            return best_match
        return None
