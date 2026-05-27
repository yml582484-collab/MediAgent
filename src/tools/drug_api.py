"""
外部药品信息 API 抓取工具 - MediAgent 智慧医疗助手
从公开药品信息网站获取药品数据，扩展药品查询范围
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from .base import BaseTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DrugApiTool(BaseTool):
    """
    外部药品信息 API 抓取工具

    功能：
    - 从多个公开药品信息网站抓取药品数据
    - 支持药品名称、通用名搜索
    - 自动解析药品说明书信息
    - 作为内置药品数据库的补充

    数据来源（公开网站）：
    1. 丁香园用药助手 (drugs.dxy.cn) - 药品说明书
    2. 国家药监局数据查询 (nmpa.gov.cn) - 基础信息
    3. 百度百科/维基百科 - 药品百科

    注意：此工具仅用于补充查询，不替代专业药品数据库
    """

    name = "drug_api"
    description = (
        "从外部药品信息网站获取药品数据，扩展药品查询范围。"
        "当内置药品数据库未找到时，自动从公开网站搜索药品信息。"
        "支持查询药品说明书、用法用量、不良反应等信息。"
    )

    # 数据源配置
    DATA_SOURCES = {
        "dxy": {
            "name": "丁香园用药助手",
            "search_url": "https://drugs.dxy.cn/search",
            "detail_url": "https://drugs.dxy.cn/drug/",
            "enabled": True,
            "priority": 1,  # 最高优先级
        },
        "baike": {
            "name": "百度百科",
            "search_url": "https://baike.baidu.com/item/",
            "enabled": True,
            "priority": 2,
        },
        "nmpa": {
            "name": "国家药监局",
            "search_url": "https://www.nmpa.gov.cn/datasearch/home.html",
            "enabled": False,  # 需要复杂认证，暂不启用
            "priority": 3,
        },
    }

    def __init__(self):
        super().__init__()
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = 15.0
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "药品名称（通用名或商品名）",
                },
                "query_type": {
                    "type": "string",
                    "description": "查询类型",
                    "enum": ["基本信息", "用法用量", "不良反应", "药物相互作用", "完整说明书"],
                    "default": "基本信息",
                },
                "source": {
                    "type": "string",
                    "description": "数据来源（可选）",
                    "enum": ["auto", "dxy", "baike"],
                    "default": "auto",
                },
            },
            "required": ["drug_name"],
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行外部药品信息查询

        Args:
            drug_name: 药品名称
            query_type: 查询类型
            source: 数据来源

        Returns:
            药品信息查询结果
        """
        drug_name = kwargs.get("drug_name", "").strip()
        query_type = kwargs.get("query_type", "基本信息").strip()
        source = kwargs.get("source", "auto")

        if not drug_name:
            return {
                "success": False,
                "error": "请提供药品名称",
            }

        logger.info(f"外部药品查询: drug_name='{drug_name}', query_type='{query_type}', source='{source}'")

        # 初始化 HTTP 客户端
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._headers,
                follow_redirects=True,
            )

        try:
            # 根据数据源选择查询策略
            if source == "auto":
                # 自动选择：按优先级依次尝试
                result = await self._auto_search(drug_name, query_type)
            elif source == "dxy":
                result = await self._search_dxy(drug_name, query_type)
            elif source == "baike":
                result = await self._search_baike(drug_name, query_type)
            else:
                result = await self._auto_search(drug_name, query_type)

            return result

        except Exception as e:
            logger.error(f"外部药品查询失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "drug_name": drug_name,
                "suggestion": "外部药品数据源暂时不可用，请稍后重试或前往正规医疗机构咨询药师。",
            }

    async def _auto_search(self, drug_name: str, query_type: str) -> Dict[str, Any]:
        """
        自动选择数据源进行搜索

        按优先级依次尝试各数据源，直到找到结果
        """
        # 按优先级排序的数据源
        sources = sorted(
            [s for s in self.DATA_SOURCES.values() if s["enabled"]],
            key=lambda x: x["priority"],
        )

        for source in sources:
            try:
                if source["name"] == "丁香园用药助手":
                    result = await self._search_dxy(drug_name, query_type)
                elif source["name"] == "百度百科":
                    result = await self._search_baike(drug_name, query_type)
                else:
                    continue

                if result.get("success"):
                    result["source"] = source["name"]
                    return result

            except Exception as e:
                logger.warning(f"数据源 {source['name']} 查询失败: {e}")
                continue

        # 所有数据源都失败
        return {
            "success": False,
            "error": "未找到药品信息",
            "drug_name": drug_name,
            "suggestion": "请确认药品名称是否正确，或前往正规医疗机构咨询药师。",
        }

    async def _search_dxy(self, drug_name: str, query_type: str) -> Dict[str, Any]:
        """
        从丁香园用药助手搜索药品信息

        注意：丁香园网站有反爬机制，此方法可能不稳定
        """
        try:
            # 搜索药品
            search_url = f"https://drugs.dxy.cn/search?keyword={quote_plus(drug_name)}"
            response = await self._client.get(search_url)

            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}"}

            html = response.text

            # 解析搜索结果页面，提取药品链接
            # 丁香园搜索结果通常包含药品列表
            drug_links = self._parse_dxy_search_results(html, drug_name)

            if not drug_links:
                # 没找到，尝试直接访问药品页面
                return await self._fallback_dxy_search(drug_name, query_type)

            # 获取第一个匹配药品的详情
            first_drug_url = drug_links[0]
            detail_response = await self._client.get(first_drug_url)

            if detail_response.status_code != 200:
                return {"success": False, "error": f"详情页 HTTP {detail_response.status_code}"}

            detail_html = detail_response.text
            drug_info = self._parse_dxy_drug_detail(detail_html, query_type)

            if drug_info:
                drug_info["source"] = "丁香园用药助手"
                drug_info["source_url"] = first_drug_url
                return {"success": True, **drug_info}

            return {"success": False, "error": "解析药品详情失败"}

        except Exception as e:
            logger.error(f"丁香园查询失败: {e}")
            return {"success": False, "error": str(e)}

    def _parse_dxy_search_results(self, html: str, drug_name: str) -> List[str]:
        """
        解析丁香园搜索结果页面，提取药品链接
        """
        links = []

        # 简单的正则匹配药品链接
        # 丁香园药品详情页格式: /drug/xxxxx.htm 或 /drugs/xxxx
        patterns = [
            r'href="(/drug/[a-zA-Z0-9\-]+\.htm)"',
            r'href="(/drugs/\d+)"',
            r'href="(https://drugs\.dxy\.cn/drug/[a-zA-Z0-9\-]+)"',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html)
            for match in matches:
                if match.startswith("/"):
                    full_url = f"https://drugs.dxy.cn{match}"
                else:
                    full_url = match
                links.append(full_url)

        # 过滤掉重复链接
        return list(set(links))[:5]  #最多取5个

    async def _fallback_dxy_search(self, drug_name: str, query_type: str) -> Dict[str, Any]:
        """
        丁香园搜索失败时的备用方案：直接尝试药品详情页
        """
        # 一些常见药品的丁香园页面格式
        # 例如：布洛芬 -> https://drugs.dxy.cn/drug/buluofen
        # 这需要根据药品名称推断URL，成功率不高

        # 转换药品名称为可能的URL格式
        url_name = drug_name.lower().replace(" ", "-")

        possible_urls = [
            f"https://drugs.dxy.cn/drug/{url_name}",
            f"https://drugs.dxy.cn/drug/{quote_plus(drug_name)}",
        ]

        for url in possible_urls:
            try:
                response = await self._client.get(url)
                if response.status_code == 200:
                    drug_info = self._parse_dxy_drug_detail(response.text, query_type)
                    if drug_info:
                        drug_info["source"] = "丁香园用药助手"
                        drug_info["source_url"] = url
                        return {"success": True, **drug_info}
            except Exception:
                continue

        return {"success": False, "error": "备用搜索也失败"}

    def _parse_dxy_drug_detail(self, html: str, query_type: str) -> Optional[Dict[str, Any]]:
        """
        解析丁香园药品详情页面，提取药品信息

        丁香园药品详情页结构：
        - 药品名称
        - 适应症
        - 用法用量
        - 不良反应
        - 禁忌
        - 注意事项
        """
        drug_info = {}

        # 提取药品名称
        name_match = re.search(r'<h1[^>]*class="[^"]*drug-name[^"]*"[^>]*>([^<]+)</h1>', html)
        if name_match:
            drug_info["drug_name"] = name_match.group(1).strip()
        else:
            # 备用匹配
            name_match = re.search(r'<title>([^<]+)_丁香园用药助手</title>', html)
            if name_match:
                drug_info["drug_name"] = name_match.group(1).strip()

        # 提取适应症
        indication_match = re.search(
            r'<div[^>]*class="[^"]*indication[^"]*"[^>]*>.*?<p[^>]*>([^<]+)</p>',
            html, re.DOTALL
        )
        if indication_match:
            drug_info["indications"] = indication_match.group(1).strip()

        # 提取用法用量
        dosage_match = re.search(
            r'<div[^>]*class="[^"]*dosage[^"]*"[^>]*>.*?<p[^>]*>([^<]+)</p>',
            html, re.DOTALL
        )
        if dosage_match:
            drug_info["dosage"] = dosage_match.group(1).strip()

        # 提取不良反应
        adverse_match = re.search(
            r'<div[^>]*class="[^"]*adverse[^"]*"[^>]*>.*?<p[^>]*>([^<]+)</p>',
            html, re.DOTALL
        )
        if adverse_match:
            drug_info["adverse_reactions"] = adverse_match.group(1).strip()

        # 提取禁忌
        contraindication_match = re.search(
            r'<div[^>]*class="[^"]*contraindication[^"]*"[^>]*>.*?<p[^>]*>([^<]+)</p>',
            html, re.DOTALL
        )
        if contraindication_match:
            drug_info["contraindications"] = contraindication_match.group(1).strip()

        # 如果没有提取到任何信息，返回 None
        if not drug_info:
            return None

        # 根据查询类型筛选返回内容
        result = {"drug_name": drug_info.get("drug_name", "")}

        if query_type == "基本信息":
            result["indications"] = drug_info.get("indications", "")
            result["contraindications"] = drug_info.get("contraindications", "")
        elif query_type == "用法用量":
            result["dosage"] = drug_info.get("dosage", "")
        elif query_type == "不良反应":
            result["adverse_reactions"] = drug_info.get("adverse_reactions", "")
        elif query_type == "完整说明书":
            result.update(drug_info)

        return result

    async def _search_baike(self, drug_name: str, query_type: str) -> Dict[str, Any]:
        """
        从百度百科搜索药品信息

        百度百科药品词条通常包含：
        - 药品名称
        - 成分/适应症
        - 用法用量
        - 不良反应
        """
        try:
            # 直接访问百度百科词条
            baike_url = f"https://baike.baidu.com/item/{quote_plus(drug_name)}"
            response = await self._client.get(baike_url)

            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}"}

            html = response.text
            drug_info = self._parse_baike_drug_page(html, drug_name)

            if drug_info:
                drug_info["source"] = "百度百科"
                drug_info["source_url"] = baike_url
                return {"success": True, **drug_info}

            return {"success": False, "error": "未找到药品词条"}

        except Exception as e:
            logger.error(f"百度百科查询失败: {e}")
            return {"success": False, "error": str(e)}

    def _parse_baike_drug_page(self, html: str, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        解析百度百科药品页面

        百度百科药品页面结构：
        - 词条标题
        - 基本信息（表格形式）
        - 正文内容
        """
        drug_info = {"drug_name": drug_name}

        # 提取词条标题
        title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
        if title_match:
            drug_info["title"] = title_match.group(1).strip()

        # 提取基本信息表格
        # 百度百科的基本信息通常在 class="basicInfo" 的表格中
        basic_info_match = re.search(
            r'<div[^>]*class="[^"]*basicInfo[^"]*"[^>]*>(.*?)</div>',
            html, re.DOTALL
        )

        if basic_info_match:
            basic_info_html = basic_info_match.group(1)
            # 解析表格中的键值对
            pairs = re.findall(
                r'<dt[^>]*>([^<]+)</dt>.*?<dd[^>]*>([^<]+)</dd>',
                basic_info_html, re.DOTALL
            )
            for key, value in pairs:
                key = key.strip()
                value = value.strip()
                if key in ["适应症", "用途", "功能主治"]:
                    drug_info["indications"] = value
                elif key in ["用法用量", "用量"]:
                    drug_info["dosage"] = value
                elif key in ["不良反应", "副作用"]:
                    drug_info["adverse_reactions"] = value
                elif key in ["禁忌", "注意事项"]:
                    drug_info["contraindications"] = value
                elif key in ["成分", "主要成分"]:
                    drug_info["composition"] = value

        # 从正文提取更多信息
        # 百度百科正文在 class="para" 的段落中
        paragraphs = re.findall(r'<div[^>]*class="[^"]*para[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)

        full_text = "\n".join([re.sub(r'<[^>]+>', '', p).strip() for p in paragraphs])

        # 如果表格信息不完整，从正文提取
        if "indications" not in drug_info:
            indication_match = re.search(
                r'(适应症|功能主治|用途)[：:]\s*([^\n]+)',
                full_text
            )
            if indication_match:
                drug_info["indications"] = indication_match.group(2).strip()

        if "dosage" not in drug_info:
            dosage_match = re.search(
                r'(用法用量|用量|用法)[：:]\s*([^\n]+)',
                full_text
            )
            if dosage_match:
                drug_info["dosage"] = dosage_match.group(2).strip()

        if "adverse_reactions" not in drug_info:
            adverse_match = re.search(
                r'(不良反应|副作用)[：:]\s*([^\n]+)',
                full_text
            )
            if adverse_match:
                drug_info["adverse_reactions"] = adverse_match.group(2).strip()

        # 添加免责声明
        drug_info["disclaimer"] = "以上信息来自百度百科，仅供参考，不能替代专业医生的诊断和治疗建议。"

        return drug_info if len(drug_info) > 2 else None

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def __repr__(self) -> str:
        return f"DrugApiTool(timeout={self._timeout})"