"""
Web Search Tool
Searches the web for information using multiple search engines with fallback
"""
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from .base import BaseTool, tool_decorator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """
    Web Search Tool with multiple fallback strategies
    
    Search order:
    1. duckduckgo-search package (if installed)
    2. DuckDuckGo Lite HTML via httpx
    3. Bing Search via httpx (always available fallback)
    """
    
    name = "web_search"
    description = (
        "Search the web for current information, news, facts, and other data. "
        "Useful when you need up-to-date information or specific details not in your training data."
    )
    
    def __init__(self):
        super().__init__()
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (what to search for)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }
    
    async def execute(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute web search with multiple fallback strategies
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        logger.info(f"Searching web for: {query}")
        
        # 策略1: 尝试 duckduckgo-search 包
        try:
            results = await self._duckduckgo_search(query, max_results)
            if results:
                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "source": "duckduckgo-search",
                }
        except Exception as e:
            logger.warning(f"duckduckgo-search failed: {e}")
        
        # 策略2: DuckDuckGo Lite HTML
        try:
            results = await self._duckduckgo_html_search(query, max_results)
            if results:
                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "source": "duckduckgo-html",
                }
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML search failed: {e}")
        
        # 策略3: Bing Search
        try:
            results = await self._bing_search(query, max_results)
            if results:
                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "source": "bing",
                }
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")
        
        # 所有策略都失败
        logger.error("All search strategies failed")
        return {
            "success": False,
            "query": query,
            "error": "Search service temporarily unavailable",
            "suggestion": (
                f"无法搜索 '{query}'。建议：\n"
                "1. 稍后重试\n"
                "2. 直接提供相关信息\n"
                "3. 手动搜索后告知结果"
            ),
            "results": [],
        }
    
    async def _duckduckgo_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict]:
        """使用 duckduckgo-search 包搜索"""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = [
                    {
                        "title": r["title"],
                        "url": r["href"],
                        "snippet": r["body"],
                        "source": "duckduckgo",
                    }
                    for r in ddgs.text(query, max_results=max_results)
                ]
                
            return results
            
        except ImportError:
            logger.debug("duckduckgo-search not installed, skipping")
            raise
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            raise
    
    async def _duckduckgo_html_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict]:
        """
        使用 httpx 调用 DuckDuckGo Lite (HTML) 版本搜索
        """
        import httpx
        
        encoded_query = quote_plus(query)
        url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            html = response.text
            results = self._parse_ddg_lite_html(html, max_results)
            
            return results
    
    def _parse_ddg_lite_html(self, html: str, max_results: int) -> List[Dict]:
        """解析 DuckDuckGo Lite 版本的 HTML 响应"""
        results = []
        
        # DuckDuckGo Lite 使用表格布局
        link_pattern = r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
        snippet_pattern = r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>'
        
        links = re.findall(link_pattern, html, re.DOTALL)
        snippets = re.findall(snippet_pattern, html, re.DOTALL)
        
        def strip_html(text):
            text = re.sub(r'<[^>]+>', '', text)
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&quot;', '"').replace('&#39;', "'")
            return text.strip()
        
        for i in range(min(len(links), max_results)):
            url, title = links[i]
            title = strip_html(title)
            
            snippet = ""
            if i < len(snippets):
                snippet = strip_html(snippets[i])
            
            if title:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "duckduckgo-lite",
                })
        
        return results
    
    async def _bing_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict]:
        """
        使用 httpx 调用 Bing 搜索（无需 API Key）
        通过解析 Bing 搜索页面获取结果
        """
        import httpx
        
        encoded_query = quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded_query}&count={max_results}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            html = response.text
            results = self._parse_bing_html(html, max_results)
            
            return results
    
    def _parse_bing_html(self, html: str, max_results: int) -> List[Dict]:
        """解析 Bing 搜索结果页面"""
        results = []
        
        # Bing 搜索结果结构
        # 每个结果在 <li class="b_algo"> 中
        # 标题: <h2><a href="...">标题</a></h2>
        # 摘要: <p> 或 <div class="b_caption">
        
        # 匹配 b_algo 块
        algo_pattern = r'<li[^>]+class="b_algo"[^>]*>(.*?)</li>'
        blocks = re.findall(algo_pattern, html, re.DOTALL)
        
        def strip_html(text):
            text = re.sub(r'<[^>]+>', '', text)
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&quot;', '"').replace('&#39;', "'")
            text = text.replace('\r', '').replace('\n', ' ').strip()
            # 压缩多余空格
            text = re.sub(r'\s+', ' ', text)
            return text
        
        for block in blocks[:max_results]:
            # 提取链接和标题
            link_match = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            if not link_match:
                continue
            
            url = link_match.group(1)
            title = strip_html(link_match.group(2))
            
            # 提取摘要
            snippet = ""
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
            if snippet_match:
                snippet = strip_html(snippet_match.group(1))
            
            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "bing",
                })
        
        return results
