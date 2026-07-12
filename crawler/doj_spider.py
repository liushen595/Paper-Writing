## Copyright [2026] [Yijun Liu, Soochow University]
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

"""
DOJ News Press Release Spider
==============================
基于 Scrapling Spider 框架，爬取美国司法部新闻稿（Press Release）。
- 自动绕过 Cloudflare 防护
- 通过本地代理 + DoH 防止 IP/DNS 泄露
- 串行爬取，严格控制频率
- JSONL 增量保存，支持 Ctrl+C 中断恢复
"""

import json
import logging
import os
import sys

# 将项目根目录加入 sys.path，便于直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrapling.spiders import Spider, Response, Request
from scrapling.fetchers import AsyncStealthySession

import config


class DojNewsSpider(Spider):
    """爬取 justice.gov 新闻稿的 Spider。"""

    # ── Spider 元信息 ──────────────────────────────────────────────
    name = "doj_news"
    start_urls = [config.START_URL]
    allowed_domains = {"justice.gov"}
    # 测试限制：最多爬取的列表页数（None = 不限）
    max_pages = None

    # ── 速率控制 ────────────────────────────────────────────────────
    concurrent_requests = config.CONCURRENT_REQUESTS
    concurrent_requests_per_domain = config.CONCURRENT_REQUESTS_PER_DOMAIN
    download_delay = config.DOWNLOAD_DELAY
    robots_txt_obey = False  # DOJ robots.txt 过度屏蔽 ?page= (网站自身用此分页)，但仍通过 download_delay=12 遵守 Crawl-delay:10

    # ── 日志 ────────────────────────────────────────────────────────
    logging_level = logging.INFO
    log_file = os.path.join(config.OUTPUT_DIR, "crawl.log")

    # ═══════════════════════════════════════════════════════════════
    #  CSS 选择器（基于实际页面结构确认）
    # ═══════════════════════════════════════════════════════════════

    # --- 列表页：每个 Press Release 卡片 ---
    ARTICLE_CARD = "div.views-row"
    # 标题 & 链接
    ARTICLE_TITLE_LINK = "h2.news-title a"
    # 发布日期
    ARTICLE_DATE = ".node-date time"
    # 摘要
    ARTICLE_SUMMARY = ".field_teaser"

    # --- 详情页 ---
    DETAIL_TITLE = "h1.page-title"
    DETAIL_DATE = ".node-date time"
    DETAIL_BODY = "div.node-body"
    DETAIL_SECTION = "div.node-content.node-press-release"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_file = None
        self._pages_processed = 0  # 已处理的列表页数

    # ── 会话配置 ──────────────────────────────────────────────────
    def configure_sessions(self, manager):
        """配置 StealthySession — 自动绕过 Cloudflare + 代理 + DoH。"""
        session = AsyncStealthySession(
            # 代理 & DNS 泄露防护
            proxy=config.PROXY_URL,
            dns_over_https=config.DNS_OVER_HTTPS,
            # Cloudflare 绕过
            solve_cloudflare=config.SOLVE_CLOUDFLARE,
            # 反指纹 & 隐私
            headless=config.HEADLESS,
            block_webrtc=config.BLOCK_WEBRTC,
            hide_canvas=config.HIDE_CANVAS,
            google_search=config.GOOGLE_SEARCH,
            block_ads=config.BLOCK_ADS,
            # 等待网络空闲（确保 JS 渲染完成）
            network_idle=config.NETWORK_IDLE,
            load_dom=config.LOAD_DOM,
            # 超时
            timeout=config.TIMEOUT,
        )
        manager.add("default", session)

    # ── 生命周期钩子 ──────────────────────────────────────────────
    async def on_start(self, resuming: bool = False):
        """爬取开始前打开输出文件。"""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        # 以追加模式打开 JSONL 输出文件
        self._output_file = open(config.OUTPUT_FILE, "a", encoding="utf-8")
        if resuming:
            self.logger.info(
                "🔄 从检查点恢复，输出文件以追加模式打开: %s", config.OUTPUT_FILE
            )
        else:
            self.logger.info(
                "🚀 全新爬取，输出文件: %s", config.OUTPUT_FILE
            )
        self.logger.info("代理: %s | DoH: %s", config.PROXY_URL, config.DNS_OVER_HTTPS)

    async def on_close(self):
        """爬取结束后关闭输出文件。"""
        if self._output_file and not self._output_file.closed:
            self._output_file.flush()
            self._output_file.close()
            self.logger.info("📄 输出文件已关闭")

    # ── 列表页解析 ────────────────────────────────────────────────
    async def parse(self, response: Response):
        """
        解析新闻列表页，识别 Press Release 并跟进详情页。
        同时处理分页。
        """
        url = response.url
        self._pages_processed += 1
        self.logger.info(
            "📃 解析列表页 (第 %d 页): %s", self._pages_processed, url
        )

        # 首次运行时打印页面信息以便调试选择器
        title_text = response.css("title::text").get("")
        self.logger.info("页面标题: %s", title_text)
        self.logger.info("列表页 URL: %s", response.url)

        # ── 查找所有文章卡片 ──
        articles = response.css(self.ARTICLE_CARD)
        self.logger.info(
            "找到 %d 篇文章（选择器: %s）", len(articles), self.ARTICLE_CARD
        )

        if not articles:
            self.logger.warning("未找到文章容器")
            return

        for i, article in enumerate(articles):
            # 提取标题 & 链接
            title_links = article.css(self.ARTICLE_TITLE_LINK)
            if not title_links:
                # 备用：找任意链接
                all_links = article.css("a[href]")
                if not all_links:
                    continue
                title = all_links[0].get_all_text(strip=True)
                link = all_links[0].attrib.get("href", "")
            else:
                title = title_links[0].get_all_text(strip=True)
                link = title_links[0].attrib.get("href", "")

            if not link or not title:
                self.logger.debug("  文章 %d: 跳过（无标题/链接）", i)
                continue

            if not link.startswith("http"):
                link = response.urljoin(link)

            # 提取日期
            date_els = article.css(self.ARTICLE_DATE)
            date_text = date_els[0].get_all_text(strip=True) if date_els else ""

            # 提取摘要
            summary_els = article.css(self.ARTICLE_SUMMARY)
            summary = summary_els[0].get_all_text(strip=True) if summary_els else ""

            self.logger.info("  📰 [%d/%d] %s (%s)", i + 1, len(articles), title[:80], date_text)
            yield Request(
                url=link,
                callback=self.parse_article,
                meta={
                    "listing_title": title,
                    "listing_date": date_text,
                    "listing_summary": summary,
                },
            )

        # ── 分页：查找下一页 ──
        async for item in self._follow_next_page(response):
            yield item

    # ── 详情页解析 ────────────────────────────────────────────────
    async def parse_article(self, response: Response):
        """解析新闻稿详情页，提取完整内容。"""
        url = response.url
        self.logger.info("📄 解析详情页: %s", url)

        meta = response.meta or {}

        # 标题（详情页优先）
        title_els = response.css(self.DETAIL_TITLE)
        title = meta.get("listing_title", "")
        if title_els:
            title = title_els[0].get_all_text(strip=True) or title

        # 日期（详情页优先，取第一个 time 元素）
        date_els = response.css(self.DETAIL_DATE)
        date_text = meta.get("listing_date", "")
        if date_els:
            date_text = date_els[0].get_all_text(strip=True) or date_text

        # 正文：从确认的容器中提取
        body_els = response.css(self.DETAIL_BODY)
        body_text = ""
        if body_els:
            body_text = body_els[0].get_all_text(separator="\n", strip=True)

        # 标签：已通过 Press Releases 页面确认，添加固定标签
        tags = ["Press Release"]

        # 摘要
        summary = meta.get("listing_summary", "")

        item = {
            "url": url,
            "title": title,
            "date": date_text,
            "tags": tags,
            "summary": summary,
            "body": body_text,
        }

        yield item

    # ── on_scraped_item 钩子：增量写入 JSONL ─────────────────────
    async def on_scraped_item(self, item: dict) -> dict | None:
        """
        每当爬取到一个有效条目，立即以 JSONL 格式追加到输出文件。
        返回 item 以保留在 Spider 的结果列表中。
        """
        if not item.get("title"):
            self.logger.warning("⏭️ 丢弃无标题条目: %s", item.get("url", ""))
            return None

        # 写入 JSONL（每行一个 JSON 对象）
        line = json.dumps(item, ensure_ascii=False) + "\n"
        if self._output_file and not self._output_file.closed:
            self._output_file.write(line)
            self._output_file.flush()  # 立即落盘，防止中断丢失
            self.logger.info("💾 已保存: %s", item.get("title", "")[:80])
        else:
            self.logger.error("❌ 输出文件未打开，无法写入: %s", item.get("url"))

        return item

    # ── 辅助方法 ──────────────────────────────────────────────────

    async def _follow_next_page(self, response: Response):
        """查找并跟进下一页链接（DOJ 使用 ?page=N 分页，USWDS 分页组件）。"""
        # 检查是否达到最大页数限制
        if self.max_pages is not None and self._pages_processed >= self.max_pages:
            self.logger.info(
                "🛑 已达最大页数限制 (%d)，停止爬取", self.max_pages
            )
            return

        # USWDS 分页：查找 .usa-pagination 中的 Next 链接
        for a in response.css("a[href*='page=']"):
            text = a.get_all_text(strip=True).lower()
            href = a.attrib.get("href", "")
            if text == "next" and href:
                next_url = response.urljoin(href)
                self.logger.info("➡️ 发现下一页: %s", next_url)
                yield Request(url=next_url, callback=self.parse)
                return

        # 备用：找 rel="next"
        for a in response.css("a[rel='next']"):
            href = a.attrib.get("href", "")
            if href:
                next_url = response.urljoin(href)
                self.logger.info("➡️ 发现下一页 (rel=next): %s", next_url)
                yield Request(url=next_url, callback=self.parse)
                return

        self.logger.info("🏁 已无下一页，爬取完成")

    def _is_press_release(self, tags: list[str], title: str, summary: str) -> bool:
        """
        判断一篇文章是否为 Press Release。
        优先检查标签，其次标题和摘要。
        """
        # 检查标签
        for tag in tags:
            tag_lower = tag.lower().strip()
            for kw in self.PRESS_RELEASE_KEYWORDS:
                if kw in tag_lower:
                    return True

        # 检查标题
        title_lower = title.lower().strip()
        for kw in self.PRESS_RELEASE_KEYWORDS:
            if kw in title_lower:
                return True

        # 检查摘要
        summary_lower = summary.lower().strip()
        for kw in self.PRESS_RELEASE_KEYWORDS:
            if kw in summary_lower:
                return True

        return False
