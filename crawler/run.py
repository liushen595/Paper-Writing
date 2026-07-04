#!/usr/bin/env python3
"""
DOJ News Press Release Crawler — 入口脚本
==========================================
用法:
    python run.py                    # 正常爬取（支持 Ctrl+C 暂停/恢复）
    python run.py --dev              # 开发模式（缓存响应，反复调试选择器）
    python run.py --fresh            # 清除检查点，重新开始
    python run.py --max-pages 5      # 仅爬取前 5 页（测试用）
"""

import argparse
import logging
import os
import shutil
import sys

# 确保在 crawler/ 目录下可运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doj_spider import DojNewsSpider
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="爬取美国司法部 (DOJ) 新闻稿 (Press Release)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="开发模式：缓存响应到磁盘，反复运行不重复请求",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="清除检查点和缓存，从第一页重新开始",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="最多爬取的页数（不包括已缓存的页面）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── --fresh: 清除检查点和缓存 ────────────────────────────────
    if args.fresh:
        if os.path.exists(config.CRAWL_DIR):
            shutil.rmtree(config.CRAWL_DIR)
            print(f"🧹 已清除检查点: {config.CRAWL_DIR}")
        cache_dir = os.path.join(config.OUTPUT_DIR, ".scrapling_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"🧹 已清除缓存: {cache_dir}")
        # 注意：不清除输出文件，防止误删已有数据

    # ── 构造 Spider ──────────────────────────────────────────────
    spider = DojNewsSpider(
        crawldir=config.CRAWL_DIR,
        interval=config.CHECKPOINT_INTERVAL,
    )

    # 开发模式
    if args.dev:
        spider.development_mode = True
        spider.development_cache_dir = config.CACHE_DIR
        print("🔧 开发模式已启用 — 响应将被缓存，重复运行不会发起网络请求")

    # 最大页数限制
    if args.max_pages:
        spider.max_pages = args.max_pages
        print(f"📄 最大页数限制: {args.max_pages}")

    # ── 打印配置信息 ─────────────────────────────────────────────
    print("=" * 60)
    print("  DOJ 新闻稿爬虫")
    print("=" * 60)
    print(f"  目标:     {config.START_URL}")
    print(f"  代理:     {config.PROXY_URL}")
    print(f"  DoH:      {config.DNS_OVER_HTTPS}")
    print(f"  输出:     {config.OUTPUT_FILE}")
    print(f"  检查点:   {config.CRAWL_DIR}")
    print(f"  延迟:     {config.DOWNLOAD_DELAY}s")
    print(f"  超时:     {config.TIMEOUT / 1000}s")
    print("=" * 60)
    print("  按 Ctrl+C 优雅暂停（再次按 Ctrl+C 强制退出）")
    print("=" * 60)

    # ── 运行 ──────────────────────────────────────────────────────
    try:
        result = spider.start()
    except KeyboardInterrupt:
        print("\n⚠️  用户中断，等待当前任务完成后退出...")
        # Spider 内部已处理 KeyboardInterrupt
        return

    # ── 结果统计 ──────────────────────────────────────────────────
    stats = result.stats
    print("\n" + "=" * 60)
    print("  爬取完成！" if result.completed else "  爬取已暂停（可重新运行恢复）")
    print("=" * 60)
    print(f"  请求总数:     {stats.requests_count}")
    print(f"  成功请求:     {stats.requests_count - stats.failed_requests_count}")
    print(f"  失败请求:     {stats.failed_requests_count}")
    print(f"  被拦截请求:   {stats.blocked_requests_count}")
    print(f"  爬取条目:     {stats.items_scraped}")
    print(f"  丢弃条目:     {stats.items_dropped}")
    print(f"  响应大小:     {stats.response_bytes / 1024:.1f} KB")
    print(f"  耗时:         {stats.elapsed_seconds:.1f} 秒")
    print(f"  速度:         {stats.requests_per_second:.2f} req/s")
    if hasattr(stats, "cache_hits") and stats.cache_hits:
        print(f"  缓存命中:     {stats.cache_hits}")
        print(f"  缓存未命中:   {stats.cache_misses}")
    print(f"  输出文件:     {config.OUTPUT_FILE}")
    print("=" * 60)

    if result.paused:
        print("\n💡 提示: 重新运行 `python run.py` 即可从断点继续爬取\n")
    else:
        # 输出文件行数统计
        if os.path.exists(config.OUTPUT_FILE):
            with open(config.OUTPUT_FILE, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            print(f"\n📊 输出文件共有 {line_count} 条新闻稿记录\n")


if __name__ == "__main__":
    main()
