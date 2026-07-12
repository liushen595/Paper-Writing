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
DOJ News Press Release Crawler — 配置模块
============================================
所有可调参数集中管理，方便调试和调整。
"""

import os

# ─── 代理 ───────────────────────────────────────────────────────────────
# 从环境变量读取代理地址，若未设置则使用默认值
PROXY_URL = os.environ.get("DOJ_PROXY", "http://127.0.0.1:2778")

# ─── 目标 ───────────────────────────────────────────────────────────────
START_URL = "https://www.justice.gov/news/press-releases"

# ─── 输出路径 ───────────────────────────────────────────────────────────
# 所有输出文件均在 output/ 目录下
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "doj_raw.jsonl")
CRAWL_DIR = os.path.join(OUTPUT_DIR, "crawl_checkpoint")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".scrapling_cache")  # development mode 缓存

# ─── 爬取速率控制 ───────────────────────────────────────────────────────
DOWNLOAD_DELAY = 12           # 每次请求前等待秒数（robots.txt Crawl-delay:10 + 缓冲）
CONCURRENT_REQUESTS = 1       # 全局并发数（1 = 完全串行）
CONCURRENT_REQUESTS_PER_DOMAIN = 1  # 单域名并发数

# ─── StealthyFetcher / StealthySession 参数 ───────────────────────────
TIMEOUT = 90_000              # 浏览器操作超时（毫秒），Cloudflare 建议 ≥60s
HEADLESS = True               # 无头模式
SOLVE_CLOUDFLARE = True       # 自动绕过 Cloudflare Turnstile/Interstitial
BLOCK_WEBRTC = True           # 防止 WebRTC 泄露真实 IP
HIDE_CANVAS = True            # Canvas 指纹噪声
GOOGLE_SEARCH = True          # 设置 Google Referer
DNS_OVER_HTTPS = True         # 通过 Cloudflare DoH 防止 DNS 泄露
BLOCK_ADS = True              # 阻止广告/跟踪域名请求（节省带宽）
NETWORK_IDLE = True           # 等待网络空闲，确保 JS 渲染完成
LOAD_DOM = True               # 等待 DOM 加载完成

# ─── Spider 检查点 ─────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 120.0   # 检查点保存间隔（秒）
