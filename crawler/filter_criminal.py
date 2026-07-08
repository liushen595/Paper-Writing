#!/usr/bin/env python3
"""
DOJ Press Release 犯罪类稿件筛选器
===================================
基于标题+摘要的关键词规则，从全量新闻稿中筛选出涉及刑事犯罪的稿件。

用法:
    python filter_criminal.py              # 覆盖模式（默认，清空旧结果重写）
    python filter_criminal.py --append     # 追加模式（不清空，新结果追加到已有文件末尾）

输入:  output/doj_press_releases.jsonl
输出:  output/doj_criminal.jsonl      (犯罪类)
       output/doj_non_criminal.jsonl  (非犯罪类)
"""

import json
import os
import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple

# ── 路径 ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INPUT_FILE = os.path.join(OUTPUT_DIR, "doj_press_releases.jsonl")
CRIMINAL_OUT = os.path.join(OUTPUT_DIR, "doj_criminal.jsonl")
NON_CRIMINAL_OUT = os.path.join(OUTPUT_DIR, "doj_non_criminal.jsonl")

# ═══════════════════════════════════════════════════════════════════
#  关键词规则 — 全部小写匹配
# ═══════════════════════════════════════════════════════════════════

# ── A. 强犯罪信号（命中任意一条 → 标记为犯罪类）─────────────────

CRIMINAL_PATTERNS = [
    # --- 定罪/量刑（最可靠）---
    r"\bsentenced\b",                    # 含 sentenced to/for
    r"pleads?\s+guilty",                 # pleads guilty / plead guilty
    r"pleaded\s+guilty",
    r"pled\s+guilty",
    r"\bfound\s+guilty\b",               # jury found guilty
    r"\bguilty\s+of\b",                  # guilty of [crime]
    r"\bconvicted\b",
    r"\bconviction\b",
    r"\bcriminal\s+fine\b",              # 刑事罚款

    # --- 逮捕/起诉 ---
    r"\barrested\b",                     # 逮捕（DOJ语境下均为刑事）
    r"\bindicted\b",
    r"\bindictment\b",
    r"criminal\s+complaint",
    r"criminal\s+charges?\b",
    r"\bextradited?\b",
    r"\bfugitive\b",                     # 逃犯

    # --- 刑罚/监禁 ---
    r"\bprison\s+for\b",
    r"\bprison\s+sentence\b",
    r"\byears?\s+in\s+prison\b",
    r"\bmonths?\s+in\s+prison\b",
    r"\blife\s+in\s+prison\b",
    r"maximum\s+penalty\s+of\b",
    r"faces?\s+a\s+maximum\b",
    r"\bprison\s+term\b",
    r"\bjail\b",

    # --- 特定刑事罪名 ---
    r"\bracketeering\b",
    r"\brico\b",
    r"sex\s+trafficking",
    r"child\s+sexual\s+abuse",
    r"\bcsam\b",
    r"child\s+pornography",
    r"child\s+exploitation",
    r"narcotics?\s+trafficking",
    r"drug\s+trafficking",
    r"\bcocaine\b",
    r"\bfentanyl\b",
    r"\bmethamphetamine\b",
    r"\bheroin\b",
    r"\bmurder\b",
    r"\bkidnapping\b",
    r"\bkidnapped\b",
    r"\bcarjacking\b",
    r"\brobbery\b",
    r"hate\s+crime",
    r"money\s+laundering",
    r"\bbribery\b",
    r"\bribery\b",
    r"\bkickback",
    r"alien\s+smuggling",
    r"human\s+smuggling",
    r"\bespionage\b",
    r"forced\s+labor",
    r"health\s+care\s+fraud",
    r"medicare\s+fraud",
    r"medicaid\s+fraud",
    r"bank\s+secrecy\s+act",
    r"\bwire\s+fraud\b",
    r"\bmail\s+fraud\b",
    r"securities\s+fraud",
    r"tax\s+evasion",
    r"\barson\b",
    r"\bobstruction\s+of\s+justice\b",
    r"\bprostitution\b",
    r"\bsex\s+offender\b",

    # --- 恐怖主义/国家安全 ---
    r"material\s+support.*terroris",
    r"material\s+support.*\bfto\b",
    r"material\s+support.*\bhamas\b",
    r"material\s+support.*\bhezbollah\b",
    r"material\s+support.*\bisis\b",
    r"designated\s+foreign\s+terrorist\s+organization",
    r"\bterroris[mt]\b",

    # --- 火器/暴力犯罪 ---
    r"firearms?\s+trafficking",
    r"gun\s+trafficking",
    r"armed\s+robbery",
    r"aggravated\s+assault",
    r"sexual\s+assault",

    # --- 网络犯罪 ---
    r"cyber\s+(crime|attack|intrusion|hacking)",
    r"computer\s+intrusion",
    r"ransomware",

    # --- 协议类（NPA/DPA，底层是刑事）---
    r"non.prosecution\s+agreement",
    r"deferred\s+prosecution\s+agreement",

    # --- 部门信号 ---
    r"criminal\s+division",
    r"national\s+security\s+division",
    r"national\s+fraud\s+enforcement\s+division",
    r"\bfbi\b",                           # FBI 参与通常为刑事调查

    # --- 法规引用（刑事法典）---
    r"18\s*u\.?\s*s\.?\s*c\.?\s*§",
    r"21\s*u\.?\s*s\.?\s*c\.?\s*§",
    r"title\s+18",
    r"sherman\s+act.*criminal",
    r"controlled\s+substances?\s+act",
    r"arms\s+export\s+control\s+act",

    # --- 判决/刑罚相关 ---
    r"\bprobation\b",                    # 缓刑（刑事）
    r"\brestitution\b",                  # 赔偿（刑事附带）
    r"\bconspir(a|ed|ing)\b",            # 共谋（刑事）
    r"\bfraud\s+scheme\b",               # 欺诈计划
    r"\bdefraud\b",                      # 诈骗
    r"\bperjury\b",                      # 伪证
    r"\bobstruction\b",                  # 妨碍司法
    r"\bcounterfeiting\b",               # 伪造
    r"\bforgery\b",                      # 伪造
    r"\bembezzlement\b",                 # 贪污
    r"\bextortion\b",                    # 敲诈
]

# ── B. 强排除信号（命中 → 排除为非犯罪类）─────────────────────────

NON_CRIMINAL_PATTERNS = [
    # --- 民事诉讼 ---
    r"justice\s+department\s+sues?\b",
    r"justice\s+department\s+files?\s+(suit|a\s+lawsuit|a\s+complaint|complaint)",
    r"justice\s+department\s+seeks?\s+to\s+intervene",
    r"\bfiles?\s+a?\s+lawsuit\b",
    r"\bfiles?\s+suit\b",
    r"\bfiles?\s+a?\s+civil\b",
    r"\bcivil\s+lawsuit\b",
    r"\bcivil\s+complaint\b",
    r"\bcivil\s+suit\b",
    r"\bconsent\s+decree\b",
    r"denaturalization\s+complaint",     # 剥夺国籍是民事诉讼

    # --- 行政/人事 ---
    r"\bappoints?\b",
    r"\bdesignated\s+as\b",
    r"\bnamed\s+as\b",
    r"\brenames?\b.*division",
    r"\breports?\s+to\s+congress\b",
    r"\bdelivers?\s+report\b",

    # --- 民权调查结论（非刑事）---
    r"opens?\s+investigation\b",
    r"justice\s+department\s+finds?\b",
    r"findings?\s+letter",
    r"concludes?\s+investigation",
    r"statement\s+of\s+interest",

    # --- 政策/倡议 ---
    r"seeks?\s+public\s+comment",
    r"announces?\s+(new\s+)?(policy|initiative|program|grant|guidance|task\s+force)",
    r"prioritization\s+of\b",
    r"launches?\s+(new\s+)?(initiative|program)",

    # --- 纯反垄断民事 ---
    r"antitrust\s+division.*(?:lawsuit|settlement|civil|complaint|sues?)",
    r"merger\s+(clearance|review|challenge)",
    r"premerger\s+notification",

    # --- 纯民事和解（仅在不涉及刑事协议时）---
    r"false\s+claims\s+act.*settlement",
    # "agrees to pay" 仅在无 NPA/DPA/guilty plea 语境下为排除信号
    r"civil\s+penalty.*(?:settlement|agreement)",
    r"civil\s+settlement",
    r"\bcivil\s+rights\s+division\b.*\bconcludes?\b",
    r"\bcivil\s+rights\s+division\b.*\bfinds?\b",
    r"\bcivil\s+rights\s+division\b.*\breport\b",
    r"agrees?\s+to\s+pay\s+\$[\d,.]+.*(?:resolve|civil)\s+allegations?\b",
    r"agrees?\s+to\s+pay\s+\$[\d,.]+.*false\s+claims",

    # --- 法院驳回类（政府防御性胜诉）---
    r"court\s+dismisses\b",
    r"court\s+grants.*motion\s+to\s+dismiss",
    r"\bdismissal\s+of\b.*\blawsuit\b",

    # --- 反垄断民事判决 ---
    r"antitrust\s+division.*\bsettlement\b",
    r"requires?\s+.*\bto\s+divest\b",

    # --- 环境/监管民事（仅当不涉及刑事定罪时）---
    # 注意：CWA/RCRA 等有刑事条款，因此这些词不作为强排除信号，
    # 仅在无刑事信号时辅助判断
    r"\bepa\b.*rulemaking",
    r"\bnotice\s+of\s+proposed\b",
]

# ── C. 排除信号"覆盖检查"：即使用了排除词但仍然是刑事案件的例外 ──

# 如果文本匹配了任一排除模式，但同时匹配了以下强刑事覆盖信号，
# 则仍然保留为犯罪类
CRIMINAL_OVERRIDE = [
    r"sentenced\s+to\s+\d+\s+(year|month|day)",
    r"sentenced\s+to\s+(prison|jail|life)",
    r"pleads?\s+guilty",
    r"pleaded\s+guilty",
    r"\bprison\s+sentence\b",
    r"\byears?\s+in\s+prison\b",
    r"\bcriminal\s+complaint\b",
    r"\bindicted\b",
    r"\bextradited\b",
    # NPA/DPA 是刑事替代处置，应覆盖民事排除信号
    r"non.prosecution\s+agreement",
    r"deferred\s+prosecution\s+agreement",
    # 动物斗殴（刑事犯罪）覆盖 AWA 民事排除
    r"\bdog\s+fighting\b",
    r"\bcockfighting\b",
    r"animal\s+(cruelty|fighting)",
]


# ═══════════════════════════════════════════════════════════════════
#  分类逻辑
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ClassifyResult:
    is_criminal: bool
    matched_criminal: List[str] = field(default_factory=list)
    matched_non_criminal: List[str] = field(default_factory=list)
    overridden: bool = False


def classify(title: str, summary: str, body: str = "") -> ClassifyResult:
    """基于 title + summary 判断是否为犯罪类稿件。"""
    # 拼接文本（标题放前面，权重更高）
    text = f"{title} {summary}".lower()

    result = ClassifyResult(is_criminal=False)

    # Step 1: 匹配强犯罪信号
    for pat in CRIMINAL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            result.matched_criminal.append(pat)
            result.is_criminal = True

    # Step 2: 匹配强排除信号
    hit_exclusion = False
    for pat in NON_CRIMINAL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            result.matched_non_criminal.append(pat)
            hit_exclusion = True

    # Step 3: 若同时命中犯罪信号和排除信号，检查是否有强覆盖信号
    if result.is_criminal and hit_exclusion:
        # 有排除信号，检查是否可以覆盖
        for pat in CRIMINAL_OVERRIDE:
            if re.search(pat, text, re.IGNORECASE):
                result.overridden = True
                break
        else:
            # 没有强覆盖信号 → 排除信号生效，反转分类
            if len(result.matched_non_criminal) >= 2:
                # 多个排除信号 → 排除
                result.is_criminal = False
            elif len(result.matched_criminal) <= 1:
                # 排除信号 ≥ 犯罪信号 → 排除
                result.is_criminal = False
            # 否则保留犯罪类（只有一个排除信号但犯罪信号更多）

    # Step 4: 如果只有排除信号，没有犯罪信号 → 非犯罪
    if not result.matched_criminal and hit_exclusion:
        result.is_criminal = False

    # Step 5: 既无犯罪信号也无排除信号 → 默认为非犯罪（保守）
    if not result.matched_criminal and not hit_exclusion:
        result.is_criminal = False

    return result


# ═══════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DOJ Press Release 犯罪类筛选")
    parser.add_argument("--append", action="store_true", help="追加模式：不清空已有输出文件，新结果追加到末尾")
    args = parser.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        sys.exit(1)

    total = 0
    criminal = 0
    non_criminal = 0
    overridden_count = 0
    borderline: List[dict] = []  # 边界案例（同时命中犯罪+排除信号）

    # 追加模式用 "a"，覆盖模式用 "w"
    file_mode = "a" if args.append else "w"
    if args.append:
        print("📎 追加模式：新结果将追加到已有输出文件末尾")

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(CRIMINAL_OUT, file_mode, encoding="utf-8") as f_crim, \
         open(NON_CRIMINAL_OUT, file_mode, encoding="utf-8") as f_noncrim:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            title = record.get("title", "")
            summary = record.get("summary", "")
            body = record.get("body", "")

            result = classify(title, summary, body)

            if result.overridden:
                overridden_count += 1

            if result.is_criminal:
                criminal += 1
                f_crim.write(line + "\n")
            else:
                non_criminal += 1
                f_noncrim.write(line + "\n")

            # 收集边界案例
            if result.matched_criminal and result.matched_non_criminal:
                borderline.append({
                    "title": title,
                    "criminal_matches": result.matched_criminal[:5],
                    "non_criminal_matches": result.matched_non_criminal[:5],
                    "overridden": result.overridden,
                    "final": "CRIMINAL" if result.is_criminal else "NON-CRIMINAL",
                })

    # ── 输出统计 ──────────────────────────────────────────────────
    print("=" * 60)
    print("  DOJ Press Release 犯罪类筛选报告")
    print("=" * 60)
    print(f"  总记录数:       {total}")
    print(f"  犯罪类:         {criminal}  ({criminal/total*100:.1f}%)")
    print(f"  非犯罪类:       {non_criminal}  ({non_criminal/total*100:.1f}%)")
    print(f"  其中覆盖纠正:   {overridden_count}")
    print(f"  边界模糊案例:   {len(borderline)}")
    print("-" * 60)
    print(f"  输出文件:")
    print(f"    犯罪类 → {CRIMINAL_OUT}")
    print(f"    非犯罪 → {NON_CRIMINAL_OUT}")
    print("=" * 60)

    # ── 列出边界案例 ──────────────────────────────────────────────
    if borderline:
        print(f"\n⚠️  边界模糊案例 ({len(borderline)} 条):")
        print("-" * 60)
        for i, b in enumerate(borderline[:30], 1):
            print(f"\n[{i}] [{b['final']}] {b['title'][:120]}")
            if not b['overridden']:
                print(f"    犯罪信号: {b['criminal_matches'][:3]}")
                print(f"    排除信号: {b['non_criminal_matches'][:3]}")
        if len(borderline) > 30:
            print(f"\n    ... 还有 {len(borderline) - 30} 条边界案例未显示")


if __name__ == "__main__":
    main()
