"""从 GeoNames 扩充 A 组词表（城市），并让条目自带可复核的元数据。

**为什么需要这个脚本**

A 组只有 108 个词，学习曲线显示样本量是当前最大的瓶颈
（error 与 n 的指数约 -0.4~-0.5，而参数量只有 -0.12）。但扩词表不能随便加词，
必须满足几条硬约束，否则会引入新的混杂：

1. **真值必须无歧义** —— 同名城市（Springfield、Victoria）会让「正确坐标」不唯一
2. **必须地理均衡** —— 否则会变成「欧美地名识别」而不是「地理知识」
3. **必须跨人口规模** —— 只挑超大城市会引入「知名度」混杂
4. **必须与已有词表不冲突** —— A/B/C 三组都不能重复
5. **必须标注难度档** —— 见下

**difficulty 字段是什么**

它是**目标点的空间尺度**，不是模型属性：

* 粗（coarse）：指代一个**区域**，特征尺度 >= 100 km。
  答错 500 km 可能仍落在「正确区域内」。例：国家、构成国、大岛。
* 精（fine）：指代一个**点状实体**，特征尺度 < 50 km。
  答错 500 km 就是「错城市」。例：城市、城镇。

为什么要区分：探针误差以 km 计，而不同尺度的目标有**不同的固有可容忍半径**。
把国家和城市混在一起算总误差，会把「粗任务已经饱和」和「精任务还在进步」
平均成一个看不出趋势的数字（本项目实测：0.5B 的国家误差只有城市的 0.71 倍）。

本脚本用 expected_radius_km 作为客观依据来推导 difficulty：

* 城市：0.01 * sqrt(人口)，截断到 [1, 30] km（100 万人 -> 10 km，10 万人 -> 3 km）
* 国家：500 km；构成国/地区：200 km

用法::

    python scripts/expand_wordlist.py                 # 生成候选 + 复核报告
    python scripts/expand_wordlist.py --per-continent 40
    python scripts/expand_wordlist.py --no-write      # 只看统计
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

FIELDS = (
    "geonameid", "name", "asciiname", "alternatenames", "lat", "lon",
    "feature_class", "feature_code", "country_code", "cc2",
    "admin1", "admin2", "admin3", "admin4",
    "population", "elevation", "dem", "timezone", "modification",
)

CONTINENTS: Dict[str, Tuple[str, ...]] = {
    "Africa": (
        "DZ AO BJ BW BF BI CM CV CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH GN "
        "GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD "
        "TZ TG TN UG ZM ZW EH"
    ).split(),
    "Asia": (
        "AF AM AZ BH BD BT BN KH CN CY GE IN ID IR IQ IL JP JO KZ KP KR KW KG LA "
        "LB MY MV MN MM NP OM PK PS PH QA SA SG LK SY TW TJ TH TL TR TM AE UZ VN YE"
    ).split(),
    "Europe": (
        "AL AD AT BY BE BA BG HR CZ DK EE FO FI FR DE GI GR GG HU IS IE IM IT JE LV "
        "LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA AX"
    ).split(),
    "North America": (
        "AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC TT US PR VI"
    ).split(),
    "South America": "AR BO BR CL CO EC FK GF GY PY PE SR UY VE".split(),
    "Oceania": "AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU NC PF GU AS MP".split(),
    "Antarctica": ["AQ"],
}

EXTRA_REGIONS: Dict[str, str] = {
    "HK": "Asia", "MO": "Asia", "XK": "Europe",
    "RE": "Africa", "YT": "Africa",
    "GP": "North America", "MQ": "North America", "AW": "North America",
    "CW": "North America", "KY": "North America", "TC": "North America",
    "VG": "North America", "BM": "North America", "GL": "North America",
}

COUNTRY_TO_CONTINENT: Dict[str, str] = {}
for _continent, _codes in CONTINENTS.items():
    for _code in _codes:
        COUNTRY_TO_CONTINENT[_code] = _continent
COUNTRY_TO_CONTINENT.update(EXTRA_REGIONS)

#: 手写条目里的国家名 -> ISO 国家码。
#:
#: **为什么必须显式列表**：国家不能拿去和 cities15000（城市库）匹配 ——
#: 那会产生可笑的误匹配（本项目实测：Venezuela 匹配到了古巴一个叫 Venezuela 的镇）。
COUNTRY_NAME_TO_CODE: Dict[str, str] = {
    "Brazil": "BR", "Argentina": "AR", "Kenya": "KE", "Norway": "NO",
    "Sweden": "SE", "Finland": "FI", "Poland": "PL", "Portugal": "PT",
    "Germany": "DE", "Italy": "IT", "Japan": "JP", "Canada": "CA",
    "Australia": "AU", "Egypt": "EG", "Morocco": "MA", "Vietnam": "VN",
    "Thailand": "TH", "Indonesia": "ID", "Philippines": "PH", "Ethiopia": "ET",
    "Bolivia": "BO", "Uruguay": "UY", "Paraguay": "PY", "Ecuador": "EC",
    "Iceland": "IS", "Ireland": "IE", "Greece": "GR", "Hungary": "HU",
    "Romania": "RO", "Bulgaria": "BG", "Croatia": "HR", "Ukraine": "UA",
    "Peru": "PE", "Venezuela": "VE", "Guatemala": "GT", "Mongolia": "MN",
    "Nepal": "NP",
    "Scotland": "GB", "Wales": "GB",
}

#: 手写条目里「其他」类的实际类型：英国构成国属于区域（粗尺度），不是城市。
REGION_WORDS = {"Scotland", "Wales"}

#: 罗马化变体：手写拼写与 GeoNames 的 asciiname 不一致时用它搭桥。
#: 例：Ulaanbaatar 在 GeoNames 里叫 Ulan Bator（旧译）。
#: 只用于查元数据（国家码/大洲/人口），不改变用作模型输入的词。
GEONAMES_ALIASES: Dict[str, str] = {
    "Ulaanbaatar": "Ulan Bator",
}

#: 城市名同时是**日常高频英文词**的情况（Nice / Split / Bath / Mobile ...）。
#: 这些词的「主要含义」不是城市，属于歧义范畴，应进 B 组而不是 A 组。
AMBIGUOUS_CITY_NAMES = {
    "nice", "split", "bath", "mobile", "reading", "why", "born", "brand", "bride",
    "cork", "deal", "earl", "fear", "frank", "hope", "joy", "lack", "march", "mark",
    "orange", "pen", "plain", "point", "port", "post", "price", "rock", "rose",
    "safe", "seal", "sound", "spare", "spirit", "stand", "star", "state", "stone",
    "store", "test", "top", "total", "touch", "town", "trade", "train", "wall",
    "water", "west", "white", "wind", "window", "work", "young",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点大圆距离（km）。用于判断「匹配到的城市是不是同一个地方」。"""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def expected_radius_km(population: int) -> float:
    """按人口估计城市特征半径（km）。

    0.01 * sqrt(人口)：100 万人 -> 10 km，10 万人 -> 3 km，1 万人 -> 1 km。
    只需量级正确 —— 它只用来把词分档。
    """
    return min(30.0, max(1.0, 0.01 * math.sqrt(max(population, 1))))


def difficulty_from_radius(radius_km: float) -> str:
    """按特征尺度分档：>= 100 km 为粗（区域），否则为精（点状）。"""
    return "粗" if radius_km >= 100.0 else "精"


def is_clean_name(name: str) -> bool:
    """名字必须是纯拉丁**字母+空格**，长度 4~24 个字符。

    三条硬性排除，都是实测踩出来的：

    1. **不含撇号**（Xi'an、Lang'ata）—— BPE 几乎必然把词切碎，
       破坏 entity_token 池化「这个词的 token 是连续的」这一假设
    2. **不含连字符**（Saint-Dizier、Nicetown-Tioga）—— 同样会切碎，
       而且复合地名往往不是「一个地方」的常用指称
    3. **长度 >= 4**（Man、Kai、Vic、Ipu）—— 三字母地名几乎都会被日常词抢走
       （Man 是英文词，Kai/Vic 是常见人名缩写）
    """
    if not name or len(name) > 24 or len(name) < 4:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z ]*", name))


def read_geonames(zip_path: Path) -> List[Dict[str, Any]]:
    """读取 cities15000.zip，返回字典列表。"""
    if not zip_path.exists():
        raise SystemExit(f"找不到 {zip_path}，请把 cities15000.zip 放到 data/ 下")
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        inner = archive.namelist()[0]
        with archive.open(inner) as handle:
            for raw in handle:
                parts = raw.decode("utf-8").rstrip("\n").split("\t")
                if len(parts) < len(FIELDS):
                    continue
                record = dict(zip(FIELDS, parts))
                try:
                    record["lat"] = float(record["lat"])
                    record["lon"] = float(record["lon"])
                    record["population"] = int(record["population"] or 0)
                except (TypeError, ValueError):
                    continue
                record["country_code"] = (record["country_code"] or "").upper()
                rows.append(record)
    return rows


def existing_words() -> set:
    """三组已用过的词（小写）。"""
    used = set()
    for filename in ("words_control.json", "words_ambiguous.json", "words_nongeo.json"):
        path = ROOT / "data" / filename
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            word = str(entry.get("word", "")).strip().lower()
            if word:
                used.add(word)
    return used


def select_candidates(
    rows: List[Dict[str, Any]],
    used: set,
    per_continent: int,
    min_population: int,
    max_population: int,
) -> Tuple[List[Dict[str, Any]], Counter, Counter]:
    """按规则挑城市，返回 (候选, 排除原因, 未映射国家码)。"""
    reasons: Counter = Counter()
    unmapped: Counter = Counter()

    maximum_population: Dict[str, int] = {}
    for record in rows:
        key = record["asciiname"].lower()
        maximum_population[key] = max(maximum_population.get(key, 0), record["population"])

    by_continent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in rows:
        if record["feature_class"] != "P":
            reasons["非人口聚居地"] += 1
            continue
        name = record["asciiname"]
        if not is_clean_name(name):
            reasons["名字含非拉丁字符/撇号/过长"] += 1
            continue
        population = record["population"]
        if population < min_population:
            reasons[f"人口 < {min_population}"] += 1
            continue
        if population > max_population:
            reasons[f"人口 > {max_population}"] += 1
            continue
        if name.lower() in used:
            reasons["与已有词表重名"] += 1
            continue
        if maximum_population.get(name.lower(), 0) > population * 2:
            reasons["同名歧义（有更大的同名城市）"] += 1
            continue
        if name.lower() in AMBIGUOUS_CITY_NAMES:
            reasons["与常用英文词同形"] += 1
            continue

        continent = COUNTRY_TO_CONTINENT.get(record["country_code"], "")
        if not continent:
            reasons["国家码未映射到大洲"] += 1
            unmapped[record["country_code"]] += 1
            continue
        by_continent[continent].append(record)

    selected: List[Dict[str, Any]] = []
    for continent, records in sorted(by_continent.items()):
        records.sort(key=lambda item: -item["population"])
        quota = per_continent
        if len(records) <= quota:
            chosen = records
        else:
            # 纯分层（等分取中点）会把最大城市全丢掉 —— 实测欧洲最大只到 49 万，
            # 分布被截断。所以先取 1/4 配额给头部，其余再分层覆盖中段与下端。
            top_n = max(1, quota // 4)
            chosen = records[:top_n]
            rest = records[top_n:]
            remaining = quota - top_n
            total = len(rest)
            for index in range(remaining):
                start = index * total // remaining
                end = (index + 1) * total // remaining
                chosen.append(rest[(start + end) // 2])

        for record in chosen:
            radius = expected_radius_km(record["population"])
            selected.append(
                {
                    "word": record["asciiname"],
                    "lat": round(record["lat"], 5),
                    "lon": round(record["lon"], 5),
                    "label": f"{record['name']}（{record['country_code']}）",
                    "category": "城市",
                    "difficulty": difficulty_from_radius(radius),
                    "expected_radius_km": round(radius, 1),
                    "continent": continent,
                    "country_code": record["country_code"],
                    "population": record["population"],
                    "geonameid": int(record["geonameid"]),
                    "source": "geonames:cities15000",
                }
            )

    return selected, reasons, unmapped


def dedupe_by_distance(
    entries: List[Dict[str, Any]], min_km: float = 50.0
) -> Tuple[List[Dict[str, Any]], int]:
    """去掉彼此太近的候选（同一都市圈的重复点）。"""
    kept: List[Dict[str, Any]] = []
    removed = 0
    for entry in sorted(entries, key=lambda item: -item["population"]):
        too_close = False
        for other in kept:
            dlat = (entry["lat"] - other["lat"]) * 111.0
            dlon = (entry["lon"] - other["lon"]) * 111.0 * math.cos(math.radians(entry["lat"]))
            if math.hypot(dlat, dlon) < min_km:
                too_close = True
                break
        if too_close:
            removed += 1
        else:
            kept.append(entry)
    return kept, removed


def enrich_manual_entries(
    manual: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    match_tolerance_km: float = 500.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """给手写条目补全 country_code / continent / population，并校准坐标。

    两类分别处理：

    * **国家 / 地区**：只查 COUNTRY_NAME_TO_CODE，**绝不**去城市库匹配
    * **城市**：按 asciiname 匹配，取人口最大者；再用坐标距离校验，
      相差 > 500 km 视为误匹配 -> 保留手写坐标，且**不写它的国家码**
      （写了会让下游误以为地理位置可信）
    """
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in rows:
        by_name[record["asciiname"].lower()].append(record)

    enriched: List[Dict[str, Any]] = []
    flagged: List[Dict[str, Any]] = []

    for entry in manual:
        word = str(entry.get("word", ""))
        note = str(entry.get("note", ""))

        if word in REGION_WORDS:
            category = "地区"
        elif entry.get("category"):
            category = str(entry["category"])
        elif "城市" in note:
            # ⚠ 顺序很重要：必须先判「城市」再判「国家」。
            #   Singapore 的 note 是「城市国家」——两个词都含，
            #   先判国家就会把它误分类成国家，进而查不到国家码、continent 为空。
            category = "城市"
        elif "国家" in note:
            category = "国家"
        else:
            category = "其他"

        if category == "国家":
            radius = 500.0
        elif category == "地区":
            radius = 200.0
        else:
            radius = expected_radius_km(0)

        base: Dict[str, Any] = {
            **entry,
            "category": category,
            "difficulty": difficulty_from_radius(radius),
            "expected_radius_km": radius,
            "source": entry.get("source", "manual"),
        }

        if category in ("国家", "地区"):
            code = COUNTRY_NAME_TO_CODE.get(word, "")
            base["country_code"] = code
            base["continent"] = COUNTRY_TO_CONTINENT.get(code, "")
            base.setdefault("population", 0)
            if not code:
                flagged.append({**base, "_flag": "国家名不在映射表里，continent 为空"})
            enriched.append(base)
            continue

        candidates = by_name.get(word.lower(), [])
        if not candidates:
            alias = GEONAMES_ALIASES.get(word, "")
            if alias:
                candidates = by_name.get(alias.lower(), [])
        if not candidates:
            for record in rows:
                if record["name"].lower() == word.lower():
                    candidates.append(record)
        best = max(candidates, key=lambda record: record["population"], default=None)

        if best is None:
            base.update({"continent": "", "country_code": "", "population": 0})
            flagged.append({**base, "_flag": "GeoNames 里找不到同名城市"})
            enriched.append(base)
            continue

        distance = haversine_km(entry["lat"], entry["lon"], best["lat"], best["lon"])
        continent = COUNTRY_TO_CONTINENT.get(best["country_code"], "")

        if distance > match_tolerance_km:
            base.update({"continent": continent, "population": best["population"]})
            flagged.append(
                {
                    **base,
                    "_flag": (
                        "匹配到 "
                        f"{best['name']}（{best['country_code']}），相距 {distance:,.0f} km；"
                        "已保留手写坐标且不采用其国家码"
                    ),
                }
            )
            enriched.append(base)
            continue

        base.update(
            {
                "lat": round(best["lat"], 5),
                "lon": round(best["lon"], 5),
                "continent": continent,
                "country_code": best["country_code"],
                "population": best["population"],
                "geonameid": int(best["geonameid"]),
                "source": "geonames:cities15000（坐标已校准）",
            }
        )
        enriched.append(base)

    return enriched, flagged


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从 GeoNames 扩充 A 组城市词表")
    parser.add_argument("--per-continent", type=int, default=40, help="每个大洲取多少个城市")
    parser.add_argument("--min-population", type=int, default=15000)
    parser.add_argument(
        "--max-population",
        type=int,
        default=15_000_000,
        help="人口上限。默认几乎不限 —— 知名度效应应当测量（population 已记录），"
        "而不是靠剔掉大城市来回避",
    )
    parser.add_argument("--min-distance-km", type=float, default=50.0)
    parser.add_argument("--no-write", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args(argv)

    zip_path = ROOT / "data" / "cities15000.zip"
    print(f"读取 {zip_path.name} ...")
    rows = read_geonames(zip_path)
    print(f"  共 {len(rows):,} 条聚居地记录")

    used = existing_words()
    print(f"  已有词表占用 {len(used)} 个词")

    candidates, reasons, unmapped = select_candidates(
        rows, used, args.per_continent, args.min_population, args.max_population
    )
    candidates, removed_near = dedupe_by_distance(candidates, args.min_distance_km)

    print()
    print("=== 排除原因 ===")
    for reason, count in reasons.most_common():
        print(f"  {reason:<38}{count:>8,}")
    near_label = f"相邻重复（< {args.min_distance_km:.0f} km）"
    print(f"  {near_label:<38}{removed_near:>8,}")
    if unmapped:
        sample = ", ".join(f"{code}x{n}" for code, n in unmapped.most_common(6))
        print(f"  （未映射的国家码：{sample}）")

    by_continent = Counter(entry["continent"] for entry in candidates)
    print()
    print(f"=== 选中 {len(candidates)} 个城市 ===")
    for continent, count in sorted(by_continent.items()):
        pops = [entry["population"] for entry in candidates if entry["continent"] == continent]
        print(f"  {continent:<16}{count:>4} 个  人口 {min(pops):>10,} ~ {max(pops):>10,}")

    print()
    print(f"难度分档：{dict(Counter(entry['difficulty'] for entry in candidates))}")

    if args.no_write:
        print()
        print("--no-write：未写文件")
        return 0

    manual_raw = json.loads((ROOT / "data" / "words_control.json").read_text(encoding="utf-8"))
    manual, flagged_manual = enrich_manual_entries(manual_raw, rows)

    combined = manual + candidates
    out_path = ROOT / "data" / "words_control_expanded.json"
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    calibrated = sum(1 for entry in manual if "校准" in str(entry.get("source", "")))
    print()
    print(f"[OK] 候选词表：{out_path.name}（{len(combined)} 条 = 原 {len(manual)} + 新 {len(candidates)}）")
    print(f"     手写条目 {len(manual)} 条，其中 {calibrated} 条坐标已用 GeoNames 校准")
    if flagged_manual:
        print(f"     ! {len(flagged_manual)} 条需人工确认（见复核清单）")

    report: List[str] = [
        "# 词表扩充复核清单",
        "",
        f"> 来源 GeoNames `cities15000`　每大洲上限 {args.per_continent}"
        f"　人口区间 [{args.min_population:,}, {args.max_population:,}]",
        f"> 生成：`python scripts/expand_wordlist.py --per-continent {args.per_continent}`",
        "",
        "## 怎么复核",
        "",
        "对每一行只检查三件事：",
        "",
        "1. **这个名字在英文里主要指这个城市吗？**（若主要含义是别的，应移到 B 组）",
        "2. **坐标看着对吗？**（大体在正确的国家/大洲即可，±100 km 不必纠）",
        "3. **difficulty 合理吗？**（城市应为「精」；国家/地区应为「粗」）",
        "",
        "改完后把 `words_control_expanded.json` 改名为 `words_control.json`（旧的先备份）即可生效。",
        "",
        "## 排除统计（供审计）",
        "",
        "| 原因 | 条数 |",
        "|---|---|",
    ]
    for reason, count in reasons.most_common():
        report.append(f"| {reason} | {count:,} |")
    report.append(f"| 相邻重复（<{args.min_distance_km:.0f} km） | {removed_near:,} |")
    report.append("")

    if flagged_manual:
        report += [
            f"## 手写条目中需确认的 {len(flagged_manual)} 条",
            "",
            "这些条目在手写坐标附近找不到同名城市。**坐标仍用手写值**，请确认。",
            "",
            "| 词 | 手写坐标 | 问题 |",
            "|---|---|---|",
        ]
        for entry in flagged_manual:
            report.append(
                f"| `{entry['word']}` | {entry.get('lat')}, {entry.get('lon')} | {entry['_flag']} |"
            )
        report.append("")

    report += ["## 候选城市（按大洲，人口降序）", ""]
    for continent in sorted(by_continent):
        items = sorted(
            [entry for entry in candidates if entry["continent"] == continent],
            key=lambda item: -item["population"],
        )
        report.append(f"### {continent}（{len(items)} 个）")
        report.append("")
        report.append("| 词 | 坐标 | 人口 | 国家 | difficulty |")
        report.append("|---|---|---|---|---|")
        for entry in items:
            report.append(
                f"| `{entry['word']}` | {entry['lat']:.3f}, {entry['lon']:.3f} | "
                f"{entry['population']:,} | {entry['country_code']} | {entry['difficulty']} |"
            )
        report.append("")

    report += [
        "## 原有条目（已补 difficulty / continent，城市坐标已校准）",
        "",
        "| 词 | 坐标 | 类别 | difficulty | 国家码 | 大洲 | 来源 |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in manual:
        report.append(
            f"| `{entry['word']}` | {entry.get('lat')}, {entry.get('lon')} | "
            f"{entry.get('category')} | {entry.get('difficulty')} | "
            f"{entry.get('country_code') or '—'} | {entry.get('continent') or '—'} | "
            f"{entry.get('source')} |"
        )
    report.append("")

    report += [
        "## 已知局限",
        "",
        "1. **AMBIGUOUS_CITY_NAMES 不完整** —— 只能挡住最常见的日常词同形城市，",
        "   复核时若发现「这名字主要是别的意思」，删掉即可。",
        "2. **同名歧义用人口比阈值**（有 2 倍以上更大的同名城市则排除）。",
        "   能挡住 Springfield 这类，挡不住「同名且规模相近」的情况。",
        "3. **坐标用 GeoNames 聚居地点**，精度通常 < 10 km，完全够用。",
        "4. **difficulty 只有粗/精两档**，没有中间档。",
        "",
    ]

    report_path = ROOT / "docs" / "词表扩充复核.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] 复核清单：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
