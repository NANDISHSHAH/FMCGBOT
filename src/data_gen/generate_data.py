"""
Synthetic FMCG data generator for FMCGQABOT QNA Agent.

Generates a coherent "universe" of entities (brands, SKUs, regions, channels,
campaigns) and produces:
  1. A structured SQLite warehouse (sales, inventory, promotions) at
     data/structured/fmcg.db
  2. A set of unstructured markdown documents (market reports, launch memos,
     promo playbooks, category reviews) at data/unstructured/*.md

The two sides deliberately reference the SAME entities (brand names, SKU
codes, region names, campaign names) so that hybrid retrieval and
cross-source synthesis can be demonstrated meaningfully -- e.g. a question
like "Why did NutriOat Gold sales dip in North region in Feb 2025?" requires
pulling a number from SQL AND a causal explanation from a document that
both mention "NutriOat Gold" + "North".

Deterministic: seeded RNG so re-running produces identical data (important
so the notebook's pre-run outputs stay reproducible).
"""
import os
import random
import sqlite3
import datetime as dt
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_DIR = ROOT / "data" / "structured"
UNSTRUCTURED_DIR = ROOT / "data" / "unstructured"
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
UNSTRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STRUCTURED_DIR / "fmcg.db"

# ---------------------------------------------------------------------------
# Entity universe (shared across structured + unstructured data)
# ---------------------------------------------------------------------------
BRANDS = ["NutriOat", "SunFresh", "CrispCo", "PureWave", "HomeGlow"]

CATEGORIES = {
    "NutriOat": "Breakfast Cereals",
    "SunFresh": "Juices & Beverages",
    "CrispCo": "Savory Snacks",
    "PureWave": "Home Care",
    "HomeGlow": "Personal Care",
}

SKUS = [
    # brand, sku_name, sku_code, category, unit, list_price_inr
    ("NutriOat", "NutriOat Gold 500g", "NO-GLD-500", "Breakfast Cereals", "pack", 210),
    ("NutriOat", "NutriOat Classic 1kg", "NO-CLS-1000", "Breakfast Cereals", "pack", 340),
    ("SunFresh", "SunFresh Orange 1L", "SF-ORG-1000", "Juices & Beverages", "bottle", 120),
    ("SunFresh", "SunFresh Mixed Fruit 200ml", "SF-MXF-200", "Juices & Beverages", "bottle", 35),
    ("CrispCo", "CrispCo Masala Chips 90g", "CC-MSL-90", "Savory Snacks", "pack", 20),
    ("CrispCo", "CrispCo Peri Peri 90g", "CC-PRP-90", "Savory Snacks", "pack", 20),
    ("PureWave", "PureWave Floor Cleaner 1L", "PW-FLR-1000", "Home Care", "bottle", 145),
    ("PureWave", "PureWave Dish Gel 500ml", "PW-DSH-500", "Home Care", "bottle", 99),
    ("HomeGlow", "HomeGlow Body Wash 250ml", "HG-BDW-250", "Personal Care", "bottle", 165),
    ("HomeGlow", "HomeGlow Talc 200g", "HG-TLC-200", "Personal Care", "pack", 90),
]

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Modern Trade", "General Trade", "E-commerce", "Quick Commerce"]

CAMPAIGNS = [
    ("Festive Harvest 2024", "NutriOat", "North", "2024-10-01", "2024-11-15"),
    ("Summer Chill Fest", "SunFresh", "South", "2025-03-01", "2025-04-30"),
    ("Crunch Time Combo", "CrispCo", "West", "2025-01-05", "2025-02-10"),
    ("Clean Home Drive", "PureWave", "East", "2025-02-01", "2025-02-28"),
    ("Glow Up Weekend", "HomeGlow", "South", "2025-04-10", "2025-04-20"),
]

MONTHS = []
d = dt.date(2024, 7, 1)
while d <= dt.date(2025, 6, 1):
    MONTHS.append(d.strftime("%Y-%m"))
    # advance one month
    if d.month == 12:
        d = dt.date(d.year + 1, 1, 1)
    else:
        d = dt.date(d.year, d.month + 1, 1)


def seasonal_factor(sku_category, month):
    m = int(month.split("-")[1])
    if sku_category == "Breakfast Cereals" and m in (10, 11):  # festive uptick
        return 1.35
    if sku_category == "Juices & Beverages" and m in (3, 4, 5, 6):  # summer
        return 1.4
    if sku_category == "Home Care" and m == 2:  # spring clean promo dip-then-rise
        return 1.2
    if sku_category == "Personal Care" and m == 4:
        return 1.25
    return 1.0


def build_database():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE brands (
            brand_id INTEGER PRIMARY KEY,
            brand_name TEXT UNIQUE,
            category TEXT
        );

        CREATE TABLE skus (
            sku_id INTEGER PRIMARY KEY,
            sku_code TEXT UNIQUE,
            sku_name TEXT,
            brand_id INTEGER,
            category TEXT,
            unit TEXT,
            list_price_inr REAL,
            FOREIGN KEY(brand_id) REFERENCES brands(brand_id)
        );

        CREATE TABLE regions (
            region_id INTEGER PRIMARY KEY,
            region_name TEXT UNIQUE
        );

        CREATE TABLE channels (
            channel_id INTEGER PRIMARY KEY,
            channel_name TEXT UNIQUE
        );

        CREATE TABLE campaigns (
            campaign_id INTEGER PRIMARY KEY,
            campaign_name TEXT,
            brand_id INTEGER,
            region_id INTEGER,
            start_date TEXT,
            end_date TEXT,
            FOREIGN KEY(brand_id) REFERENCES brands(brand_id),
            FOREIGN KEY(region_id) REFERENCES regions(region_id)
        );

        -- Monthly grain fact table: one row per SKU x region x channel x month
        CREATE TABLE sales_fact (
            fact_id INTEGER PRIMARY KEY,
            sku_id INTEGER,
            region_id INTEGER,
            channel_id INTEGER,
            month TEXT,               -- 'YYYY-MM'
            units_sold INTEGER,
            gross_revenue_inr REAL,
            discount_inr REAL,
            net_revenue_inr REAL,
            FOREIGN KEY(sku_id) REFERENCES skus(sku_id),
            FOREIGN KEY(region_id) REFERENCES regions(region_id),
            FOREIGN KEY(channel_id) REFERENCES channels(channel_id)
        );

        CREATE TABLE inventory_snapshot (
            snap_id INTEGER PRIMARY KEY,
            sku_id INTEGER,
            region_id INTEGER,
            month TEXT,
            closing_stock_units INTEGER,
            days_of_cover REAL,
            FOREIGN KEY(sku_id) REFERENCES skus(sku_id),
            FOREIGN KEY(region_id) REFERENCES regions(region_id)
        );
        """
    )

    # Brands
    brand_ids = {}
    for i, b in enumerate(BRANDS, start=1):
        cur.execute("INSERT INTO brands VALUES (?,?,?)", (i, b, CATEGORIES[b]))
        brand_ids[b] = i

    # SKUs
    sku_ids = {}
    for i, (brand, name, code, cat, unit, price) in enumerate(SKUS, start=1):
        cur.execute(
            "INSERT INTO skus VALUES (?,?,?,?,?,?,?)",
            (i, code, name, brand_ids[brand], cat, unit, price),
        )
        sku_ids[code] = (i, brand, cat, unit, price)

    # Regions / Channels
    region_ids = {}
    for i, r in enumerate(REGIONS, start=1):
        cur.execute("INSERT INTO regions VALUES (?,?)", (i, r))
        region_ids[r] = i

    channel_ids = {}
    for i, c in enumerate(CHANNELS, start=1):
        cur.execute("INSERT INTO channels VALUES (?,?)", (i, c))
        channel_ids[c] = i

    # Campaigns
    for i, (name, brand, region, start, end) in enumerate(CAMPAIGNS, start=1):
        cur.execute(
            "INSERT INTO campaigns VALUES (?,?,?,?,?,?)",
            (i, name, brand_ids[brand], region_ids[region], start, end),
        )

    # Sales fact — synthetic but internally consistent with seasonality + campaigns
    fact_id = 1
    for code, (sid, brand, cat, unit, price) in sku_ids.items():
        base_units = random.randint(4000, 12000)
        for region in REGIONS:
            region_mult = random.uniform(0.7, 1.3)
            for channel in CHANNELS:
                channel_mult = {"Modern Trade": 1.0, "General Trade": 1.15,
                                 "E-commerce": 0.55, "Quick Commerce": 0.4}[channel]
                for month in MONTHS:
                    sfac = seasonal_factor(cat, month)
                    # campaign boost if this brand+region had an active campaign that month
                    campaign_boost = 1.0
                    for cname, cbrand, cregion, cstart, cend in CAMPAIGNS:
                        if cbrand == brand and cregion == region:
                            if cstart[:7] <= month <= cend[:7]:
                                campaign_boost = 1.5
                    noise = random.uniform(0.85, 1.15)
                    units = max(
                        0,
                        int(base_units * region_mult * channel_mult * sfac * campaign_boost * noise / 12)
                    )
                    gross = units * price
                    discount_rate = random.uniform(0.03, 0.12) if campaign_boost > 1 else random.uniform(0.0, 0.05)
                    discount = round(gross * discount_rate, 2)
                    net = round(gross - discount, 2)
                    cur.execute(
                        "INSERT INTO sales_fact VALUES (?,?,?,?,?,?,?,?,?)",
                        (fact_id, sid, region_ids[region], channel_ids[channel],
                         month, units, gross, discount, net),
                    )
                    fact_id += 1

    # Inventory snapshot — sku x region x month
    snap_id = 1
    for code, (sid, brand, cat, unit, price) in sku_ids.items():
        for region in REGIONS:
            stock = random.randint(2000, 20000)
            for month in MONTHS:
                stock = max(500, int(stock * random.uniform(0.85, 1.1)))
                cover = round(stock / max(1, random.randint(150, 900)), 1)
                cur.execute(
                    "INSERT INTO inventory_snapshot VALUES (?,?,?,?,?,?)",
                    (snap_id, sid, region_ids[region], month, stock, cover),
                )
                snap_id += 1

    conn.commit()
    conn.close()
    print(f"Structured DB written to {DB_PATH}")


# ---------------------------------------------------------------------------
# Unstructured documents — deliberately reference the same brands/SKUs/regions/
# campaigns as the structured DB so hybrid queries have something to join on.
# ---------------------------------------------------------------------------
DOCS = []

DOCS.append((
    "market_report_breakfast_cereals_fy25_q3.md",
    {"doc_type": "market_report", "category": "Breakfast Cereals", "period": "2024-Q3",
     "tags": ["cereals", "festive", "north"], "published": "2024-11-20"},
    """# Breakfast Cereals Category — FY25 Q3 Market Report

## Summary
The Breakfast Cereals category grew 18% YoY in Q3 FY25, led by **NutriOat Gold 500g**
(SKU: NO-GLD-500), which posted its strongest quarter on record in the **North** region.
The uplift is attributed to the **Festive Harvest 2024** campaign (Oct 1 – Nov 15, 2024),
which combined in-store sampling with a 8-12% price-off promotion in General Trade outlets.

## Regional Performance
- **North**: Outperformed plan due to Festive Harvest 2024 tie-in with regional
  festival calendars (Diwali). General Trade channel drove the bulk of incremental volume.
- **South**: Flat growth; NutriOat Classic 1kg saw modest cannibalisation from Gold variant.
- **East / West**: In line with category average, no major promotional activity.

## Risks flagged
- Discounting depth in North (up to 12%) compressed net realization; Finance flagged margin
  erosion risk if the same depth is repeated outside festive windows.
- Competitor "OatPro" launched a value pack in South, expected to pressure NutriOat Classic.

## Recommendation
Sustain premium positioning for NutriOat Gold; avoid extending festive discount depth into Q4.
"""
))

DOCS.append((
    "product_launch_memo_nutrioat_gold.md",
    {"doc_type": "launch_memo", "category": "Breakfast Cereals", "period": "2024-09",
     "tags": ["cereals", "launch", "nutrioat"], "published": "2024-09-05"},
    """# Product Launch Memo — NutriOat Gold 500g (SKU: NO-GLD-500)

## Positioning
NutriOat Gold is a premium whole-grain oat variant targeted at health-conscious urban
households, priced at INR 210 (approx. 15% premium over NutriOat Classic).

## Launch Plan
- Phase 1 (Sept 2024): Modern Trade + E-commerce listing across North and West.
- Phase 2 (Oct–Nov 2024): General Trade rollout in North, timed with Festive Harvest 2024
  campaign for maximum trial generation during the festive season.

## Success Metrics
- Target: 10,000 units/month in North by end of Q3 FY25.
- Distribution target: 60% weighted distribution in General Trade, North, by Nov 2024.

## Notes
Marketing has requested that any future discounting stay under 10% to protect the
premium brand narrative established at launch.
"""
))

DOCS.append((
    "promo_playbook_festive_harvest_2024.md",
    {"doc_type": "promo_playbook", "category": "Breakfast Cereals", "period": "2024-10",
     "tags": ["campaign", "festive harvest", "nutrioat", "north"], "published": "2024-09-25"},
    """# Promo Playbook — Festive Harvest 2024

**Brand:** NutriOat | **Region:** North | **Duration:** Oct 1 – Nov 15, 2024

## Mechanics
- In-store sampling at 400+ General Trade outlets across North.
- Price-off promotion: 8% standard, up to 12% at high-footfall outlets during festival week.
- Bundled offer: NutriOat Gold 500g + recipe booklet.

## Objective
Drive trial of NutriOat Gold among existing NutriOat Classic buyers and new-to-brand
households during the high-consumption festive period.

## Post-campaign Read (preliminary)
Early POS data shows a step-up in offtake in North versus the prior quarter; category
team to validate against final sales reconciliation in the Q3 market report.
"""
))

DOCS.append((
    "market_report_beverages_summer_2025.md",
    {"doc_type": "market_report", "category": "Juices & Beverages", "period": "2025-Q2",
     "tags": ["beverages", "summer", "south"], "published": "2025-05-10"},
    """# Juices & Beverages — Summer 2025 Market Report

## Summary
**SunFresh Orange 1L** (SKU: SF-ORG-1000) led category growth through the summer window
(Mar–Jun 2025), aided by the **Summer Chill Fest** campaign in the **South** region.

## Campaign Read
Summer Chill Fest (Mar 1 – Apr 30, 2025) combined chilled-display placement in Modern
Trade with a value pack in E-commerce. South region net revenue grew faster than units,
implying limited discounting relative to volume gained — a healthier growth pattern than
the Q3 FY25 cereals campaign in North.

## Watch-outs
Quick Commerce penetration for SunFresh Mixed Fruit 200ml remains below category average;
category team recommends a dedicated quick-commerce assortment review in Q3.
"""
))

DOCS.append((
    "category_review_snacks_fy25.md",
    {"doc_type": "category_review", "category": "Savory Snacks", "period": "2025-Q1",
     "tags": ["snacks", "crispco", "west"], "published": "2025-02-15"},
    """# Savory Snacks — FY25 Category Review

## Summary
**CrispCo** brand held steady share through FY25 H1. The **Crunch Time Combo** campaign
(Jan 5 – Feb 10, 2025) in the **West** region bundled CrispCo Masala Chips 90g and
CrispCo Peri Peri 90g at a combo price point aimed at General Trade impulse purchase.

## Observations
- Combo mechanics lifted units but at a higher discount depth (avg. ~10%) than typical
  snacks promotions (~4-5%), consistent with a deliberate trial-generation push rather
  than a margin-accretive play.
- No major stockout incidents reported in West during the campaign window.
"""
))

DOCS.append((
    "promo_playbook_clean_home_drive.md",
    {"doc_type": "promo_playbook", "category": "Home Care", "period": "2025-02",
     "tags": ["campaign", "purewave", "east"], "published": "2025-01-28"},
    """# Promo Playbook — Clean Home Drive

**Brand:** PureWave | **Region:** East | **Duration:** Feb 1 – Feb 28, 2025

## Mechanics
Floor Cleaner + Dish Gel cross-promotion in Modern Trade and General Trade, East region,
timed with the pre-spring cleaning season. Discount depth capped at 8% per Finance
guidance following the Q3 FY25 cereals campaign learnings on margin erosion.

## Objective
Grow PureWave household penetration in East ahead of the April festive/spring cleaning
period, without repeating the margin compression seen in the North cereals campaign.
"""
))

DOCS.append((
    "product_launch_memo_homeglow_bodywash.md",
    {"doc_type": "launch_memo", "category": "Personal Care", "period": "2025-03",
     "tags": ["homeglow", "launch", "south"], "published": "2025-03-10"},
    """# Product Launch Memo — HomeGlow Body Wash 250ml (SKU: HG-BDW-250)

## Positioning
Entry-premium body wash targeted at Gen-Z consumers in urban South and West markets.

## Launch Plan
Tied to the **Glow Up Weekend** campaign (Apr 10–20, 2025) in **South**, featuring
influencer-led sampling in Modern Trade and E-commerce.

## Success Metrics
Target: 60% distribution in Modern Trade, South, within 2 months of the Glow Up Weekend
campaign window.
"""
))

DOCS.append((
    "fy25_annual_business_review_summary.md",
    {"doc_type": "annual_review", "category": "Cross-category", "period": "FY25",
     "tags": ["annual review", "portfolio", "all brands"], "published": "2025-06-20"},
    """# FY25 Annual Business Review — Executive Summary

## Portfolio Highlights
- **NutriOat**: Strongest brand growth in FY25, driven by NutriOat Gold's North festive
  performance (see Q3 FY25 market report). Margin discipline needed outside festive windows.
- **SunFresh**: Healthy summer-led growth in South with better revenue-to-volume ratio
  than the NutriOat festive campaign — held up as the "efficient promotion" benchmark
  for FY26 planning.
- **CrispCo**: Stable share; Crunch Time Combo in West prioritized trial over margin.
- **PureWave**: Clean Home Drive in East applied capped discounting (8%) per Finance
  guidance, avoiding the margin erosion flagged in the cereals category.
- **HomeGlow**: Body Wash launch in South via Glow Up Weekend is the newest bet, results
  pending full-quarter read.

## FY26 Planning Note
Finance has recommended an org-wide discount depth ceiling of 10% for any single-brand
campaign outside of the festive quarter (Oct-Nov), based on the FY25 NutriOat learning.
"""
))

# Some intentionally near-duplicate / overlapping-theme docs from different
# departments, to exercise recency-based filtering and de-duplication logic.
DOCS.append((
    "finance_note_discount_depth_guidance.md",
    {"doc_type": "finance_note", "category": "Cross-category", "period": "2025-04",
     "tags": ["finance", "discount policy", "margin"], "published": "2025-04-02"},
    """# Finance Note — Discount Depth Guidance (Superseding May 2024 version)

Following margin erosion observed in the NutriOat Gold Festive Harvest 2024 campaign
(North, up to 12% discount depth), Finance is issuing updated guidance:

- Standard campaigns: cap discount depth at 8% (see Clean Home Drive, East, as the
  compliant reference case).
- Festive-quarter campaigns (Oct-Nov) may go up to 12% with category-head sign-off.
- All other campaigns require Finance approval to exceed 8%.

This note supersedes the discount guidance circulated in May 2024 and should be treated
as the current policy of record.
"""
))

DOCS.append((
    "finance_note_discount_depth_guidance_may2024_superseded.md",
    {"doc_type": "finance_note", "category": "Cross-category", "period": "2024-05",
     "tags": ["finance", "discount policy", "margin", "superseded"], "published": "2024-05-15"},
    """# Finance Note — Discount Depth Guidance (SUPERSEDED — see April 2025 version)

[Historical] Standard campaigns were capped at 6% discount depth, with festive-quarter
campaigns allowed up to 10%. This guidance was revised in April 2025 following the
NutriOat Gold Festive Harvest 2024 campaign read-out.
"""
))


def build_documents():
    for filename, meta, body in DOCS:
        path = UNSTRUCTURED_DIR / filename
        front_matter = "---\n" + "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n---\n\n"
        path.write_text(front_matter + body.strip() + "\n", encoding="utf-8")
    print(f"{len(DOCS)} unstructured documents written to {UNSTRUCTURED_DIR}")


if __name__ == "__main__":
    build_database()
    build_documents()
