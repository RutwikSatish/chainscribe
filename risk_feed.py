"""
ChainScribe — Live Risk Feed Module
====================================
Fetches real-time supplier news and uses local Ollama AI to assess
supply chain risk — geopolitical, financial, operational, and more.

How it works:
  1. User types a supplier name
  2. We fetch recent headlines from Google News RSS (free, no API key)
  3. Headlines are sent to your local Ollama model for risk analysis
  4. AI returns structured risk categories, score, and procurement actions
  5. User can generate a formal risk alert email in one click

Cost:   $0 — Google News RSS is free, Ollama runs locally
Privacy: 100% local — no supplier data leaves your machine
"""

import streamlit as st
import feedparser
import requests
import urllib.parse
from datetime import datetime, date

OLLAMA_URL = "http://localhost:11434/api/chat"


# ─── STEP 1: FETCH NEWS ──────────────────────────────────────────────────────

def fetch_news(supplier_name: str, max_results: int = 10) -> list:
    """
    Fetches recent news about a supplier using Google News RSS.
    No API key needed — Google News RSS is completely free.

    We search for the supplier name + supply chain keywords to filter
    out irrelevant results (e.g. stock price articles).
    """
    # Build a targeted search query
    query = urllib.parse.quote(
        f'"{supplier_name}" '
        f'supply chain OR supplier OR procurement OR logistics OR factory OR shortage'
    )
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_results]:
            # Parse the published date into a readable string
            try:
                pub_date = datetime(*entry.published_parsed[:6]).strftime("%b %d, %Y")
            except Exception:
                pub_date = "Recent"

            articles.append({
                "title":   entry.title,
                "link":    entry.link,
                "date":    pub_date,
                "summary": getattr(entry, "summary", "")[:300],
                "source":  getattr(entry, "source", {}).get("title", "News"),
            })
        return articles

    except Exception as e:
        return []


# ─── STEP 2: ANALYSE WITH OLLAMA ─────────────────────────────────────────────

def analyze_risk(supplier_name: str, articles: list, model: str) -> str:
    """
    Passes the news headlines to your local Ollama model.
    Returns a structured text response that we parse for display.

    The prompt uses a strict format so we can reliably extract
    risk levels, categories, and actions from the AI's response.
    """
    headlines = "\n".join([
        f"[{a['date']}] {a['title']} — {a['source']}"
        for a in articles
    ])

    system = (
        "You are a supply chain risk intelligence analyst. "
        "You assess news about suppliers and identify risks relevant to procurement teams. "
        "You are precise, data-focused, and always give actionable recommendations. "
        "You respond ONLY in the exact structured format requested — no extra commentary."
    )

    user = f"""Analyse these recent news headlines about supplier: {supplier_name}

NEWS HEADLINES:
{headlines}

Respond in EXACTLY this format (no deviations):

OVERALL_RISK: [HIGH / MEDIUM / LOW / MONITORING]
RISK_SCORE: [integer 0-100]
SUMMARY: [2-3 sentences summarising the supply chain risk situation]

TOP_RISKS:
- CATEGORY: [Geopolitical/Financial/Operational/Weather-Natural/Compliance-Legal/Supply-Logistics] | LEVEL: [HIGH/MEDIUM/LOW] | DETAIL: [one specific sentence citing a news item]
(repeat for 2-4 risks maximum — only include genuinely relevant risks)

PROCUREMENT_ACTIONS:
1. [Specific action with owner placeholder and timeline]
2. [Specific action]
3. [Specific action]
(maximum 3 actions — be specific, not generic)

WATCH_FOR:
[One sentence: what procurement should monitor over the next 30 days]

If no relevant supply chain risks are found, use:
OVERALL_RISK: MONITORING
RISK_SCORE: 5
SUMMARY: No significant supply chain risks identified in recent news for this supplier.
TOP_RISKS:
- CATEGORY: Supply-Logistics | LEVEL: LOW | DETAIL: No material risks found in current news cycle.
PROCUREMENT_ACTIONS:
1. Maintain standard supplier monitoring cadence.
WATCH_FOR: No immediate action required — continue routine monitoring."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":    model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        return "CONNECTION_ERROR"
    except Exception as e:
        return f"ERROR: {e}"


# ─── STEP 3: PARSE AI RESPONSE ────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    """
    Converts the AI's structured text response into a Python dict
    so we can display each part separately in the UI.
    """
    result = {
        "overall_risk": "UNKNOWN",
        "risk_score":   0,
        "summary":      "",
        "top_risks":    [],
        "actions":      [],
        "watch_for":    "",
    }

    if not text or text in ("CONNECTION_ERROR",):
        result["overall_risk"] = "ERROR"
        return result

    section = None
    for raw_line in text.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("OVERALL_RISK:"):
            result["overall_risk"] = line.split(":", 1)[1].strip()
        elif line.startswith("RISK_SCORE:"):
            try:
                result["risk_score"] = int("".join(filter(str.isdigit, line.split(":", 1)[1])))
            except Exception:
                result["risk_score"] = 0
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
            section = "summary"
        elif line.startswith("TOP_RISKS:"):
            section = "risks"
        elif line.startswith("PROCUREMENT_ACTIONS:"):
            section = "actions"
        elif line.startswith("WATCH_FOR:"):
            result["watch_for"] = line.split(":", 1)[1].strip()
            section = "watch"
        elif section == "summary":
            # Multi-line summary
            result["summary"] += " " + line
        elif section == "risks" and line.startswith("-"):
            # Parse "- CATEGORY: X | LEVEL: Y | DETAIL: Z"
            parts = {}
            for part in line[1:].split("|"):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    parts[k.strip().upper()] = v.strip()
            if parts:
                result["top_risks"].append(parts)
        elif section == "actions" and line and (line[0].isdigit() or line.startswith("-")):
            clean = line.lstrip("0123456789.-) ").strip()
            if clean:
                result["actions"].append(clean)
        elif section == "watch" and not result["watch_for"]:
            result["watch_for"] = line

    return result


# ─── STEP 4: UI ───────────────────────────────────────────────────────────────

def render(model: str):
    """
    Renders the full Live Risk Feed page.
    Called from app.py when the user selects "Live Risk Feed" in the sidebar.
    """

    st.markdown("## 📰 Live Supplier Risk Feed")
    st.caption(
        "Searches real-time news and uses AI to assess supply chain risk — "
        "no API key required, runs 100% locally"
    )

    # ── Search bar ──────────────────────────────────────────────────────────
    col_s, col_b = st.columns([4, 1])
    supplier_input = col_s.text_input(
        "supplier",
        placeholder="Type a supplier name — e.g. TSMC, Maersk, FedEx, Samsung, BASF...",
        label_visibility="collapsed",
    )
    search_clicked = col_b.button("🔍  Analyse", type="primary", use_container_width=True)

    # Quick-try examples
    st.caption("Quick examples:")
    ex_cols = st.columns(7)
    for i, ex in enumerate(["TSMC", "Maersk", "FedEx", "BASF", "Foxconn", "Vale", "Nippon Steel"]):
        if ex_cols[i].button(ex, key=f"ex_{ex}", use_container_width=True):
            supplier_input = ex
            search_clicked = True

    # ── Landing state (no search yet) ──────────────────────────────────────
    if not search_clicked or not supplier_input.strip():
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown("**How it works**")
        c1.markdown(
            "Searches Google News RSS for the supplier name + supply chain keywords. "
            "Completely free — no API key, no signup, no cost."
        )
        c2.markdown("**What the AI does**")
        c2.markdown(
            "Your local Ollama model reads the headlines and identifies risks by category: "
            "Geopolitical, Financial, Operational, Weather, Compliance, and Logistics."
        )
        c3.markdown("**What you get**")
        c3.markdown(
            "A risk score (0–100), categorised risk breakdown, 3 specific procurement actions, "
            "and a 30-day watchlist. One click to generate a formal risk alert email."
        )
        st.divider()
        st.markdown(
            "> *\"77% of supply chain executives say gen AI models successfully identify "
            "geopolitical and climate risks, enabling proactive mitigation.\"*  \n"
            "> — IBM Institute for Business Value, 2024"
        )
        return

    # ── Run the analysis ────────────────────────────────────────────────────
    supplier = supplier_input.strip()

    with st.spinner(f"Fetching live news for **{supplier}**..."):
        articles = fetch_news(supplier)

    if not articles:
        st.warning(
            f"No recent news found for **{supplier}**. "
            "Try a more widely-covered supplier, or check your internet connection."
        )
        return

    with st.spinner(f"AI analysing {len(articles)} headlines..."):
        raw = analyze_risk(supplier, articles, model)

    if raw == "CONNECTION_ERROR":
        st.error(
            "Cannot connect to Ollama. "
            "Make sure `ollama serve` is running in a separate terminal tab."
        )
        return

    data = parse_response(raw)

    # ── Risk header card ────────────────────────────────────────────────────
    RISK_STYLE = {
        "HIGH":       ("#fef2f2", "#dc2626", "🔴"),
        "MEDIUM":     ("#fffbeb", "#d97706", "🟡"),
        "LOW":        ("#f0fdf4", "#16a34a", "🟢"),
        "MONITORING": ("#eff6ff", "#2563eb", "🔵"),
        "UNKNOWN":    ("#f9fafb", "#6b7280", "⚪"),
        "ERROR":      ("#fef2f2", "#dc2626", "⚠️"),
    }
    level = data["overall_risk"]
    bg, fg, icon = RISK_STYLE.get(level, RISK_STYLE["UNKNOWN"])
    score = min(max(data["risk_score"], 0), 100)

    st.markdown(f"""
    <div style="background:{bg};border:1px solid {fg}44;border-radius:10px;
                padding:18px 22px;margin:16px 0;
                display:flex;align-items:center;justify-content:space-between;gap:16px">
        <div style="flex:1">
            <div style="font-size:11px;color:{fg};font-weight:600;
                        text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">
                Supply Chain Risk Assessment
            </div>
            <div style="font-size:20px;font-weight:600;color:{fg};margin-bottom:6px">
                {icon}&nbsp; {level} — {supplier}
            </div>
            <div style="font-size:13px;color:#444;line-height:1.6;max-width:580px">
                {data['summary']}
            </div>
        </div>
        <div style="text-align:center;min-width:72px;flex-shrink:0">
            <div style="font-size:38px;font-weight:700;color:{fg};line-height:1">{score}</div>
            <div style="font-size:11px;color:#888;margin-top:2px">/ 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risks + Actions (side by side) ─────────────────────────────────────
    col_l, col_r = st.columns(2)

    # Left: identified risks
    with col_l:
        st.markdown("#### ⚠️ Identified Risks")

        CAT_COLOR = {
            "GEOPOLITICAL":     "#ef4444",
            "FINANCIAL":        "#f59e0b",
            "OPERATIONAL":      "#f97316",
            "WEATHER-NATURAL":  "#3b82f6",
            "COMPLIANCE-LEGAL": "#8b5cf6",
            "SUPPLY-LOGISTICS": "#06b6d4",
        }
        CAT_ICON = {
            "GEOPOLITICAL":     "🌍",
            "FINANCIAL":        "💸",
            "OPERATIONAL":      "⚙️",
            "WEATHER-NATURAL":  "🌪️",
            "COMPLIANCE-LEGAL": "⚖️",
            "SUPPLY-LOGISTICS": "📦",
        }
        LVL_STYLE = {
            "HIGH":   ("#fef2f2", "#dc2626"),
            "MEDIUM": ("#fffbeb", "#d97706"),
            "LOW":    ("#f0fdf4", "#16a34a"),
        }

        if data["top_risks"]:
            for risk in data["top_risks"]:
                cat    = risk.get("CATEGORY", "OPERATIONAL").upper().replace("/","-").replace(" ","-")
                lvl    = risk.get("LEVEL",    "MEDIUM").upper()
                detail = risk.get("DETAIL",   "")
                c      = CAT_COLOR.get(cat, "#888")
                ico    = CAT_ICON.get(cat, "⚡")
                lb_bg, lb_fg = LVL_STYLE.get(lvl, ("#f3f4f6", "#666"))
                label  = cat.replace("-", " ").title()

                st.markdown(f"""
                <div style="border:0.5px solid {c}44;border-left:3px solid {c};
                            border-radius:8px;padding:10px 14px;margin-bottom:8px">
                    <div style="display:flex;align-items:center;
                                justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:13px;font-weight:500;
                                     color:var(--color-text-primary,#111)">
                            {ico} {label}
                        </span>
                        <span style="font-size:11px;font-weight:600;padding:2px 8px;
                                     border-radius:10px;
                                     background:{lb_bg};color:{lb_fg}">
                            {lvl}
                        </span>
                    </div>
                    <div style="font-size:12px;color:#555;line-height:1.55">{detail}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant risks identified in current news.")

    # Right: actions + watchlist
    with col_r:
        st.markdown("#### ✅ Procurement Actions")

        for i, action in enumerate(data["actions"], 1):
            st.markdown(f"""
            <div style="border:0.5px solid var(--color-border-tertiary,#e5e7eb);
                        border-radius:8px;padding:10px 14px;margin-bottom:8px;
                        display:flex;gap:10px;align-items:flex-start">
                <span style="background:#f0fdf4;color:#16a34a;border-radius:50%;
                             width:22px;height:22px;min-width:22px;
                             display:flex;align-items:center;justify-content:center;
                             font-size:11px;font-weight:700">{i}</span>
                <span style="font-size:13px;color:#333;line-height:1.55">{action}</span>
            </div>
            """, unsafe_allow_html=True)

        if data["watch_for"]:
            st.markdown(f"""
            <div style="background:#eff6ff;border:0.5px solid #bfdbfe;
                        border-radius:8px;padding:12px 14px;margin-top:6px">
                <div style="font-size:11px;font-weight:600;color:#1d4ed8;
                            text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">
                    📡 30-Day Watchlist
                </div>
                <div style="font-size:12px;color:#1e40af;line-height:1.55">
                    {data['watch_for']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Source articles ─────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"#### 📰 Source Articles — {len(articles)} headlines analysed")
    st.caption("These are the exact headlines the AI read. Click any to open the full article.")

    for a in articles:
        label = f"**{a['date']}** — {a['title'][:88]}{'...' if len(a['title'])>88 else ''}"
        with st.expander(label):
            st.markdown(f"**Source:** {a['source']}")
            if a["summary"]:
                st.caption(a["summary"])
            st.markdown(f"[Read full article ↗]({a['link']})")

    # ── Generate risk alert email ───────────────────────────────────────────
    st.divider()
    st.markdown("#### 📬 Generate Risk Alert Email")
    st.caption("Turn this analysis into a formal email ready to send to your team or management.")

    if st.button("✍️  Write Risk Alert Email", type="primary", use_container_width=True):
        risks_text   = "\n".join([
            f"- {r.get('CATEGORY','')} [{r.get('LEVEL','')}]: {r.get('DETAIL','')}"
            for r in data["top_risks"]
        ]) or "No critical risks identified."
        actions_text = "\n".join(
            [f"{i+1}. {a}" for i, a in enumerate(data["actions"])]
        )

        email_prompt = f"""Write a formal supplier risk alert email for internal distribution.

SUPPLIER:    {supplier}
RISK LEVEL:  {data['overall_risk']} (Score: {score}/100)
DATE:        {date.today().strftime('%B %d, %Y')}

RISK SUMMARY:
{data['summary']}

IDENTIFIED RISKS:
{risks_text}

RECOMMENDED ACTIONS:
{actions_text}

30-DAY WATCH: {data['watch_for']}

FORMAT REQUIREMENTS:
Subject line: [RISK ALERT] {supplier} — {data['overall_risk']} Risk Level — Action Required

Sections:
1. Opening (1 sentence: purpose and risk level)
2. Risk Overview paragraph (cite the score and summary)
3. Risk Details (bullet points with severity badges in brackets)
4. Required Actions (numbered, with [Owner] placeholders and specific deadlines)
5. Monitoring Note (30-day watchlist item)
6. Professional close

Style: Formal procurement language. Under 280 words. Every action has an owner and deadline."""

        with st.spinner("Writing risk alert email..."):
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model":    model,
                        "messages": [
                            {
                                "role":    "system",
                                "content": (
                                    "You are a senior procurement manager writing a formal "
                                    "internal supplier risk alert. You are precise, factual, "
                                    "and action-oriented."
                                ),
                            },
                            {"role": "user", "content": email_prompt},
                        ],
                        "stream": False,
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                email_text = resp.json()["message"]["content"]

                st.markdown(
                    f'<div style="background:#fefefe;border:0.5px solid #d4d4d0;'
                    f'border-radius:8px;padding:22px 26px;font-family:Georgia,serif;'
                    f'font-size:14px;line-height:1.9;white-space:pre-wrap;margin-top:12px">'
                    f'{email_text}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "📥 Download Risk Alert (.txt)",
                    email_text,
                    file_name=f"risk_alert_{supplier.replace(' ','_')}_{date.today()}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error generating email: {e}")

