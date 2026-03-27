"""
ChainScribe: AI Supply Chain Document Writer
============================================
Turns raw supply chain data into professional, ready-to-send documents.

Why this exists:
  Research of 100+ supply chain analyst job descriptions shows that
  88% of roles require "translating complex data into stakeholder communications."
  Analysts spend ~110 minutes/day writing the same types of documents.
  ChainScribe automates exactly that — using a free, local AI model.

6 document types:
  1. Supplier Performance Review Letter
  2. Executive KPI Summary
  3. Supplier Escalation Email
  4. Weekly Operations Briefing
  5. Request for Quote (RFQ)
  6. Cost Savings Report

Stack:  Python · Streamlit · Ollama (llama3.2:3b) · Requests
Author: Rutwik Satish
"""

import streamlit as st
import requests
from datetime import date

# ─── SETTINGS ───────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:3b"
TODAY         = date.today().strftime("%B %d, %Y")

# Estimated manual writing time per document type (minutes).
# Source: industry surveys of supply chain professionals.
TIME_ESTIMATES = {
    "supplier_letter":  45,
    "exec_kpi":         60,
    "escalation_email": 25,
    "weekly_brief":     40,
    "rfq":              90,
    "savings_report":   60,
}

# ─── PAGE CONFIGURATION ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChainScribe AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS: styled output box + time-saved badge
st.markdown("""
<style>
.doc-output {
    background: var(--background-color, #fefefe);
    border: 0.5px solid #d4d4d0;
    border-radius: 8px;
    padding: 24px 28px;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 14.5px;
    line-height: 1.88;
    color: #1a1a1a;
    white-space: pre-wrap;
    word-break: break-word;
}
.saved-badge {
    background: #f0fdf4;
    color: #166534;
    border: 0.5px solid #bbf7d0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction.
# Session state keeps values across those re-runs.
if "document"   not in st.session_state: st.session_state.document   = ""
if "doc_id"     not in st.session_state: st.session_state.doc_id     = ""

# ─── OLLAMA HELPER ────────────────────────────────────────────────────────────

def ask_ollama(system: str, user: str, model: str = DEFAULT_MODEL) -> str:
    """
    Sends a two-message conversation to your local Ollama model.
    - system: gives the AI its persona and rules
    - user:   the actual document request with all the data
    
    Runs 100% on your Mac — no internet required, no API keys, no cost.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
            },
            timeout=180,  # writing a full RFQ can take a minute
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "⚠️  CANNOT CONNECT TO OLLAMA\n\n"
            "To fix this:\n"
            "  1. Open a new Terminal tab\n"
            "  2. Run the command:  ollama serve\n"
            "  3. Come back here and click Generate again\n\n"
            "Keep that terminal tab running while you use ChainScribe."
        )
    except Exception as e:
        return f"⚠️  Error: {e}"


def refine(original: str, instruction: str, model: str) -> str:
    """Applies a user's refinement instruction to an already-generated document."""
    return ask_ollama(
        system=(
            "You are a professional supply chain document editor. "
            "Apply the requested changes precisely while keeping the "
            "professional tone, structure, and all specific numbers intact. "
            "Return only the complete revised document — no commentary."
        ),
        user=(
            f"ORIGINAL DOCUMENT:\n{original}\n\n"
            f"CHANGE REQUESTED:\n{instruction}\n\n"
            "Return the full revised document:"
        ),
        model=model,
    )


# ─── PROMPT LIBRARY ───────────────────────────────────────────────────────────
# Each function receives a dict of form values and returns (system, user) prompts.
# The quality of these prompts is what makes ChainScribe's output genuinely usable.

def prompt_supplier_letter(v: dict) -> tuple:
    # Select tone guidance based on the supplier relationship status
    tone_guide = {
        "Strategic Partner":  "warm but accountable — this is a valued long-term partner",
        "Preferred Supplier": "professional and direct — maintain relationship, drive improvement",
        "Standard Supplier":  "formal and business-like — clear expectations and consequences",
        "Under Review":       "firm and factual — this letter may be used in contract proceedings",
    }.get(v.get("relationship", "Standard Supplier"), "professional and direct")

    # Determine pass/fail for each metric
    try:
        otd_ok   = float(v.get("otd","0"))    >= float(v.get("otd_target","95"))
        defect_ok= float(v.get("defect","0")) <= float(v.get("defect_target","2"))
    except ValueError:
        otd_ok = defect_ok = False

    sys = (
        "You are a senior supply chain manager writing an official supplier performance review letter. "
        "You always cite specific numbers, set measurable improvement targets with deadlines, and are "
        "professionally firm without being hostile. Use proper business letter format."
    )
    usr = f"""Write a formal supplier performance review letter.

DATE: {TODAY}
SUPPLIER: {v['supplier_name']}
REVIEW PERIOD: {v['period']}
RELATIONSHIP: {v['relationship']} → tone: {tone_guide}

PERFORMANCE vs TARGETS:
  On-Time Delivery:  {v['otd']}%   (Target: {v['otd_target']}%)  → {'MEETS TARGET ✓' if otd_ok else 'BELOW TARGET ✗ — IMPROVEMENT REQUIRED'}
  Defect Rate:       {v['defect']}% (Target: <{v['defect_target']}%) → {'WITHIN LIMIT ✓' if defect_ok else 'EXCEEDS LIMIT ✗ — IMMEDIATE ACTION REQUIRED'}
  Contract Value:    ${v['contract_value']}
  Open POs:          {v.get('open_pos', 'N/A')}

KEY ISSUES THIS PERIOD:
{v['issues']}

REQUIRED DOCUMENT STRUCTURE:
1. Date and address block
2. Subject line: Performance Review — [Period]
3. Opening: state purpose clearly in one paragraph
4. Performance summary: cite every metric above with its actual vs target
5. Areas of concern: specific issues with dates/PO numbers if given
6. Required actions: numbered list — each action must have a deadline
7. Consequences of non-improvement (proportionate to relationship status)
8. Request for written response within 5 business days
9. Professional close + signature line: "[Your Name] | Supply Chain Management"

Rules: No vague language. Every concern must have a specific required action."""
    return sys, usr


def prompt_exec_kpi(v: dict) -> tuple:
    sys = (
        "You are a VP of Supply Chain writing a monthly performance summary for the CEO and CFO. "
        "You write with executive precision: every sentence has a data point or a decision. "
        "You never use filler phrases like 'it should be noted.' "
        "You bold critical numbers. You use → for trends. Under 400 words — executives read fast."
    )
    usr = f"""Write an executive supply chain KPI summary.

COMPANY / DIVISION: {v['company']}
REPORTING PERIOD:   {v['period']}
REPORT DATE:        {TODAY}

KPI DATA:
{v['kpi_data']}

CONTEXT / NOTABLE EVENTS:
{v['context']}

REQUIRED STRUCTURE (use these exact bold headers):
**OVERALL HEALTH:** [🟢 / 🟡 / 🔴] — one sentence explanation

**KEY WINS THIS PERIOD:** (max 3 bullets — each must have a number)

**AREAS OF CONCERN:** (max 3 bullets — each: Issue → Root Cause → Owner)

**TOP 2 RISKS:**
  Risk 1: [description] | Likelihood: [H/M/L] | Mitigation: [action + owner]
  Risk 2: [description] | Likelihood: [H/M/L] | Mitigation: [action + owner]

**LEADERSHIP ACTIONS REQUIRED:** (numbered, max 3)

Style: Bold all KPI numbers. Use → for trend direction. No passive voice.
Target: 280–380 words total."""
    return sys, usr


def prompt_escalation(v: dict) -> tuple:
    ref = f"ESC-{date.today().strftime('%Y%m%d')}-001"
    sys = (
        "You are a procurement director writing a formal escalation email. "
        "Your emails are documented business records — firm, factual, legally precise. "
        "You quantify everything: days late, dollars at risk, downstream impact. "
        "You never threaten, but you always make consequences clear."
    )
    usr = f"""Write a formal supplier escalation email.

REFERENCE: {ref}
DATE:       {TODAY}
TO:         {v['contact']} — {v['supplier_name']}
FROM:       Supply Chain / Procurement Team

ISSUE:           {v['issue_type']}
DURATION:        {v['duration']}
BUSINESS IMPACT: {v['impact']}
PRIOR ACTIONS:   {v['prior_actions']}
REQUIRED BY:     {v['deadline']}
RESOLUTION NEEDED: {v['resolution']}

REQUIRED DOCUMENT STRUCTURE:
Subject line: [ESCALATION {ref}] {v['issue_type']} — {v['supplier_name']} — Action Required by {v['deadline']}

1. Opening (2 sentences): state the issue and reference number factually
2. Business impact paragraph: quantify time, money, downstream effects
3. Timeline of events: when first raised, what commitments were made, what was missed
4. Required actions (numbered):
   - Action 1 | Owner: [supplier name] | Deadline: [specific date]
   - Action 2 | Owner: [supplier name] | Deadline: [specific date]
5. Consequences paragraph: factual, not threatening (e.g. contract review, sourcing alternatives)
6. Close: offer a call to resolve + placeholder for contact info

Tone: Firm, factual, professional. Every statement is traceable."""
    return sys, usr


def prompt_weekly(v: dict) -> tuple:
    sys = (
        "You are a supply chain operations manager writing the Monday team briefing. "
        "Your briefings are structured, scannable, and action-oriented. "
        "You use RAG status. Every issue has an owner. Every action has a who and when."
    )
    usr = f"""Write a weekly supply chain operations briefing.

WEEK:  {v['week']}
DATE:  {TODAY}
FOR:   Operations leadership and cross-functional stakeholders

INPUT DATA:
WINS:        {v['wins']}
ISSUES:      {v['issues']}
METRICS:     {v['metrics']}
RISKS:       {v['risks']}
ACTIONS:     {v['actions']}
NEXT WEEK:   {v['upcoming']}

REQUIRED FORMAT:
OVERALL STATUS: 🟢/🟡/🔴 — [one sentence reason]

✓ WINS THIS WEEK:
• [bullet per win with number]

✗ ISSUES:
Issue | Root Cause | Owner | ETA
[one row per issue]

📊 METRICS:
Metric | Actual | Target | Trend (↑ ↓ →)
[one row per metric]

⚠️ RISK REGISTER:
Risk | Likelihood (H/M/L) | Impact (H/M/L) | Owner
[top 3 risks]

✅ ACTIONS:
• WHO → WHAT → BY WHEN
[one per action]

🎯 NEXT WEEK FOCUS:
[2-3 priorities]

Keep it scannable. Bold important numbers. ~280-350 words."""
    return sys, usr


def prompt_rfq(v: dict) -> tuple:
    ref = f"RFQ-{date.today().strftime('%Y-%m')}-001"
    sys = (
        "You are a senior procurement manager writing a formal Request for Quote. "
        "Your RFQs leave zero ambiguity — suppliers know exactly what is required. "
        "You use numbered sections and professional procurement document formatting."
    )
    usr = f"""Write a complete, formal Request for Quote (RFQ) document.

REFERENCE:     {ref}
ISSUE DATE:    {TODAY}
ISSUED BY:     {v['company']}
CATEGORY:      {v['category']}
QUANTITY:      {v['quantity']}
DELIVERY TO:   {v['location']}
DELIVERY BY:   {v['delivery_date']}
QUOTE DEADLINE:{v['quote_deadline']}

TECHNICAL SPECIFICATIONS:
{v['specs']}

EVALUATION CRITERIA:
{v['criteria']}

SPECIAL REQUIREMENTS:
{v['special_reqs']}

REQUIRED SECTIONS (number each one):
1. Document Header (reference, dates, issuing company, contact placeholders)
2. Purpose & Scope
3. Technical Specifications (formatted as a table: Requirement | Specification | Mandatory Y/N)
4. Quantity & Delivery Schedule
5. Commercial Requirements (payment terms, warranty, required certifications, insurance)
6. Evaluation Criteria (list with weightings that total 100%)
7. Submission Instructions (format, deadline, where to send)
8. Terms & Conditions (brief — 4-5 bullet points)
9. Supplier Acknowledgement (signature/acceptance section)

Professional procurement document. Number all sections. Use tables where noted."""
    return sys, usr


def prompt_savings(v: dict) -> tuple:
    # Safe number parsing
    try:
        base  = float(str(v.get("baseline","0")).replace(",","").replace("$",""))
        new   = float(str(v.get("new_spend","0")).replace(",","").replace("$",""))
        saved = base - new
        pct   = (saved / base * 100) if base > 0 else 0
        months = float(v.get("months","12") or "12")
        annual = (saved / months * 12) if months > 0 else saved
    except ValueError:
        base = new = saved = pct = annual = 0

    sys = (
        "You are a procurement manager presenting cost savings to the CFO and board. "
        "Your reports are auditable, conservative, and transparent about methodology. "
        "You clearly label saving type (hard/soft/avoidance). Every number is traceable."
    )
    usr = f"""Write a formal procurement cost savings report.

DATE:        {TODAY}
PERIOD:      {v['period']}
CATEGORY:    {v['category']}
SAVING TYPE: {v['saving_type']}

FINANCIALS:
  Baseline Spend:       ${base:,.0f}
  New / Negotiated:     ${new:,.0f}
  Gross Saving:         ${saved:,.0f}  ({pct:.1f}%)
  Annualized Projection:${annual:,.0f}
  Period Covered:       {v['months']} months

STRATEGY USED:
{v['strategy']}

SUPPLIERS:
{v['suppliers']}

RISKS TO REALIZATION:
{v['risks']}

REQUIRED SECTIONS:
1. Headline (${saved:,.0f} / {pct:.1f}% — type: {v['saving_type']})
2. Executive Summary (4-5 sentences: what, how, validated how)
3. Baseline Methodology (how the pre-saving number was calculated)
4. Strategy & Approach (what negotiation/sourcing approach was used)
5. Saving Validation (how the saving was confirmed — benchmarks, contracts, invoices)
6. Risks to Realization (what could prevent the saving being achieved)
7. Annualized Projection (${annual:,.0f} based on {v['months']}-month run-rate)
8. Recommended Next Steps (2-3 actions for leadership)

Format: Professional finance document. All $ with commas. Label saving type clearly."""
    return sys, usr


# ─── MAP DOC IDs TO PROMPT FUNCTIONS ─────────────────────────────────────────
PROMPTS = {
    "supplier_letter":  prompt_supplier_letter,
    "exec_kpi":         prompt_exec_kpi,
    "escalation_email": prompt_escalation,
    "weekly_brief":     prompt_weekly,
    "rfq":              prompt_rfq,
    "savings_report":   prompt_savings,
}

DISPLAY_TO_ID = {
    "📬 Supplier Performance Review Letter": "supplier_letter",
    "📊 Executive KPI Summary":              "exec_kpi",
    "🚨 Supplier Escalation Email":          "escalation_email",
    "📋 Weekly Operations Briefing":         "weekly_brief",
    "📝 Request for Quote (RFQ)":            "rfq",
    "💰 Cost Savings Report":                "savings_report",
}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📝 ChainScribe")
    st.caption("AI Document Writer for Supply Chain")
    st.divider()

    st.markdown("### What do you need to write?")
    choice = st.selectbox("", list(DISPLAY_TO_ID.keys()), label_visibility="collapsed")
    doc_id = DISPLAY_TO_ID[choice]

    st.divider()
    st.markdown("### AI Model")
    model = st.selectbox(
        "",
        ["llama3.2:3b", "llama3.2:1b", "mistral:7b", "llama3:8b"],
        label_visibility="collapsed",
        help="3b = best balance | 1b = fastest | 7b/8b = highest quality"
    )

    st.divider()
    est = TIME_ESTIMATES.get(doc_id, 45)
    st.markdown("### ⏱ Time Saved")
    st.markdown(f"Manual writing: **~{est} min**")
    st.markdown(f"With ChainScribe: **~2 min**")
    st.markdown(
        f"<span class='saved-badge'>Saves ~{est-2} minutes</span>",
        unsafe_allow_html=True
    )

# ─── MAIN LAYOUT: FORM (left) | OUTPUT (right) ───────────────────────────────
st.markdown(f"## {choice}")
st.caption("Fill in the fields below, then click Generate.")
st.divider()

col_form, col_out = st.columns([1, 1], gap="large")

# We initialize submitted here so it's always defined
submitted = False
inputs    = {}

# ─── FORM: SUPPLIER PERFORMANCE REVIEW LETTER ────────────────────────────────
with col_form:
    st.markdown("### 📋 Document Details")

    if doc_id == "supplier_letter":
        with st.form("f_supplier"):
            r1c1, r1c2 = st.columns(2)
            inputs["supplier_name"] = r1c1.text_input("Supplier Name *", placeholder="GlobalTech Parts Inc.")
            inputs["period"]        = r1c2.text_input("Review Period *",  placeholder="Q2 2025")

            r2c1, r2c2 = st.columns(2)
            inputs["otd"]        = r2c1.text_input("Actual On-Time Delivery %", value="87")
            inputs["otd_target"] = r2c2.text_input("OTD Target %",             value="95")

            r3c1, r3c2 = st.columns(2)
            inputs["defect"]        = r3c1.text_input("Actual Defect Rate %",       value="3.2")
            inputs["defect_target"] = r3c2.text_input("Max Acceptable Defect Rate", value="2.0")

            r4c1, r4c2 = st.columns(2)
            inputs["contract_value"] = r4c1.text_input("Annual Contract Value ($)", value="250,000")
            inputs["open_pos"]       = r4c2.text_input("Open Purchase Orders",      value="12")

            inputs["relationship"] = st.selectbox(
                "Supplier Relationship Status",
                ["Strategic Partner", "Preferred Supplier", "Standard Supplier", "Under Review"]
            )
            inputs["issues"] = st.text_area(
                "Key Issues During Period *",
                placeholder=(
                    "- 3 shipments delayed 5+ days in January\n"
                    "- Incorrect spec on PO #4521\n"
                    "- Non-compliance with packaging standards on 2 orders"
                ),
                height=120
            )
            submitted = st.form_submit_button("✍️  Generate Letter", type="primary", use_container_width=True)

    # ─── FORM: EXECUTIVE KPI SUMMARY ─────────────────────────────────────────
    elif doc_id == "exec_kpi":
        with st.form("f_kpi"):
            c1, c2 = st.columns(2)
            inputs["company"] = c1.text_input("Company / Division", placeholder="ACME Corp — North America Ops")
            inputs["period"]  = c2.text_input("Reporting Period",   placeholder="April 2025")

            inputs["kpi_data"] = st.text_area(
                "KPI Data — paste your metrics here *",
                placeholder=(
                    "On-Time Delivery:       88.4%   (Target: 95%)   ↓ from 91.2%\n"
                    "Inventory Turns:         6.2    (Target: 7.0)   → flat\n"
                    "Supplier Lead Time:     18 days (Target: 14)    ↑ improved\n"
                    "PO Fill Rate:           94.1%   (Target: 96%)   ↓ slight decline\n"
                    "Supply Chain Cost %:    11.3%   (Target: 10%)   ↓ over budget"
                ),
                height=160
            )
            inputs["context"] = st.text_area(
                "Context / Notable Events",
                placeholder="e.g. Supplier X factory shutdown 5 days. New 3PL launched in Texas.",
                height=80
            )
            submitted = st.form_submit_button("✍️  Generate KPI Summary", type="primary", use_container_width=True)

    # ─── FORM: ESCALATION EMAIL ───────────────────────────────────────────────
    elif doc_id == "escalation_email":
        with st.form("f_escalation"):
            c1, c2 = st.columns(2)
            inputs["supplier_name"] = c1.text_input("Supplier Name *",     placeholder="FastFreight Co.")
            inputs["contact"]       = c2.text_input("Contact Name/Title",  placeholder="John Smith, Account Manager")

            inputs["issue_type"] = st.selectbox(
                "Issue Type",
                ["Late Delivery", "Quality / Defects", "Pricing Dispute",
                 "Documentation Error", "Capacity / Shortage", "Compliance Breach", "Other"]
            )
            c3, c4 = st.columns(2)
            inputs["duration"] = c3.text_input("Duration of Issue", placeholder="3 weeks / since Jan 15")
            inputs["deadline"] = c4.text_input("Resolution Deadline", placeholder="March 31, 2025")

            inputs["impact"] = st.text_area(
                "Business Impact *",
                placeholder="3 customer orders delayed, ~$45,000 in pending shipments, production halted 2 days",
                height=80
            )
            inputs["prior_actions"] = st.text_area(
                "Prior Actions / Communications",
                placeholder="Emailed Feb 10 — no response. Called Feb 14 — promised resolution Feb 17 (missed).",
                height=70
            )
            inputs["resolution"] = st.text_area(
                "Required Resolution *",
                placeholder="Ship all 3 delayed orders by March 31. Submit root cause analysis. Corrective action plan.",
                height=70
            )
            submitted = st.form_submit_button("✍️  Generate Escalation Email", type="primary", use_container_width=True)

    # ─── FORM: WEEKLY BRIEFING ────────────────────────────────────────────────
    elif doc_id == "weekly_brief":
        with st.form("f_weekly"):
            inputs["week"] = st.text_input("Week Of", placeholder="Week of March 24, 2025")
            inputs["wins"] = st.text_area(
                "Wins This Week ✓",
                placeholder=(
                    "- Completed Q1 supplier audits for all Tier-1 suppliers\n"
                    "- New 3PL contract signed — saves $180K annually\n"
                    "- Achieved 97% OTIF this week"
                ),
                height=90
            )
            inputs["issues"] = st.text_area(
                "Issues / Lowlights ✗",
                placeholder=(
                    "- Apex Steel delayed — 2 POs at risk\n"
                    "- Inventory variance flagged in Atlanta DC\n"
                    "- ERP causing PO duplication errors since Monday"
                ),
                height=90
            )
            inputs["metrics"] = st.text_area(
                "Key Metrics",
                placeholder="OTIF: 91% (target 95%) ↓\nInventory Turns: 6.1 →\nDefect Rate: 1.8% ✓",
                height=80
            )
            inputs["risks"] = st.text_area(
                "Risks & Watchlist",
                placeholder="- Port congestion LA: 4 inbound shipments at risk\n- Apex Steel capacity issue ongoing",
                height=80
            )
            inputs["actions"] = st.text_area(
                "Actions / In Progress",
                placeholder="[Done] Sarah: Emergency PO to backup supplier\n[WIP] James: Negotiating expedite fee",
                height=70
            )
            inputs["upcoming"] = st.text_area(
                "Next Week Focus",
                placeholder="Q2 demand planning session Thu. Annual supplier review kickoff.",
                height=60
            )
            submitted = st.form_submit_button("✍️  Generate Weekly Briefing", type="primary", use_container_width=True)

    # ─── FORM: RFQ ────────────────────────────────────────────────────────────
    elif doc_id == "rfq":
        with st.form("f_rfq"):
            c1, c2 = st.columns(2)
            inputs["company"]  = c1.text_input("Your Company Name",     placeholder="ACME Manufacturing Inc.")
            inputs["category"] = c2.text_input("Category / Item *",     placeholder="Industrial Steel Tubing Grade 316")

            c3, c4 = st.columns(2)
            inputs["quantity"] = c3.text_input("Quantity",               placeholder="50,000 units / 200 metric tons")
            inputs["location"] = c4.text_input("Delivery Location",      placeholder="Atlanta, GA 30301")

            c5, c6 = st.columns(2)
            inputs["delivery_date"]  = c5.text_input("Required Delivery Date",    placeholder="June 30, 2025")
            inputs["quote_deadline"] = c6.text_input("Quote Submission Deadline", placeholder="April 15, 2025")

            inputs["specs"] = st.text_area(
                "Technical Specifications *",
                placeholder=(
                    "Material: 316L stainless steel\n"
                    "Wall thickness: 2mm ± 0.1mm\n"
                    "Length: 6m standard\n"
                    "Required certifications: ISO 9001, ASTM A269"
                ),
                height=120
            )
            inputs["criteria"] = st.text_area(
                "Evaluation Criteria",
                placeholder="Price 50%, Quality certs 25%, Lead time 15%, Sustainability 10%",
                height=60
            )
            inputs["special_reqs"] = st.text_area(
                "Special Requirements",
                placeholder="Supplier must hold ISO 9001. Min 3 years in category. QBR required.",
                height=60
            )
            submitted = st.form_submit_button("✍️  Generate RFQ Document", type="primary", use_container_width=True)

    # ─── FORM: COST SAVINGS REPORT ────────────────────────────────────────────
    elif doc_id == "savings_report":
        with st.form("f_savings"):
            c1, c2 = st.columns(2)
            inputs["period"]   = c1.text_input("Reporting Period",   placeholder="Q1 2025")
            inputs["category"] = c2.text_input("Spend Category",     placeholder="Indirect Procurement — MRO")

            c3, c4 = st.columns(2)
            inputs["baseline"]  = c3.text_input("Baseline Spend ($)", placeholder="1200000")
            inputs["new_spend"] = c4.text_input("Negotiated Spend ($)", placeholder="950000")

            c5, c6 = st.columns(2)
            inputs["saving_type"] = c5.selectbox(
                "Saving Type",
                ["Hard Saving (Cash Out)", "Cost Avoidance", "Soft Saving / Efficiency", "Rebate / Volume Discount"]
            )
            inputs["months"] = c6.text_input("Months Covered", value="12")

            inputs["strategy"] = st.text_area(
                "Strategy Used *",
                placeholder="Competitive RFQ to 6 suppliers. Consolidated from 3 to 1. 2-year contract with volume commitment.",
                height=80
            )
            inputs["suppliers"] = st.text_input(
                "Suppliers Involved",
                placeholder="OfficeSupply Co. (incumbent), SupplyMax Inc. (new), ProSource Ltd."
            )
            inputs["risks"] = st.text_area(
                "Risks to Realization",
                placeholder="Volume commitment — if usage drops >20%, unit price increases. New supplier performance unproven.",
                height=70
            )
            submitted = st.form_submit_button("✍️  Generate Savings Report", type="primary", use_container_width=True)

    # ─── TRIGGER GENERATION ───────────────────────────────────────────────────
    if submitted:
        prompt_fn = PROMPTS.get(doc_id)
        if prompt_fn:
            with st.spinner("ChainScribe is writing your document…"):
                sys_p, usr_p = prompt_fn(inputs)
                result = ask_ollama(sys_p, usr_p, model)
            st.session_state.document = result
            st.session_state.doc_id   = doc_id
            st.rerun()  # refresh to show the output column immediately

# ─── OUTPUT COLUMN ────────────────────────────────────────────────────────────
with col_out:
    st.markdown("### 📄 Generated Document")

    if not st.session_state.document:
        st.info("👈  Fill in the form and click Generate to create your document.")
        st.markdown("")
        st.markdown("**What ChainScribe does:**")
        st.markdown("- Takes your raw data and context")
        st.markdown("- Generates a complete, professional document")
        st.markdown("- Runs entirely on your Mac via Ollama — no internet, no cost")
        st.markdown("- Output is ready to copy into email or download")
    else:
        doc = st.session_state.document

        # Render the document in a professional-looking styled container
        st.markdown(
            f'<div class="doc-output">{doc}</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Download button
        filename = f"ChainScribe_{doc_id}_{date.today().strftime('%Y%m%d')}.txt"
        st.download_button(
            label="📥 Download Document (.txt)",
            data=doc,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

        # ── REFINEMENT SECTION ────────────────────────────────────────────────
        st.markdown("### ✏️ Refine This Document")
        st.caption("Tell ChainScribe what to change and it will rewrite the document.")

        # Contextual placeholder for each doc type
        placeholders = {
            "supplier_letter":  "Make the tone firmer. Reference that the contract is under review.",
            "exec_kpi":         "Shorten to 200 words. Focus only on the two critical risks.",
            "escalation_email": "Add that we are already evaluating alternative suppliers.",
            "weekly_brief":     "Add a leadership decision needed at the end of the risks section.",
            "rfq":              "Add a section requiring suppliers to confirm minority-owned status.",
            "savings_report":   "Be more conservative. Add more caveats about risk to realization.",
        }

        refine_input = st.text_area(
            "What should change?",
            placeholder=placeholders.get(doc_id, "Describe what you'd like to change…"),
            height=80
        )
        if st.button("🔄 Apply Changes", use_container_width=True):
            if refine_input.strip():
                with st.spinner("Applying changes…"):
                    revised = refine(st.session_state.document, refine_input, model)
                st.session_state.document = revised
                st.rerun()
            else:
                st.warning("Please describe what you'd like to change.")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "ChainScribe · AI Supply Chain Document Writer · "
    "Powered by Ollama (runs locally — zero cost, fully private) · "
    "Research source: IBM Institute for Business Value, 2024"
)
