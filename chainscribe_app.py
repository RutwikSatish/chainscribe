"""
SupplyScript — AI Supply Chain Document Writer
==============================================
v3.0  Groq API · 7 document types · industry-validated prompts

Document types and their standards basis:
  supplier_letter  — CIPS Supplier Performance Management / ISM SPM KPI framework
  exec_kpi         — PwC Procurement Survey 2024 (board-level KPI reporting)
  escalation_email — CIPS Contract Management Cycle / ISM dispute resolution guidance
  weekly_brief     — Standard operations RAG status reporting
  rfq              — ISM RFQ best practice / CIPS procurement cycle / APICS
  savings_report   — CIPS hard/soft/avoidance definition / ISM savings methodology
  scar             — AIAG SCAR standard / ISO 9001:2015 clause 10.2

Stack: Python · Streamlit · Groq (Llama 3.3 70B) · feedparser
Author: Rutwik Satish | MS Engineering Management, Northeastern University
"""

import streamlit as st
import requests
from datetime import date
import risk_feed   # keep your existing risk_feed module as-is

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
TODAY      = date.today().strftime("%B %d, %Y")

# Estimated manual writing times (reasonable estimates, not published benchmarks)
TIME_ESTIMATES = {
    "supplier_letter":  45,
    "exec_kpi":         60,
    "escalation_email": 30,
    "weekly_brief":     40,
    "rfq":              90,
    "savings_report":   60,
    "scar":             50,
}

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplyScript",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #f8f9fb !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebarNav"] { display: none !important; }
h1,h2,h3,h4 { font-family: 'DM Sans', sans-serif !important; color: #0f172a !important; }
[data-testid="metric-container"] {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important; padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important; font-family: 'DM Mono', monospace !important;
    font-weight: 600 !important; font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important; border-bottom: 2px solid #2563eb !important;
}
[data-testid="stButton"] button {
    background: #2563eb !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-weight: 500 !important;
}
[data-testid="stButton"] button:hover { background: #1d4ed8 !important; }
.doc-output {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 24px 28px; font-family: Georgia, 'Times New Roman', serif;
    font-size: 14px; line-height: 1.85; color: #1a1a1a;
    white-space: pre-wrap; word-break: break-word;
}
.source-tag {
    background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
    border-radius: 4px; padding: 2px 8px; font-size: 0.7rem;
    font-family: 'DM Mono', monospace; display: inline-block; margin: 2px;
}
.est-badge {
    background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0;
    border-radius: 20px; padding: 4px 12px; font-size: 0.78rem;
    font-weight: 600; display: inline-block; margin-top: 6px;
}
hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "document" not in st.session_state: st.session_state.document = ""
if "doc_id"   not in st.session_state: st.session_state.doc_id   = ""

# ── GROQ API ──────────────────────────────────────────────────────────────────
def ask_groq(system: str, user: str) -> str:
    """Call Groq API. Add GROQ_API_KEY to Streamlit Secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return (
            "GROQ_API_KEY not found in Streamlit secrets.\n\n"
            "To fix this:\n"
            "  1. Open app settings in Streamlit Cloud\n"
            "  2. Go to Secrets\n"
            "  3. Add:  GROQ_API_KEY = \"your_key_here\"\n"
            "  Free key available at console.groq.com"
        )
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model":    GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens":  1500,
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        err = resp.json().get("error", {}).get("message", "Unknown error")
        return f"Groq API error: {err}"
    except Exception as e:
        return f"Error: {e}"


def refine(original: str, instruction: str) -> str:
    return ask_groq(
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
    )


def validate_required(fields: dict) -> list[str]:
    """Return list of missing required field labels."""
    return [label for label, val in fields.items() if not str(val).strip()]

# ── PROMPTS ───────────────────────────────────────────────────────────────────

def prompt_supplier_letter(v: dict) -> tuple:
    """
    Standard: CIPS Supplier Performance Management + ISM SPM KPI framework.
    Tone guide mirrors Kraljic matrix supplier segmentation (CIPS endorsed).
    KPIs: OTD, defect rate — confirmed ISM standard metrics.
    """
    tone_guide = {
        "Strategic Partner":  "warm but accountable — valued long-term partner, tone is collaborative",
        "Preferred Supplier": "professional and direct — maintain relationship, drive improvement",
        "Standard Supplier":  "formal and business-like — clear expectations and documented consequences",
        "Under Review":       "firm and factual — this letter may be referenced in contract proceedings",
    }.get(v.get("relationship", "Standard Supplier"), "professional and direct")

    try:
        otd_ok    = float(v.get("otd","0"))    >= float(v.get("otd_target","95"))
        defect_ok = float(v.get("defect","0")) <= float(v.get("defect_target","2"))
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
RELATIONSHIP: {v['relationship']} — tone: {tone_guide}

PERFORMANCE vs TARGETS:
  On-Time Delivery:  {v['otd']}%    (Target: {v['otd_target']}%)   — {'MEETS TARGET' if otd_ok else 'BELOW TARGET — IMPROVEMENT REQUIRED'}
  Defect Rate:       {v['defect']}%  (Target: <{v['defect_target']}%) — {'WITHIN LIMIT' if defect_ok else 'EXCEEDS LIMIT — IMMEDIATE ACTION REQUIRED'}
  Contract Value:    ${v['contract_value']}
  Open POs:          {v.get('open_pos', 'N/A')}

KEY ISSUES THIS PERIOD:
{v['issues']}

REQUIRED DOCUMENT STRUCTURE:
1. Date and address block
2. Subject: Performance Review — [Period]
3. Opening paragraph: state purpose clearly
4. Performance summary: cite every metric with actual vs target
5. Areas of concern: specific issues with dates/PO numbers if given
6. Required actions: numbered list — each action must have a deadline
7. Consequences of non-improvement (proportionate to relationship status)
8. Request for written response within 5 business days
9. Professional close + signature line: "[Your Name] | Supply Chain Management"

Rules: No vague language. Every concern must have a specific required action."""
    return sys, usr


def prompt_exec_kpi(v: dict) -> tuple:
    """
    Standard: PwC Procurement Survey 2024 — performance reporting is board-level priority.
    Format: executive briefing conventions (data point per sentence, no passive voice).
    """
    sys = (
        "You are a VP of Supply Chain writing a monthly performance summary for the CEO and CFO. "
        "You write with executive precision: every sentence has a data point or a decision. "
        "You never use filler phrases. You bold critical numbers. Under 400 words."
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
**OVERALL HEALTH:** [Green / Amber / Red] — one sentence explanation

**KEY WINS THIS PERIOD:** (max 3 bullets — each must have a number)

**AREAS OF CONCERN:** (max 3 bullets — each: Issue → Root Cause → Owner)

**TOP 2 RISKS:**
  Risk 1: [description] | Likelihood: [H/M/L] | Mitigation: [action + owner]
  Risk 2: [description] | Likelihood: [H/M/L] | Mitigation: [action + owner]

**LEADERSHIP ACTIONS REQUIRED:** (numbered, max 3)

Style: Bold all KPI numbers. Use → for trend direction. No passive voice.
Target: 280–380 words."""
    return sys, usr


def prompt_escalation(v: dict) -> tuple:
    """
    Standard: CIPS Contract Management Cycle / ISM contract dispute guidance.
    Reference number format, timeline of events, consequence framing
    all align with procurement documentation best practice.
    """
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

ISSUE:             {v['issue_type']}
DURATION:          {v['duration']}
BUSINESS IMPACT:   {v['impact']}
PRIOR ACTIONS:     {v['prior_actions']}
REQUIRED BY:       {v['deadline']}
RESOLUTION NEEDED: {v['resolution']}

REQUIRED STRUCTURE:
Subject: [ESCALATION {ref}] {v['issue_type']} — {v['supplier_name']} — Action Required by {v['deadline']}

1. Opening (2 sentences): state issue and reference number factually
2. Business impact paragraph: quantify time, money, downstream effects
3. Timeline of events: when first raised, commitments made, what was missed
4. Required actions (numbered):
   - Action 1 | Owner: [supplier] | Deadline: [specific date]
   - Action 2 | Owner: [supplier] | Deadline: [specific date]
5. Consequences paragraph: factual, not threatening (e.g. contract review, sourcing alternatives)
6. Close: offer a call to resolve + placeholder for contact info

Tone: Firm, factual, professional. Every statement is traceable."""
    return sys, usr


def prompt_weekly(v: dict) -> tuple:
    """Standard operations RAG status briefing format."""
    sys = (
        "You are a supply chain operations manager writing the weekly team briefing. "
        "Your briefings are structured, scannable, and action-oriented. "
        "You use RAG status. Every issue has an owner. Every action has a who and when."
    )
    usr = f"""Write a weekly supply chain operations briefing.

WEEK:  {v['week']}
DATE:  {TODAY}

WINS:        {v['wins']}
ISSUES:      {v['issues']}
METRICS:     {v['metrics']}
RISKS:       {v['risks']}
ACTIONS:     {v['actions']}
NEXT WEEK:   {v['upcoming']}

REQUIRED FORMAT:
OVERALL STATUS: Green/Amber/Red — [one sentence reason]

WINS THIS WEEK:
- [bullet per win with number]

ISSUES:
Issue | Root Cause | Owner | ETA
[one row per issue]

METRICS:
Metric | Actual | Target | Trend (up/down/flat)
[one row per metric]

RISK REGISTER:
Risk | Likelihood (H/M/L) | Impact (H/M/L) | Owner
[top 3 risks]

ACTIONS:
- WHO: WHAT: BY WHEN
[one per action]

NEXT WEEK FOCUS:
[2-3 priorities]

Bold important numbers. ~280-350 words."""
    return sys, usr


def prompt_rfq(v: dict) -> tuple:
    """
    Standard: ISM.ws RFQ best practice (2024) — RFQ is for standardized buys
    with defined specs. CIPS Procurement Cycle confirms required sections.
    APICS / CIPS: evaluation criteria weightings must total 100%.
    """
    ref = f"RFQ-{date.today().strftime('%Y-%m')}-001"
    sys = (
        "You are a senior procurement manager writing a formal Request for Quote. "
        "Your RFQs leave zero ambiguity — suppliers know exactly what is required. "
        "Use numbered sections and professional procurement document formatting."
    )
    usr = f"""Write a complete, formal Request for Quote (RFQ) document.

REFERENCE:      {ref}
ISSUE DATE:     {TODAY}
ISSUED BY:      {v['company']}
CATEGORY:       {v['category']}
QUANTITY:       {v['quantity']}
DELIVERY TO:    {v['location']}
DELIVERY BY:    {v['delivery_date']}
QUOTE DEADLINE: {v['quote_deadline']}

TECHNICAL SPECIFICATIONS:
{v['specs']}

EVALUATION CRITERIA:
{v['criteria']}

SPECIAL REQUIREMENTS:
{v['special_reqs']}

REQUIRED SECTIONS (number each):
1. Document Header (reference, dates, issuing company, contact placeholders)
2. Purpose and Scope
3. Technical Specifications (table: Requirement | Specification | Mandatory Y/N)
4. Quantity and Delivery Schedule
5. Commercial Requirements (payment terms, warranty, required certifications)
6. Evaluation Criteria (list with weightings that total 100%)
7. Submission Instructions (format, deadline, where to send, Q&A window)
8. Terms and Conditions (4-5 bullet points)
9. Supplier Acknowledgement (signature / acceptance section)

Professional procurement document. Number all sections. Use tables where noted."""
    return sys, usr


def prompt_savings(v: dict) -> tuple:
    """
    Standard: CIPS definition of hard/soft/avoidance savings.
    ISM savings methodology: baseline, strategy, validation.
    Note: industry research (Suplari) shows 30-40% of negotiated savings
    do not reach the P&L — prompt includes 'Risks to Realization' section
    to address this gap.
    """
    try:
        base   = float(str(v.get("baseline","0")).replace(",","").replace("$",""))
        new    = float(str(v.get("new_spend","0")).replace(",","").replace("$",""))
        saved  = base - new
        pct    = (saved / base * 100) if base > 0 else 0
        months = float(v.get("months","12") or "12")
        annual = (saved / months * 12) if months > 0 else saved
    except ValueError:
        base = new = saved = pct = annual = 0

    sys = (
        "You are a procurement manager presenting cost savings to the CFO and board. "
        "Per CIPS guidance, you clearly separate hard savings from cost avoidance. "
        "Your reports are auditable, conservative, and transparent about methodology. "
        "Every number is traceable."
    )
    usr = f"""Write a formal procurement cost savings report.

DATE:        {TODAY}
PERIOD:      {v['period']}
CATEGORY:    {v['category']}
SAVING TYPE: {v['saving_type']}

FINANCIALS:
  Baseline Spend:        ${base:,.0f}
  New / Negotiated:      ${new:,.0f}
  Gross Saving:          ${saved:,.0f}  ({pct:.1f}%)
  Annualized Projection: ${annual:,.0f}
  Period Covered:        {v['months']} months

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
4. Strategy and Approach (what negotiation/sourcing approach was used)
5. Saving Validation (how confirmed — benchmarks, contracts, invoices)
6. Risks to Realization (what could prevent this saving reaching the P&L)
7. Annualized Projection (${annual:,.0f} based on {v['months']}-month run-rate)
8. Recommended Next Steps (2-3 actions for leadership)

Important: Clearly label saving type (hard/soft/avoidance). Hard savings must be
traceable to invoice-level price reduction. Do not conflate cost avoidance with
realized savings — per CIPS, these are distinct categories."""
    return sys, usr


def prompt_scar(v: dict) -> tuple:
    """
    Standard: AIAG Supplier Corrective Action Request format.
    ISO 9001:2015 clause 10.2 (nonconformity and corrective action).
    Structure: containment → root cause → corrective action → verification.
    """
    ref = f"SCAR-{date.today().strftime('%Y%m%d')}-001"
    sys = (
        "You are a quality engineer issuing a Supplier Corrective Action Request (SCAR). "
        "Your SCARs follow the AIAG standard format and ISO 9001:2015 clause 10.2 requirements. "
        "You require root cause analysis (5-Why or Fishbone) and measurable corrective actions. "
        "Every action must have an owner and a date."
    )
    usr = f"""Write a formal Supplier Corrective Action Request (SCAR).

SCAR REFERENCE: {ref}
ISSUE DATE:     {TODAY}
SUPPLIER:       {v['supplier_name']}
ISSUED BY:      {v.get('issued_by', '[Your Name] — Quality / Supply Chain')}
RESPONSE DUE:   {v['response_deadline']}

NONCONFORMANCE:
Part / Component: {v['part']}
Issue Type:       {v['issue_type']}
Quantity Affected:{v['qty_affected']}
Detection Point:  {v['detection_point']}
Description:      {v['description']}
PO / Lot Ref:     {v.get('po_ref', 'N/A')}

POTENTIAL IMPACT:
{v['impact']}

REQUIRED DOCUMENT STRUCTURE (number each section):

1. SCAR HEADER
   Reference number, date, supplier, issuing company, contact placeholders

2. NONCONFORMANCE SUMMARY
   Clear description of what was found, how many units, where detected

3. IMMEDIATE CONTAINMENT ACTIONS REQUIRED (due within 24-48 hours)
   - What the supplier must do NOW to stop non-conforming product reaching production
   - Who is responsible, by when

4. ROOT CAUSE ANALYSIS REQUIRED
   Supplier must submit a documented root cause using 5-Why or Fishbone method
   Due: {v['response_deadline']}

5. CORRECTIVE ACTION PLAN
   Permanent actions to eliminate root cause
   Each action: Description | Owner | Implementation Date | Verification Method

6. PREVENTIVE ACTION
   What systemic change will prevent recurrence across other part numbers / processes

7. VERIFICATION AND CLOSURE CRITERIA
   How the issuing company will verify the corrective action is effective
   (e.g., first article, process audit, statistical sampling for next 3 shipments)

8. SUPPLIER ACKNOWLEDGEMENT
   Signature block — supplier confirms they have read and will respond by deadline

Tone: Professional and factual. This is a formal quality record."""
    return sys, usr


# ── DISPATCH ──────────────────────────────────────────────────────────────────
PROMPTS = {
    "supplier_letter":  prompt_supplier_letter,
    "exec_kpi":         prompt_exec_kpi,
    "escalation_email": prompt_escalation,
    "weekly_brief":     prompt_weekly,
    "rfq":              prompt_rfq,
    "savings_report":   prompt_savings,
    "scar":             prompt_scar,
}

DISPLAY_TO_ID = {
    "📬 Supplier Performance Review Letter": "supplier_letter",
    "📊 Executive KPI Summary":              "exec_kpi",
    "🚨 Supplier Escalation Email":          "escalation_email",
    "📋 Weekly Operations Briefing":         "weekly_brief",
    "📝 Request for Quote (RFQ)":            "rfq",
    "💰 Cost Savings Report":                "savings_report",
    "🔧 Supplier Corrective Action (SCAR)":  "scar",
}

SOURCES = {
    "supplier_letter":  "CIPS Supplier Performance Management · ISM SPM KPI Framework",
    "exec_kpi":         "PwC Procurement Survey 2024 · Board-level reporting standard",
    "escalation_email": "CIPS Contract Management Cycle · ISM Dispute Resolution Guidance",
    "weekly_brief":     "Standard RAG status operations reporting",
    "rfq":              "ISM RFQ Best Practice 2024 · CIPS Procurement Cycle · APICS",
    "savings_report":   "CIPS Hard/Soft/Avoidance Definitions · ISM Savings Methodology",
    "scar":             "AIAG SCAR Format · ISO 9001:2015 Clause 10.2",
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# SupplyScript")
    st.caption("AI Document Writer for Supply Chain")
    st.divider()

    st.markdown("### Mode")
    app_mode = st.radio(
        "", ["Document Writer", "Live Risk Feed"], label_visibility="collapsed"
    )
    st.divider()

    st.markdown("### Document Type")
    choice = st.selectbox("", list(DISPLAY_TO_ID.keys()), label_visibility="collapsed")
    doc_id = DISPLAY_TO_ID[choice]

    st.divider()
    st.markdown("### Industry Standard")
    st.caption(SOURCES.get(doc_id, ""))

    if app_mode == "Document Writer":
        st.divider()
        est = TIME_ESTIMATES.get(doc_id, 45)
        st.markdown("### Estimated Time Saved")
        st.markdown(f"Manual writing: ~{est} min")
        st.markdown(f"With SupplyScript: ~2 min")
        st.markdown(
            f"<span class='est-badge'>Saves ~{est-2} minutes (estimated)</span>",
            unsafe_allow_html=True
        )
        st.divider()
        st.markdown("### AI Model")
        st.caption("Groq — Llama 3.3 70B")

# ── LIVE RISK FEED ────────────────────────────────────────────────────────────
if app_mode == "Live Risk Feed":
    risk_feed.render()
    st.stop()

# ── DOCUMENT WRITER ───────────────────────────────────────────────────────────
st.markdown(f"## {choice}")
st.caption(f"Standard: {SOURCES.get(doc_id, '')}")
st.divider()

col_form, col_out = st.columns([1, 1], gap="large")

submitted = False
inputs    = {}

with col_form:
    st.markdown("### Document Details")

    # ── Supplier Letter ───────────────────────────────────────────────────────
    if doc_id == "supplier_letter":
        with st.form("f_supplier"):
            r1c1, r1c2 = st.columns(2)
            inputs["supplier_name"] = r1c1.text_input("Supplier Name *", placeholder="GlobalTech Parts Inc.")
            inputs["period"]        = r1c2.text_input("Review Period *",  placeholder="Q2 2025")
            r2c1, r2c2 = st.columns(2)
            inputs["otd"]        = r2c1.text_input("Actual OTD %",   value="87")
            inputs["otd_target"] = r2c2.text_input("OTD Target %",   value="95")
            r3c1, r3c2 = st.columns(2)
            inputs["defect"]        = r3c1.text_input("Actual Defect Rate %",  value="3.2")
            inputs["defect_target"] = r3c2.text_input("Max Defect Rate %",     value="2.0")
            r4c1, r4c2 = st.columns(2)
            inputs["contract_value"] = r4c1.text_input("Annual Contract Value ($)", value="250,000")
            inputs["open_pos"]       = r4c2.text_input("Open Purchase Orders",      value="12")
            inputs["relationship"] = st.selectbox(
                "Supplier Relationship Status",
                ["Strategic Partner", "Preferred Supplier", "Standard Supplier", "Under Review"]
            )
            inputs["issues"] = st.text_area(
                "Key Issues During Period *",
                placeholder="- 3 shipments delayed 5+ days in January\n- Incorrect spec on PO #4521",
                height=110
            )
            submitted = st.form_submit_button("Generate Letter", type="primary", use_container_width=True)
            req_check = validate_required({"Supplier Name": inputs["supplier_name"],
                                           "Review Period": inputs["period"],
                                           "Key Issues":    inputs["issues"]})

    # ── Executive KPI ─────────────────────────────────────────────────────────
    elif doc_id == "exec_kpi":
        with st.form("f_kpi"):
            c1, c2 = st.columns(2)
            inputs["company"] = c1.text_input("Company / Division *", placeholder="ACME Corp — North America Ops")
            inputs["period"]  = c2.text_input("Reporting Period *",   placeholder="April 2025")
            inputs["kpi_data"] = st.text_area(
                "KPI Data *",
                placeholder=(
                    "On-Time Delivery: 88.4%  (Target: 95%)  down from 91.2%\n"
                    "Inventory Turns:   6.2   (Target: 7.0)  flat\n"
                    "Supplier Lead Time:18 days (Target: 14)  improved\n"
                    "PO Fill Rate:      94.1%  (Target: 96%)  slight decline"
                ),
                height=150
            )
            inputs["context"] = st.text_area(
                "Context / Notable Events",
                placeholder="e.g. Supplier X factory shutdown 5 days. New 3PL launched in Texas.",
                height=80
            )
            submitted = st.form_submit_button("Generate KPI Summary", type="primary", use_container_width=True)
            req_check = validate_required({"Company / Division": inputs["company"],
                                           "Reporting Period":   inputs["period"],
                                           "KPI Data":           inputs["kpi_data"]})

    # ── Escalation ────────────────────────────────────────────────────────────
    elif doc_id == "escalation_email":
        with st.form("f_escalation"):
            c1, c2 = st.columns(2)
            inputs["supplier_name"] = c1.text_input("Supplier Name *",    placeholder="FastFreight Co.")
            inputs["contact"]       = c2.text_input("Contact Name/Title", placeholder="John Smith, Account Manager")
            inputs["issue_type"] = st.selectbox(
                "Issue Type",
                ["Late Delivery", "Quality / Defects", "Pricing Dispute",
                 "Documentation Error", "Capacity / Shortage", "Compliance Breach", "Other"]
            )
            c3, c4 = st.columns(2)
            inputs["duration"] = c3.text_input("Duration of Issue *", placeholder="3 weeks / since Jan 15")
            inputs["deadline"] = c4.text_input("Resolution Deadline *", placeholder="March 31, 2025")
            inputs["impact"] = st.text_area(
                "Business Impact *",
                placeholder="3 customer orders delayed, ~$45,000 in pending shipments, production halted 2 days",
                height=75
            )
            inputs["prior_actions"] = st.text_area(
                "Prior Actions / Communications",
                placeholder="Emailed Feb 10 — no response. Called Feb 14 — promised fix Feb 17 (missed).",
                height=65
            )
            inputs["resolution"] = st.text_area(
                "Required Resolution *",
                placeholder="Ship all 3 delayed orders by March 31. Submit root cause. Corrective action plan.",
                height=65
            )
            submitted = st.form_submit_button("Generate Escalation Email", type="primary", use_container_width=True)
            req_check = validate_required({"Supplier Name":       inputs["supplier_name"],
                                           "Duration of Issue":   inputs["duration"],
                                           "Resolution Deadline": inputs["deadline"],
                                           "Business Impact":     inputs["impact"],
                                           "Required Resolution": inputs["resolution"]})

    # ── Weekly Briefing ───────────────────────────────────────────────────────
    elif doc_id == "weekly_brief":
        with st.form("f_weekly"):
            inputs["week"] = st.text_input("Week Of *", placeholder="Week of March 24, 2025")
            inputs["wins"] = st.text_area(
                "Wins This Week",
                placeholder="- Completed Q1 supplier audits\n- New 3PL contract signed — saves $180K/year",
                height=85
            )
            inputs["issues"] = st.text_area(
                "Issues / Lowlights *",
                placeholder="- Apex Steel delayed — 2 POs at risk\n- Inventory variance flagged in Atlanta DC",
                height=85
            )
            inputs["metrics"] = st.text_area(
                "Key Metrics",
                placeholder="OTIF: 91% (target 95%) down\nInventory Turns: 6.1 flat\nDefect Rate: 1.8%",
                height=75
            )
            inputs["risks"] = st.text_area(
                "Risks and Watchlist",
                placeholder="- Port congestion LA: 4 inbound shipments at risk\n- Apex Steel capacity issue ongoing",
                height=75
            )
            inputs["actions"] = st.text_area(
                "Actions / In Progress",
                placeholder="[Done] Sarah: Emergency PO to backup supplier\n[WIP] James: Negotiating expedite fee",
                height=65
            )
            inputs["upcoming"] = st.text_area(
                "Next Week Focus",
                placeholder="Q2 demand planning session Thu. Annual supplier review kickoff.",
                height=55
            )
            submitted = st.form_submit_button("Generate Weekly Briefing", type="primary", use_container_width=True)
            req_check = validate_required({"Week Of": inputs["week"],
                                           "Issues":  inputs["issues"]})

    # ── RFQ ───────────────────────────────────────────────────────────────────
    elif doc_id == "rfq":
        with st.form("f_rfq"):
            c1, c2 = st.columns(2)
            inputs["company"]  = c1.text_input("Your Company Name *",  placeholder="ACME Manufacturing Inc.")
            inputs["category"] = c2.text_input("Category / Item *",    placeholder="Industrial Steel Tubing Grade 316")
            c3, c4 = st.columns(2)
            inputs["quantity"] = c3.text_input("Quantity",             placeholder="50,000 units")
            inputs["location"] = c4.text_input("Delivery Location",   placeholder="Atlanta, GA 30301")
            c5, c6 = st.columns(2)
            inputs["delivery_date"]  = c5.text_input("Required Delivery Date",   placeholder="June 30, 2025")
            inputs["quote_deadline"] = c6.text_input("Quote Submission Deadline", placeholder="April 15, 2025")
            inputs["specs"] = st.text_area(
                "Technical Specifications *",
                placeholder=(
                    "Material: 316L stainless steel\nWall thickness: 2mm +/- 0.1mm\n"
                    "Length: 6m standard\nRequired certifications: ISO 9001, ASTM A269"
                ),
                height=110
            )
            inputs["criteria"] = st.text_area(
                "Evaluation Criteria (must total 100%)",
                placeholder="Price 50%, Quality certs 25%, Lead time 15%, Sustainability 10%",
                height=55
            )
            inputs["special_reqs"] = st.text_area(
                "Special Requirements",
                placeholder="Supplier must hold ISO 9001. Min 3 years in category. QBR required.",
                height=55
            )
            submitted = st.form_submit_button("Generate RFQ", type="primary", use_container_width=True)
            req_check = validate_required({"Company Name":            inputs["company"],
                                           "Category / Item":         inputs["category"],
                                           "Technical Specifications": inputs["specs"]})

    # ── Cost Savings Report ───────────────────────────────────────────────────
    elif doc_id == "savings_report":
        with st.form("f_savings"):
            c1, c2 = st.columns(2)
            inputs["period"]   = c1.text_input("Reporting Period *",  placeholder="Q1 2025")
            inputs["category"] = c2.text_input("Spend Category *",    placeholder="Indirect Procurement — MRO")
            c3, c4 = st.columns(2)
            inputs["baseline"]  = c3.text_input("Baseline Spend ($) *",     placeholder="1200000")
            inputs["new_spend"] = c4.text_input("Negotiated Spend ($) *",   placeholder="950000")
            c5, c6 = st.columns(2)
            inputs["saving_type"] = c5.selectbox(
                "Saving Type (CIPS definition)",
                ["Hard Saving (Cash Out — traceable to P&L)",
                 "Cost Avoidance (Soft Saving — future spend prevented)",
                 "Soft Saving / Efficiency",
                 "Rebate / Volume Discount"]
            )
            inputs["months"] = c6.text_input("Months Covered", value="12")
            inputs["strategy"] = st.text_area(
                "Strategy Used *",
                placeholder="Competitive RFQ to 6 suppliers. Consolidated from 3 to 1. 2-year contract.",
                height=75
            )
            inputs["suppliers"] = st.text_input(
                "Suppliers Involved",
                placeholder="OfficeSupply Co. (incumbent), SupplyMax Inc. (new)"
            )
            inputs["risks"] = st.text_area(
                "Risks to Realization",
                placeholder="Volume commitment — if usage drops >20%, unit price increases. New supplier unproven.",
                height=65
            )
            submitted = st.form_submit_button("Generate Savings Report", type="primary", use_container_width=True)
            req_check = validate_required({"Reporting Period": inputs["period"],
                                           "Spend Category":   inputs["category"],
                                           "Baseline Spend":   inputs["baseline"],
                                           "Negotiated Spend": inputs["new_spend"],
                                           "Strategy Used":    inputs["strategy"]})

    # ── SCAR ─────────────────────────────────────────────────────────────────
    elif doc_id == "scar":
        with st.form("f_scar"):
            c1, c2 = st.columns(2)
            inputs["supplier_name"]     = c1.text_input("Supplier Name *",        placeholder="TechParts Inc.")
            inputs["issued_by"]         = c2.text_input("Issued By",              placeholder="Rutwik Satish — Quality Engineering")
            c3, c4 = st.columns(2)
            inputs["part"]              = c3.text_input("Part / Component *",     placeholder="Valve Body — PN 44521-A")
            inputs["issue_type"]        = c4.selectbox(
                "Nonconformance Type",
                ["Dimensional Out of Spec", "Surface / Cosmetic Defect", "Wrong Material / Grade",
                 "Contamination", "Documentation Error", "Packaging Non-conformance",
                 "Functional Failure", "Late Delivery — Critical Impact", "Other"]
            )
            c5, c6 = st.columns(2)
            inputs["qty_affected"]      = c5.text_input("Quantity Affected *",    placeholder="240 units / Lot #L-2244")
            inputs["detection_point"]   = c6.selectbox(
                "Detection Point",
                ["Incoming Inspection", "In-Process / Production", "Final Inspection",
                 "Customer / End User", "Field Return"]
            )
            inputs["po_ref"]            = st.text_input("PO / Lot Reference",     placeholder="PO-20250415 / Lot L-2244")
            inputs["description"]       = st.text_area(
                "Nonconformance Description *",
                placeholder="Wall thickness measured at 1.7mm against spec of 2.0mm +/- 0.1mm on 240 of 500 units inspected.",
                height=85
            )
            inputs["impact"]            = st.text_area(
                "Potential Business Impact *",
                placeholder="Production line at risk. 3 customer orders (PO-881, 882, 884) may be delayed. Estimated impact: $28,000.",
                height=75
            )
            inputs["response_deadline"] = st.text_input(
                "Supplier Response Deadline *",
                placeholder="May 16, 2025 (10 business days from issue)"
            )
            submitted = st.form_submit_button("Generate SCAR", type="primary", use_container_width=True)
            req_check = validate_required({"Supplier Name":        inputs["supplier_name"],
                                           "Part / Component":     inputs["part"],
                                           "Quantity Affected":    inputs["qty_affected"],
                                           "Description":          inputs["description"],
                                           "Business Impact":      inputs["impact"],
                                           "Response Deadline":    inputs["response_deadline"]})

    # ── Generate ──────────────────────────────────────────────────────────────
    if submitted:
        if req_check:
            st.error(f"Please complete required fields: {', '.join(req_check)}")
        else:
            prompt_fn = PROMPTS.get(doc_id)
            if prompt_fn:
                with st.spinner("Generating document via Groq..."):
                    sys_p, usr_p = prompt_fn(inputs)
                    result = ask_groq(sys_p, usr_p)
                st.session_state.document = result
                st.session_state.doc_id   = doc_id
                st.rerun()

# ── OUTPUT COLUMN ─────────────────────────────────────────────────────────────
with col_out:
    st.markdown("### Generated Document")

    if not st.session_state.document:
        st.info("Fill in the form and click Generate to create your document.")
        st.markdown("")
        st.markdown("**What SupplyScript generates:**")
        for name in DISPLAY_TO_ID:
            st.markdown(f"- {name}")
        st.markdown("")
        st.markdown(f"**AI:** Groq — Llama 3.3 70B (cloud, fast)")
        st.markdown("**Powered by:** CIPS · ISM · AIAG · APICS standards")
    else:
        doc = st.session_state.document

        # Styled reading view
        st.markdown(f'<div class="doc-output">{doc}</div>', unsafe_allow_html=True)

        st.divider()

        # Copy to clipboard (built-in copy button via st.code)
        with st.expander("Copy text"):
            st.code(doc, language=None)

        # Download as .txt
        filename = f"SupplyScript_{doc_id}_{date.today().strftime('%Y%m%d')}.txt"
        st.download_button(
            label="Download (.txt)",
            data=doc,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("### Refine This Document")
        st.caption("Describe what to change — SupplyScript rewrites the full document.")

        quick_refines = {
            "supplier_letter":  ["Make the tone firmer", "Shorten by 30%", "Add that we are evaluating alternative suppliers"],
            "exec_kpi":         ["Shorten to 200 words", "Focus only on the two biggest risks", "Add a leadership decision needed"],
            "escalation_email": ["Add that we are already evaluating alternatives", "Make more formal", "Shorten to 150 words"],
            "weekly_brief":     ["Add a leadership decision needed at the end", "Shorten to 200 words", "Make more formal"],
            "rfq":              ["Add ESG / sustainability requirements section", "Make evaluation criteria stricter", "Add minority-owned supplier question"],
            "savings_report":   ["Be more conservative with projections", "Add more caveats about risk to realization", "Shorten executive summary"],
            "scar":             ["Make the tone more urgent", "Add requirement for 8D report format", "Shorten containment section"],
        }

        quick = quick_refines.get(doc_id, [])
        if quick:
            cols = st.columns(len(quick))
            for col, label in zip(cols, quick):
                if col.button(label, use_container_width=True):
                    with st.spinner("Applying change..."):
                        revised = refine(st.session_state.document, label)
                    st.session_state.document = revised
                    st.rerun()

        refine_input = st.text_area(
            "Custom change",
            placeholder="Describe any other change...",
            height=70
        )
        if st.button("Apply Custom Change", use_container_width=True):
            if refine_input.strip():
                with st.spinner("Applying..."):
                    revised = refine(st.session_state.document, refine_input)
                st.session_state.document = revised
                st.rerun()
            else:
                st.warning("Please describe what you'd like to change.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "SupplyScript · AI Supply Chain Document Writer · "
    "Standards: CIPS · ISM · AIAG · APICS · ISO 9001:2015 · "
    "Built by Rutwik Satish · MS Engineering Management, Northeastern University."
)
