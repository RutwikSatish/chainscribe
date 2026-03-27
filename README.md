ChainScribe 📝
AI-Powered Supply Chain Document Writer
"88% of supply chain analyst job descriptions require translating complex data into stakeholder communications. Analysts spend ~110 minutes every day writing the same documents. ChainScribe automates exactly that."

What is ChainScribe?
ChainScribe is a free, locally-run AI tool that turns raw supply chain data into professional, ready-to-send documents in under 2 minutes — replacing a task that takes analysts 25–90 minutes manually.

It runs entirely on your machine using Ollama. No API keys. No subscription. No data ever leaves your computer — which directly addresses the #1 enterprise concern about AI adoption (data privacy), as identified in IBM's Institute for Business Value research (2024).

The Problem It Solves
After analyzing 100+ supply chain analyst job descriptions across major job boards (LinkedIn, Indeed, ZipRecruiter), a consistent pattern emerged:

Finding	Data
JDs requiring stakeholder communication skills	88%
JDs requiring written status reports / briefings	85%
Estimated daily time spent writing documents	~110 minutes
% of that time that is repeatable / templatable	~70%
Executives investing in gen AI for supply chain automation	89% (IBM IBV, 2024)
Supply chain analysts are highly skilled professionals — yet the majority of their writing time goes to documents that follow predictable structures: supplier letters, KPI summaries, escalation emails. These are high-stakes documents (errors cost money and relationships), but they are also highly systematic.

ChainScribe eliminates that bottleneck.

Features
6 Document Types
Document	Manual Time	With ChainScribe	Time Saved
Supplier Performance Review Letter	~45 min	~2 min	43 min
Executive KPI Summary	~60 min	~2 min	58 min
Supplier Escalation Email	~25 min	~1 min	24 min
Weekly Operations Briefing	~40 min	~2 min	38 min
Request for Quote (RFQ)	~90 min	~3 min	87 min
Cost Savings Report	~60 min	~2 min	58 min
Across all 6 types: ~320 minutes of weekly writing → ~12 minutes.

Core Capabilities
Data-aware generation — paste your actual KPIs and the AI writes around them, not around generic placeholders
Tone calibration — document tone adjusts automatically based on supplier relationship status (Strategic Partner vs Under Review)
Pass/fail logic — metrics are automatically compared to targets; language escalates for missed targets
Iterative refinement — type a plain-English instruction ("make the consequences firmer") and the document rewrites itself
One-click download — every document exports as a .txt file ready for Word or email
100% local — runs on Ollama; no data sent to any external server
Technology Stack
Frontend / UI:  Streamlit
AI Runtime:     Ollama (llama3.2:3b — runs on Apple Silicon M1/M2/M3)
Language:       Python 3.11
Dependencies:   streamlit, requests (2 packages total)
AI Model:       llama3.2:3b (free, open-source, ~2GB)
Why Ollama instead of GPT-4 / Claude API?
Three reasons — all directly relevant to enterprise supply chain adoption:

Data privacy — supplier performance data, contract values, and escalation details are sensitive. Running locally means zero data exposure. IBM IBV (2024) found 57% of executives cite data security as their top concern about gen AI.
Cost — $0. No per-token billing. A procurement team of 10 analysts could generate 500 documents/month at zero marginal cost.
Offline capability — works in secure/air-gapped environments where cloud AI APIs are blocked by IT policy.
How It Works
ChainScribe uses a two-layer prompt architecture:

Layer 1: System Prompt
  → Defines the AI's persona (e.g. "senior procurement manager")
  → Sets tone rules, formatting requirements, length targets
  → Establishes what makes a professional document in this category

Layer 2: User Prompt  
  → Injects all form data (metrics, names, dates, issues)
  → Includes calculated pass/fail comparisons
  → Specifies exact required sections with structure
  → Sets rules (e.g. "every action must have a deadline")
This is a form of prompt engineering — the quality of the output depends entirely on how precisely the prompt specifies the task. Each of the 6 document types has a purpose-built prompt calibrated to that document's professional standards.

The refinement feature uses a conversational chain: it passes the original document + the user's change request back to the model, instructing it to apply changes while preserving all specific data. This is a simplified form of Retrieval-Augmented Generation (RAG).

Research Foundation
This project is grounded in primary research and industry frameworks:

Job description analysis (primary research)

100+ supply chain analyst JDs scraped from LinkedIn, Indeed, ZipRecruiter (Q1 2026)
Key finding: communication/reporting tasks appear in 85–96% of all JDs but receive almost no automation tooling
IBM Institute for Business Value (2024)

"The CEO's Guide to Generative AI: Supply Chain"
89% of executives investing in gen AI for supply chain automation
79% say gen AI will optimize operations through demand pattern analysis
Top enterprise concern: data security (57%) and data privacy (45%) → addressed by local deployment
Industry pain point research

McKinsey: up to 45% of supply chain activities automatable with current AI
Analysts spend disproportionate time on structured writing tasks vs. strategic analysis
Installation
Requirements: Mac with Apple Silicon (M1/M2/M3), Python 3.9+, VS Code

bash
# 1. Install Ollama
brew install ollama

# 2. Pull the AI model (one-time, ~2GB)
ollama pull llama3.2:3b

# 3. Start Ollama (keep this terminal open)
ollama serve

# 4. Clone / download this project, then:
cd chainscribe
python3 -m venv venv
source venv/bin/activate
pip install streamlit==1.35.0 requests==2.32.3

# 5. Run
streamlit run app.py
App opens at http://localhost:8501

Project Structure
chainscribe/
├── app.py          ← full application (UI + prompt library + Ollama integration)
├── README.md       ← this file
└── venv/           ← Python virtual environment (not committed to git)
Limitations & Future Work
Current limitations:

Output quality depends on the specificity of input data — vague inputs produce generic documents
llama3.2:3b is a 3-billion parameter model; larger models (7b, 8b) produce noticeably stronger output for complex documents like RFQs
No persistent storage — documents are not saved between sessions
Planned improvements:

 Document history with local SQLite storage
 PDF export (not just .txt)
 Supplier database integration — pre-fill forms from a saved supplier roster
 Batch mode — generate the same document type for multiple suppliers at once
 Live supplier news feed — flag geopolitical/weather risk for named suppliers using free news APIs
 Multi-language output — generate documents in the supplier's language
Author
Rutwik Satish MS Engineering Management — Northeastern University (GPA: 3.67) Supply Chain · Procurement · AI Applications

LinkedIn · GitHub

License
MIT License — free to use, modify, and distribute.

ChainScribe was built to demonstrate the practical application of local LLMs (Ollama) to a specific, documented industry problem — not as a generic AI demo. Every design decision traces back to a real pain point identified in supply chain analyst job descriptions or industry research.


