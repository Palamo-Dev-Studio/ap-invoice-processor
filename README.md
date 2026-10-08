# 🚀 AP Copilot — Autonomous Accounts Payable Agent

**Kaggle Capstone Submission: 5-Day AI Agents Intensive Vibe Coding Course With Google**  
*Track: Agents for Business | Framework: Google ADK 2.3.0 & Antigravity*

[![CI](https://github.com/hectorluisalamo/ap-invoice-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/hectorluisalamo/ap-invoice-processor/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Built with Google ADK](https://img.shields.io/badge/Built%20with-Google%20ADK%202.3.0-4285F4?logo=google&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-enabled-FF6F00)
![Tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)
[![Watch the Demo](https://img.shields.io/badge/%E2%96%B6%20Watch-Demo%20Video-FF0000?logo=youtube&logoColor=white)](https://youtu.be/5spsyj1pMpY)

---

## 🎬 Demo (≤5 min)

[![AP Copilot — full walkthrough](https://img.youtube.com/vi/5spsyj1pMpY/maxresdefault.jpg)](https://youtu.be/5spsyj1pMpY)

▶️ **[Watch the full walkthrough on YouTube](https://youtu.be/5spsyj1pMpY)** — invoice intake → field extraction → GL-coding → the human safety gate → posting to a mock NetSuite ERP over a real MCP server.

---

## 💡 Executive Summary & ROI Value Proposition

Manual Accounts Payable (AP) processing is an enterprise bottleneck. Independent benchmarks put the fully-loaded cost of processing a single invoice manually at **$10–$40**: APQC's 2024–2025 Open Standards data reports a **$21.40 median** (and **$10.18** for top-quartile organizations)[^apqc], while Ardent Partners' *AP Metrics That Matter (2025)* cites **$15–$40 for primarily-manual workflows** and a **$12.88 average** across non-best-in-class teams[^ardent]. This model uses a conservative mid-range **$14.50** manual baseline. 

**AP Copilot** is an autonomous accounts-payable agent built on **Google Agent Development Kit (ADK 2.3.0)** that automates invoice extraction, GL account coding, and ERP posting through a local **NetSuite MCP server**—with a **hard Human-in-the-Loop safety gate** on high-dollar or risky invoices. (All data is 100% synthetic; the NetSuite target is a mock ERP behind a real MCP interface.)

### 📈 Benchmark Performance & ROI Highlights (N=50 Synthetic Invoices)
* **Autonomous Auto-Post Rate:** **44.0%** of routine invoices posted instantly with zero human intervention.
* **GL Coding Accuracy:** **82.0%** correct GL account assignment across the synthetic invoice set.
* **Safety Routing Accuracy:** **94.0%** of risky invoices correctly routed to a human reviewer.
* **Blended Cost per Invoice:** Reduced from **$14.50** (manual baseline) to **$1.75** (blended compute + triage cost).
* **Net Cost Reduction:** **87.9% Total Cost Savings**.

> **ROI methodology & assumptions (read this before quoting the number).** The blended cost is **computed, not hardcoded** — it is driven by the *measured* auto-post rate from the N=50 eval run (`eval/eval_harness.py`), so it moves with the agent's actual performance. The formula is a standard blended cost-per-invoice:
>
> ```
> blended = auto_post_rate × compute_cost + (1 − auto_post_rate) × (compute_cost + human_triage_cost)
> savings% = (manual_cost − blended) / manual_cost
> ```
>
> **Measured input:** `auto_post_rate = 44.0%` (share of the 50 synthetic invoices that cleared autonomously vs. routed to the human gate).
> **Stated modeling assumptions** (unit costs; swap them for your own to re-scope the ROI):
> | Parameter | Value | Basis |
> |---|---|---|
> | `manual_cost` | **$14.50 / invoice** | Conservative mid-range of the APQC / Ardent benchmarks cited above. |
> | `compute_cost` | **$0.35 / invoice** | Estimated agent compute per invoice (the agent itself makes no live LLM/API calls — see the keyless note). |
> | `human_triage_cost` | **$2.50 / invoice** | Estimated cost of a human reviewing one flagged invoice at the safety gate. |
>
> Worked: `0.44 × $0.35 + 0.56 × ($0.35 + $2.50) = $1.75`; savings `= ($14.50 − $1.75) / $14.50 = 87.9%`. The `compute_cost` and `human_triage_cost` figures are internal estimates, not benchmarked — they are exposed as constants at the top of the ROI block in `eval/eval_harness.py` so the model is fully transparent and adjustable.

[^apqc]: APQC, *Total Cost to Perform the Process "Process Accounts Payable (AP)" per Invoice Processed*, Open Standards Benchmarking (2024–2025). https://www.apqc.org/resource-library/resource/total-cost-process-accounts-payable-invoice-processed
[^ardent]: Ardent Partners, *Accounts Payable Metrics That Matter in 2025*. https://ardentpartners.com/ap-metrics-that-matter-in-2025/

---

## 🏗️ System Architecture

AP Copilot is structured as a branched 6-node ADK 2.3.0 `Workflow` graph—linear through extraction and validation, then forking at the Policy-Validator into an `auto_post` path or a `human_review` path—threading shared state (`InvoiceState`) and decision audit logs across every execution step:

```
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌──────────────────┐
│  Intake  │ ──> │ Extractor │ ──> │ GL-Coder │ ──> │ Policy-Validator │
└──────────┘     └───────────┘     └──────────┘     └──────────────────┘
                                                             │
                                      ┌──────────────────────┴──────────────────────┐
                                      ▼ (auto_post)                                 ▼ (human_review)
                                ┌──────────┐                                ┌────────────┐
                                │  Poster  │ <───────────────────────────── │ Human Gate │ (Approved)
                                └──────────┘                                └────────────┘
                            (posts via NetSuite MCP)                         (Safety Rail)
```

### 🧩 Node Breakdown

| Node | Responsibilities | Shared State Transition |
|---|---|---|
| **Intake** | Ingests raw invoice payloads (JSON/text/scans) and initializes normalized shared state. | `raw invoice` → `normalized state` |
| **Extractor** | Reads the structured fields (vendor, total amount, date, line items, PO#) from the synthetic invoice payload and surfaces a per-field confidence score for each. | `invoice doc` → `structured fields + confidence` |
| **GL-Coder** | Maps line items to GL accounts and department codes using rules loaded from portable `SKILL.md`. | `line items` → `GL-coded entries` |
| **Policy-Validator** | Evaluates compliance: duplicate checks, PO database verification, unknown-vendor detection, low confidence (<0.85), and the **$5,000 auto-post ceiling** (≥ $5,000 routes to a human). | `coded entry` → `auto_post` vs `human_review` |
| **Human Gate** | Intercepts flagged entries via ADK `RequestInput` for interactive human triage (Approve/Reject). | `flagged entry` → `approved / rejected` |
| **Poster** | Posts approved entries through a local **NetSuite MCP server** (`mcp_server/`)—a mock NetSuite ERP exposed over a real MCP interface—and records the returned transaction ID (`NS-POST-XXXXX`). | `approved entry` → `posted GL entry` |

---

## 🎯 Competition Concepts Demonstrated (5 of 6 Named Concepts)

Four are implemented directly in code — the ADK graph, the MCP server, the Agent Skill, and the Security/HITL rails — plus Google Antigravity as the build surface: **5 of the 6** named concepts. Only Deployability is not claimed.

1. **ADK 2.3.0 Workflow Multi-Agent Graph:** Built natively using `google.adk.workflow.Workflow`, `@node` decorators, and explicit conditional routing (`Edge(from_node=..., to_node=..., route=...)`).
2. **MCP Server:** A local **NetSuite MCP server** and client (`mcp_server/`)—the Poster node posts approved entries through the real MCP interface to a mock NetSuite ERP.
3. **Agent Skill (`SKILL.md`):** Portable skill definition in `skills/ap_invoice_skill/SKILL.md` encapsulating GL mapping tables, keyword fallbacks, and policy thresholds, parsed dynamically at runtime.
4. **Security & Human-in-the-Loop Safety Rails:** Hard **$5,000 auto-post dollar ceiling**, duplicate invoice prevention, and interactive `RequestInput` human gate.
5. **Google Antigravity:** Used as the primary build surface; the round-by-round build artifacts are preserved under [`docs/`](docs/AUDIT.md).

---

## 🛠️ Installation & Setup

### Prerequisites
* Python **3.11+**
* Virtual Environment manager (`uv` or standard `venv`)

### Setup Commands
```bash
# Clone the repository
git clone https://github.com/hectorluisalamo/ap-invoice-processor.git
cd ap-invoice-processor

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies (includes the `mcp` package)
pip install -r requirements.txt
```

> **No API key required.** The demo runs entirely on 100% synthetic data and makes no live LLM calls—the Extractor consumes pre-structured synthetic fields, so there is nothing to authenticate. Just install and run.

---

## 🚀 How to Run the Demo & Evaluation

### 1. Launch the Live Web Dashboard UI
Experience the interactive visual dashboard featuring real-time node pipeline state animations, human-in-the-loop triage intervention desk, and audit logs:
```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn web.server:web_app --reload --port 8000
```
👉 Open your browser to **`http://localhost:8000`**

Run any of the bundled synthetic scenarios, or use the **Test Your Own Invoice** panel to submit a custom vendor/amount/PO/line-item and watch it run through the same graph (`POST /api/run-custom`). Inputs above the $5,000 ceiling, with an unknown vendor, or with a bad/missing PO route to the Human Gate—exactly like the pre-baked cases.

### 2. Run the Interactive CLI Demo
Run sample invoice scenarios directly in your console:
```bash
source .venv/bin/activate
PYTHONPATH=. python cli_demo.py
```

### 3. Run the Automated Evaluation Benchmark (N=50)
Execute the batch evaluation harness across 50 synthetic invoices to verify accuracy and generate the statistical ROI readout:
```bash
source .venv/bin/activate
PYTHONPATH=. python eval/eval_harness.py
```

### 4. Run Unit Tests
```bash
source .venv/bin/activate
PYTHONPATH=. pytest tests/
```

---

## 📄 Optional Document Intake Path (reader → LLM extraction → LLM GL coder)

Besides the simulated-extraction payloads above, the graph accepts a payload with a `document_path` (a PDF or image of an invoice). Payloads without `document_path` behave exactly as before.

1. **Reader** (`ap_invoice_processor/reader.py`): turns the document into text with `pdftotext`, or Tesseract OCR for scans and photos (needs `pdftotext`, `pdftoppm` and `tesseract` on `PATH`; no network).
2. **Extraction** (`ap_invoice_processor/llm/extraction.py`): the reader text goes to the configured LLM provider, which returns structured invoice fields. Field confidence on this path is presence-based (1.0 if the extraction returned a value, 0.0 if it left the field empty), not a model confidence. A document that cannot be read or extracted leaves empty fields at zero confidence, so the validator routes it to a human.
3. **GL coder** (`ap_invoice_processor/llm/gl.py`): each line is coded against the chart of accounts. A line the provider cannot code validly (off-chart account, malformed or missing entry) falls back to the keyword coder, and the decision trail records per line which coder answered (`source`: `llm` or `keyword_fallback`) and why.

**Provider.** `AP_LLM_PROVIDER` selects it: `fixture` (the default) serves hand-authored responses from `tests/fixtures/llm/`; `anthropic` calls Claude Haiku 5.5 (`claude-haiku-5-5`) through the official `anthropic` SDK, with the reply constrained to a JSON schema (structured outputs) and validated by the same extraction schema as the fixtures. Nothing makes a live call unless `AP_LLM_PROVIDER=anthropic` (or `--provider anthropic`) is set. The live provider needs `ANTHROPIC_API_KEY` and `AP_LLM_SPEND_CAP_USD`, read from the environment or from `~/.config/ap-intake/anthropic.env` (mode 600, never committed; `AP_INTAKE_ENV_FILE` points elsewhere); the file's own `AP_LLM_PROVIDER` line is ignored so holding the key never switches a run to live. Spend is tracked in a running total in `eval/out/spend.json` (gitignored). A call first reserves a conservative estimate (one per possible SDK attempt) in that total under a file lock and is refused if the total plus every pending reservation would pass the cap, so concurrent runs cannot overshoot it; the reservation is then replaced by the cost priced from the response `usage`, and it stays counted as spent when the outcome is unknown (a timeout, a dropped connection, a crash). Any failed, refused or truncated call surfaces as an extraction error (zero confidence, human review) or a keyword-coder fallback line; no value is ever filled in. The web dashboard does not surface `document_path` yet.

**Synthetic corpus.** `data/corpus/` holds synthetic invoices (English and Spanish; PDFs plus scan and photo variants) with ground truth. See `data/corpus/README.md`; scans and photos are OCR'd with English plus Spanish data by default (`AP_OCR_LANGS` overrides; `chi_sim` and `chi_tra` are accepted options).

**Offline eval.** `eval/extraction_eval.py` runs the path end to end over the corpus and scores it against ground truth:

```bash
PYTHONPATH=. python eval/extraction_eval.py
```

This is a plumbing check, not a model evaluation: the fixtures are hand-authored from reader text and no model runs, so its numbers are not model accuracy and must not be quoted as such. Details: [`eval/README_extraction_eval.md`](eval/README_extraction_eval.md).

---

## 📂 Repository Structure

```text
ap-invoice-processor/
├── ap_invoice_processor/    # Core ADK agent graph & node implementations
│   ├── graph.py             # ADK 2.3.0 Workflow graph assembly & edges
│   ├── nodes.py             # Intake, Extractor, GL-Coder, Validator, Gate, Poster nodes
│   ├── models.py            # Pydantic shared state models (InvoiceState, DecisionStep)
│   └── skill_loader.py      # Portable SKILL.md rules loader
├── mcp_server/              # Local NetSuite MCP server & client (mock ERP, real MCP interface)
│   ├── netsuite_mcp_server.py  # MCP server exposing the mock NetSuite posting tool
│   └── netsuite_mcp_client.py  # MCP client the Poster node uses to post entries
├── skills/
│   └── ap_invoice_skill/
│       └── SKILL.md         # Portable agent skill (GL mapping & policy thresholds)
├── data/                    # Master tables & synthetic dataset generator
│   ├── gl_chart_of_accounts.json
│   ├── vendor_master.json
│   ├── po_database.json
│   ├── historical_invoices.json
│   ├── synthetic_invoices/      # Generated invoice set the runners load (invoices.json)
│   └── generate_synthetic_data.py
├── web/                     # FastAPI dashboard backend & web UI
│   ├── server.py            # FastAPI server with session & triage endpoints
│   └── static/              # HTML/CSS/JS frontend files (index.html, style.css, app.js)
├── eval/
│   └── eval_harness.py      # Batch benchmark script (N=50) & ROI readout
├── tests/                   # Pytest unit testing suite
├── docs/                    # Project specifications & Kaggle writeup draft
├── cli_demo.py              # Terminal interactive demo script
└── README.md                # Project documentation
```

---

## ⚖️ Development, Validation & License

* **Development & Validation:** Built with Google Antigravity, then independently audited and remediated with Claude Code across multiple fresh-context review rounds (confidentiality, rules-compliance, security, code/architecture, and evaluation integrity). The full round-by-round trail is in [`docs/AUDIT.md`](docs/AUDIT.md).
* **Commercial Tools Used:** Google Antigravity (Gemini-powered build surface), Google Agent Development Kit (ADK 2.3.0), and Claude Code. The agent itself makes no live LLM/API calls — see the keyless note above.
* **Data:** 100% synthetic—no real vendor, invoice, or financial data is used.
* **License:** Code is licensed under the [Apache 2.0 License](LICENSE). Per competition terms, the submission content is offered under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
