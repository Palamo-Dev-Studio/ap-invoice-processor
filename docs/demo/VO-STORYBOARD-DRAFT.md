DRAFT for Hector's rewrite — generated 2026-10-07 by Nico (Night Shift). No figures are final.

# AP invoice intake: demo VO + storyboard

- **Purpose:** show a concrete invoice-intake pipeline with a human approval step, as a Palamo pitch demo.
- **Length:** about 80 s vertical (9:16): 75 s of content plus the 5 s PALAMO + palamo.ai end card.
- **Audience:** owners and controllers at SMBs drowning in AP.
- **Angle:** the pipeline's steps, not a claim about "AI".

## Beat table

| # | time | on-screen | VO draft | notes |
|---|---|---|---|---|
| 1 | 0-6 s | Motion graphic: a stack of invoices (EN and ES, a few look like phone photos or scans) slides in. Lower-third: "All data shown is synthetic." | "This is a normal week in accounts payable. A pile of invoices. Some in Spanish. Some photographed on a desk." | Synthetic corpus built tonight. The lower-third appears once, here. |
| 2 | 6-15 s | Screen recording: drop a folder of invoices into intake. One PDF opens, then its extracted text appears beside it. Then a scan: the OCR text appears, with a few damaged characters. | "First, the system reads each document. Text from the PDF. For a scan or a photo, it reads the image." | Reader layer (pdftotext, OCR). No UI for this exists yet: needs a simple viewer. Scans and photos are read with `eng+spa` OCR, so Spanish accents mostly survive. The offline fixtures were written from earlier `-l eng` text. |
| 3 | 15-26 s | Recording: the invoice beside a field panel. Vendor, date, PO number and line items fill in; a field it could not read is flagged empty. | "Then it pulls the fields: vendor, date, PO number, each line item. Anything it couldn't read is flagged empty, so a person knows what to check." | Confidence today is presence-based (a field is 1.0 if the extraction returned a value, 0.0 if it left it null); it is not a model confidence, so show filled versus empty fields, not a score. The dashboard's Extractor node reads pre-structured fields, so the recording must come from the document path. See open question 1. |
| 4 | 26-37 s | Recording: each line gets a proposed GL code and a one-line reason in the decision trail. Cut to a second line the model was unsure about, flagged. | "Next, it proposes a GL code for each line, and says why. When it isn't sure, it says so." | GL coder behind the provider interface. On the document path the GL-Coder node records, per line, which coder answered (`source`: `llm` or `keyword_fallback`) and its reason in the decision trail; a line the LLM coder cannot code validly falls back to the keyword coder. Fixture-backed tonight, so the recording shows fixture responses. |
| 5 | 37-46 s | Recording: the Validator node lights up. Checks tick through: vendor master, PO match, duplicate check, the $5,000 ceiling. Label on the ERP panel: "MOCK ERP, synthetic." | "Then it checks the vendor, the PO, and whether you've seen this invoice before. Anything over the ceiling goes to a person." | Mock NetSuite connector. Say "mock" on screen. The $5,000 ceiling is the existing policy rule. |
| 6 | 46-60 s | Recording: one invoice stops at the Human Gate. The triage desk opens with the flag "unknown vendor." Hector's cursor types a short reason and clicks Approve. | "One invoice stops here. Unknown vendor. The system doesn't guess. A person looks, writes a reason, and approves it." | The dashboard already has this triage desk. Hector's cursor, recorded live. |
| 7 | 60-68 s | Recording: the Poster node lights up. A transaction ID appears in the mock ERP list. The decision trail scrolls: every step, with its reason. | "It posts, and every step is on the record. What it read, what it coded, who approved." | Posts over the MCP server to the mock ERP. |
| 8 | 68-75 s | Motion graphic over the eval screen: extraction and GL coding scored against an answer key. Number slot: [FIGURE — Hector to supply after live eval]. | "We score it against a known answer key, so you can see where it's right and where it isn't." | Offline eval built tonight is fixture-backed. Show no figure unless a live run supports it. |
| 9 | 75-80 s | End card: PALAMO, palamo.ai. Music resolves. | (none) | 5 s end card, per Hector's rule. |

**16:9 variant:** same beats, same VO, about 90 s with a longer beat 6 (about 8 s more). For 9:16, crop to one panel per beat: the invoice/field panel in beats 2-4, the node graph in beat 5, the triage desk in beat 6.

## Open questions for Hector

> **Gate:** no recording until Rocio has tested the demo and given her OK (Hector's standing rule, 2026-10-07).

- Will the extraction and GL coding run live on camera? The live provider (Claude Haiku 5.5), the key and the spend cap are now in place, so a live recording can say "the model." The first live eval ran on 2026-10-08 against the synthetic corpus.
- Does the dashboard header's cost/ROI banner come off for recording? It is hard-coded in `web/static/index.html` and must not appear on screen. The GL-Coder node subtitle ("SKILL.md Rules") must also match the new coder.
- Should a Spanish invoice be on screen in beats 2-3? Spanish OCR data is now installed and used by default (`eng+spa`).
- What belongs in the eval beat's number slot? Left as [FIGURE — Hector to supply after live eval].

## Assets needed

- Synthetic invoice corpus: EN and ES, including phone-photo and scan variants (built tonight).
- A reader view showing the extracted or OCR text beside the document (not built).
- Extraction layer and GL coder: wired into the graph and nodes through the `document_path` intake payload (built tonight). The web dashboard does not surface `document_path` yet (`web/server.py` builds only simulated-extraction payloads), so the dashboard view for the document path is not built.
- A dashboard variant with the ROI banner removed or masked.
- "MOCK ERP, synthetic" label on the ERP panel.
- An eval view or motion-graphic card for beat 8 (figure is a placeholder).
- Hector VO takes (his rewrite); music bed; PALAMO end card (5 s) with palamo.ai.
