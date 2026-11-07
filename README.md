# RAG Literature Gap Matrix — ChromaDB

Prototype for a master's-level literature review of RAG applications in food manufacturing quality.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # On Windows, copy manually
```

Set `OPENAI_API_KEY` in `.env` (required for generating matrices; indexing/search work without it).

Put **full-text, text-readable PDF papers** into `papers/`. Rename files to DOI or title if useful. The prototype does not download publisher PDFs or claim any results before the PDFs are supplied.

```bash
python rag_pipeline.py index
python rag_pipeline.py search "limitations of RAG for food manufacturing quality" --top-k 5
python rag_pipeline.py generate
```

Outputs: `outputs/Literature_Gap_Matrix.xlsx` with Literature Matrix, Research Gap Matrix, and Evidence sheets, plus `outputs/literature_matrix.json` and `outputs/research_gaps.json`.

## Important safeguards

- The PDF text must be readable; scanned-only PDFs need separate OCR.
- The PDF filename is not accepted as proof of DOI; DOI is extracted from the first pages and must be verified by the researcher.
- Extraction is scoped to each paper's chunks to avoid mixing papers.
- Each field stores a source quote and page reference. Unverified fields are flagged.
- Cross-paper gaps are **hypotheses**, not proof of absence in all published literature.
- Human review of every generated claim and DOI is mandatory before submitting an academic paper.
- The system uses remote LLM inference when `generate` is run. Do not upload confidential papers without appropriate authorization.
