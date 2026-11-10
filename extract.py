#!/usr/bin/env python3
"""
extract.py — schema-constrained invoice extraction with a local LLM via Ollama.

  extract.py DIR [--dsn DSN]        extract invoice PDFs in DIR -> Postgres
  extract.py DIR --dry-run          extract and print results, no database
  extract.py DIR --model MODEL      override the default model
"""
import argparse
import json
import os
import pathlib
import sys
import time
from datetime import date

import requests
from pydantic import BaseModel, ValidationError, field_validator

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"
EXTRACT_SYSTEM = (
    "Extract invoice fields exactly as written in the document. "
    "Use null if a field is absent. Do not guess or infer values."
)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def structured_call(messages: list, schema: dict, model: str) -> tuple[str, int]:
    """Non-streaming call with schema-constrained output. Returns (raw_json, latency_ms)."""
    t0 = time.perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema,  # Ollama constrains decoding to this JSON schema
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if response.status_code != 200:
        error = response.json().get("error", response.text)
        raise RuntimeError(f"Ollama error: {error}")
    return response.json()["message"]["content"], latency_ms


# ---------------------------------------------------------------------------
# Schema, validation, business rules
# ---------------------------------------------------------------------------

class InvoiceLine(BaseModel):
    description: str
    quantity: float
    unit_price: float
    line_total: float


class Invoice(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: date
    po_number: str | None
    lines: list[InvoiceLine]
    total: float

    @field_validator("total")
    @classmethod
    def total_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("total must be positive")
        return v


def business_checks(inv: Invoice, tolerance: float = 0.05) -> list[str]:
    """Rules the schema can't express. Any hit routes the document to human review."""
    issues = []
    if not inv.lines:
        issues.append("no line items")
    line_sum = round(sum(line.line_total for line in inv.lines), 2)
    if abs(line_sum - inv.total) > tolerance:
        issues.append(f"line sum {line_sum} != total {inv.total}")
    for line in inv.lines:
        expected = round(line.quantity * line.unit_price, 2)
        if abs(expected - line.line_total) > tolerance:
            issues.append(f"line math off: {line.description} ({expected} vs {line.line_total})")
    if not inv.po_number:
        issues.append("missing PO number")
    return issues


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def pdf_text(pdf_path: pathlib.Path) -> str:
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)


def extract(pdf_path: pathlib.Path, model: str) -> dict:
    record = {
        "file_name": pdf_path.name,
        "model": model,
        "status": "failed",
        "latency_ms": None,
        "issues": [],
        "raw_json": None,
        "payload": None,
    }

    try:
        text = pdf_text(pdf_path)
    except Exception as e:
        record["issues"] = [f"pdf read error: {e}"]
        return record

    if not text.strip():
        record["issues"] = ["no extractable text (scanned PDF? needs OCR)"]
        return record

    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": text},
    ]

    try:
        raw, latency_ms = structured_call(messages, Invoice.model_json_schema(), model)
    except (requests.RequestException, RuntimeError) as e:
        record["issues"] = [f"model call failed: {e}"]
        return record

    record["raw_json"] = raw
    record["latency_ms"] = latency_ms

    try:
        inv = Invoice.model_validate_json(raw)
    except ValidationError as e:
        record["issues"] = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        return record

    record["payload"] = inv.model_dump(mode="json")
    record["issues"] = business_checks(inv)
    record["status"] = "ok" if not record["issues"] else "review"
    return record


def load(rows: list[dict], dsn: str) -> None:
    import psycopg
    from psycopg.types.json import Jsonb

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """
                insert into ai.doc_extraction
                    (file_name, model, status, latency_ms, issues, raw_json, payload)
                values (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    r["file_name"],
                    r["model"],
                    r["status"],
                    r["latency_ms"],
                    Jsonb(r["issues"]),
                    r["raw_json"],
                    Jsonb(r["payload"]) if r["payload"] is not None else None,
                ),
            )


def run_extract(directory: pathlib.Path, model: str, dsn: str | None, dry_run: bool) -> None:
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {directory}")
        return

    rows = []
    for i, pdf in enumerate(pdfs, 1):
        r = extract(pdf, model)
        rows.append(r)
        latency = f"{r['latency_ms']} ms" if r["latency_ms"] is not None else "-"
        print(f"[{i}/{len(pdfs)}] {r['status']:<6} {latency:>9}  {pdf.name}")
        for issue in r["issues"]:
            print(f"           - {issue}")

    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "review", "failed")}
    print(f"\nok: {counts['ok']}  review: {counts['review']}  failed: {counts['failed']}  "
          f"straight-through: {counts['ok'] / len(rows):.0%}")

    if dry_run:
        print(json.dumps(rows, indent=2, default=str))
        return

    if not dsn:
        sys.exit("No DSN: pass --dsn or set DATABASE_URL (or use --dry-run).")
    load(rows, dsn)
    print(f"Loaded {len(rows)} rows into ai.doc_extraction.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract invoice PDFs to Postgres via a local LLM.")
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_extract(args.directory, args.model, args.dsn, args.dry_run)


if __name__ == "__main__":
    main()
