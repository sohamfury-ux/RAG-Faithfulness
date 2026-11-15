"""
Phase 1 — Ingestion: PDF -> clean text -> section-aware chunks with citation IDs.

Why section-aware chunking?
  In legal text the *section* is the natural unit of meaning AND the natural unit
  of citation ("RTI Act s.7"). If we chunk by fixed 500-char windows, a chunk can
  straddle two sections, and then "which source supports this claim?" becomes
  ambiguous. Faithfulness at citation level needs crisp citation units.

Output: data/chunks.jsonl, one JSON per chunk:
  {chunk_id, doc, section, title, text}
  chunk_id = "<DOC>-s<section>#<part>", e.g. "RTI2005-s7#0"
  The CITATION unit is (doc, section). Long sections are split into parts, but
  every part keeps the same citation.

Run:  py -3.8 src/ingest.py
"""
import json
import re
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "chunks.jsonl"

DOCS = {
    "rti_act_2005": "RTI2005",
    "it_act_2000": "IT2000",
    "consumer_protection_act_2019": "CPA2019",
    "dpdp_act_2023": "DPDP2023",
}

MAX_WORDS = 250   # chunk size for splitting long sections
OVERLAP = 50      # words of overlap between parts of one section


# ---------------------------------------------------------------- cleaning
NOISE_LINE = re.compile(
    r"^\s*("
    r"\d{1,3}"                                   # bare page numbers
    r"|SEC\.\s*\d+\].*"                          # gazette running header
    r".*THE GAZETTE OF INDIA EXTRAORDINARY.*"
    r"|Right to Information Act[.,] 2005"        # RTI running header
    r"|\(Chapter .*\)"
    r")\s*$",
    re.IGNORECASE,
)
# amendment footnotes, e.g. "5. Clause (n) omitted by Act 7 of 2017, s.169 (w.e.f. 26-5-2017)."
FOOTNOTE = re.compile(r"^\s*\d+\.\s*((Subs\.|Ins\.|Omitted|Added|The words|Cl\.|Sub-section)|.*(by Act\s*\d+\s*of\s*\d{4}|w\.e\.f))")


def clean_text(raw: str) -> str:
    lines = []
    for ln in raw.splitlines():
        if NOISE_LINE.match(ln) or FOOTNOTE.match(ln):
            continue
        lines.append(ln.rstrip())
    t = "\n".join(lines)
    t = re.sub(r"\d+\[", "[", t)            # IT Act amendment markers: "1[electronic signatures]"
    t = t.replace("�", '"')            # broken quote glyphs in CPA
    t = re.sub(r"\(/\)", "(1)", t)          # OCR: "(/)" -> "(1)"  (RTI scan)
    t = re.sub(r"[ \t]+", " ", t)
    return t


# ------------------------------------------------------------- sectioning
# A candidate section header: line starts with "12." or "66A." followed by text.
# Optional leading "[" = section substituted by an amendment, e.g. "6[66. Computer related offences".
HEADER = re.compile(r"^\s*\[?(\d{1,3})([A-Z]{0,2})\.\s*(.*)$")
BODY_START = re.compile(r"enacted by Parliament", re.IGNORECASE)
# DPDP Act: pypdf emits the penalty TABLE *before* the "THE SCHEDULE" heading, so the
# heading rule alone leaves the table glued onto s.44. Detect the table's first row.
SCHEDULE_TABLE = re.compile(r"^\s*Breach of provisions of\s+this Act or rules made thereunder", re.IGNORECASE)
# Schedule headings become their OWN citation units (they hold real answers: the IT Act's
# excluded documents, the RTI Act's exempt agencies). Optional "[" = amendment marker
# ("1[THE FIRST SCHEDULE") — missing it once glued the IT First Schedule onto s.91.
SCHED_HEAD = re.compile(r"^\s*\[?\s*THE\s+(?:(FIRST|SECOND|THIRD|FOURTH)\s+)?SCHEDULE\b", re.IGNORECASE)
ORDINAL = {"FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4"}


def is_next_section(num, suf, prev):
    """Monotonic filter: real sections appear in order 1,2,3,3A,4...
    This rejects table-of-contents echoes, footnotes and schedule list items."""
    pnum, psuf = prev
    if num == pnum and suf > psuf:          # 66 -> 66A -> 66B
        return True
    return pnum < num <= pnum + 3          # allow small gaps (missed/omitted sections)


def split_sections(text: str):
    lines = text.splitlines()
    # skip preamble + table of contents
    start = next((i for i, l in enumerate(lines) if BODY_START.search(l)), 0)
    sections, cur, prev = [], None, (0, "")
    for ln in lines[start + 1:]:
        hm = SCHED_HEAD.match(ln)
        if hm and prev[0] > 5:
            if cur and cur["section"] == "Schedule" and not hm.group(1):
                break      # DPDP: the heading arrives AFTER its table; what follows is signatures
            if cur:
                sections.append(cur)
            n = ORDINAL.get((hm.group(1) or "").upper(), "")
            cur = {"section": f"Schedule{n}", "lines": [ln],
                   "title": f"{(hm.group(1) or '').title()} Schedule".strip()}
            continue
        if SCHEDULE_TABLE.match(ln) and prev[0] > 5 and cur and cur["section"] != "Schedule":
            sections.append(cur)
            cur = {"section": "Schedule", "title": "Schedule (table of penalties)", "lines": [ln]}
            continue
        m = HEADER.match(ln)
        if m and m.group(3)[:1] and (m.group(3)[0].isupper() or m.group(3)[0] in "(["):
            num, suf = int(m.group(1)), m.group(2)
            if is_next_section(num, suf, prev):
                if cur:
                    sections.append(cur)
                rest = m.group(3)
                # India Code style "Title.—body"; gazette style has no title on the line
                tm = re.match(r"(.{3,160}?)\s*[.]?\s*[—–-]{1,2}\s*(.*)", rest)
                title = tm.group(1).strip(" .[]") if tm and not rest.startswith("(") else ""
                cur = {"section": f"{num}{suf}", "title": title, "lines": [rest]}
                prev = (num, suf)
                continue
        if cur:
            cur["lines"].append(ln)
    if cur:
        sections.append(cur)
    return sections


def chunk_section(doc, sec):
    text = re.sub(r"\s+", " ", " ".join(sec["lines"])).strip()
    words = text.split()
    title = sec["title"] or " ".join(words[:10])
    parts, i = [], 0
    while True:
        parts.append(" ".join(words[i:i + MAX_WORDS]))
        if i + MAX_WORDS >= len(words):
            break
        i += MAX_WORDS - OVERLAP
    for k, p in enumerate(parts):
        yield {
            "chunk_id": f"{doc}-s{sec['section']}#{k}",
            "doc": doc,
            "section": sec["section"],
            "title": title,
            # prepend doc+title: cheap "contextual chunking" — helps BM25 and dense alike
            "text": f"[{doc} s.{sec['section']} {title}] {p}",
        }


def main():
    all_chunks = []
    for stem, doc in DOCS.items():
        reader = pypdf.PdfReader(str(RAW / f"{stem}.pdf"))
        raw = "\n".join(p.extract_text() or "" for p in reader.pages)
        secs = split_sections(clean_text(raw))
        chunks = [c for s in secs for c in chunk_section(doc, s)]
        all_chunks += chunks
        print(f"{doc:9s} sections={len(secs):3d} chunks={len(chunks):4d} "
              f"last_section={secs[-1]['section'] if secs else None}")
    with open(OUT, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_chunks)} chunks -> {OUT}")


if __name__ == "__main__":
    main()
