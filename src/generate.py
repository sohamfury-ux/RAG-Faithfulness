"""
Phase 7 — Grounded answer generation with inline citations.

Two steps (so the 4GB GPU never holds retrievers AND the LLM at once):
  py -3.8 src/generate.py retrieve            # contexts for every retriever -> data/contexts.json
  py -3.8 src/generate.py generate rerank 5   # answer N questions using one retriever's contexts
  py -3.8 src/generate.py generate all        # all retrievers x all questions

Design choices worth knowing:
  * Sources are numbered [S1]..[S5]; the model cites by number and we map back to
    (act, section). Numbers are short and easy for a small model to copy.
  * The model is told to say NOT_IN_SOURCES instead of guessing. Whether it obeys is
    itself something we measure on the unanswerable questions.
  * Greedy decoding (do_sample=False): same input -> same output, so differences
    between retrievers come from the CONTEXT, not sampling noise.
  * Each answer is parsed into CLAIMS = (sentence, [cited sections]). Phase 8 scores
    each claim against the sections it cites: citation-level faithfulness.
"""
import json
import re
import sys
import time

from common import ROOT, citation, load_chunks
from validate_evalset import load_eval

CTX = ROOT / "data" / "contexts.json"
ANS = ROOT / "data" / "answers_{}.jsonl"
LLM = "Qwen/Qwen2.5-1.5B-Instruct"
NAMES = ["bm25", "dense", "hybrid", "rerank"]
K_SOURCES = 5            # sections shown to the generator
MAX_WORDS = 220          # per source, keeps the prompt inside the 4GB budget

SYSTEM = (
    "You are a careful assistant answering questions about Indian statutes. "
    "Answer using ONLY the numbered sources. After every sentence, cite the supporting "
    "source(s) like [S1] or [S2][S3]. Do not use outside knowledge and do not cite a source "
    "that does not support the sentence. If the sources do not contain the answer, reply "
    "exactly: NOT_IN_SOURCES. Keep the answer under 120 words."
)


# ----------------------------------------------------------------- step 1
def retrieve_all():
    from hybrid import build_all
    from common import to_citations
    chunks = load_chunks()
    text = {c["chunk_id"]: c["text"] for c in chunks}
    R = build_all(chunks)
    out = {n: {} for n in NAMES}
    for it in load_eval():
        for n in NAMES:
            ranked, seen, ctx = R[n].search(it["question"], 30), set(), []
            for cid, _ in ranked:               # best chunk of each of the top-K sections
                c = citation(cid)
                if c in seen:
                    continue
                seen.add(c)
                words = text[cid].split()
                ctx.append({"sid": len(ctx) + 1, "citation": c, "chunk_id": cid,
                            "text": " ".join(words[:MAX_WORDS])})
                if len(ctx) == K_SOURCES:
                    break
            out[n][it["id"]] = ctx
    CTX.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved contexts for {len(NAMES)} retrievers x {len(load_eval())} questions -> {CTX}")


# ----------------------------------------------------------------- step 2
def build_prompt(question, ctx):
    srcs = "\n\n".join(f"[S{s['sid']}] ({s['citation']}) {s['text']}" for s in ctx)
    return f"Sources:\n{srcs}\n\nQuestion: {question}\nAnswer:"


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
CITE = re.compile(r"\[S(\d+)\]")                       # requested format:  [S2]
LABEL = re.compile(r"\[([A-Z]+\d{4} s\.[0-9A-Za-z]+)\]")  # model's common slip: [RTI2005 s.19]
# models often write "... hours. [S1]" (citation AFTER the period). Move it inside the sentence.
TRAILING = re.compile(r"([.!?])\s*((?:\[(?:S\d+|[A-Z]+\d{4} s\.[0-9A-Za-z]+)\]\s*)+)")


def parse_claims(answer, ctx):
    """answer -> [{'sentence', 'cited': [citations], 'bad_ids': [...], 'style': ...}]
    Lenient about FORMAT (we measure faithfulness, not formatting) but records the slip:
      style = 'S-number' | 'label' | 'none'"""
    by_sid = {s["sid"]: s["citation"] for s in ctx}
    valid = set(by_sid.values())
    text = TRAILING.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", answer.strip())
    claims = []
    for sent in SENT.split(text):
        ids = [int(x) for x in CITE.findall(sent)]
        labels = LABEL.findall(sent)
        clean = LABEL.sub("", CITE.sub("", sent)).strip()
        if not clean:
            continue
        cited = {by_sid[i] for i in ids if i in by_sid} | {l for l in labels if l in valid}
        claims.append({"sentence": clean, "cited": sorted(cited),
                       "bad_ids": [i for i in ids if i not in by_sid]
                                  + [l for l in labels if l not in valid],
                       "style": "S-number" if ids else "label" if labels else "none"})
    return claims


def load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM)
    model = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16).to("cuda").eval()
    return tok, model


def ask(tok, model, question, ctx, max_new=200):
    import torch
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, ctx)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    n_in = enc["input_ids"].shape[1]
    return tok.decode(out[0, n_in:], skip_special_tokens=True).strip(), n_in


def generate(which, limit=None):
    contexts = json.loads(CTX.read_text(encoding="utf-8"))
    tok, model = load_llm()
    items = load_eval()[:limit] if limit else load_eval()
    for name in (NAMES if which == "all" else [which]):
        rows, t0 = [], time.time()
        for it in items:
            ctx = contexts[name][it["id"]]
            answer, n_in = ask(tok, model, it["question"], ctx)
            rows.append({"id": it["id"], "retriever": name, "question": it["question"],
                         "answer": answer, "abstained": "NOT_IN_SOURCES" in answer,
                         "prompt_tokens": n_in, "context": ctx,
                         "claims": [] if "NOT_IN_SOURCES" in answer else parse_claims(answer, ctx)})
        path = str(ANS).format(name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{name}] {len(rows)} answers in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "retrieve"
    if cmd == "retrieve":
        retrieve_all()
    else:
        generate(sys.argv[2] if len(sys.argv) > 2 else "rerank",
                 int(sys.argv[3]) if len(sys.argv) > 3 else None)
