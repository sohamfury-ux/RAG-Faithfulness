"""Labeling helper: print section text so gold labels come from the SOURCE, not memory.

  py -3.8 src/show_sections.py "RTI2005 s.4" "IT2000 s.66F" --words 120
  py -3.8 src/show_sections.py --grep "forty-five days"        # which sections contain a phrase
"""
import re
import sys
from collections import defaultdict

from common import citation, load_chunks


def section_texts():
    t = defaultdict(list)
    for c in load_chunks():
        t[citation(c)].append(c["text"].split("]", 1)[1].strip())
    return {k: " ".join(v) for k, v in t.items()}


if __name__ == "__main__":
    args, words = sys.argv[1:], 120
    if "--words" in args:
        i = args.index("--words"); words = int(args[i + 1]); args = args[:i] + args[i + 2:]
    T = section_texts()
    if args and args[0] == "--grep":
        pat = re.compile(args[1], re.I)
        for k, v in T.items():
            for m in pat.finditer(v):
                print(f"{k:16s} ...{v[max(0, m.start()-90):m.end()+90]}...")
    else:
        for c in args:
            w = T.get(c, "<<NOT IN CORPUS>>").split()
            print(f"--- {c} ({len(w)}w): {' '.join(w[:words])}\n")
