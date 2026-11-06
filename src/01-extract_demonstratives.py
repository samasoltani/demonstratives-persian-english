"""
01-extract_demonstratives.py
Extract every candidate demonstrative (in / un / an and their derived forms)
from the Persian CHAT (.cha) files and save one row per token as a CSV.

Input : data/raw/persian/**/*.cha
Output: outputs/01_extraction_summary.txt (summary of the run)
        data/interim/persian_demonstratives.csv
        data/interim/persian_utterance_counts.csv  (denominators for later rates)

Run from PyCharm (no need to set a working directory):
    python src/01-extract_demonstratives.py
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "persian"
OUT_DIR = ROOT / "data" / "interim"

# ---- demonstrative forms (romanized transcription: in = this, un/an = that) ----
# stem + optional suffix. "inke" etc. are kept as candidates; they will get the
# label "non-deictic" in the annotation step.
STEM = r"(?:in|im|un|an)"
SUFFIX = (r"(?:ja|jur|juri|jast|jayi|jan|jaro|a|aha|ahash|haha|ha|aro|amo|"
          r"o|i|e|ke|yeki|æm|ash)?")
DEM_RE = re.compile(rf"^(?P<hem>hæm)?(?P<stem>{STEM})(?P<suf>{SUFFIX})$")

TIMESTAMP_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
BRACKET_RE = re.compile(r"\[[^\]]*\]")      # [/] [= ...] [>] etc.
PUNCT_RE = re.compile(r"[<>(),.?!;:\"“”]")


def join_continuation_lines(text):
    """CHAT lines that start with a tab continue the previous line."""
    lines = []
    for line in text.splitlines():
        if line.startswith("\t") and lines:
            lines[-1] += " " + line.strip()
        else:
            lines.append(line)
    return lines


def parse_header(lines):
    info = {"child": "", "age": "", "date": "", "activity": "", "roles": {}}
    for line in lines:
        if line.startswith("@Participants:"):
            parts = line.split("\t", 1)[1].split(",")
            for p in parts:
                bits = p.strip().split()
                if len(bits) >= 3:
                    code, name, role = bits[0], bits[1], bits[2]
                    info["roles"][code] = role
                    if role == "Target_Child":
                        info["child"] = name
        elif line.startswith("@ID:"):
            f = line.split("\t", 1)[1].split("|")
            if len(f) > 3 and f[2] == "CHI":
                info["age"] = f[3]
        elif line.startswith("@Date:"):
            info["date"] = line.split("\t", 1)[1].strip()
        elif line.startswith("@Activities:"):
            info["activity"] = line.split("\t", 1)[1].strip()
    return info


def clean_utterance(raw):
    ts = TIMESTAMP_RE.search(raw)
    start, end = (ts.group(1), ts.group(2)) if ts else ("", "")
    text = TIMESTAMP_RE.sub("", raw)
    text = BRACKET_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    tokens = text.split()
    return tokens, start, end


def read_utterances(lines):
    """Return list of dicts: speaker, tokens, start, end, eng, com."""
    utts = []
    for line in lines:
        if line.startswith("*"):
            code, rest = line[1:].split(":", 1)
            tokens, start, end = clean_utterance(rest.strip())
            utts.append({"speaker": code, "tokens": tokens, "start": start,
                         "end": end, "eng": "", "com": ""})
        elif line.startswith("%eng:") and utts:
            utts[-1]["eng"] = line.split("\t", 1)[1].strip()
        elif line.startswith("%com:") and utts:
            utts[-1]["com"] = line.split("\t", 1)[1].strip()
    return utts


def classify(match):
    stem, suf = match.group("stem"), match.group("suf") or ""
    distance = "proximal" if stem in ("in", "im") else "distal"
    if suf.startswith("ja"):
        form_type = "locative (-ja)"
    elif suf.startswith("jur"):
        form_type = "manner (-jur/-juri)"
    elif suf in ("a", "aha", "ahash", "ha", "haha"):
        form_type = "plural (-a/-aha)"
    elif suf == "ke":
        form_type = "complementizer-like (-ke)"
    elif suf == "":
        form_type = "bare"
    else:
        form_type = "other suffix"
    return distance, form_type


def main():
    files = sorted(RAW_DIR.rglob("*.cha"))
    if not files:
        raise SystemExit(f"No .cha files found in {RAW_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, counts = [], []
    for path in files:
        text = path.read_text(encoding="utf-8")
        lines = join_continuation_lines(text)
        info = parse_header(lines)
        utts = read_utterances(lines)

        n_tokens = Counter()
        n_utts = Counter()
        for u in utts:
            n_utts[u["speaker"]] += 1
            n_tokens[u["speaker"]] += len(u["tokens"])
        for spk in n_utts:
            counts.append({"file": path.name, "child": info["child"],
                           "child_age": info["age"], "speaker": spk,
                           "role": info["roles"].get(spk, ""),
                           "n_utterances": n_utts[spk],
                           "n_words": n_tokens[spk]})

        for i, u in enumerate(utts):
            for j, tok in enumerate(u["tokens"]):
                m = DEM_RE.match(tok)
                if not m:
                    continue
                distance, form_type = classify(m)
                rows.append({
                    "file": path.name,
                    "child": info["child"],
                    "child_age": info["age"],
                    "date": info["date"],
                    "activity": info["activity"],
                    "utt_index": i,
                    "speaker": u["speaker"],
                    "role": info["roles"].get(u["speaker"], ""),
                    "token": tok,
                    "distance": distance,
                    "form_type": form_type,
                    "token_position": j + 1,
                    "utt_length": len(u["tokens"]),
                    "utterance": " ".join(u["tokens"]),
                    "eng_gloss": u["eng"],
                    "comment": u["com"],
                    "prev_utterance": " ".join(utts[i - 1]["tokens"]) if i > 0 else "",
                    "prev_speaker": utts[i - 1]["speaker"] if i > 0 else "",
                    "next_utterance": " ".join(utts[i + 1]["tokens"]) if i + 1 < len(utts) else "",
                    "time_start_ms": u["start"],
                    "time_end_ms": u["end"],
                    # to be filled in step 04 (annotation):
                    "function_label": "",
                })

    with open(OUT_DIR / "persian_demonstratives.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(OUT_DIR / "persian_utterance_counts.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(counts[0].keys()))
        w.writeheader()
        w.writerows(counts)

    # ---- summary: printed AND saved so it can be pushed to GitHub ----
    summary = [
        f"Files read: {len(files)}",
        f"Demonstrative tokens: {len(rows)}",
        "",
        f"By distance: {dict(Counter(r['distance'] for r in rows))}",
        f"By form type: {dict(Counter(r['form_type'] for r in rows))}",
        f"By speaker role: {dict(Counter(r['role'] for r in rows))}",
        f"By child: {dict(Counter(r['child'] for r in rows))}",
        "",
        "Top forms:",
    ]
    for tok, n in Counter(r["token"] for r in rows).most_common(25):
        summary.append(f"  {tok}: {n}")
    summary.append("")
    summary.append("Files:")
    for path in files:
        summary.append(f"  {path.name}")
    text_out = "\n".join(summary)
    print(text_out)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "01_extraction_summary.txt").write_text(text_out, encoding="utf-8")
    print(f"\nSaved to: {OUT_DIR} and outputs/01_extraction_summary.txt")


if __name__ == "__main__":
    main()
