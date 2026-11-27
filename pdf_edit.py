#!/usr/bin/env python3
"""Format-preserving PDF text editing for Athena AI (uses PyMuPDF / fitz).

The old editor extracted plain text and re-set it in a brand-new PDF, so
every trace of the original layout was lost. This helper edits the ORIGINAL
PDF in place: only the words the user changed are redacted and re-inserted
at the same position and size. Everything else (tables, fonts, layout,
images, line art) is untouched, and the result is saved as a NEW file.

Modes:
  extract <pdf>
      Prints JSON: {"text": <visual-order text>,
                    "words": [{"page":p, "x0":..,"y0":..,"x1":..,"y1":..,
                               "word":str, "size":float, "font":str}, ...]}
      Words are in the SAME order as the tokens of "text" (visual reading
      order: lines top-to-bottom, words left-to-right).
  edit <pdf> <edited_text_file> <out_pdf>
      Re-extracts the original words from <pdf>, diffs the original token
      sequence against the edited text, redacts changed words in the
      ORIGINAL pdf and inserts replacements at the same positions/sizes,
      saves a NEW pdf at <out_pdf>. Prints JSON:
      {"changes":[{"type":"replace","old":"60","new":"61","page":0}, ...],
       "out": <out_pdf>, "size": <bytes>}
      Exit 0 on success (even with zero changes), non-zero with an error
      message on stderr on failure.
"""

import difflib
import json
import sys

import fitz

LINE_TOL = 3.0  # points: words whose tops are within this are one line
MIN_FONT = 5.0  # never shrink replacement text below this size


def _span_size_and_font(page, word):
    """Find the text span containing the word centre; return (size, font)."""
    cx = (word[0] + word[2]) / 2.0
    cy = (word[1] + word[3]) / 2.0
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
                    return float(span["size"]), str(span["font"])
    # Fallback: font size is roughly the word bbox height / 1.2.
    return max(6.0, (word[3] - word[1]) / 1.2), ""


def _fontname(span_font):
    """Map an original span font name to the closest built-in PDF font."""
    f = (span_font or "").lower()
    if "bold" in f:
        return "hebo"
    if "times" in f or "tiro" in f:
        return "tiro"
    if "courier" in f or "cour" in f:
        return "cour"
    return "helv"


def page_words(page, pageno):
    """All words on a page in visual reading order, with size and font."""
    raw = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,wordno)
    enriched = []
    for w in raw:
        size, font = _span_size_and_font(page, w)
        enriched.append({
            "page": pageno,
            "x0": float(w[0]), "y0": float(w[1]),
            "x1": float(w[2]), "y1": float(w[3]),
            "word": str(w[4]), "size": size, "font": font,
        })
    # Group into lines by top (tolerance), then sort lines top-to-bottom
    # and words within a line left-to-right.
    enriched.sort(key=lambda w: (w["y0"], w["x0"]))
    lines = []
    for w in enriched:
        if lines and abs(w["y0"] - lines[-1][0]["y0"]) <= LINE_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
    out = []
    for line in lines:
        line.sort(key=lambda w: w["x0"])
        out.extend(line)
    return out


def extract_words(doc):
    """All words of the document in visual reading order, per page."""
    words = []
    for pno in range(len(doc)):
        words.extend(page_words(doc[pno], pno))
    return words


def build_text(words):
    """Rebuild readable text from visual-order words (lines + spaces)."""
    lines = []
    cur = []
    for w in words:
        if cur and abs(w["y0"] - cur[-1]["y0"]) > LINE_TOL:
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    # Pages: start a fresh line when the page changes.
    out_lines = []
    for line in lines:
        out_lines.append(" ".join(w["word"] for w in line))
    return "\n".join(out_lines)


def extract(pdf_path):
    doc = fitz.open(pdf_path)
    words = extract_words(doc)
    text = build_text(words)
    return {"text": text, "words": words}


def _baseline_y(word):
    """Approximate text baseline for a word bbox (descent ~20% of size)."""
    return word["y1"] - word["size"] * 0.2


def edit(pdf_path, edited_text, out_path):
    doc = fitz.open(pdf_path)
    words = extract_words(doc)
    orig_tokens = [w["word"] for w in words]
    new_tokens = edited_text.split()
    sm = difflib.SequenceMatcher(None, orig_tokens, new_tokens, autojunk=False)
    opcodes = sm.get_opcodes()

    # Group operations per page: redactions (word bboxes) and insertions.
    redactions = {}   # pageno -> [fitz.Rect]
    insertions = []   # (pageno, x, baseline_y, size, fontname, text, avail_width)
    changes = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if tag in ("delete", "replace"):
            for w in words[i1:i2]:
                redactions.setdefault(w["page"], []).append(
                    fitz.Rect(w["x0"] - 0.5, w["y0"] - 0.5,
                              w["x1"] + 0.5, w["y1"] + 0.5))
            old = " ".join(orig_tokens[i1:i2])
            if tag == "replace":
                anchor = words[i1]
                new_text = " ".join(new_tokens[j1:j2])
                avail = max(w["x1"] for w in words[i1:i2]) - anchor["x0"]
                insertions.append((anchor["page"], anchor["x0"],
                                   _baseline_y(anchor), anchor["size"],
                                   _fontname(anchor["font"]), new_text, avail))
                changes.append({"type": "replace", "old": old, "new": new_text,
                                "page": anchor["page"]})
            else:
                changes.append({"type": "delete", "old": old, "new": "",
                                "page": words[i1]["page"]})
        elif tag == "insert":
            new_text = " ".join(new_tokens[j1:j2])
            if i1 > 0:
                # Insert after the previous word, on its line.
                prev = words[i1 - 1]
                size = prev["size"]
                font = _fontname(prev["font"])
                space_w = fitz.get_text_length(" ", fontname=font, fontsize=size)
                x = prev["x1"] + space_w
                y = _baseline_y(prev)
                # Available width: up to the next word on the same line.
                avail = None
                if i1 < len(words):
                    nxt = words[i1]
                    if nxt["page"] == prev["page"] and abs(nxt["y0"] - prev["y0"]) <= LINE_TOL:
                        avail = max(20.0, nxt["x0"] - x - 2.0)
                if avail is None:
                    avail = max(40.0, doc[prev["page"]].rect.width - x - 36.0)
                insertions.append((prev["page"], x, y, size, font, new_text, avail))
                changes.append({"type": "insert", "old": "", "new": new_text,
                                "page": prev["page"]})
            elif words:
                nxt = words[0]
                insertions.append((nxt["page"], nxt["x0"], _baseline_y(nxt),
                                   nxt["size"], _fontname(nxt["font"]), new_text,
                                   max(40.0, nxt["x1"] - nxt["x0"])))
                changes.append({"type": "insert", "old": "", "new": new_text,
                                "page": nxt["page"]})

    # Apply redactions page by page. PyMuPDF 1.23.7 only takes `images`:
    # NONE leaves images untouched; text is always removed and line art
    # (table borders) is left alone.
    for pageno, rects in redactions.items():
        page = doc[pageno]
        for r in rects:
            page.add_redact_annot(r, fill=(1, 1, 1), cross_out=False)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    # Insert replacement text, shrinking to fit the available width.
    for (pageno, x, y, size, font, text, avail) in insertions:
        if not text:
            continue
        fs = size
        w = fitz.get_text_length(text, fontname=font, fontsize=fs)
        if w > avail and w > 0:
            fs = max(MIN_FONT, size * avail / w)
        doc[pageno].insert_text((x, y), text, fontsize=fs, fontname=font)

    doc.save(out_path, garbage=4, deflate=True)
    import os
    return {"changes": changes, "out": out_path,
            "size": os.path.getsize(out_path)}


def main(argv):
    if len(argv) < 3:
        print("usage: pdf_edit.py extract <pdf> | edit <pdf> <edited_txt> <out_pdf>",
              file=sys.stderr)
        return 2
    mode = argv[1]
    try:
        if mode == "extract":
            print(json.dumps(extract(argv[2])))
            return 0
        if mode == "edit":
            with open(argv[3], "r", encoding="utf-8") as f:
                edited = f.read()
            print(json.dumps(edit(argv[2], edited, argv[4])))
            return 0
        print("unknown mode: %s" % mode, file=sys.stderr)
        return 2
    except Exception as e:
        print("pdf_edit error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
