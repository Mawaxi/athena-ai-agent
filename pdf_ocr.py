#!/usr/bin/env python3
"""OCR a scanned PDF with Mirage (on-device vision).
Renders each page to an image with PyMuPDF, reads the text with Mirage,
prints JSON {"text": "...", "pages": n} to stdout.
Usage: pdf_ocr.py <pdf-path> [max-pages]
Run with /root/venv/bin/python3 (has fitz and cactus).
"""
import json
import os
import sys
import tempfile


def main():
    pdf_path = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    import fitz
    import cactus

    doc = fitz.open(pdf_path)
    vision = cactus.Mirage()
    pages_text = []
    n = min(len(doc), max_pages)
    for pno in range(n):
        page = doc[pno]
        pix = page.get_pixmap(dpi=150)
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pix.save(tmp)
            r = vision.describe(tmp)
            texts = r.get("text") or []
            strings = []
            for t in texts:
                s = (t.get("string") or "").strip()
                if s and s not in strings:
                    strings.append(s)
            if strings:
                pages_text.append("\n".join(strings))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    text = "\n\n".join(pages_text)
    print(json.dumps({"text": text, "pages": n}))


if __name__ == "__main__":
    main()
