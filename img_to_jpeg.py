#!/usr/bin/env python3
"""Convert any image on stdin to JPEG on stdout.

Used by the Athena AI PDF writer (src/main.rs) so chat PDFs can embed
photos that are not already JPEG (PNG, GIF, CMYK JPEG, oversized JPEG).
Reads raw image bytes from stdin, writes JPEG bytes to stdout.
Exits non-zero with a message on stderr when the input cannot be read.

Notes:
- `Pixmap(data)` preserves the image's pixel size; the `open()` fallback
  renders at 72 dpi and can shrink images that carry DPI metadata, so it
  is only used when Pixmap fails or when downscaling a large image.
- Alpha is dropped (JPEG has none) by compositing onto black via
  `Pixmap(pix, 0)`; for photos this is fine.
- Large images are downscaled so the long side is at most 1200 px, and the
  JPEG is written at quality 85, so PDFs stay a reasonable size.
- WebP is NOT supported by the MuPDF build on this phone and exits 1;
  the PDF writer then leaves a placeholder line with the source URL.
"""
import sys

MAX_SIDE = 1200
JPEG_QUALITY = 85

# Magic bytes of the image formats we accept. Anything else (notably an
# HTML error page from a dead link) is rejected, so a 404 page can never
# be rendered as a fake photo in the PDF.
KNOWN_MAGIC = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG",  # PNG
    b"GIF8",  # GIF
    b"BM",  # BMP
    b"II*\x00",  # TIFF, little-endian
    b"MM\x00*",  # TIFF, big-endian
)


def is_known_image(data: bytes) -> bool:
    return any(data.startswith(m) for m in KNOWN_MAGIC)


def to_jpeg(pix, fitz):
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    return pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)


def main() -> int:
    # Optional file path argument; otherwise read stdin. The PDF writer
    # passes a temp file so it can use Command::output() simply.
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], "rb") as f:
                data = f.read()
        except Exception as e:
            print(f"img_to_jpeg: cannot read {sys.argv[1]}: {e}", file=sys.stderr)
            return 1
    else:
        data = sys.stdin.buffer.read()
    if not data:
        print("img_to_jpeg: empty input", file=sys.stderr)
        return 1
    if not is_known_image(data):
        print("img_to_jpeg: not a known image format", file=sys.stderr)
        return 1
    try:
        import pymupdf as fitz
    except Exception as e:  # pragma: no cover - environment problem
        print(f"img_to_jpeg: cannot import pymupdf: {e}", file=sys.stderr)
        return 1
    # Fast path: Pixmap from bytes keeps the true pixel size.
    try:
        pix = fitz.Pixmap(data)
        if max(pix.w, pix.h) <= MAX_SIDE:
            sys.stdout.buffer.write(to_jpeg(pix, fitz))
            return 0
    except Exception:
        pix = None
    # Large image, or Pixmap could not read it: render via a document,
    # downscaling so the long side is at most MAX_SIDE.
    try:
        doc = fitz.open(stream=data)
        if doc.page_count < 1:
            print("img_to_jpeg: no pages in image", file=sys.stderr)
            return 1
        page = doc[0]
        w, h = page.rect.width, page.rect.height
        scale = min(1.0, MAX_SIDE / max(w, h)) if max(w, h) > 0 else 1.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        sys.stdout.buffer.write(to_jpeg(pix, fitz))
        return 0
    except Exception as e:
        print(f"img_to_jpeg: cannot convert image: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
