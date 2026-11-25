#!/usr/bin/env python3
"""Describe an image with Cactus Mirage (on-device vision) and print a
concise plain-text description for a text-only local model to read.

Usage: mirage_describe.py <image-path>
Prints the description to stdout. Exits non-zero on failure.

Mirage returns structured JSON (scene, objects, regions, text/OCR,
relations, events, reasoning). A tiny text model (qwen3-0.6b) reads plain
prose far better than raw JSON, so this converts the record into short
plain sentences and caps the length to fit a 4k context.
"""
import json
import sys


def describe(path: str) -> str:
    import cactus

    vision = cactus.Mirage()
    r = vision.describe(path)
    parts: list[str] = []

    scene = r.get("scene") or {}
    bits = []
    if scene.get("setting"):
        bits.append(f"a {scene['setting']}")
    if scene.get("background"):
        bits.append(f"background is {scene['background']}")
    if scene.get("style"):
        bits.append(f"in {scene['style']} style")
    if bits:
        parts.append("The photo shows " + ", ".join(bits) + ".")

    objects = r.get("objects") or []
    labels = []
    for o in objects:
        lab = o.get("label") or o.get("name")
        if lab and lab not in labels:
            labels.append(lab)
    if labels:
        parts.append("Objects visible: " + ", ".join(labels[:12]) + ".")

    texts = r.get("text") or []
    strings = []
    for t in texts:
        s = (t.get("string") or "").strip()
        if s and s not in strings:
            strings.append(s)
    if strings:
        parts.append("Text in the photo reads: " + " | ".join(strings[:12]))

    reasoning = (r.get("reasoning") or "").strip()
    if reasoning:
        # The reasoning is one long |-separated trace; keep the first
        # two segments so it stays short.
        segs = [s.strip() for s in reasoning.split("|") if s.strip()]
        if segs:
            parts.append("Description: " + ". ".join(segs[:2]) + ".")

    out = " ".join(parts).strip()
    return out if out else "The photo could not be described."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: mirage_describe.py <image-path>", file=sys.stderr)
        sys.exit(2)
    try:
        print(describe(sys.argv[1]))
    except Exception as e:  # noqa: BLE001 - report and fail cleanly
        print(f"mirage failed: {e}", file=sys.stderr)
        sys.exit(1)
