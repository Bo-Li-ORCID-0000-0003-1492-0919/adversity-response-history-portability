#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops
from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[1]


def normalized(text: str) -> str:
    return " ".join(text.replace("−", "-").replace("Δ", "Delta ").replace("²", "2").split())


def number_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", normalized(text))


def compare(reference: Path, generated: Path, temp: Path) -> dict[str, object]:
    a, b = PdfReader(reference), PdfReader(generated)
    assert len(a.pages) == len(b.pages) == 1
    box_a, box_b = a.pages[0].mediabox, b.pages[0].mediabox
    size_a = (float(box_a.width), float(box_a.height))
    size_b = (float(box_b.width), float(box_b.height))
    assert all(abs(x - y) <= 0.02 for x, y in zip(size_a, size_b)), (reference.name, size_a, size_b)
    text_a, text_b = normalized(a.pages[0].extract_text() or ""), normalized(b.pages[0].extract_text() or "")
    assert number_tokens(text_a) == number_tokens(text_b), f"numeric text mismatch: {reference.name}"
    assert text_a == text_b, f"label/text mismatch: {reference.name}"
    for label, path in [("reference", reference), ("generated", generated)]:
        subprocess.run(["pdftoppm", "-singlefile", "-r", "150", "-png", str(path), str(temp / f"{reference.stem}_{label}")], check=True, capture_output=True)
    ia = Image.open(temp / f"{reference.stem}_reference.png").convert("RGB")
    ib = Image.open(temp / f"{reference.stem}_generated.png").convert("RGB")
    assert ia.size == ib.size
    diff = ImageChops.difference(ia, ib)
    hist = diff.histogram()
    rms = math.sqrt(sum((index % 256) ** 2 * count for index, count in enumerate(hist)) / (ia.width * ia.height * 3))
    return {"figure": reference.name, "page_points": size_a, "text_match": True, "numeric_match": True, "render_rms": rms}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aggregate-data figure reproduction with reference outputs.")
    parser.add_argument("--generated", type=Path, required=True)
    args = parser.parse_args()
    pairs = []
    for i in range(1, 5):
        pairs.append((REPO / f"expected_outputs/main_figures/Figure_{i}.pdf", args.generated / f"main/figures/Figure_{i}.pdf"))
    for i in range(1, 9):
        pairs.append((REPO / f"expected_outputs/extended_data_figures/Extended_Data_Figure_{i}.pdf", args.generated / f"extended_data/extended_figures/Extended_Data_Figure_{i}.pdf"))
    with tempfile.TemporaryDirectory() as folder:
        report = [compare(a, b, Path(folder)) for a, b in pairs]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
