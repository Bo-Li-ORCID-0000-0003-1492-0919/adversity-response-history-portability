"""Reproduce the final Extended Data figures from aggregate source data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
from decimal import Decimal
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/figure_reproduction/extended_data"
FIG_OUT = OUT / "extended_figures"
RENDERER = Path(__file__).with_name("base_render_figures.py")
PDFTOPPM = shutil.which("pdftoppm")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_renderer():
    spec = importlib.util.spec_from_file_location("nmh_renderer", RENDERER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def configure_style(m):
    override = os.environ.get("NMH_ARIAL_FONT_DIR")
    candidates = ([Path(override)] if override else []) + [
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
    ]
    fontdir = next((folder for folder in candidates if (folder / "Arial.ttf").exists() and (folder / "Arial Bold.ttf").exists()), None)
    if fontdir is not None:
        regular_font = fontdir / "Arial.ttf"
        bold_font = fontdir / "Arial Bold.ttf"
    else:
        fallback_pairs = [
            (Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'), Path('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf')),
            (Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'), Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf')),
        ]
        pair = next(((regular, bold) for regular, bold in fallback_pairs if regular.exists() and bold.exists()), None)
        if pair is None:
            raise FileNotFoundError("Set NMH_ARIAL_FONT_DIR to a directory containing Arial.ttf and Arial Bold.ttf.")
        regular_font, bold_font = pair
    pdfmetrics.registerFont(TTFont("ArialRef", str(regular_font)))
    pdfmetrics.registerFont(TTFont("ArialRefBold", str(bold_font)))
    black, grey = "#000000", "#666666"
    m.BLACK, m.GRAY, m.LIGHT = black, grey, "#D9D9D9"
    m.COLORS = {"UKHLS": grey, "HRS": grey}

    def text(self, x, y, s, size=9, color=black, bold=False, anchor="start"):
        s = str(s).replace("−", "-").replace("≥", ">=")
        fn = "ArialRefBold" if bold else "ArialRef"
        self.c.setFillColor(HexColor(color))
        self.c.setFont(fn, size)
        {"start": self.c.drawString, "middle": self.c.drawCentredString, "end": self.c.drawRightString}[anchor](x, self.h - y, s)
        self.elements.append(
            f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'text-anchor="{anchor}" fill="{color}">{html.escape(s)}</text>'
        )

    def wrap(self, x, y, s, width, size=9, bold=False):
        words = str(s).split()
        lines, line = [], ""
        fn = "ArialRefBold" if bold else "ArialRef"
        for word in words:
            proposal = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(proposal, fn, size) > width:
                lines.append(line)
                line = word
            else:
                line = proposal
        if line:
            lines.append(line)
        for j, item in enumerate(lines):
            self.text(x, y + j * (size + 3), item, size, bold=bold)
        return len(lines) * (size + 3)

    def point(self, x, y, color=black, open=False, r=2.5):
        color = black
        self.c.setDash([])
        self.c.setLineWidth(1.0)
        self.c.setStrokeColor(HexColor(color))
        self.c.setFillColor(HexColor("#FFFFFF" if open else color))
        self.c.circle(x, self.h - y, r, stroke=1, fill=1)
        self.elements.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{color}" '
            f'stroke-width="1" fill="{"#FFFFFF" if open else color}"/>'
        )

    m.Dual.text = text
    m.Dual.wrap = wrap
    m.Dual.point = point

    def value_text(row, with_ci=True):
        if not m.finite(row.estimate):
            return "not available"
        estimate = f"{float(row.estimate):.4f}"
        if with_ci and m.finite(row.ci_lower) and m.finite(row.ci_upper):
            return f"{estimate} ({float(row.ci_lower):.4f}, {float(row.ci_upper):.4f})"
        return estimate

    def ref_forest(key, meta, df, stem):
        panels = list(df.panel.unique())
        ncol = 2 if len(panels) > 1 else 1
        panel_width = 365 if ncol == 2 else 650
        width = panel_width * ncol + 40
        blocks, cursor = [], 80
        for start in range(0, len(panels), ncol):
            row_panels = panels[start:start+ncol]
            heights = []
            for panel in row_panels:
                g = df.loc[df.panel.eq(panel)]
                heights.append(62 + 13 * len(g) + 17 * g.label.nunique())
            row_height = max(heights)
            for j, panel in enumerate(row_panels):
                blocks.append((panel, 20 + j * panel_width, cursor, panel_width - 16, row_height))
            cursor += row_height + 24
        d = m.Dual(stem, width, cursor + 30)
        d.wrap(20, 27, meta["title"], width - 40, 14, True)
        d.text(20, 52, "Filled points: grouped cross-validation; open points: temporal holdout. P: primary; S: secondary/sensitivity.", 8.5)
        plotted = []
        for panel, x, top, pwidth, _ in blocks:
            g = df.loc[df.panel.eq(panel)]
            d.text(x, top, panel, 10.5, bold=True)
            d.text(x, top + 18, "Comparison / cohort (n)", 7.2)
            d.text(x + pwidth, top + 18, "Estimate (95% CI)", 7.2, anchor="end")
            d.line(x, top + 22, x + pwidth, top + 22, "#000000", .5)
            label_width = 138 if ncol == 2 else 260
            value_width = 112 if ncol == 2 else 132
            plot_x = x + label_width
            plot_end = x + pwidth - value_width
            values = [float(v) for c in ["estimate", "ci_lower", "ci_upper"] for v in g[c] if m.finite(v)]
            lo, hi = min([0, *values]), max([0, *values])
            span = max(hi - lo, .005)
            lo, hi = lo - .08 * span, hi + .08 * span
            scale = lambda v: plot_x + (float(v) - lo) / (hi - lo) * (plot_end - plot_x)
            y = top + 37
            for label in g.label.unique():
                gg = g.loc[g.label.eq(label)].sort_values(["cohort", "validation"], ascending=[False, True])
                d.wrap(x, y, label, label_width - 6, 7.4, True)
                y += 12
                for _, row in gg.iterrows():
                    validation = str(row.validation)
                    temporal = "temporal" in validation
                    short = "T" if temporal else "CV" if "grouped" in validation else ""
                    role = "P" if row.analysis_role == "primary" else "S" if any(term in str(row.analysis_role) for term in ["secondary", "sensitivity"]) else ""
                    n = "" if not m.finite(row.n) else f"n={int(row.n):,}"
                    d.text(x + 5, y, f"{row.cohort} {short} {role}  {n}", 7.0, "#555555")
                    if m.finite(row.estimate):
                        py = y - 2.5
                        if m.finite(row.ci_lower) and m.finite(row.ci_upper):
                            a, b = scale(row.ci_lower), scale(row.ci_upper)
                            d.line(a, py, b, py, "#777777", .55)
                            d.line(a, py - 2, a, py + 2, "#777777", .45)
                            d.line(b, py - 2, b, py + 2, "#777777", .45)
                        d.point(scale(row.estimate), py, open=temporal, r=2.6)
                    d.text(x + pwidth, y, value_text(row), 6.8, anchor="end")
                    plotted.append(int(row.display_row))
                    y += 11
                y += 5
            axis_y = y + 1
            d.line(plot_x, axis_y, plot_end, axis_y, "#555555", .45)
            if lo <= 0 <= hi:
                d.line(scale(0), top + 25, scale(0), axis_y, "#777777", .45, True)
            for tick in np.linspace(lo, hi, 3):
                tx = scale(tick)
                d.line(tx, axis_y, tx, axis_y + 3, "#555555", .45)
                d.text(tx, axis_y + 12, f"{tick:.3f}", 6.8, anchor="middle")
            label = "Delta predictive R2" if "predictive" in str(g.unit.iloc[0]) else ("Proportion" if "proportion" in str(g.unit.iloc[0]) else "Coefficient")
            d.text((plot_x + plot_end) / 2, axis_y + 25, label, 7.0, anchor="middle")
        d.save(meta["source"], plotted)

    def ref_descriptive(key, meta, df, stem):
        panels = list(df.panel.unique())
        ncol = 2 if len(panels) > 1 else 1
        panel_width = 365 if ncol == 2 else 650
        width = panel_width * ncol + 40
        blocks, cursor = [], 72
        for start in range(0, len(panels), ncol):
            row_panels = panels[start:start+ncol]
            row_height = max(84 + 20 * len(df.loc[df.panel.eq(panel)]) for panel in row_panels)
            for j, panel in enumerate(row_panels):
                blocks.append((panel, 20 + j * panel_width, cursor, panel_width - 16))
            cursor += row_height + 24
        d = m.Dual(stem, width, cursor + 28)
        d.wrap(20, 27, meta["title"], width - 40, 14, True)
        d.text(20, 50, "Points show descriptive estimates; confidence intervals were not estimated.", 8.5)
        plotted = []
        for panel, x, top, pwidth in blocks:
            g = df.loc[df.panel.eq(panel)]
            d.text(x, top, panel, 10.5, bold=True)
            d.text(x, top + 18, "Measure / cohort (n)", 7.2)
            d.text(x + pwidth, top + 18, "Estimate", 7.2, anchor="end")
            d.line(x, top + 22, x + pwidth, top + 22, "#000000", .5)
            values = [float(v) for v in g.estimate if m.finite(v)]
            lo, hi = min([0, *values]), max([0, *values])
            span = max(hi - lo, .005)
            lo, hi = lo - .08 * span, hi + .08 * span
            label_width, value_width = 165, 65
            plot_x, plot_end = x + label_width, x + pwidth - value_width
            scale = lambda v: plot_x + (float(v) - lo) / (hi - lo) * (plot_end - plot_x)
            y = top + 38
            for _, row in g.iterrows():
                label = str(row.label).replace("IPOW_truncated_1_99", "IPOW").replace("complete_four_points", "Complete case")
                n = "" if not m.finite(row.n) else f" ({row.cohort}; n={int(row.n):,})"
                used = d.wrap(x, y, label + n, label_width - 6, 6.9)
                if m.finite(row.estimate):
                    d.point(scale(row.estimate), y - 2.5, r=2.4)
                d.text(x + pwidth, y, value_text(row, False), 7.0, anchor="end")
                plotted.append(int(row.display_row))
                y += max(13, used + 3)
            axis_y = y + 2
            d.line(plot_x, axis_y, plot_end, axis_y, "#555555", .45)
            for tick in np.linspace(lo, hi, 3):
                tx = scale(tick)
                d.line(tx, axis_y, tx, axis_y + 3, "#555555", .45)
                d.text(tx, axis_y + 12, f"{tick:.2f}", 6.8, anchor="middle")
        d.save(meta["source"], plotted)

    m.forest = ref_forest
    m.descriptive = ref_descriptive


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_source_record(record: dict[str, str]) -> None:
    source = REPO / record["source_file"]
    if not source.is_file():
        raise AssertionError(f"Missing source file: {record['source_file']}")
    original = read_rows(source)[int(record["source_row_0based"])]
    for key, value in json.loads(record["source_filter"]).items():
        if str(original[key]) != str(value):
            raise AssertionError((source, key, original[key], value))
    for target, column_key in (("estimate", "source_estimate_column"), ("n", "source_n_column")):
        value = record.get(target, "")
        column = record.get(column_key, "")
        if value == "" or column == "":
            continue
        tolerance = Decimal("0") if target == "n" else Decimal("1e-14")
        if abs(Decimal(value) - Decimal(original[column])) > tolerance:
            raise AssertionError((source, target, value, original[column]))
    ci_columns = [item for item in record.get("source_ci_columns", "").split(";") if item]
    for target, column in zip(("ci_lower", "ci_upper"), ci_columns):
        value = record.get(target, "")
        if value == "":
            continue
        if abs(Decimal(value) - Decimal(original[column])) > Decimal("1e-14"):
            raise AssertionError((source, target, value, original[column]))


def main():
    global OUT, FIG_OUT
    parser = argparse.ArgumentParser(description="Reproduce final Extended Data figures from aggregate source data only.")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Output directory; expected outputs are never overwritten.")
    parser.add_argument(
        "--render-review-pngs",
        action="store_true",
        help="Also render 300 dpi PNG review copies. This is optional and can be slow.",
    )
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    FIG_OUT = OUT / "extended_figures"
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    review = OUT / "review"
    if args.render_review_pngs:
        review.mkdir(parents=True, exist_ok=True)
    m = load_renderer()
    m.RENDER_PNGS = False
    configure_style(m)
    registry = json.loads((REPO / "config/display_registry.json").read_text())["figures"]
    audits = []
    for key, meta in registry.items():
        if not key.startswith("Extended_Data_Figure_"):
            continue
        source = REPO / "source_data/extended_data" / Path(meta["source"]).name
        before = sha256(source)
        frame = m.pd.read_csv(source)
        for record in read_rows(source):
            verify_source_record(record)
        stem = FIG_OUT / key
        renderer = {
            "design": m.design,
            "trajectory": m.trajectory,
            "flow": m.flow,
            "measurement": m.descriptive,
            "points": m.descriptive,
        }.get(meta["kind"], m.forest)
        renderer(key, meta, frame, stem)
        assert sha256(source) == before
        displayed = sorted(int(x) for x in frame["display_row"])
        svg = stem.with_suffix(".svg").read_text()
        metadata = html.unescape(svg.split("<metadata>", 1)[1].split("</metadata>", 1)[0])
        plotted = sorted(json.loads(metadata)["plotted_source_rows"])
        if displayed != plotted:
            raise AssertionError((key, len(displayed), len(plotted)))
        if args.render_review_pngs:
            if PDFTOPPM is None:
                raise RuntimeError("pdftoppm is required for PNG rendering and was not found on PATH.")
            subprocess.run(
                [PDFTOPPM, "-singlefile", "-r", "300", "-png", str(stem.with_suffix(".pdf")), str(review / (key + "_300dpi"))],
                check=True,
                capture_output=True,
            )
        audits.append(
            {
                "display": key,
                "source_file": str(source.relative_to(REPO)),
                "source_sha256": before,
                "source_rows": len(frame),
                "plotted_rows": len(plotted),
                "all_source_rows_plotted": "PASS",
            }
        )
    with (OUT / "extended_figure_style_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)
    (OUT / "README_EXTENDED_FIGURES.txt").write_text(
        "EXTENDED DATA FIGURE REPRODUCTION\n\n"
        "The figures were generated from the aggregate source-data files in this repository.\n"
        "All source rows were checked against the plotted-row metadata.\n"
        "The PDFs use embedded Arial, white backgrounds and vector marks.\n"
        "Use --render-review-pngs to create optional 300 dpi PNG copies.\n",
        encoding="utf-8",
    )
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
