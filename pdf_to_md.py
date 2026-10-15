#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pdf_to_md.py

Stage 1 of the paper-researcher pipeline:

    PDF -> Markdown + attachment folder + metadata JSON

Design:
  - text is keep as text, not OCR
  - images (raster and vector figures such as logic gates) become attachments
  - tables are stored as JSON rows and rendered as Markdown tables
  - image links use standard Markdown syntax: ![alt](relative/path.png)
  - tables are emitted as Markdown tables, with metadata sidecar JSON
  - one PDF at a time, one output directory, no LLM

Typical use:
  python3 pdf_to_md.py --out data/md data/papers/foo.pdf

Unix-friendly:
  python3 pdf_to_md.py --print-paths --out data/md data/papers/*.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from paper_common import ensure_dir, image_markdown, safe_stem, table_to_markdown

try:
    import pdfplumber  # optional, preferred table engine
except Exception:  # pragma: no cover
    pdfplumber = None


BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class Table:
    id: str
    page: int
    bbox: BBox
    data: List[List[str]]
    markdown: str
    format: str = "md"


@dataclass
class Figure:
    id: str
    page: int
    bbox: BBox
    path: str  # filename inside <stem>.assets/
    kind: str  # "raster" or "vector"
    labels_text: List[str] = field(default_factory=list)


@dataclass
class Block:
    id: str
    page: int
    bbox: BBox
    kind: str
    text: str = ""
    table_id: Optional[str] = None
    image_id: Optional[str] = None


class Ids:
    def __init__(self) -> None:
        self.fig = 0
        self.tbl = 0
        self.block = 0

    def fig_id(self) -> str:
        self.fig += 1
        return "fig-%04d" % self.fig

    def table_id(self) -> str:
        self.tbl += 1
        return "tbl-%04d" % self.tbl

    def block_id(self) -> str:
        self.block += 1
        return "blk-%04d" % self.block


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def rect_tuple(rect: Any) -> BBox:
    return (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))


def _intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    return float(inter.width * inter.height)


def overlap_ratio(inner: fitz.Rect, outer: fitz.Rect) -> float:
    if inner.is_empty:
        return 0.0
    area = float(inner.width * inner.height)
    if area <= 0:
        return 0.0
    return _intersection_area(inner, outer) / area


def _expand(rect: fitz.Rect, amount: float) -> fitz.Rect:
    return fitz.Rect(rect.x0 - amount, rect.y0 - amount, rect.x1 + amount, rect.y1 + amount)


def _table_has_content(rows: Sequence[Sequence[str]]) -> bool:
    if len(rows) < 2:
        return False
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False
    return any((cell or "").strip() for row in rows for cell in row)


def _is_boilerplate_text(text: str) -> bool:
    plain = re.sub(r"[*_`]", "", text).strip()
    if not plain:
        return True
    if plain.upper() in ("PMT",):
        return True
    # Barcode / registration-number lines: mostly digits, spaces and stars.
    if re.fullmatch(r"[0-9*+\-\s]{12,}", plain):
        return True
    # Cambridge / PMT running footers.
    if re.search(r"©\s*UCLES", plain, re.I):
        return True
    if re.fullmatch(r"\[?Turn over\]?", plain, re.I):
        return True
    if re.fullmatch(r"\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}", plain):
        return True
    if re.fullmatch(r"DC\s*\([^)]*\)\s*\d+/\d+", plain, re.I):
        return True
    return False


def _remove_empty_file(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# text extraction
# ---------------------------------------------------------------------------

def _span_to_markdown(text: str, flags: int) -> str:
    if not text:
        return ""
    bold = bool(flags & (1 << 4))  # TEXT_FONT_BOLD
    italic = bool(flags & (1 << 1))  # TEXT_FONT_ITALIC
    mono = bool(flags & (1 << 3))  # TEXT_FONT_MONOSPACED
    superscript = bool(flags & (1 << 0))  # TEXT_FONT_SUPERSCRIPT

    if superscript:
        text = "<sup>%s</sup>" % text
    if mono:
        return text
    if bold:
        return "**%s**" % text
    if italic:
        return "*%s*" % text
    return text


def format_text_block(block: Dict[str, Any]) -> str:
    """Reconstruct a PyMuPDF text block as Markdown-ish raw text."""
    lines_out: List[str] = []
    mono_spans = 0
    total_spans = 0

    for line in block.get("lines", []):
        parts: List[str] = []
        for span in line.get("spans", []):
            text = span.get("text", "")
            if text == "":
                continue
            flags = int(span.get("flags", 0))
            total_spans += 1
            if flags & (1 << 3):
                mono_spans += 1
            parts.append(_span_to_markdown(text, flags))
        line_text = "".join(parts).rstrip()
        if line_text.strip():
            lines_out.append(line_text)

    if not lines_out:
        return ""
    raw = "\n".join(lines_out)
    # A mostly-monospaced block is probably pseudocode; preserve it verbatim.
    if total_spans and len(lines_out) >= 2 and mono_spans / float(total_spans) >= 0.8:
        return "```\n%s\n```" % raw
    return raw


# ---------------------------------------------------------------------------
# table extraction
# ---------------------------------------------------------------------------

def extract_tables_pdfplumber_page(plumber_pdf: Any, page_index: int) -> List[Dict[str, Any]]:
    if plumber_pdf is None:
        return []
    try:
        page = plumber_pdf.pages[page_index]
        out: List[Dict[str, Any]] = []
        for table in page.find_tables():
            rows = table.extract()
            if not rows:
                continue
            rows = [["" if c is None else str(c) for c in row] for row in rows]
            out.append({"bbox": rect_tuple(table.bbox), "rows": rows})
        return out
    except Exception:
        return []


def extract_tables_pymupdf(page: fitz.Page) -> List[Dict[str, Any]]:
    try:
        finder = page.find_tables()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for table in finder.tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        if not rows:
            continue
        rows = [["" if c is None else str(c) for c in row] for row in rows]
        out.append({"bbox": rect_tuple(table.bbox), "rows": rows})
    return out


# ---------------------------------------------------------------------------
# figure extraction
# ---------------------------------------------------------------------------

def _labels_inside(text_blocks: Sequence[Dict[str, Any]], rect: fitz.Rect) -> List[str]:
    labels: List[str] = []
    expanded = _expand(rect, 8.0)
    for tb in text_blocks:
        tr = fitz.Rect(tb["bbox"])
        if overlap_ratio(tr, expanded) >= 0.5:
            labels.append(tb["text"])
    return labels


def extract_raster_figures(
    doc: fitz.Document,
    page: fitz.Page,
    page_no: int,
    assets_dir: Path,
    ids: Ids,
    table_rects: Sequence[fitz.Rect],
    text_blocks: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Figure]:
    figures: List[Figure] = []
    for img in page.get_images(full=True):
        xref = int(img[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            r = fitz.Rect(rect)
            area = r.width * r.height
            if r.width < args.min_figure_size or r.height < args.min_figure_size:
                continue
            if area < args.min_image_area:
                continue
            if any(overlap_ratio(r, tr) >= 0.3 for tr in table_rects):
                continue

            fig_id = ids.fig_id()
            filename = "%s.png" % fig_id
            path = assets_dir / filename
            try:
                pix = fitz.Pixmap(doc, xref)
                # CMYK / extra channels: normalise to RGB so the PNG is viewable.
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(str(path))
                pix = None
                if not path.exists() or path.stat().st_size == 0:
                    _remove_empty_file(path)
                    continue
            except Exception:
                # Some PDFs contain unsupported mask/colourspaces. Skipping is
                # better than crashing the whole extraction.
                _remove_empty_file(path)
                continue

            figures.append(
                Figure(
                    id=fig_id,
                    page=page_no,
                    bbox=rect_tuple(r),
                    path=filename,
                    kind="raster",
                    labels_text=_labels_inside(text_blocks, r),
                )
            )
    return figures


def extract_vector_figures(
    page: fitz.Page,
    page_no: int,
    assets_dir: Path,
    ids: Ids,
    table_rects: Sequence[fitz.Rect],
    text_blocks: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Figure]:
    figures: List[Figure] = []
    try:
        clusters = page.cluster_drawings()
    except Exception:
        return figures

    page_rect = page.rect
    page_area = float(page_rect.width * page_rect.height)
    for rect in clusters:
        r = fitz.Rect(rect)
        area = r.width * r.height
        if r.width < args.min_figure_size or r.height < args.min_figure_size:
            continue
        if area < args.min_figure_area:
            continue
        if area > page_area * 0.45:
            continue
        if r.width > 20.0 * r.height or r.height > 20.0 * r.width:
            continue
        if any(overlap_ratio(r, tr) >= 0.3 for tr in table_rects):
            continue
        # Ignore page-border / header / footer rules.
        if (
            r.x0 <= 4
            or r.y0 <= 4
            or r.x1 >= page_rect.width - 4
            or r.y1 >= page_rect.height - 4
        ):
            continue
        if r.y1 < page_rect.height * 0.12 or r.y0 > page_rect.height * 0.90:
            continue
        # A wide, very short cluster is usually a ruled line / answer line.
        if r.height < page_rect.height * 0.06 and r.width > page_rect.width * 0.30:
            continue

        labels = _labels_inside(text_blocks, r)
        large_diagram = area >= args.diagram_min_area
        if args.vector_figures == "labels" and not labels:
            continue
        if args.vector_figures == "auto" and not labels and not large_diagram:
            continue

        fig_id = ids.fig_id()
        filename = "%s.png" % fig_id
        path = assets_dir / filename
        try:
            pix = page.get_pixmap(clip=r, dpi=args.dpi, alpha=False)
            pix.save(str(path))
            pix = None
            if not path.exists() or path.stat().st_size == 0:
                _remove_empty_file(path)
                continue
        except Exception:
            _remove_empty_file(path)
            continue

        figures.append(
            Figure(
                id=fig_id,
                page=page_no,
                bbox=rect_tuple(r),
                path=filename,
                kind="vector",
                labels_text=labels,
            )
        )
    return figures


# ---------------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------------

def _element_sort_key(element: Dict[str, Any]) -> Tuple[float, float, int]:
    bbox = element["bbox"]
    return (float(bbox[1]), float(bbox[0]), int(element.get("seq", 0)))


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def resolve_stem(pdf_path: Path, out_dir: Path, args: argparse.Namespace) -> str:
    """Return a collision-safe output stem.

    PMT and similar archives often use the same basename for different
    subjects, e.g. History and CS both have "June 2021 (v1) QP.pdf".
    """
    base = safe_stem(pdf_path.name)
    if args.stem_mode == "basename":
        return base
    if args.stem_mode == "hash":
        return "%s-%s" % (base, _short_hash(str(pdf_path.resolve())))

    # auto: keep the readable name unless it already belongs to another PDF.
    md_path = out_dir / ("%s.md" % base)
    meta_path = out_dir / ("%s.meta.json" % base)
    if args.overwrite or not md_path.exists():
        return base
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_pdf") == str(pdf_path.resolve()):
            return base
    except Exception:
        pass
    return "%s-%s" % (base, _short_hash(str(pdf_path.resolve())))


def convert_pdf(pdf_path: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    stem = resolve_stem(pdf_path, out_dir, args)
    assets_dir = ensure_dir(out_dir / ("%s.assets" % stem))

    doc = fitz.open(str(pdf_path))
    ids = Ids()

    plumber_pdf = None
    if args.table_engine in ("pdfplumber", "auto") and pdfplumber is not None:
        try:
            plumber_pdf = pdfplumber.open(str(pdf_path))
        except Exception:
            plumber_pdf = None

    all_blocks: List[Block] = []
    all_figures: List[Figure] = []
    all_tables: List[Table] = []
    md_lines: List[str] = []

    page_count = doc.page_count
    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else page_count

    for page_index in range(min(page_count, max_pages)):
        page = doc.load_page(page_index)
        page_no = page_index + 1

        # 1) raw text blocks
        raw_text_blocks: List[Dict[str, Any]] = []
        try:
            page_dict = page.get_text("dict")
        except Exception:
            page_dict = {"blocks": []}
        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                continue
            text = format_text_block(raw_block)
            if not text.strip() or _is_boilerplate_text(text):
                continue
            raw_text_blocks.append(
                {
                    "page": page_no,
                    "bbox": rect_tuple(raw_block["bbox"]),
                    "text": text,
                }
            )

        # 2) tables
        page_tables: List[Dict[str, Any]] = []
        if args.table_engine == "pymupdf":
            page_tables = extract_tables_pymupdf(page)
        else:
            page_tables = extract_tables_pdfplumber_page(plumber_pdf, page_index)
            if not page_tables and args.table_engine == "auto":
                page_tables = extract_tables_pymupdf(page)
        page_tables = [t for t in page_tables if _table_has_content(t["rows"])]

        table_objects: List[Table] = []
        table_rects: List[fitz.Rect] = []
        for raw_table in page_tables:
            rows = raw_table["rows"]
            table_id = ids.table_id()
            markdown = table_to_markdown(rows)
            table = Table(
                id=table_id,
                page=page_no,
                bbox=raw_table["bbox"],
                data=rows,
                markdown=markdown,
                format="md",
            )
            table_objects.append(table)
            all_tables.append(table)
            table_rects.append(fitz.Rect(raw_table["bbox"]))

        # 3) figures
        page_figures: List[Figure] = []
        if not args.no_images:
            page_figures.extend(
                extract_raster_figures(
                    doc,
                    page,
                    page_no,
                    assets_dir,
                    ids,
                    table_rects,
                    raw_text_blocks,
                    args,
                )
            )
            if args.vector_figures != "none":
                page_figures.extend(
                    extract_vector_figures(
                        page,
                        page_no,
                        assets_dir,
                        ids,
                        table_rects,
                        raw_text_blocks,
                        args,
                    )
                )
        figure_rects = [fitz.Rect(f.bbox) for f in page_figures]

        # 4) keep only text that does not belong to a table or figure
        kept_text: List[Dict[str, Any]] = []
        for tb in raw_text_blocks:
            tr = fitz.Rect(tb["bbox"])
            if any(overlap_ratio(tr, table_rect) >= 0.3 for table_rect in table_rects):
                continue
            if any(overlap_ratio(tr, figure_rect) >= 0.5 for figure_rect in figure_rects):
                continue
            kept_text.append(tb)

        # 5) build page elements in reading order
        elements: List[Dict[str, Any]] = []
        for i, tb in enumerate(kept_text):
            elements.append(
                {
                    "kind": "text",
                    "page": page_no,
                    "bbox": tb["bbox"],
                    "text": tb["text"],
                    "seq": i,
                }
            )
        for i, table in enumerate(table_objects):
            elements.append(
                {
                    "kind": "table",
                    "page": page_no,
                    "bbox": table.bbox,
                    "table_id": table.id,
                    "seq": 100000 + i,
                }
            )
        for i, figure in enumerate(page_figures):
            elements.append(
                {
                    "kind": "figure",
                    "page": page_no,
                    "bbox": figure.bbox,
                    "image_id": figure.id,
                    "path": figure.path,
                    "seq": 200000 + i,
                }
            )
        elements.sort(key=_element_sort_key)

        if args.page_comments:
            md_lines.append("<!-- page: %d -->" % page_no)

        for element in elements:
            if element["kind"] == "text":
                md_lines.append(element["text"])
                md_lines.append("")
            elif element["kind"] == "table":
                table = next(t for t in table_objects if t.id == element["table_id"])
                md_lines.append("<!-- table:%s -->" % table.id)
                md_lines.append(table.markdown)
                md_lines.append("")
            elif element["kind"] == "figure":
                figure = next(f for f in page_figures if f.id == element["image_id"])
                rel = "%s.assets/%s" % (stem, figure.path)
                md_lines.append(image_markdown(figure.id, rel))
                md_lines.append("")

            block = Block(
                id=ids.block_id(),
                page=page_no,
                bbox=element["bbox"],
                kind=element["kind"],
                text=element.get("text", ""),
                table_id=element.get("table_id"),
                image_id=element.get("image_id"),
            )
            all_blocks.append(block)

        all_figures.extend(page_figures)

    doc.close()
    if plumber_pdf is not None:
        try:
            plumber_pdf.close()
        except Exception:
            pass

    md_path = out_dir / ("%s.md" % stem)
    meta_path = out_dir / ("%s.meta.json" % stem)
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    meta = {
        "source_pdf": str(pdf_path.resolve()),
        "stem": stem,
        "generator": "pdf_to_md.py",
        "image_link_style": "markdown",
        "table_format": "md",
        "pages": page_count,
        "images": [asdict(f) for f in all_figures],
        "tables": [asdict(t) for t in all_tables],
        "blocks": [asdict(b) for b in all_blocks],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"pdf": pdf_path, "md": md_path, "meta": meta_path, "assets": assets_dir}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PDF -> Markdown + attachments + metadata JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdfs", nargs="+", help="input PDF path(s)")
    parser.add_argument("--out", "-o", default="data/md", help="output directory")
    parser.add_argument(
        "--stem-mode",
        choices=("auto", "basename", "hash"),
        default="auto",
        help="how to disambiguate PDFs with the same basename",
    )
    parser.add_argument(
        "--table-engine",
        choices=("auto", "pdfplumber", "pymupdf", "none"),
        default="auto",
        help="table extraction backend",
    )
    parser.add_argument("--no-images", action="store_true", help="do not extract image/figure attachments")
    parser.add_argument("--no-page-comments", action="store_true", help="omit <!-- page: N --> comments")
    parser.add_argument("--min-figure-size", type=float, default=24.0, help="minimum figure width/height in points")
    parser.add_argument("--min-figure-area", type=float, default=1600.0, help="minimum vector figure area in square points")
    parser.add_argument("--min-image-area", type=float, default=10000.0, help="minimum raster image area in square points")
    parser.add_argument("--diagram-min-area", type=float, default=30000.0, help="vector clusters smaller than this need text labels to be kept")
    parser.add_argument(
        "--vector-figures",
        choices=("auto", "labels", "all", "none"),
        default="auto",
        help="vector figure policy: auto=labels or large diagram, labels=only labelled, all=everything, none=disable",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering vector figures")
    parser.add_argument("--max-pages", type=int, default=0, help="max pages per PDF; 0 = all")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--print-paths", action="store_true", help="print generated .md paths to stdout")
    parser.add_argument(
        "--print0",
        action="store_true",
        help="like --print-paths but NUL-delimited for xargs -0",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.page_comments = not args.no_page_comments

    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    failures = 0
    for raw in args.pdfs:
        pdf_path = Path(raw).expanduser()
        if not pdf_path.exists():
            print("skip missing PDF: %s" % pdf_path, file=sys.stderr)
            failures += 1
            continue
        if not pdf_path.suffix.lower() == ".pdf":
            print("skip non-PDF: %s" % pdf_path, file=sys.stderr)
            failures += 1
            continue

        try:
            result = convert_pdf(pdf_path, out_dir, args)
        except Exception as exc:
            print("error %s: %s" % (pdf_path, exc), file=sys.stderr)
            failures += 1
            continue

        if args.print_paths or args.print0:
            if args.print0:
                sys.stdout.buffer.write(str(result["md"].resolve()).encode("utf-8") + b"\0")
                sys.stdout.buffer.flush()
            else:
                print(str(result["md"].resolve()), flush=True)
        else:
            print("wrote %s" % result["md"], file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
