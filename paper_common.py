#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the paper-researcher pipeline.

Keep this tiny and dependency-free. Both pdf_to_md.py and split_questions.py
import the table renderer from here so the Markdown table format is defined in
exactly one place.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(name: str) -> str:
    """Turn a filename or path into a safe, stable stem."""
    stem = Path(name).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return stem or "document"


def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    text = str(cell)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "<br>")
    text = text.replace("|", "\\|")
    return text.strip()


def table_to_markdown(rows: Sequence[Sequence[Any]]) -> str:
    """Render a rectangular JSON table as a GitHub-flavoured Markdown table.

    This is the single shared table renderer.  Rows may have different
    lengths; missing cells are padded with empty strings.
    """
    if not rows:
        return ""

    normalized: List[List[str]] = []
    width = max(len(row) for row in rows)
    for row in rows:
        padded = ["" for _ in range(width)]
        for i, cell in enumerate(row):
            padded[i] = _cell_text(cell)
        normalized.append(padded)

    header = normalized[0]
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_markdown_table(lines: Sequence[str]) -> List[List[str]]:
    """Parse a simple Markdown table back into JSON rows.

    This is intentionally conservative; it is the reverse of
    table_to_markdown() for the simple tables produced by the pipeline.
    """
    rows: List[List[str]] = []
    for raw in lines:
        line = raw.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        inner = line.strip()
        if inner.startswith("|"):
            inner = inner[1:]
        if inner.endswith("|"):
            inner = inner[:-1]
        cells = [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", inner)]
        rows.append(cells)
    return rows


def image_markdown(alt: str, rel_path: str) -> str:
    """Return a standard Markdown image link."""
    alt = alt.replace("]", "\\]")
    return "![%s](%s)" % (alt, rel_path)
