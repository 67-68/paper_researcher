#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split_questions.py

Stage 2 of the paper-researcher pipeline:

    Markdown + .meta.json -> hierarchical questions JSON + flat JSONL

Design:
  - no PDF access; this script only reads Markdown and metadata
  - no LLM by default; deterministic rule-based segmentation
  - rules can be selected with --profile auto|history|cs
  - tables are stored as JSON rows, with format="md" and a Markdown rendering
  - image links are standard Markdown: ![fig-0001](relative/path.png)

Typical use:
  python3 split_questions.py --out data/questions data/md/foo.md

Unix-friendly:
  pdf_to_md.py --print0 --out data/md data/papers/*.pdf \
    | xargs -0 -n1 python3 split_questions.py --out data/questions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from paper_common import (
    ensure_dir,
    parse_markdown_table,
    safe_stem,
    table_to_markdown,
)


# ---------------------------------------------------------------------------
# regex rules
# ---------------------------------------------------------------------------

PAGE_COMMENT_RE = re.compile(r"^\s*<!--\s*page:\s*(\d+)\s*-->\s*$")
TABLE_COMMENT_RE = re.compile(r"^\s*<!--\s*table:([A-Za-z0-9_\-]+)\s*-->\s*$")
# Greedy on the path so image filenames may contain spaces and nested
# parentheses, e.g. ![fig-0001](June 2021 (v1) QP.assets/fig-0001.png)
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((.*)\)\s*$")

# main question: 1. / 1) / 1、 / Q1 / Question 1 / bare 1
# NOTE: allow at most one space between the number and the punctuation, so
# answer lines such as "1  ........................" are not mistaken for "1.".
MAIN_Q_RE = re.compile(r"^\s*(?:Q(?:uestion)?\s*)?(\d{1,3})\s{0,1}[\.、)]\s*(.*)$", re.I)
MAIN_Q_ALT_RE = re.compile(r"^\s*(?:Q(?:uestion)?\s*)(\d{1,3})\b\s*(.*)$", re.I)
MAIN_Q_LONE_RE = re.compile(r"^\s*(\d{1,3})\s*$", re.I)

# sub-question forms:
#   (a), (b)
#   (i), (ii)
#   (1), (2)
#   a), b)
SUB_LETTER_RE = re.compile(r"^\s*\(([a-zA-Z])\)\s*(.*)$")
SUB_ROMAN_RE = re.compile(r"^\s*\(([ivxlcdmIVXLCDM]{1,6})\)\s*(.*)$")
SUB_NUM_RE = re.compile(r"^\s*\((\d{1,2})\)\s*(.*)$")
SUB_TRAILING_LETTER_RE = re.compile(r"^\s*([a-zA-Z])\)\s*(.*)$")

# options: A. / A) / (A)
OPT_RE = re.compile(r"^\s*\(?([A-D])[\)\.、]\s*(.*)$")

# marks: [2] or (2 marks)
MARKS_RE = re.compile(r"(?:\[(\d{1,3})\]|\((\d{1,3})\s*marks?\))\s*$", re.I)

# section headings create a new group
SECTION_RE = re.compile(
    r"^\s*(Section\s+[A-Z]\b.*|Part\s+[IVX\d]+\b.*|Paper\s+\d+.*)\s*$",
    re.I,
)

# shared-stimulus headings mark the following text as shared_stem, but do not
# reset the current question (History: Source A / Passage A, etc.)
STIMULUS_RE = re.compile(
    r"^\s*(Reading Comprehension.*|Comprehension.*|Passage\s+[A-Z]\b.*|Source\s+[A-Z]\b.*)\s*$",
    re.I,
)


def _strip_md_formatting(line: str) -> str:
    """Strip inline emphasis markers used by pdf_to_md.py.

    The raw Markdown stays the source of truth; this lightweight normalisation
    is only for rule matching and clean JSON text.
    """
    return re.sub(r"[*`]", "", line).strip()


def _is_boilerplate_line(plain: str) -> bool:
    if not plain:
        return True
    if re.search(r"©\s*UCLES", plain, re.I):
        return True
    if re.fullmatch(r"\[?Turn over\]?", plain, re.I):
        return True
    if re.fullmatch(r"\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}", plain):
        return True
    if re.fullmatch(r"DC\s*\([^)]*\)\s*\d+/\d+", plain, re.I):
        return True
    return False


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class Attachment:
    id: str
    path: str
    alt: str


@dataclass
class TableItem:
    id: str
    format: str
    markdown: str
    data: List[List[str]]


@dataclass
class SubQuestion:
    label: str
    stem_lines: List[str] = field(default_factory=list)
    marks: Optional[int] = None
    tables: List[TableItem] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Question:
    question_no: str
    stem_lines: List[str] = field(default_factory=list)
    sub_questions: List[SubQuestion] = field(default_factory=list)
    options: Dict[str, str] = field(default_factory=dict)
    marks: Optional[int] = None
    tables: List[TableItem] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    section: str = ""
    group_id: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class Group:
    id: str
    section: str = ""
    shared_stem_lines: List[str] = field(default_factory=list)
    tables: List[TableItem] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

class QuestionSplitter:
    def __init__(self, profile: str = "auto", meta: Optional[Dict[str, Any]] = None) -> None:
        self.profile = profile
        self.meta = meta or {}
        self.meta_tables = {t.get("id"): t for t in self.meta.get("tables", []) if t.get("id")}
        self.groups: List[Group] = []
        self.current_group: Optional[Group] = None
        self.current_question: Optional[Question] = None
        self.current_sub: Optional[SubQuestion] = None
        self.shared_mode = False
        self.pending_table_lines: List[str] = []
        self.pending_table_id: Optional[str] = None
        self.in_code = False
        self.current_page: Optional[int] = None
        self.lines_since_page_comment = 0

    # ---------- context helpers ----------

    def start_group(self, title: str = "") -> Group:
        group = Group(id="g%04d" % (len(self.groups) + 1), section=title.strip())
        self.groups.append(group)
        self.current_group = group
        self.current_question = None
        self.current_sub = None
        self.shared_mode = True
        return group

    def ensure_group(self) -> Group:
        return self.current_group if self.current_group is not None else self.start_group()

    def start_question(self, question_no: str) -> Question:
        group = self.ensure_group()
        question = Question(
            question_no=question_no,
            section=group.section,
            group_id=group.id,
        )
        group.questions.append(question)
        self.current_question = question
        self.current_sub = None
        self.shared_mode = False
        return question

    def start_sub(self, label: str) -> SubQuestion:
        if self.current_question is None:
            # Be forgiving: a sub-question without a main number still gets a
            # synthetic parent so no text is silently dropped.
            self.start_question("?")
            if self.current_question is not None:
                self.current_question.warnings.append("sub_question_before_main_question")
        sub = SubQuestion(label=label)
        if self.current_question is not None:
            self.current_question.sub_questions.append(sub)
        self.current_sub = sub
        self.shared_mode = False
        return sub

    def append_text(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        if self.shared_mode and self.current_group is not None:
            self.current_group.shared_stem_lines.append(line)
        elif self.current_sub is not None:
            self.current_sub.stem_lines.append(line)
        elif self.current_question is not None:
            self.current_question.stem_lines.append(line)
        else:
            self.ensure_group().shared_stem_lines.append(line)

    # ---------- table helpers ----------

    def _table_from_meta(self, table_id: str) -> Optional[TableItem]:
        raw = self.meta_tables.get(table_id)
        if not raw:
            return None
        data = raw.get("data") or raw.get("rows") or []
        if not isinstance(data, list):
            data = []
        data = [[("" if c is None else str(c)) for c in row] for row in data]
        markdown = raw.get("markdown") or table_to_markdown(data)
        return TableItem(id=table_id, format="md", markdown=markdown, data=data)

    def flush_table(self) -> None:
        if not self.pending_table_lines:
            self.pending_table_id = None
            return
        table: Optional[TableItem] = None
        if self.pending_table_id:
            table = self._table_from_meta(self.pending_table_id)
        if table is None:
            data = parse_markdown_table(self.pending_table_lines)
            if data:
                table = TableItem(
                    id=self.pending_table_id or "",
                    format="md",
                    markdown=table_to_markdown(data),
                    data=data,
                )
        if table is not None:
            if self.current_sub is not None:
                self.current_sub.tables.append(table)
            elif self.current_question is not None:
                self.current_question.tables.append(table)
            else:
                self.ensure_group().tables.append(table)
        self.pending_table_lines = []
        self.pending_table_id = None

    # ---------- parsing ----------

    def parse(self, text: str) -> None:
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            self.lines_since_page_comment += 1

            if self.in_code:
                self.append_text(line)
                if line.strip().startswith("```"):
                    self.in_code = False
                continue

            if line.strip().startswith("```"):
                self.append_text(line)
                self.in_code = True
                continue

            page_comment = PAGE_COMMENT_RE.match(line)
            if page_comment:
                self.current_page = int(page_comment.group(1))
                self.lines_since_page_comment = 0
                continue

            table_comment = TABLE_COMMENT_RE.match(line)
            if table_comment:
                self.flush_table()
                self.pending_table_id = table_comment.group(1)
                continue

            if line.strip().startswith("|"):
                self.pending_table_lines.append(line)
                continue

            self.flush_table()

            plain = _strip_md_formatting(line)
            if not plain or _is_boilerplate_line(plain):
                continue

            # images: keep the original Markdown path, but attach it to the
            # current question/sub-question/group.
            image_match = IMAGE_RE.search(line)
            if image_match:
                attachment = Attachment(
                    id=image_match.group(1).strip() or "image",
                    path=image_match.group(2).strip(),
                    alt=image_match.group(1).strip(),
                )
                if self.current_sub is not None:
                    self.current_sub.attachments.append(attachment)
                elif self.current_question is not None:
                    self.current_question.attachments.append(attachment)
                else:
                    self.ensure_group().attachments.append(attachment)
                continue

            # headings / shared-stimulus boundaries
            if SECTION_RE.match(plain):
                self.start_group(plain)
                continue
            if STIMULUS_RE.match(plain):
                group = self.ensure_group()
                group.shared_stem_lines.append(plain)
                self.shared_mode = True
                continue

            # main question: 1. / 1) / 1、 / Q1 / Question 1
            main_match = MAIN_Q_RE.match(plain) or MAIN_Q_ALT_RE.match(plain)
            if main_match:
                question_no = main_match.group(1)
                rest = main_match.group(2).strip()
                self.start_question(question_no)
                if rest:
                    sub_match = self._match_sub(rest)
                    if sub_match:
                        label, sub_rest = sub_match
                        self.start_sub(label)
                        if sub_rest:
                            self.append_text(sub_rest)
                    else:
                        self.append_text(rest)
                continue

            # bare question number on its own line: "1"
            lone_match = MAIN_Q_LONE_RE.match(plain)
            if lone_match:
                value = lone_match.group(1)
                # A bare page number just after <!-- page: N --> is boilerplate,
                # not a question.  Give the real header a few lines of grace.
                if (
                    self.current_page is not None
                    and value == str(self.current_page)
                    and self.lines_since_page_comment <= 6
                ):
                    continue
                self.start_question(value)
                continue

            # sub-question
            sub_match = self._match_sub(plain)
            if sub_match and self.current_question is not None:
                label, rest = sub_match
                self.start_sub(label)
                if rest:
                    self.append_text(rest)
                continue

            # options
            opt_match = OPT_RE.match(plain)
            if opt_match and self.current_question is not None:
                label = opt_match.group(1).upper()
                rest = opt_match.group(2).strip()
                if rest:
                    self.current_question.options[label] = rest
                continue

            # marks: [2] or (2 marks)
            marks_match = MARKS_RE.search(plain)
            if marks_match:
                marks = int(marks_match.group(1) or marks_match.group(2))
                if self.current_sub is not None:
                    self.current_sub.marks = marks
                elif self.current_question is not None:
                    self.current_question.marks = marks
                plain = MARKS_RE.sub("", plain).strip()
                if not plain:
                    continue

            self.append_text(plain)

        self.flush_table()
        self._post_process()

    def _match_sub(self, line: str) -> Optional[Tuple[str, str]]:
        """Return (label, rest) for the first matching sub-question form."""
        patterns = []
        if self.profile == "cs":
            patterns = [SUB_LETTER_RE, SUB_ROMAN_RE, SUB_NUM_RE, SUB_TRAILING_LETTER_RE]
        elif self.profile == "history":
            patterns = [SUB_LETTER_RE, SUB_ROMAN_RE, SUB_NUM_RE]
        else:
            patterns = [SUB_LETTER_RE, SUB_ROMAN_RE, SUB_NUM_RE, SUB_TRAILING_LETTER_RE]
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                return "(%s)" % match.group(1), match.group(2).strip()
        return None

    def _post_process(self) -> None:
        for group in self.groups:
            group.shared_stem_lines = _clean_lines(group.shared_stem_lines)
            for question in group.questions:
                question.stem_lines = _clean_lines(question.stem_lines)
                for sub in question.sub_questions:
                    sub.stem_lines = _clean_lines(sub.stem_lines)
                    if not sub.stem_lines:
                        sub.warnings.append("empty_sub_question_stem")
                if not question.stem_lines and not question.sub_questions:
                    question.warnings.append("empty_question_stem")

    # ---------- output ----------

    def to_hierarchical(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for group in self.groups:
            out.append(
                {
                    "group_id": group.id,
                    "section": group.section,
                    "shared_stem": "\n".join(group.shared_stem_lines).strip(),
                    "tables": [asdict(t) for t in group.tables],
                    "attachments": [asdict(a) for a in group.attachments],
                    "questions": [
                        {
                            "question_no": q.question_no,
                            "section": q.section,
                            "group_id": q.group_id,
                            "stem": "\n".join(q.stem_lines).strip(),
                            "options": q.options,
                            "marks": q.marks,
                            "tables": [asdict(t) for t in q.tables],
                            "attachments": [asdict(a) for a in q.attachments],
                            "warnings": q.warnings,
                            "sub_questions": [
                                {
                                    "label": s.label,
                                    "stem": "\n".join(s.stem_lines).strip(),
                                    "marks": s.marks,
                                    "tables": [asdict(t) for t in s.tables],
                                    "attachments": [asdict(a) for a in s.attachments],
                                    "warnings": s.warnings,
                                }
                                for s in q.sub_questions
                            ],
                        }
                        for q in group.questions
                    ],
                }
            )
        return out

    def to_flat(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for group in self.groups:
            shared_stem = "\n".join(group.shared_stem_lines).strip()
            for question in group.questions:
                base = {
                    "question_no": question.question_no,
                    "section": question.section,
                    "group_id": question.group_id,
                    "parent_stem": "\n".join(question.stem_lines).strip(),
                    "shared_stem": shared_stem,
                    "options": question.options,
                    "marks": question.marks,
                    "tables": [asdict(t) for t in question.tables],
                    "attachments": [asdict(a) for a in question.attachments],
                    "warnings": list(question.warnings),
                }
                if question.sub_questions:
                    for index, sub in enumerate(question.sub_questions, 1):
                        record = dict(base)
                        record.update(
                            {
                                "id": "%s-%02d-%s"
                                % (
                                    question.question_no,
                                    index,
                                    sub.label.replace("(", "").replace(")", ""),
                                ),
                                "sub_question": sub.label,
                                "sub_question_index": index,
                                "stem": "\n".join(sub.stem_lines).strip(),
                                "marks": sub.marks if sub.marks is not None else question.marks,
                                "tables": [asdict(t) for t in sub.tables] + [asdict(t) for t in question.tables],
                                "attachments": [asdict(a) for a in sub.attachments]
                                + [asdict(a) for a in question.attachments],
                                "warnings": list(question.warnings) + list(sub.warnings),
                            }
                        )
                        records.append(record)
                else:
                    record = dict(base)
                    record.update(
                        {
                            "id": question.question_no,
                            "sub_question": None,
                            "stem": "\n".join(question.stem_lines).strip(),
                        }
                    )
                    records.append(record)
        return records


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean_lines(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        cleaned.append(line.strip())
    return cleaned


def load_meta(md_path: Path) -> Dict[str, Any]:
    meta_path = md_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Markdown exam papers into question JSON/JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("markdown", nargs="+", help="input Markdown path(s)")
    parser.add_argument("--out", "-o", default="data/questions", help="output directory")
    parser.add_argument(
        "--profile",
        choices=("auto", "history", "cs"),
        default="auto",
        help="rule profile for AS History / AS CS style papers",
    )
    parser.add_argument("--print-paths", action="store_true", help="print generated .jsonl paths to stdout")
    parser.add_argument(
        "--print0",
        action="store_true",
        help="like --print-paths but NUL-delimited for xargs -0",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = ensure_dir(Path(args.out).expanduser().resolve())

    failures = 0
    for raw in args.markdown:
        md_path = Path(raw).expanduser()
        if not md_path.exists():
            print("skip missing Markdown: %s" % md_path, file=sys.stderr)
            failures += 1
            continue

        try:
            text = md_path.read_text(encoding="utf-8")
            meta = load_meta(md_path)
            splitter = QuestionSplitter(profile=args.profile, meta=meta)
            splitter.parse(text)

            stem = safe_stem(md_path.name)
            hierarchical = splitter.to_hierarchical()
            flat = splitter.to_flat()

            questions_path = out_dir / ("%s.questions.json" % stem)
            jsonl_path = out_dir / ("%s.questions.jsonl" % stem)
            questions_path.write_text(
                json.dumps(hierarchical, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            jsonl_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in flat) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            print("error %s: %s" % (md_path, exc), file=sys.stderr)
            failures += 1
            continue

        if args.print_paths or args.print0:
            if args.print0:
                sys.stdout.buffer.write(str(jsonl_path.resolve()).encode("utf-8") + b"\0")
                sys.stdout.buffer.flush()
            else:
                print(str(jsonl_path.resolve()), flush=True)
        else:
            print("wrote %s" % jsonl_path, file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
