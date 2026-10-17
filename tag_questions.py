#!/usr/bin/env python3
"""tag_questions.py - assign profile topic-tree tags to questions with Jev.

For every indexed main/sub question this script builds a compact but complete
context (paper metadata + shared source + parent stem + stem + tables +
attachments) and asks Jev to choose exactly one topic from the profile's topic
tree.  The result is written to ``question_topics`` with ``source='jev'``.

Positional (``source='auto'``) tags such as ``Paper 2`` / ``Question 1`` /
``(a)`` are refreshed first, so they are present even when Jev is not run.

Typical use::

    export OPENCODE_API_KEY=...
    python3 tag_questions.py --profile history --paper <pdf_id> --dry-run
    python3 tag_questions.py --profile history --paper <pdf_id>

The script is deliberately resumable: without ``--refresh`` it skips questions
that already have a Jev topic tag.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jev  # noqa: E402
import paper_common  # noqa: E402
import paperdb  # noqa: E402
import topic_tree  # noqa: E402

DEFAULT_DB = "data/papers.db"
DEFAULT_TOPICS_DIR = topic_tree.DEFAULT_TOPICS_DIR
MAX_CHOICES = topic_tree.MAX_CHOICES


class TaggingError(Exception):
    """Raised for user-facing tagging failures."""


def _warn(message: str) -> None:
    sys.stderr.write("tag_questions: %s\n" % message)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# question context
# ---------------------------------------------------------------------------

def _tables_text(tables: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for table in tables or []:
        if not isinstance(table, dict):
            continue
        markdown = table.get("markdown") or ""
        if not markdown and table.get("data"):
            markdown = paper_common.table_to_markdown(table["data"])
        if markdown:
            parts.append(markdown)
    return "\n\n".join(parts)


def _attachments_text(attachments: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for attachment in attachments or []:
        if not isinstance(attachment, dict):
            continue
        alt = attachment.get("alt") or attachment.get("id") or "image"
        path = attachment.get("path") or ""
        lines.append("- %s (%s)" % (alt, path))
    return "\n".join(lines)


def _question_children(
    conn: sqlite3.Connection, question: Dict[str, Any]
) -> List[Dict[str, Any]]:
    children = paperdb.get_children(conn, question["question_id"])
    result: List[Dict[str, Any]] = []
    for child in children:
        result.append(child)
        result.extend(_question_children(conn, child))
    return result


def build_question_context(
    conn: sqlite3.Connection,
    question: Dict[str, Any],
    tree: topic_tree.TopicTree,
    *,
    include_children: bool = True,
) -> Dict[str, Any]:
    """Build the JSON-serialisable state for one Jev request."""
    paper = question.get("paper") or {}
    content_lines: List[str] = []
    content_lines.append(
        "Paper: %s %s %s %s %s"
        % (
            paper.get("subject") or "",
            paper.get("syllabus") or "",
            paper.get("paper_code") or "",
            paper.get("session") or "",
            paper.get("kind") or "",
        )
    )
    if question.get("section"):
        content_lines.append("Section: %s" % question["section"])
    label = str(question.get("number") or "")
    if question.get("sub_label"):
        label = ("%s%s" % (label, question["sub_label"])).strip()
    content_lines.append("Question label: %s" % (label or question.get("flat_id") or ""))
    if question.get("marks") is not None:
        content_lines.append("Marks: %s" % question["marks"])
    if question.get("shared_stem"):
        content_lines.append("Shared source / stimulus:\n%s" % question["shared_stem"])
    if question.get("parent_stem"):
        content_lines.append("Parent question stem:\n%s" % question["parent_stem"])
    if question.get("stem"):
        content_lines.append("Question stem:\n%s" % question["stem"])
    if question.get("tables"):
        content_lines.append("Tables:\n%s" % _tables_text(question["tables"]))
    if question.get("attachments"):
        content_lines.append("Attachments:\n%s" % _attachments_text(question["attachments"]))

    if include_children and question.get("kind") == "main":
        children = _question_children(conn, question)
        if children:
            child_lines = []
            for child in children:
                child_label = "%s%s" % (
                    child.get("number") or "",
                    child.get("sub_label") or "",
                )
                child_stem = (child.get("stem") or "").strip() or "(no stem)"
                child_lines.append("- %s: %s" % (child_label, child_stem))
            content_lines.append("Sub-questions:\n" + "\n".join(child_lines))

    topics = [
        {
            "id": node.topic_id,
            "name": node.name,
            "path": node.path,
            "intro": node.intro,
        }
        for node in tree.walk_nodes()
    ]
    return {
        "background": {"title": tree.title, "intro": tree.intro},
        "paper": {
            "pdf_id": paper.get("pdf_id") or question.get("pdf_id"),
            "subject": paper.get("subject") or "",
            "syllabus": paper.get("syllabus") or "",
            "paper_code": paper.get("paper_code") or "",
            "session": paper.get("session") or "",
            "variant": paper.get("variant") or "",
            "kind": paper.get("kind") or "",
        },
        "question": {
            "id": question.get("question_id"),
            "kind": question.get("kind"),
            "section": question.get("section") or "",
            "number": question.get("number") or "",
            "sub_label": question.get("sub_label") or "",
            "flat_id": question.get("flat_id") or "",
            "marks": question.get("marks"),
            "text": "\n\n".join(content_lines),
        },
        "topics": topics,
    }


# ---------------------------------------------------------------------------
# choices / Jev call
# ---------------------------------------------------------------------------

def build_choices(
    tree: topic_tree.TopicTree, *, leaves_only: bool = False
) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    nodes = list(tree.walk_nodes())
    for node in nodes:
        if leaves_only and node.children:
            continue
        if not node.topic_id:
            continue
        description = node.path or node.name
        if node.intro:
            description = "%s — %s" % (description, node.intro)
        choices[node.topic_id] = description
    if not choices:
        raise TaggingError("topic tree has no usable choices")
    if len(choices) > MAX_CHOICES:
        raise TaggingError(
            "topic tree has %d choices; Jev supports at most %d (use --leaves-only)"
            % (len(choices), MAX_CHOICES)
        )
    return choices


def build_payload(
    state: Dict[str, Any], choices: Dict[str, str], *, leaves_only: bool = False
) -> Dict[str, Any]:
    instructions = (
        "You are tagging one exam question with the single best topic from the "
        "provided topic tree. Use the question text, the shared source/stimulus, "
        "the marks, and the command word. Prefer the most specific matching "
        "topic. If no specific topic fits, choose the closest ancestor. "
        "Return exactly one option key."
    )
    if leaves_only:
        instructions += " Only leaf topics are provided as choices."
    return {
        "state": state,
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": instructions,
                "criteria": choices,
            }
        },
    }


def _parse_answer(response: Any) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
    if not isinstance(response, dict):
        raise TaggingError("unexpected Jev response: %r" % (response,))
    answers = response.get("answers") or {}
    answer = answers.get("topic") or {}
    if not isinstance(answer, dict):
        raise TaggingError("Jev answer for 'topic' is not an object")
    choice = answer.get("choice")
    if not isinstance(choice, str):
        return None, None, answer
    confidence = answer.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
    return choice, confidence, answer


# ---------------------------------------------------------------------------
# selection / storage
# ---------------------------------------------------------------------------

def resolve_paper_arg(conn: sqlite3.Connection, value: str) -> Optional[str]:
    """Resolve a pdf_id, absolute path, or bare filename to a pdf_id."""
    row = conn.execute("SELECT pdf_id FROM papers WHERE pdf_id = ?", (value,)).fetchone()
    if row is not None:
        return row["pdf_id"]
    path = Path(value)
    candidates = []
    if path.exists():
        candidates.append(str(path.resolve()))
    candidates.append(str(path))
    for candidate in candidates:
        row = conn.execute(
            "SELECT pdf_id FROM papers WHERE path = ? OR path = ?",
            (candidate, value),
        ).fetchone()
        if row is not None:
            return row["pdf_id"]
    row = conn.execute(
        "SELECT pdf_id FROM papers WHERE path LIKE ? ORDER BY path LIMIT 2",
        ("%" + Path(value).name,),
    ).fetchall()
    if len(row) == 1:
        return row[0]["pdf_id"]
    return None


def select_questions(
    conn: sqlite3.Connection,
    *,
    pdf_ids: Optional[Sequence[str]] = None,
    include_ms: bool = False,
    refresh: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sql = [paperdb.QUESTION_SELECT, "WHERE q.kind IN ('main', 'sub')"]
    params: List[Any] = []
    if pdf_ids:
        placeholders = ", ".join("?" for _ in pdf_ids)
        sql.append("AND q.pdf_id IN (%s)" % placeholders)
        params.extend(pdf_ids)
    if not include_ms:
        sql.append("AND upper(coalesce(p.kind, '')) IN ('', 'QP')")
    if not refresh:
        sql.append(
            "AND NOT EXISTS (SELECT 1 FROM question_topics qt "
            "WHERE qt.question_id = q.question_id AND qt.source = 'jev')"
        )
    sql.append("ORDER BY p.path, q.order_index")
    if limit is not None:
        sql.append("LIMIT ?")
        params.append(limit)
    rows = conn.execute(" ".join(sql), params).fetchall()
    return [paperdb.row_to_question(row) for row in rows]


def store_topic_tag(
    conn: sqlite3.Connection,
    question_id: str,
    topic_id: str,
    confidence: Optional[float],
) -> None:
    conn.execute(
        "DELETE FROM question_topics WHERE question_id = ? AND source = 'jev'",
        (question_id,),
    )
    conn.execute(
        "INSERT INTO question_topics (question_id, topic_id, source, confidence, created_at) "
        "VALUES (?, ?, 'jev', ?, ?)",
        (question_id, topic_id, confidence, paperdb.now_utc()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tag_questions.py",
        description="Tag indexed questions with a profile topic tree using Jev.",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument(
        "--topics-dir", default=DEFAULT_TOPICS_DIR, help="topic tree directory"
    )
    parser.add_argument("--profile", default="history", help="profile / tree name")
    parser.add_argument(
        "--paper",
        action="append",
        default=[],
        help="pdf_id, PDF path, or bare filename; repeatable",
    )
    parser.add_argument("--all", action="store_true", help="tag every indexed QP question")
    parser.add_argument(
        "--include-ms", action="store_true", help="also tag MS papers (normally empty)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-tag questions that already have a Jev tag",
    )
    parser.add_argument("--limit", type=int, help="maximum number of questions to process")
    parser.add_argument(
        "--leaves-only",
        action="store_true",
        help="offer only leaf topics to Jev",
    )
    parser.add_argument("--dry-run", action="store_true", help="print payloads; do not call Jev")
    parser.add_argument("--auto-only", action="store_true", help="only refresh positional tags")
    parser.add_argument("--api-key", help="Jev API key (default: environment)")
    parser.add_argument("--api-key-file", help="file containing a Jev API key")
    parser.add_argument("--base-url", default=jev.DEFAULT_BASE_URL, help="Jev base URL")
    parser.add_argument("--model", default=jev.DEFAULT_MODEL, help="Jev model")
    parser.add_argument("--timeout", type=float, default=jev.DEFAULT_TIMEOUT, help="HTTP timeout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    topics_dir = Path(args.topics_dir).expanduser()
    try:
        conn = paperdb.connect(db_path)
        paperdb.init_schema(conn)
    except sqlite3.Error as exc:
        _warn("cannot open database: %s" % exc)
        return 2

    try:
        if args.auto_only:
            auto_pdf_ids: List[str] = []
            for value in args.paper:
                pdf_id = resolve_paper_arg(conn, value)
                if not pdf_id:
                    _warn("paper not found: %s" % value)
                    return 2
                auto_pdf_ids.append(pdf_id)
            count = topic_tree.refresh_auto_tags(conn, auto_pdf_ids or None)
            sys.stdout.write(json.dumps({"auto_tag_questions": count}, ensure_ascii=False) + "\n")
            return 0

        if args.dry_run:
            tree = topic_tree.get_tree(conn, args.profile)
            if tree is None or not tree.nodes:
                path = topic_tree.canonical_path(topics_dir, args.profile)
                if path.exists():
                    tree = topic_tree.load_tree_file(path, default_profile=args.profile)
                    tree = topic_tree.assign_topic_ids(tree)
                else:
                    raise topic_tree.TopicTreeMissing(
                        "profile %r has no topic tree; create one with `topics edit %s`"
                        % (args.profile, args.profile)
                    )
        else:
            tree = topic_tree.ensure_tree(conn, args.profile, topics_dir)
        choices = build_choices(tree, leaves_only=args.leaves_only)

        pdf_ids: List[str] = []
        for value in args.paper:
            pdf_id = resolve_paper_arg(conn, value)
            if not pdf_id:
                _warn("paper not found: %s" % value)
                return 2
            pdf_ids.append(pdf_id)
        if not args.paper and not args.all:
            # Default to every indexed QP question.  This is the common
            # "tag the corpus" path; --paper is available for a narrower run.
            args.all = True

        questions = select_questions(
            conn,
            pdf_ids=pdf_ids or None,
            include_ms=args.include_ms,
            refresh=args.refresh,
            limit=args.limit,
        )
        auto_count = (
            0
            if args.dry_run
            else topic_tree.refresh_auto_tags(conn, pdf_ids or None)
        )

        if not questions:
            sys.stdout.write(
                json.dumps(
                    {
                        "profile": args.profile,
                        "tree_nodes": tree.node_count(),
                        "auto_tag_questions": auto_count,
                        "questions": 0,
                        "tagged": 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            return 0

        if not args.dry_run:
            api_key = jev.resolve_api_key(args.api_key, args.api_key_file)
            if not api_key:
                _warn(
                    "missing API key: set OPENCODE_API_KEY (or use --api-key/--api-key-file)"
                )
                return 2
        else:
            api_key = None

        tagged = 0
        skipped = 0
        usage_totals: Dict[str, int] = {}
        for index, question in enumerate(questions, 1):
            context = build_question_context(conn, question, tree)
            payload = build_payload(context, choices, leaves_only=args.leaves_only)
            if args.dry_run:
                sys.stdout.write(
                    json.dumps(
                        {
                            "question_id": question["question_id"],
                            "context": context,
                            "choices": list(choices.keys()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            _warn(
                "[%d/%d] %s %s"
                % (
                    index,
                    len(questions),
                    question.get("pdf_id") or "",
                    question["question_id"],
                )
            )
            try:
                response = jev.evaluate(
                    payload["state"],
                    payload["questions"],
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    timeout=args.timeout,
                )
            except jev.JevError as exc:
                _warn("Jev error on %s: %s" % (question["question_id"], exc))
                return 1

            choice, confidence, answer = _parse_answer(response)
            if not choice or choice not in choices:
                _warn(
                    "question %s returned unknown choice %r; skipped"
                    % (question["question_id"], choice)
                )
                skipped += 1
                continue
            store_topic_tag(conn, question["question_id"], choice, confidence)
            tagged += 1
            usage = response.get("usage") or {}
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, int):
                        usage_totals[key] = usage_totals.get(key, 0) + value

        summary = {
            "profile": args.profile,
            "tree_nodes": tree.node_count(),
            "choices": len(choices),
            "auto_tag_questions": auto_count,
            "questions": len(questions),
            "tagged": tagged,
            "skipped": skipped,
            "dry_run": bool(args.dry_run),
        }
        if usage_totals:
            summary["usage"] = usage_totals
        sys.stdout.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
        return 0
    except topic_tree.TopicTreeError as exc:
        _warn("%s" % exc)
        return 2
    except sqlite3.Error as exc:
        _warn("sqlite error: %s" % exc)
        return 2
    except KeyboardInterrupt:
        _warn("interrupted")
        return 130
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
