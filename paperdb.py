#!/usr/bin/env python3
"""paperdb.py - minimal SQLite index for exam paper corpus.

Source of truth stays on disk:

    PDF  ->  Markdown + assets + meta  ->  questions.json

This script builds a *rebuildable derived index* at ``data/papers.db``.  It
does not modify PDFs, Markdown or question JSON files.  Deleting the database
and running ``sync`` again must reproduce it.

Design:
  * ``papers``     : one row per unique PDF, keyed by sha256 prefix.
  * ``documents``  : one row per Markdown conversion, linked to a PDF.
  * ``questions``  : one row per stimulus / main question / sub-question,
                     linked to a document and, when applicable, to its parent.
  * ``topic_trees`` / ``topics`` : profile topic taxonomy; the canonical text
                     source is ``data/topics/<profile>.tree.md``.
  * ``question_topics`` / ``question_tags`` : Jev topic choices and positional
                     tags such as Paper 2 / Question 1 / (a).

Other tools can either import the query helpers or call the CLI.  The CLI
writes data to stdout (JSONL by default) and logs to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys

sys.dont_write_bytecode = True
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 2
DEFAULT_DB = "data/papers.db"
DEFAULT_PAPERS_DIR = "data/papers"
DEFAULT_MD_DIR = "data/md"
DEFAULT_QUESTIONS_DIR = "data/questions"
DEFAULT_TOPICS_DIR = "data/topics"

# Subject -> syllabus code.  Best effort; a URL or filename can override it.
SUBJECT_SYLLABUS = {
    "history": "9489",
    "computer-science": "9618",
    "computer science": "9618",
    "cs": "9618",
}

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
SESSION_RE = re.compile(
    r"(?i)\b((?:%s)\s+\d{4}|Specimen\s*(?:\(\s*\d{4}\s*\)|\d{4}))(?=\s|$|\.)"
    % MONTHS
)
VARIANT_RE = re.compile(r"(?i)\(\s*v(\d+)\s*\)")
KIND_RE = re.compile(r"(?i)(?:^|[\s_\-])(QP|MS)(?:[\s_.\-]|$)")
PAPER_CODE_RE = re.compile(r"(?i)Paper[\s_\-]*(\d+)")
# Four-digit syllabus codes (e.g. 9489, 9618) but not years such as 2021.
SYLLABUS_RE = re.compile(r"(?<!\d)((?!19\d\d|20\d\d)\d{4})(?!\d)")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    pdf_id       TEXT PRIMARY KEY,
    sha256       TEXT NOT NULL UNIQUE,
    path         TEXT NOT NULL,
    bytes        INTEGER,
    subject      TEXT,
    syllabus     TEXT,
    paper_code   TEXT,
    session      TEXT,
    variant      TEXT,
    kind         TEXT,
    source_url   TEXT,
    added_at     TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    md_id            TEXT PRIMARY KEY,
    pdf_id           TEXT NOT NULL REFERENCES papers(pdf_id) ON DELETE CASCADE,
    stem             TEXT NOT NULL,
    md_path          TEXT NOT NULL,
    meta_path        TEXT,
    assets_dir       TEXT,
    extractor        TEXT,
    extractor_version TEXT,
    content_hash     TEXT,
    created_at       TEXT
);

CREATE TABLE IF NOT EXISTS questions (
    question_id    TEXT PRIMARY KEY,
    md_id          TEXT NOT NULL REFERENCES documents(md_id) ON DELETE CASCADE,
    pdf_id         TEXT NOT NULL REFERENCES papers(pdf_id) ON DELETE CASCADE,
    parent_id      TEXT REFERENCES questions(question_id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,
    section        TEXT,
    group_id       TEXT,
    number         TEXT,
    sub_label      TEXT,
    flat_id        TEXT,
    order_index    INTEGER NOT NULL,
    stem           TEXT,
    parent_stem    TEXT,
    shared_stem    TEXT,
    marks          INTEGER,
    options_json   TEXT,
    tables_json    TEXT,
    attachments_json TEXT,
    warnings_json  TEXT,
    created_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_pdf
    ON documents(pdf_id);
CREATE INDEX IF NOT EXISTS idx_questions_md_order
    ON questions(md_id, order_index);
CREATE INDEX IF NOT EXISTS idx_questions_pdf_order
    ON questions(pdf_id, order_index);
CREATE INDEX IF NOT EXISTS idx_questions_parent
    ON questions(parent_id);
CREATE INDEX IF NOT EXISTS idx_questions_kind
    ON questions(kind);
CREATE INDEX IF NOT EXISTS idx_questions_number
    ON questions(number);
CREATE INDEX IF NOT EXISTS idx_questions_sub_label
    ON questions(sub_label);
CREATE INDEX IF NOT EXISTS idx_papers_subject
    ON papers(subject);
CREATE INDEX IF NOT EXISTS idx_papers_kind
    ON papers(kind);
CREATE INDEX IF NOT EXISTS idx_papers_session
    ON papers(session);

CREATE TABLE IF NOT EXISTS topic_trees (
    tree_id    TEXT PRIMARY KEY,
    profile    TEXT NOT NULL UNIQUE,
    title      TEXT,
    intro      TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id  TEXT PRIMARY KEY,
    tree_id   TEXT NOT NULL REFERENCES topic_trees(tree_id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES topics(topic_id) ON DELETE CASCADE,
    name      TEXT NOT NULL,
    intro     TEXT,
    position  INTEGER NOT NULL DEFAULT 0,
    path      TEXT NOT NULL DEFAULT '',
    path_key  TEXT NOT NULL DEFAULT '',
    created_at TEXT,
    updated_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_tree_path_key
    ON topics(tree_id, path_key);
CREATE INDEX IF NOT EXISTS idx_topics_tree_parent
    ON topics(tree_id, parent_id, position);

CREATE TABLE IF NOT EXISTS question_topics (
    question_id TEXT NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    topic_id    TEXT NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    source      TEXT NOT NULL DEFAULT 'jev',
    confidence  REAL,
    created_at  TEXT,
    PRIMARY KEY (question_id, topic_id, source)
);

CREATE INDEX IF NOT EXISTS idx_question_topics_question
    ON question_topics(question_id);
CREATE INDEX IF NOT EXISTS idx_question_topics_topic
    ON question_topics(topic_id);

CREATE TABLE IF NOT EXISTS question_tags (
    question_id TEXT NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    tag         TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'auto',
    created_at  TEXT,
    PRIMARY KEY (question_id, tag, source)
);

CREATE INDEX IF NOT EXISTS idx_question_tags_question
    ON question_tags(question_id);
CREATE INDEX IF NOT EXISTS idx_question_tags_tag
    ON question_tags(tag);
"""

QUESTION_COLUMNS = (
    "question_id",
    "md_id",
    "pdf_id",
    "parent_id",
    "kind",
    "section",
    "group_id",
    "number",
    "sub_label",
    "flat_id",
    "order_index",
    "stem",
    "parent_stem",
    "shared_stem",
    "marks",
    "options_json",
    "tables_json",
    "attachments_json",
    "warnings_json",
    "created_at",
)

INSERT_QUESTION_SQL = (
    "INSERT INTO questions (%s) VALUES (%s)"
    % (", ".join(QUESTION_COLUMNS), ", ".join("?" for _ in QUESTION_COLUMNS))
)

QUESTION_SELECT = """
SELECT
    q.*,
    d.md_path       AS md_path,
    d.meta_path     AS meta_path,
    d.assets_dir    AS assets_dir,
    d.stem          AS md_stem,
    p.path          AS paper_path,
    p.subject       AS subject,
    p.syllabus      AS syllabus,
    p.paper_code    AS paper_code,
    p.session       AS session,
    p.variant       AS variant,
    p.kind          AS paper_kind,
    p.source_url    AS source_url
FROM questions q
JOIN documents d ON q.md_id = d.md_id
JOIN papers p ON q.pdf_id = p.pdf_id
"""


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def connect(db_path: Path) -> sqlite3.Connection:
    if str(db_path.parent) and str(db_path.parent) != ".":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA user_version = %d" % SCHEMA_VERSION)
    conn.commit()


def _warn(message: str) -> None:
    print("paperdb: %s" % message, file=sys.stderr)


# ---------------------------------------------------------------------------
# paper metadata
# ---------------------------------------------------------------------------

def load_manifest(papers_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return (by_sha256, by_filename) from an optional manifest.jsonl."""
    by_sha: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    manifest_path = papers_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return by_sha, by_name
    for lineno, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            _warn("manifest.jsonl line %d is not JSON; ignored" % lineno)
            continue
        if not isinstance(record, dict):
            continue
        sha = record.get("sha256")
        if isinstance(sha, str) and sha:
            by_sha[sha.lower()] = record
        name = record.get("file")
        if isinstance(name, str) and name:
            by_name[name] = record
    return by_sha, by_name


def iter_pdf_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".pdf":
            yield path


def _first_match(pattern: re.Pattern, *values: str) -> Optional[str]:
    for value in values:
        if not value:
            continue
        match = pattern.search(value)
        if match:
            return match.group(1)
    return None


def derive_paper_meta(pdf_path: Path, manifest_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-effort metadata from manifest URL and/or filename/path."""
    url = ""
    if manifest_record:
        url = str(manifest_record.get("url") or "")

    # subject: /download/History/... or /Computer-Science/...
    subject = ""
    match = re.search(r"/download/([^/]+)/", url)
    if match:
        subject = match.group(1).strip().lower().replace("_", "-")
    if not subject:
        parent = pdf_path.parent.name
        if parent and parent.lower() not in ("papers", "data", "pdf", "pdfs"):
            subject = parent.strip().lower().replace("_", "-")

    # syllabus: explicit 4-digit code in URL or filename; otherwise subject map.
    syllabus = _first_match(SYLLABUS_RE, url, pdf_path.name)
    if not syllabus:
        syllabus = SUBJECT_SYLLABUS.get(subject, "")

    paper_code = _first_match(PAPER_CODE_RE, url, pdf_path.name) or ""

    kind = _first_match(KIND_RE, pdf_path.name)
    if kind:
        kind = kind.upper()

    variant_match = VARIANT_RE.search(pdf_path.name) or VARIANT_RE.search(url)
    variant = "v%s" % variant_match.group(1) if variant_match else ""

    session_match = SESSION_RE.search(pdf_path.name) or SESSION_RE.search(url)
    session = session_match.group(1) if session_match else ""
    if session:
        session = re.sub(r"\s+", " ", session).strip()

    return {
        "subject": subject,
        "syllabus": syllabus,
        "paper_code": paper_code,
        "session": session,
        "variant": variant,
        "kind": kind,
        "source_url": url,
    }


def resolve_pdf_id(conn: sqlite3.Connection, sha: str) -> str:
    """Pick a short sha prefix that does not collide with different content."""
    for length in (16, 20, 24, 32, 64):
        candidate = sha[:length]
        row = conn.execute(
            "SELECT sha256 FROM papers WHERE pdf_id = ?", (candidate,)
        ).fetchone()
        if row is None or row["sha256"] == sha:
            return candidate
    return sha


def upsert_paper(conn: sqlite3.Connection, meta: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO papers (
            pdf_id, sha256, path, bytes, subject, syllabus, paper_code,
            session, variant, kind, source_url, added_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pdf_id) DO UPDATE SET
            sha256 = excluded.sha256,
            path = excluded.path,
            bytes = excluded.bytes,
            subject = excluded.subject,
            syllabus = excluded.syllabus,
            paper_code = excluded.paper_code,
            session = excluded.session,
            variant = excluded.variant,
            kind = excluded.kind,
            source_url = excluded.source_url
        """,
        (
            meta["pdf_id"],
            meta["sha256"],
            meta["path"],
            meta.get("bytes"),
            meta.get("subject") or None,
            meta.get("syllabus") or None,
            meta.get("paper_code") or None,
            meta.get("session") or None,
            meta.get("variant") or None,
            meta.get("kind") or None,
            meta.get("source_url") or None,
            meta["added_at"],
        ),
    )


def sync_papers(
    conn: sqlite3.Connection, papers_dir: Path
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Scan PDFs, upsert papers, return path/name lookup maps."""
    by_sha, by_name = load_manifest(papers_dir)
    path_to_id: Dict[str, str] = {}
    name_to_ids: Dict[str, List[str]] = {}
    seen = 0

    if not papers_dir.exists():
        _warn("papers dir does not exist: %s" % papers_dir)
        return path_to_id, name_to_ids

    for pdf_path in iter_pdf_files(papers_dir):
        sha = sha256_file(pdf_path)
        record = by_sha.get(sha) or by_name.get(pdf_path.name)
        meta = derive_paper_meta(pdf_path, record)
        meta["sha256"] = sha
        meta["pdf_id"] = resolve_pdf_id(conn, sha)
        meta["path"] = str(pdf_path.resolve())
        meta["bytes"] = pdf_path.stat().st_size
        meta["added_at"] = now_utc()
        upsert_paper(conn, meta)
        path_to_id[str(pdf_path.resolve())] = meta["pdf_id"]
        path_to_id[str(pdf_path)] = meta["pdf_id"]
        name_to_ids.setdefault(pdf_path.name, []).append(meta["pdf_id"])
        seen += 1

    _warn("papers: %d" % seen)
    return path_to_id, name_to_ids


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def upsert_document(conn: sqlite3.Connection, doc: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO documents (
            md_id, pdf_id, stem, md_path, meta_path, assets_dir,
            extractor, extractor_version, content_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(md_id) DO UPDATE SET
            pdf_id = excluded.pdf_id,
            stem = excluded.stem,
            md_path = excluded.md_path,
            meta_path = excluded.meta_path,
            assets_dir = excluded.assets_dir,
            extractor = excluded.extractor,
            extractor_version = excluded.extractor_version,
            content_hash = excluded.content_hash
        """,
        (
            doc["md_id"],
            doc["pdf_id"],
            doc["stem"],
            doc["md_path"],
            doc.get("meta_path"),
            doc.get("assets_dir"),
            doc.get("extractor"),
            doc.get("extractor_version"),
            doc.get("content_hash"),
            doc["created_at"],
        ),
    )


def _lookup_pdf_id(
    source_pdf: str,
    path_to_id: Dict[str, str],
    name_to_ids: Dict[str, List[str]],
) -> Optional[str]:
    if not source_pdf:
        return None
    source_path = Path(source_pdf).expanduser()
    for candidate in (str(source_path), str(source_path.resolve())):
        if candidate in path_to_id:
            return path_to_id[candidate]
    ids = name_to_ids.get(source_path.name, [])
    if len(ids) == 1:
        return ids[0]
    return None


def sync_documents(
    conn: sqlite3.Connection,
    md_dir: Path,
    path_to_id: Dict[str, str],
    name_to_ids: Dict[str, List[str]],
) -> Dict[str, str]:
    """Scan *.meta.json, upsert documents, return stem -> md_id map."""
    stem_to_md_id: Dict[str, str] = {}
    if not md_dir.exists():
        _warn("md dir does not exist: %s" % md_dir)
        return stem_to_md_id

    seen = 0
    for meta_path in sorted(md_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _warn("cannot read %s: %s" % (meta_path, exc))
            continue
        if not isinstance(meta, dict):
            _warn("not a JSON object: %s" % meta_path)
            continue

        stem = str(meta.get("stem") or meta_path.name[: -len(".meta.json")])
        md_path = md_dir / ("%s.md" % stem)
        if not md_path.exists():
            _warn("missing markdown for %s: %s" % (meta_path.name, md_path))
            continue

        source_pdf = str(meta.get("source_pdf") or "")
        pdf_id = _lookup_pdf_id(source_pdf, path_to_id, name_to_ids)
        if not pdf_id:
            _warn(
                "cannot link %s to a PDF (source_pdf=%r)"
                % (meta_path.name, source_pdf)
            )
            continue

        md_id = "%s:md:%s" % (pdf_id, stem)
        upsert_document(
            conn,
            {
                "md_id": md_id,
                "pdf_id": pdf_id,
                "stem": stem,
                "md_path": str(md_path.resolve()),
                "meta_path": str(meta_path.resolve()),
                "assets_dir": str((md_dir / ("%s.assets" % stem)).resolve()),
                "extractor": meta.get("generator"),
                "extractor_version": str(meta.get("generator_version") or ""),
                "content_hash": sha256_file(md_path),
                "created_at": now_utc(),
            },
        )
        stem_to_md_id[stem] = md_id
        seen += 1

    _warn("documents: %d" % seen)
    return stem_to_md_id


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------

def _clean_label(label: str) -> str:
    return label.strip().strip("()").strip()


def flatten_questions(
    md_id: str, pdf_id: str, groups: Sequence[Dict[str, Any]]
) -> List[Tuple[Any, ...]]:
    """Turn hierarchical questions.json into flat rows for the questions table."""
    rows: List[Tuple[Any, ...]] = []
    created_at = now_utc()
    order = 0

    def add(
        *,
        question_id: str,
        parent_id: Optional[str],
        kind: str,
        section: str,
        group_id: str,
        number: str,
        sub_label: str,
        flat_id: str,
        stem: str,
        parent_stem: str,
        shared_stem: str,
        marks: Optional[int],
        options: Any,
        tables: Any,
        attachments: Any,
        warnings: Any,
    ) -> None:
        nonlocal order
        rows.append(
            (
                question_id,
                md_id,
                pdf_id,
                parent_id,
                kind,
                section or None,
                group_id or None,
                number or None,
                sub_label or None,
                flat_id or None,
                order,
                stem or None,
                parent_stem or None,
                shared_stem or None,
                marks,
                json_dumps(options if options is not None else {}),
                json_dumps(tables if tables is not None else []),
                json_dumps(attachments if attachments is not None else []),
                json_dumps(warnings if warnings is not None else []),
                created_at,
            )
        )
        order += 1

    for group_index, group in enumerate(groups, 1):
        if not isinstance(group, dict):
            _warn("skipping non-object group in %s" % md_id)
            continue
        group_id = str(group.get("group_id") or ("g%04d" % group_index))
        section = str(group.get("section") or "")
        shared_stem = str(group.get("shared_stem") or "").strip()
        group_tables = group.get("tables") or []
        group_attachments = group.get("attachments") or []

        questions = group.get("questions") or []
        if not isinstance(questions, list):
            continue
        has_questions = bool(questions)

        stimulus_id: Optional[str] = None
        if has_questions and (shared_stem or group_tables or group_attachments):
            stimulus_id = "%s:%s:__stimulus__" % (pdf_id, group_id)
            add(
                question_id=stimulus_id,
                parent_id=None,
                kind="stimulus",
                section=section,
                group_id=group_id,
                number="",
                sub_label="",
                flat_id="__stimulus__",
                stem=shared_stem,
                parent_stem="",
                shared_stem=shared_stem,
                marks=None,
                options={},
                tables=group_tables,
                attachments=group_attachments,
                warnings=[],
            )

        for question in questions:
            if not isinstance(question, dict):
                continue
            number = str(question.get("question_no") or "?").strip() or "?"
            main_id = "%s:%s:%s" % (pdf_id, group_id, number)
            main_stem = str(question.get("stem") or "").strip()
            question_marks = question.get("marks")
            if not isinstance(question_marks, int):
                question_marks = None
            question_warnings = question.get("warnings") or []
            add(
                question_id=main_id,
                parent_id=stimulus_id,
                kind="main",
                section=section,
                group_id=group_id,
                number=number,
                sub_label="",
                flat_id=number,
                stem=main_stem,
                parent_stem="",
                shared_stem=shared_stem,
                marks=question_marks,
                options=question.get("options") or {},
                tables=question.get("tables") or [],
                attachments=question.get("attachments") or [],
                warnings=question_warnings,
            )

            sub_questions = question.get("sub_questions") or []
            if not isinstance(sub_questions, list):
                continue
            for sub_index, sub in enumerate(sub_questions, 1):
                if not isinstance(sub, dict):
                    continue
                label = str(sub.get("label") or "")
                clean = _clean_label(label) or str(sub_index)
                flat_id = "%s-%02d-%s" % (number, sub_index, clean)
                sub_id = "%s:%s:%s" % (pdf_id, group_id, flat_id)
                sub_marks = sub.get("marks")
                if not isinstance(sub_marks, int):
                    sub_marks = question_marks
                sub_warnings = list(question_warnings) + list(sub.get("warnings") or [])
                add(
                    question_id=sub_id,
                    parent_id=main_id,
                    kind="sub",
                    section=section,
                    group_id=group_id,
                    number=number,
                    sub_label=label,
                    flat_id=flat_id,
                    stem=str(sub.get("stem") or "").strip(),
                    parent_stem=main_stem,
                    shared_stem=shared_stem,
                    marks=sub_marks,
                    options={},
                    tables=sub.get("tables") or [],
                    attachments=sub.get("attachments") or [],
                    warnings=sub_warnings,
                )

    return rows


def sync_questions(
    conn: sqlite3.Connection,
    questions_dir: Path,
    stem_to_md_id: Dict[str, str],
) -> int:
    """Replace questions for every document whose questions file is present."""
    if not questions_dir.exists():
        _warn("questions dir does not exist: %s" % questions_dir)
        return 0

    inserted = 0
    files = 0
    for qpath in sorted(questions_dir.glob("*.questions.json")):
        stem = qpath.name[: -len(".questions.json")]
        md_id = stem_to_md_id.get(stem)
        if not md_id:
            _warn("no document for %s; skipped" % qpath.name)
            continue
        row = conn.execute(
            "SELECT pdf_id FROM documents WHERE md_id = ?", (md_id,)
        ).fetchone()
        if row is None:
            _warn("document disappeared for %s; skipped" % qpath.name)
            continue
        pdf_id = row["pdf_id"]

        try:
            groups = json.loads(qpath.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _warn("cannot read %s: %s" % (qpath, exc))
            continue
        if not isinstance(groups, list):
            _warn("%s does not contain a JSON list; skipped" % qpath.name)
            continue

        # Question ids are deterministic, so re-syncing a document should not
        # throw away Jev/manual topic tags.  Save them before the delete and
        # restore rows whose question id still exists afterwards.
        old_ids = [
            row["question_id"]
            for row in conn.execute(
                "SELECT question_id FROM questions WHERE md_id = ?", (md_id,)
            )
        ]
        saved_topics: List[sqlite3.Row] = []
        saved_tags: List[sqlite3.Row] = []
        if old_ids:
            placeholders = ", ".join("?" for _ in old_ids)
            saved_topics = list(
                conn.execute(
                    "SELECT * FROM question_topics WHERE question_id IN (%s)"
                    % placeholders,
                    old_ids,
                )
            )
            saved_tags = list(
                conn.execute(
                    "SELECT * FROM question_tags WHERE question_id IN (%s)"
                    % placeholders,
                    old_ids,
                )
            )

        conn.execute("DELETE FROM questions WHERE md_id = ?", (md_id,))
        rows = flatten_questions(md_id, pdf_id, groups)
        if rows:
            conn.executemany(INSERT_QUESTION_SQL, rows)
        inserted += len(rows)
        files += 1

        new_ids = {row[0] for row in rows}
        valid_topic_ids = {
            row["topic_id"] for row in conn.execute("SELECT topic_id FROM topics")
        }
        for saved in saved_topics:
            if saved["question_id"] not in new_ids:
                continue
            if saved["topic_id"] not in valid_topic_ids:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO question_topics "
                "(question_id, topic_id, source, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    saved["question_id"],
                    saved["topic_id"],
                    saved["source"],
                    saved["confidence"],
                    saved["created_at"],
                ),
            )
        for saved in saved_tags:
            if saved["question_id"] not in new_ids:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO question_tags "
                "(question_id, tag, source, created_at) VALUES (?, ?, ?, ?)",
                (
                    saved["question_id"],
                    saved["tag"],
                    saved["source"],
                    saved["created_at"],
                ),
            )

    _warn("questions: %d rows from %d files" % (inserted, files))
    return inserted


# ---------------------------------------------------------------------------
# query helpers
# ---------------------------------------------------------------------------

def row_to_paper(row: sqlite3.Row, question_count: Optional[int] = None) -> Dict[str, Any]:
    data = {
        "pdf_id": row["pdf_id"],
        "sha256": row["sha256"],
        "path": row["path"],
        "bytes": row["bytes"],
        "subject": row["subject"],
        "syllabus": row["syllabus"],
        "paper_code": row["paper_code"],
        "session": row["session"],
        "variant": row["variant"],
        "kind": row["kind"],
        "source_url": row["source_url"],
        "added_at": row["added_at"],
    }
    if question_count is not None:
        data["question_count"] = question_count
    return data


def row_to_question(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "pdf_id": row["pdf_id"],
        "md_id": row["md_id"],
        "parent_id": row["parent_id"],
        "kind": row["kind"],
        "section": row["section"],
        "group_id": row["group_id"],
        "number": row["number"],
        "sub_label": row["sub_label"],
        "flat_id": row["flat_id"],
        "order_index": row["order_index"],
        "stem": row["stem"],
        "parent_stem": row["parent_stem"],
        "shared_stem": row["shared_stem"],
        "marks": row["marks"],
        "options": json_loads(row["options_json"], {}),
        "tables": json_loads(row["tables_json"], []),
        "attachments": json_loads(row["attachments_json"], []),
        "warnings": json_loads(row["warnings_json"], []),
        "md_path": row["md_path"],
        "meta_path": row["meta_path"],
        "assets_dir": row["assets_dir"],
        "md_stem": row["md_stem"],
        "paper": {
            "path": row["paper_path"],
            "subject": row["subject"],
            "syllabus": row["syllabus"],
            "paper_code": row["paper_code"],
            "session": row["session"],
            "variant": row["variant"],
            "kind": row["paper_kind"],
            "source_url": row["source_url"],
        },
    }


def query_papers(
    conn: sqlite3.Connection,
    subject: Optional[str] = None,
    kind: Optional[str] = None,
    session: Optional[str] = None,
    syllabus: Optional[str] = None,
    paper_code: Optional[str] = None,
) -> List[sqlite3.Row]:
    sql = [
        """
        SELECT p.*,
            (SELECT COUNT(*) FROM questions q
             WHERE q.pdf_id = p.pdf_id AND q.kind IN ('main', 'sub')) AS question_count
        FROM papers p
        WHERE 1 = 1
        """
    ]
    params: List[Any] = []
    if subject:
        sql.append("AND lower(p.subject) = lower(?)")
        params.append(subject)
    if kind:
        sql.append("AND upper(p.kind) = upper(?)")
        params.append(kind)
    if session:
        sql.append("AND lower(p.session) LIKE lower(?)")
        params.append("%" + session + "%")
    if syllabus:
        sql.append("AND p.syllabus = ?")
        params.append(syllabus)
    if paper_code:
        sql.append("AND p.paper_code = ?")
        params.append(paper_code)
    sql.append("ORDER BY p.subject, p.session, p.variant, p.kind, p.path")
    return list(conn.execute(" ".join(sql), params))


def get_paper(conn: sqlite3.Connection, pdf_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT p.*,
            (SELECT COUNT(*) FROM questions q
             WHERE q.pdf_id = p.pdf_id AND q.kind IN ('main', 'sub')) AS question_count
        FROM papers p WHERE p.pdf_id = ?
        """,
        (pdf_id,),
    ).fetchone()
    if row is None:
        return None
    return row_to_paper(row, row["question_count"])


def get_question(conn: sqlite3.Connection, question_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        QUESTION_SELECT + " WHERE q.question_id = ?", (question_id,)
    ).fetchone()
    if row is None:
        return None
    return row_to_question(row)


def get_documents(conn: sqlite3.Connection, pdf_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM documents WHERE pdf_id = ? ORDER BY stem", (pdf_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def get_children(
    conn: sqlite3.Connection, question_id: str
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        QUESTION_SELECT + " WHERE q.parent_id = ? ORDER BY q.order_index",
        (question_id,),
    ).fetchall()
    return [row_to_question(row) for row in rows]


def search_questions(
    conn: sqlite3.Connection, text: str, limit: int = 200
) -> List[Dict[str, Any]]:
    like = "%" + text + "%"
    rows = conn.execute(
        QUESTION_SELECT
        + """
        WHERE q.kind IN ('main', 'sub')
          AND (q.stem LIKE ? OR q.parent_stem LIKE ? OR q.shared_stem LIKE ?)
        ORDER BY p.path, q.order_index
        LIMIT ?
        """,
        (like, like, like, limit),
    ).fetchall()
    return [row_to_question(row) for row in rows]


def query_questions(
    conn: sqlite3.Connection,
    paper: Optional[str] = None,
    subject: Optional[str] = None,
    kinds: Optional[Sequence[str]] = None,
    question_id: Optional[str] = None,
    section: Optional[str] = None,
    number: Optional[str] = None,
    sub_label: Optional[str] = None,
    include_stimulus: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[sqlite3.Row]:
    sql = [QUESTION_SELECT, "WHERE 1 = 1"]
    params: List[Any] = []
    if paper:
        sql.append("AND q.pdf_id = ?")
        params.append(paper)
    if subject:
        sql.append("AND lower(p.subject) = lower(?)")
        params.append(subject)
    if question_id:
        sql.append("AND q.question_id = ?")
        params.append(question_id)
    if section:
        sql.append("AND lower(q.section) LIKE lower(?)")
        params.append("%" + section + "%")
    if number:
        sql.append("AND q.number = ?")
        params.append(number)
    if sub_label:
        sql.append("AND q.sub_label = ?")
        params.append(sub_label)

    if kinds:
        placeholders = ", ".join("?" for _ in kinds)
        sql.append("AND q.kind IN (%s)" % placeholders)
        params.extend(kinds)
    elif not include_stimulus:
        sql.append("AND q.kind IN ('main', 'sub')")

    sql.append("ORDER BY p.path, q.order_index")
    if limit is not None:
        sql.append("LIMIT ? OFFSET ?")
        params.extend([limit, offset])
    return list(conn.execute(" ".join(sql), params))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def emit_jsonl(records: Iterable[Dict[str, Any]]) -> None:
    for record in records:
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def emit_records(records: List[Dict[str, Any]], fmt: str) -> None:
    if fmt == "json":
        sys.stdout.write(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    elif fmt == "text":
        for record in records:
            sys.stdout.write(format_text_record(record) + "\n")
    else:
        emit_jsonl(records)
    sys.stdout.flush()


def format_text_record(record: Dict[str, Any]) -> str:
    if "pdf_id" in record and "path" in record:
        return "%s  %-8s %-3s %-12s %s  questions=%s" % (
            record.get("pdf_id", ""),
            record.get("subject") or "",
            record.get("kind") or "",
            record.get("session") or "",
            record.get("path") or "",
            record.get("question_count", ""),
        )
    if "question_id" in record:
        label = record.get("flat_id") or record.get("number") or ""
        if record.get("sub_label"):
            label = "%s %s" % (label, record["sub_label"])
        stem = (record.get("stem") or "").replace("\n", " ")
        if len(stem) > 90:
            stem = stem[:87] + "..."
        marks = record.get("marks")
        return "%s  %-8s %-8s %s" % (
            record.get("question_id", ""),
            record.get("kind", ""),
            ("[%s]" % marks) if marks is not None else "",
            stem,
        )
    return json.dumps(record, ensure_ascii=False)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    conn = connect(db_path)
    init_schema(conn)
    conn.close()
    print("initialized %s (schema v%d)" % (db_path, SCHEMA_VERSION), file=sys.stderr)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    conn = connect(db_path)
    init_schema(conn)

    papers_dir = Path(args.papers_dir).expanduser()
    md_dir = Path(args.md).expanduser()
    questions_dir = Path(args.questions).expanduser()
    topics_dir = Path(
        getattr(args, "topics_dir", DEFAULT_TOPICS_DIR) or DEFAULT_TOPICS_DIR
    ).expanduser()

    path_to_id, name_to_ids = sync_papers(conn, papers_dir)
    stem_to_md_id = sync_documents(conn, md_dir, path_to_id, name_to_ids)
    sync_questions(conn, questions_dir, stem_to_md_id)

    # Topic trees are optional, but when present they are synced from the
    # canonical *.tree.md files.  Auto (positional) tags are always refreshed
    # because they depend only on the indexed questions and paper metadata.
    topic_stats = {"topic_files": 0, "topic_nodes": 0, "auto_tag_questions": 0}
    try:
        import topic_tree

        synced = topic_tree.sync_topic_trees(conn, topics_dir)
        topic_stats["topic_files"] = synced.get("files", 0)
        topic_stats["topic_nodes"] = synced.get("nodes", 0)
        topic_stats["auto_tag_questions"] = topic_tree.refresh_auto_tags(conn)
    except Exception as exc:  # pragma: no cover - defensive, keep paper sync alive
        _warn("topic tree sync failed: %s" % exc)
    conn.commit()

    stats = {
        "papers": conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
        "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "questions": conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0],
        "main_or_sub": conn.execute(
            "SELECT COUNT(*) FROM questions WHERE kind IN ('main', 'sub')"
        ).fetchone()[0],
        "topic_trees": conn.execute("SELECT COUNT(*) FROM topic_trees").fetchone()[0],
        "topics": conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0],
        "question_topics": conn.execute(
            "SELECT COUNT(*) FROM question_topics"
        ).fetchone()[0],
        "question_tags": conn.execute("SELECT COUNT(*) FROM question_tags").fetchone()[0],
    }
    stats.update(topic_stats)
    conn.close()
    _warn("database: %s" % db_path)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return 0


def cmd_papers(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    rows = query_papers(
        conn,
        subject=args.subject,
        kind=args.kind,
        session=args.session,
        syllabus=args.syllabus,
        paper_code=args.paper_code,
    )
    records = [row_to_paper(row, row["question_count"]) for row in rows]
    emit_records(records, args.format)
    conn.close()
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    kinds = None
    if args.kind:
        kinds = [k.strip() for k in args.kind.split(",") if k.strip()]
    rows = query_questions(
        conn,
        paper=args.paper,
        subject=args.subject,
        kinds=kinds,
        question_id=args.question_id,
        section=args.section,
        number=args.number,
        sub_label=args.sub_label,
        include_stimulus=args.include_stimulus,
        limit=args.limit,
        offset=args.offset,
    )
    records = [row_to_question(row) for row in rows]
    emit_records(records, args.format)
    conn.close()
    return 0


def _paper_record(conn: sqlite3.Connection, pdf_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT p.*,
            (SELECT COUNT(*) FROM questions q
             WHERE q.pdf_id = p.pdf_id AND q.kind IN ('main', 'sub')) AS question_count
        FROM papers p WHERE p.pdf_id = ?
        """,
        (pdf_id,),
    ).fetchone()
    if row is None:
        return None
    return row_to_paper(row, row["question_count"])


def cmd_show_paper(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    paper = _paper_record(conn, args.pdf_id)
    if paper is None:
        _warn("paper not found: %s" % args.pdf_id)
        conn.close()
        return 2
    documents = [
        {
            "md_id": row["md_id"],
            "stem": row["stem"],
            "md_path": row["md_path"],
            "meta_path": row["meta_path"],
            "assets_dir": row["assets_dir"],
            "content_hash": row["content_hash"],
        }
        for row in conn.execute(
            "SELECT * FROM documents WHERE pdf_id = ? ORDER BY stem", (args.pdf_id,)
        )
    ]
    questions = [
        row_to_question(row)
        for row in query_questions(conn, paper=args.pdf_id, include_stimulus=True)
    ]
    payload = {
        "paper": paper,
        "documents": documents,
        "questions": questions,
    }
    if args.format == "jsonl":
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()
    conn.close()
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    ids = []
    if args.ids:
        ids.extend(args.ids)
    if args.stdin:
        ids.extend(line.strip() for line in sys.stdin if line.strip())

    if not ids:
        _warn("no ids given; use positional ids or --stdin")
        conn.close()
        return 2

    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for ident in ids:
        paper = _paper_record(conn, ident)
        if paper is not None:
            records.append(paper)
            continue
        row = conn.execute(QUESTION_SELECT + " WHERE q.question_id = ?", (ident,)).fetchone()
        if row is not None:
            records.append(row_to_question(row))
            continue
        missing.append(ident)
        _warn("not found: %s" % ident)

    emit_records(records, args.format)
    conn.close()
    return 2 if missing else 0


def cmd_md(args: argparse.Namespace) -> int:
    """Print the Markdown text for a paper or document to stdout."""
    conn = connect(Path(args.db))
    row = conn.execute(
        """
        SELECT md_id, md_path FROM documents
        WHERE md_id = ? OR pdf_id = ?
        ORDER BY stem
        LIMIT 1
        """,
        (args.id, args.id),
    ).fetchone()
    if row is None:
        _warn("no markdown document for: %s" % args.id)
        conn.close()
        return 2
    md_path = Path(row["md_path"])
    if not md_path.exists():
        _warn("markdown file is missing: %s" % md_path)
        conn.close()
        return 2
    sys.stdout.write(md_path.read_text(encoding="utf-8"))
    sys.stdout.flush()
    conn.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    init_schema(conn)
    stats: Dict[str, Any] = {}
    stats["papers"] = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    stats["documents"] = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    stats["questions_total"] = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    stats["topic_trees"] = conn.execute("SELECT COUNT(*) FROM topic_trees").fetchone()[0]
    stats["topics"] = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    stats["question_topics"] = conn.execute(
        "SELECT COUNT(*) FROM question_topics"
    ).fetchone()[0]
    stats["question_tags"] = conn.execute("SELECT COUNT(*) FROM question_tags").fetchone()[0]
    stats["questions_by_kind"] = {
        row["kind"]: row["n"]
        for row in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM questions GROUP BY kind ORDER BY kind"
        )
    }
    stats["papers_by_subject"] = {
        row["subject"] or "": row["n"]
        for row in conn.execute(
            "SELECT subject, COUNT(*) AS n FROM papers GROUP BY subject ORDER BY subject"
        )
    }
    stats["papers_by_kind"] = {
        row["kind"] or "": row["n"]
        for row in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM papers GROUP BY kind ORDER BY kind"
        )
    }
    if args.format == "json":
        sys.stdout.write(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    elif args.format == "text":
        sys.stdout.write(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(json.dumps(stats, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    conn.close()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperdb.py",
        description="Minimal SQLite index for exam paper PDFs, Markdown and questions.",
    )
    parser.add_argument(
        "--db", default=None, help="SQLite database path (default: %s)" % DEFAULT_DB
    )
    parser.add_argument(
        "--topics-dir",
        default=DEFAULT_TOPICS_DIR,
        help="topic tree directory (default: %s)" % DEFAULT_TOPICS_DIR,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create an empty database and schema")
    p_init.set_defaults(func=cmd_init)

    p_sync = sub.add_parser(
        "sync",
        help="scan PDF / Markdown / questions / topic-tree files and upsert the index",
    )
    p_sync.add_argument("--papers-dir", default=DEFAULT_PAPERS_DIR, help="PDF corpus dir")
    p_sync.add_argument("--md", default=DEFAULT_MD_DIR, help="Markdown output dir")
    p_sync.add_argument(
        "--questions", default=DEFAULT_QUESTIONS_DIR, help="questions output dir"
    )
    p_sync.set_defaults(func=cmd_sync)

    p_papers = sub.add_parser("papers", help="list papers")
    p_papers.add_argument("--subject")
    p_papers.add_argument("--kind", help="QP or MS")
    p_papers.add_argument("--session")
    p_papers.add_argument("--syllabus")
    p_papers.add_argument("--paper-code")
    p_papers.add_argument(
        "--format", choices=("jsonl", "json", "text"), default="jsonl"
    )
    p_papers.set_defaults(func=cmd_papers)

    p_questions = sub.add_parser("questions", help="query questions")
    p_questions.add_argument("--paper", help="paper pdf_id")
    p_questions.add_argument("--subject")
    p_questions.add_argument(
        "--kind",
        help="comma separated kinds: main,sub,stimulus (default: main,sub)",
    )
    p_questions.add_argument("--question-id")
    p_questions.add_argument("--section")
    p_questions.add_argument("--number")
    p_questions.add_argument("--sub-label")
    p_questions.add_argument(
        "--include-stimulus",
        action="store_true",
        help="include kind=stimulus shared-material rows",
    )
    p_questions.add_argument("--limit", type=int)
    p_questions.add_argument("--offset", type=int, default=0)
    p_questions.add_argument(
        "--format", choices=("jsonl", "json", "text"), default="jsonl"
    )
    p_questions.set_defaults(func=cmd_questions)

    p_show = sub.add_parser("show-paper", help="show one paper with documents and questions")
    p_show.add_argument("pdf_id")
    p_show.add_argument("--format", choices=("json", "jsonl"), default="json")
    p_show.set_defaults(func=cmd_show_paper)

    p_md = sub.add_parser("md", help="print Markdown text for a paper or document")
    p_md.add_argument("id", help="pdf_id or md_id")
    p_md.set_defaults(func=cmd_md)

    p_get = sub.add_parser("get", help="get papers or questions by id")
    p_get.add_argument("ids", nargs="*", help="paper ids or question ids")
    p_get.add_argument(
        "--stdin",
        action="store_true",
        help="also read ids from stdin, one per line",
    )
    p_get.add_argument(
        "--format", choices=("jsonl", "json", "text"), default="jsonl"
    )
    p_get.set_defaults(func=cmd_get)

    p_stats = sub.add_parser("stats", help="show index statistics")
    p_stats.add_argument(
        "--format", choices=("jsonl", "json", "text"), default="json"
    )
    p_stats.set_defaults(func=cmd_stats)

    # Allow --db after the subcommand as well, e.g.
    #   paperdb.py sync --db data/papers.db
    # without clobbering the global default when it is omitted.
    for subparser in sub.choices.values():
        subparser.add_argument(
            "--db", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
        subparser.add_argument(
            "--topics-dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "db", None) is None:
        args.db = DEFAULT_DB
    try:
        return int(args.func(args))
    except sqlite3.Error as exc:
        _warn("sqlite error: %s" % exc)
        return 1
    except KeyboardInterrupt:
        _warn("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
