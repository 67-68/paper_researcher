#!/usr/bin/env python3
"""topic_tree.py - profile topic trees and question tagging helpers.

A *profile* (for example ``history`` or ``cs``) can own one topic tree.  The
tree is stored in SQLite for fast lookups, but it also has a canonical
human-editable form on disk::

    data/topics/history.tree.md

The on-disk format is a small YAML front matter block followed by a nested
unordered list::

    ---
    profile: "history"
    title: "AS History 9489"
    intro: "Thematic tree for AS History 9489."
    ---

    - Peace treaties: 1919-1923 settlements.
      - Treaty of Versailles: The 1919 treaty with Germany.
      - Treaty of Saint-Germain: The 1919 treaty with Austria.
    - Cold War: 1945-1991 superpower rivalry.

Node lines use ``name: intro``.  A missing ``: `` means the node has no
introduction.  Colons inside a *name* can be escaped as ``\\: ``.

The database is a derived index: importing a file replaces the tree rows for
that profile, but topic ids are preserved by name-path so that re-importing a
reordered tree keeps existing tags.  Pending edits live in
``data/topics/.pending/<profile>.tree.md`` and are deleted after a successful
import.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.dont_write_bytecode = True

import paperdb

DEFAULT_TOPICS_DIR = "data/topics"
PENDING_DIRNAME = ".pending"
TREE_SUFFIX = ".tree.md"

MAX_TREE_INTRO = 100
MAX_NODE_INTRO = 40
MAX_CHOICES = 255

TOPIC_COLUMNS = (
    "topic_id",
    "tree_id",
    "parent_id",
    "name",
    "intro",
    "position",
    "path",
    "path_key",
    "created_at",
    "updated_at",
)

INSERT_TOPIC_SQL = (
    "INSERT INTO topics (%s) VALUES (%s) "
    "ON CONFLICT(topic_id) DO UPDATE SET "
    "tree_id = excluded.tree_id, "
    "parent_id = excluded.parent_id, "
    "name = excluded.name, "
    "intro = excluded.intro, "
    "position = excluded.position, "
    "path = excluded.path, "
    "path_key = excluded.path_key, "
    "updated_at = excluded.updated_at"
    % (", ".join(TOPIC_COLUMNS), ", ".join("?" for _ in TOPIC_COLUMNS))
)


class TopicTreeError(Exception):
    """Base class for topic-tree errors."""


class TopicTreeMissing(TopicTreeError):
    """Raised when a profile has no usable topic tree."""


class TopicTreeValidationError(TopicTreeError):
    """Raised when an edited tree violates the format rules."""


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class TopicNode:
    name: str
    intro: str = ""
    children: List["TopicNode"] = field(default_factory=list)
    topic_id: Optional[str] = None
    path: str = ""
    path_key: str = ""
    position: int = 0

    def walk(self) -> Iterable["TopicNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class TopicTree:
    profile: str
    title: str = ""
    intro: str = ""
    nodes: List[TopicNode] = field(default_factory=list)
    tree_id: Optional[str] = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.tree_id:
            self.tree_id = self.profile.strip()

    def node_count(self) -> int:
        return sum(1 for _ in self.walk_nodes())

    def walk_nodes(self) -> Iterable[TopicNode]:
        for node in self.nodes:
            yield from node.walk()


# ---------------------------------------------------------------------------
# path / file helpers
# ---------------------------------------------------------------------------

def profile_slug(profile: str) -> str:
    """Return a filesystem-safe slug for *profile*."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(profile).strip().lower())
    slug = slug.strip("-._")
    return slug or "profile"


def canonical_path(topics_dir: Path, profile: str) -> Path:
    return Path(topics_dir) / ("%s%s" % (profile_slug(profile), TREE_SUFFIX))


def pending_path(topics_dir: Path, profile: str) -> Path:
    return Path(topics_dir) / PENDING_DIRNAME / ("%s%s" % (profile_slug(profile), TREE_SUFFIX))


def pending_files(topics_dir: Path) -> List[Path]:
    directory = Path(topics_dir) / PENDING_DIRNAME
    if not directory.exists():
        return []
    return sorted(directory.glob("*%s" % TREE_SUFFIX))


def profile_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith(TREE_SUFFIX):
        return name[: -len(TREE_SUFFIX)]
    if name.endswith(".md"):
        return name[:-3]
    return path.stem


# ---------------------------------------------------------------------------
# markdown parsing / rendering
# ---------------------------------------------------------------------------

def _parse_yaml_scalar(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if raw[0] in ('"', "'"):
        if raw[0] == '"':
            try:
                value = json.loads(raw)
                return value if isinstance(value, str) else str(value)
            except ValueError:
                pass
        else:
            if len(raw) >= 2 and raw[-1] == "'":
                return raw[1:-1].replace("''", "'")
    if raw in ("null", "~", "Null", "NULL"):
        return ""
    return raw


def _parse_front_matter(lines: Sequence[str], start: int = 1, end: Optional[int] = None) -> Dict[str, str]:
    """Parse the simple ``key: value`` YAML subset used by tree files."""
    meta: Dict[str, str] = {}
    stop = len(lines) if end is None else end
    index = start
    while index < stop:
        raw = lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        if raw[:1].isspace():
            raise TopicTreeValidationError(
                "front matter line %d is unexpectedly indented: %r" % (index + 1, raw)
            )
        key, separator, value = raw.partition(":")
        if not separator:
            raise TopicTreeValidationError(
                "front matter line %d is not 'key: value': %r" % (index + 1, raw)
            )
        key = key.strip()
        value = value.strip()
        if value in ("|", ">"):
            block: List[str] = []
            index += 1
            base_indent: Optional[int] = None
            while index < stop:
                block_raw = lines[index]
                if not block_raw.strip():
                    block.append("")
                    index += 1
                    continue
                indent = len(block_raw) - len(block_raw.lstrip(" "))
                if base_indent is None:
                    base_indent = indent
                if indent < base_indent:
                    break
                block.append(block_raw[base_indent:])
                index += 1
            if value == "|":
                meta[key] = "\n".join(block).rstrip()
            else:
                meta[key] = " ".join(part for part in block if part).strip()
            continue
        meta[key] = _parse_yaml_scalar(value)
        index += 1
    return meta


def _split_node_line(content: str) -> Tuple[str, str]:
    """Split ``name: intro`` at the first unescaped ``: ``."""
    index = 0
    while True:
        found = content.find(": ", index)
        if found == -1:
            return content, ""
        if found > 0 and content[found - 1] == "\\":
            index = found + 1
            continue
        name = content[:found]
        intro = content[found + 2 :]
        return name.strip(), intro.strip()


def _unescape_name(name: str) -> str:
    return name.replace("\\:", ":").replace("\\\\", "\\")


def _escape_name(name: str) -> str:
    return name.replace("\\", "\\\\").replace(": ", "\\: ")


def parse_topic_markdown(text: str, *, default_profile: str = "") -> TopicTree:
    """Parse a tree Markdown document into a :class:`TopicTree`."""
    lines = text.splitlines()
    meta: Dict[str, str] = {}
    body = list(lines)
    if lines and lines[0].strip() == "---":
        end = None
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                end = index
                break
        if end is None:
            raise TopicTreeValidationError("front matter starts with '---' but never ends")
        meta = _parse_front_matter(lines, 1, end)
        body = lines[end + 1 :]

    profile = str(meta.get("profile") or default_profile or "").strip()
    title = str(meta.get("title") or "").strip()
    intro = str(meta.get("intro") or "").strip()
    tree = TopicTree(profile=profile, title=title, intro=intro)

    stack: List[Tuple[int, TopicNode]] = []
    for offset, raw in enumerate(body, 1):
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        if "\t" in raw:
            raise TopicTreeValidationError("line %d contains a tab; use spaces" % offset)
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if not stripped.startswith("- "):
            raise TopicTreeValidationError(
                "line %d is not an unordered-list item: %r" % (offset, raw)
            )
        if indent % 2:
            raise TopicTreeValidationError(
                "line %d has odd indentation (%d spaces); use 2 spaces per level"
                % (offset, indent)
            )
        content = stripped[2:].strip()
        if not content:
            raise TopicTreeValidationError("line %d has an empty list item" % offset)
        name, intro = _split_node_line(content)
        name = _unescape_name(name).strip()
        if not name:
            raise TopicTreeValidationError("line %d has an empty node name" % offset)
        depth = indent // 2
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if depth > 0 and (not stack or stack[-1][0] != depth - 1):
            raise TopicTreeValidationError(
                "line %d jumps to depth %d without a parent at depth %d"
                % (offset, depth, depth - 1)
            )
        node = TopicNode(name=name, intro=intro)
        if stack:
            stack[-1][1].children.append(node)
        else:
            tree.nodes.append(node)
        stack.append((depth, node))

    return tree


def render_topic_markdown(tree: TopicTree) -> str:
    """Render *tree* as YAML front matter + nested unordered list."""
    lines = ["---"]
    lines.append("profile: %s" % json.dumps(tree.profile, ensure_ascii=False))
    lines.append("title: %s" % json.dumps(tree.title or "", ensure_ascii=False))
    lines.append("intro: %s" % json.dumps(tree.intro or "", ensure_ascii=False))
    lines.append("---")
    lines.append("")

    def render_node(node: TopicNode, depth: int) -> List[str]:
        prefix = "  " * depth + "- "
        name = _escape_name(node.name.strip())
        intro = " ".join((node.intro or "").split())
        if intro:
            line = "%s%s: %s" % (prefix, name, intro)
        else:
            line = "%s%s" % (prefix, name)
        rendered = [line]
        for child in node.children:
            rendered.extend(render_node(child, depth + 1))
        return rendered

    for node in tree.nodes:
        lines.extend(render_node(node, 0))
    return "\n".join(lines).rstrip() + "\n"


def render_tree_text(tree: TopicTree) -> str:
    """Render a compact plain-text tree for terminals."""
    lines = []
    if tree.title or tree.intro:
        lines.append("profile: %s" % (tree.profile or ""))
        if tree.title:
            lines.append("title: %s" % tree.title)
        if tree.intro:
            lines.append("intro: %s" % tree.intro)
        lines.append("")

    def walk(nodes: Sequence[TopicNode], depth: int) -> None:
        for node in nodes:
            suffix = (": %s" % node.intro) if node.intro else ""
            lines.append("%s- %s%s" % ("  " * depth, node.name, suffix))
            walk(node.children, depth + 1)

    walk(tree.nodes, 0)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def validate_tree(tree: TopicTree, *, require_nodes: bool = True) -> List[str]:
    errors: List[str] = []
    if not tree.profile.strip():
        errors.append("profile is required")
    if len(tree.intro or "") > MAX_TREE_INTRO:
        errors.append(
            "tree intro is %d characters; maximum is %d"
            % (len(tree.intro), MAX_TREE_INTRO)
        )
    if require_nodes and not tree.nodes:
        errors.append("tree has no nodes; add at least one '- node' line")

    def walk(nodes: Sequence[TopicNode], path: str, depth: int) -> None:
        if depth > 20:
            errors.append("tree is deeper than 20 levels at %s" % (path or "root"))
            return
        seen: Set[str] = set()
        for node in nodes:
            name = node.name.strip()
            if not name:
                errors.append("empty node name under %s" % (path or "root"))
                continue
            if name in seen:
                errors.append("duplicate sibling node %r under %s" % (name, path or "root"))
            seen.add(name)
            if len(node.intro or "") > MAX_NODE_INTRO:
                errors.append(
                    "node %r intro is %d characters; maximum is %d"
                    % (name, len(node.intro), MAX_NODE_INTRO)
                )
            child_path = "%s / %s" % (path, name) if path else name
            walk(node.children, child_path, depth + 1)

    walk(tree.nodes, "", 1)
    return errors


def load_tree_file(path: Path, *, default_profile: str = "") -> TopicTree:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise TopicTreeError("cannot read %s: %s" % (path, exc)) from exc
    return parse_topic_markdown(text, default_profile=default_profile)


def write_tree_file(path: Path, tree: TopicTree) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_topic_markdown(tree), encoding="utf-8")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _tree_id(profile: str) -> str:
    return str(profile).strip()


def get_tree(conn: sqlite3.Connection, profile: str) -> Optional[TopicTree]:
    tree_id = _tree_id(profile)
    row = conn.execute(
        "SELECT * FROM topic_trees WHERE tree_id = ? OR profile = ?",
        (tree_id, tree_id),
    ).fetchone()
    if row is None:
        return None
    tree = TopicTree(
        profile=row["profile"] or row["tree_id"],
        title=row["title"] or "",
        intro=row["intro"] or "",
        tree_id=row["tree_id"],
        updated_at=row["updated_at"] or "",
    )
    nodes: Dict[str, TopicNode] = {}
    order: List[Tuple[str, Optional[str]]] = []
    for topic_row in conn.execute(
        "SELECT * FROM topics WHERE tree_id = ? ORDER BY position, topic_id",
        (row["tree_id"],),
    ):
        node = TopicNode(
            name=topic_row["name"],
            intro=topic_row["intro"] or "",
            topic_id=topic_row["topic_id"],
            path=topic_row["path"] or "",
            path_key=topic_row["path_key"] or "",
            position=topic_row["position"] or 0,
        )
        nodes[topic_row["topic_id"]] = node
        order.append((topic_row["topic_id"], topic_row["parent_id"]))
    for topic_id, parent_id in order:
        node = nodes[topic_id]
        if parent_id and parent_id in nodes:
            nodes[parent_id].children.append(node)
        else:
            tree.nodes.append(node)
    return tree


def list_trees(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT t.*,
            (SELECT COUNT(*) FROM topics x WHERE x.tree_id = t.tree_id) AS node_count
        FROM topic_trees t
        ORDER BY t.profile
        """
    ).fetchall()
    return [dict(row) for row in rows]


def assign_topic_ids(tree: TopicTree) -> TopicTree:
    """Populate ``topic_id`` / ``path`` fields without touching the database."""
    tree_id = _tree_id(tree.profile)

    def walk(nodes: Sequence[TopicNode], parent_key: str, parent_path: str) -> None:
        for node in nodes:
            name = node.name.strip()
            key = name if not parent_key else parent_key + "\x1f" + name
            path = name if not parent_path else parent_path + " / " + name
            node.topic_id = "%s:%s" % (tree_id, paperdb.short_hash(key, 12))
            node.path = path
            node.path_key = key
            walk(node.children, key, path)

    walk(tree.nodes, "", "")
    return tree


def _flatten_for_save(
    tree: TopicTree, existing_by_key: Dict[str, str]
) -> List[Dict[str, Any]]:
    tree_id = _tree_id(tree.profile)
    rows: List[Dict[str, Any]] = []
    used_ids: Set[str] = set(existing_by_key.values())

    def new_topic_id(path_key: str) -> str:
        base = "%s:%s" % (tree_id, paperdb.short_hash(path_key, 12))
        candidate = base
        counter = 2
        while candidate in used_ids and existing_by_key.get(path_key) != candidate:
            candidate = "%s-%d" % (base, counter)
            counter += 1
        used_ids.add(candidate)
        return candidate

    def walk(nodes: Sequence[TopicNode], parent_id: Optional[str], parent_key: str, parent_path: str) -> None:
        for node in nodes:
            name = node.name.strip()
            key = name if not parent_key else parent_key + "\x1f" + name
            path = name if not parent_path else parent_path + " / " + name
            topic_id = existing_by_key.get(key) or new_topic_id(key)
            rows.append(
                {
                    "topic_id": topic_id,
                    "tree_id": tree_id,
                    "parent_id": parent_id,
                    "name": name,
                    "intro": (node.intro or "").strip(),
                    "position": len(rows),
                    "path": path,
                    "path_key": key,
                }
            )
            node.topic_id = topic_id
            node.path = path
            node.path_key = key
            node.position = len(rows) - 1
            walk(node.children, topic_id, key, path)

    walk(tree.nodes, None, "", "")
    return rows


def save_tree(conn: sqlite3.Connection, tree: TopicTree) -> Dict[str, int]:
    """Replace a profile's tree rows while preserving ids by name-path."""
    errors = validate_tree(tree, require_nodes=True)
    if errors:
        raise TopicTreeValidationError("; ".join(errors))

    tree_id = _tree_id(tree.profile)
    tree.tree_id = tree_id
    now = paperdb.now_utc()
    conn.execute(
        """
        INSERT INTO topic_trees (tree_id, profile, title, intro, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(tree_id) DO UPDATE SET
            profile = excluded.profile,
            title = excluded.title,
            intro = excluded.intro,
            updated_at = excluded.updated_at
        """,
        (tree_id, tree.profile.strip(), tree.title or "", tree.intro or "", now, now),
    )

    existing_by_key = {
        row["path_key"]: row["topic_id"]
        for row in conn.execute(
            "SELECT path_key, topic_id FROM topics WHERE tree_id = ?", (tree_id,)
        )
        if row["path_key"]
    }
    existing_ids = set(existing_by_key.values())
    rows = _flatten_for_save(tree, existing_by_key)
    new_ids = {row["topic_id"] for row in rows}
    stale_ids = existing_ids - new_ids

    # Remove stale rows first.  This frees unique (tree_id, path_key) slots
    # before renamed/moved rows are upserted.  Deleting a stale parent also
    # cascades to its stale descendants.
    for topic_id in stale_ids:
        conn.execute("DELETE FROM topics WHERE topic_id = ?", (topic_id,))

    for row in rows:
        conn.execute(
            INSERT_TOPIC_SQL,
            (
                row["topic_id"],
                row["tree_id"],
                row["parent_id"],
                row["name"],
                row["intro"],
                row["position"],
                row["path"],
                row["path_key"],
                now,
                now,
            ),
        )
    conn.commit()
    return {"nodes": len(rows), "removed": len(stale_ids)}


def load_tree(conn: sqlite3.Connection, profile: str, topics_dir: Path) -> Optional[TopicTree]:
    """Return the DB tree, importing the canonical file if necessary."""
    tree = get_tree(conn, profile)
    if tree is not None and tree.nodes:
        return tree
    path = canonical_path(topics_dir, profile)
    if path.exists():
        loaded = load_tree_file(path, default_profile=profile)
        save_tree(conn, loaded)
        return get_tree(conn, loaded.profile)
    return None


def ensure_tree(conn: sqlite3.Connection, profile: str, topics_dir: Path) -> TopicTree:
    tree = load_tree(conn, profile, topics_dir)
    if tree is None or not tree.nodes:
        raise TopicTreeMissing(
            "profile %r has no topic tree; create one with `topics edit %s`"
            % (profile, profile)
        )
    return tree


def sync_topic_trees(conn: sqlite3.Connection, topics_dir: Path) -> Dict[str, int]:
    """Import every canonical ``*.tree.md`` file into the database."""
    topics_dir = Path(topics_dir)
    if not topics_dir.exists():
        return {"files": 0, "nodes": 0}
    files = 0
    nodes = 0
    for path in sorted(topics_dir.glob("*%s" % TREE_SUFFIX)):
        try:
            tree = load_tree_file(path, default_profile=profile_from_filename(path))
            save_tree(conn, tree)
        except TopicTreeError as exc:
            paperdb._warn("topic tree skipped %s: %s" % (path, exc))
            continue
        files += 1
        nodes += tree.node_count()
    return {"files": files, "nodes": nodes}


def import_tree_file(
    conn: sqlite3.Connection,
    source: Path,
    *,
    topics_dir: Path,
    default_profile: str = "",
    delete_source: bool = False,
) -> Tuple[TopicTree, Dict[str, int]]:
    source = Path(source)
    tree = load_tree_file(source, default_profile=default_profile or profile_from_filename(source))
    errors = validate_tree(tree, require_nodes=True)
    if errors:
        raise TopicTreeValidationError("%s: %s" % (source, "; ".join(errors)))
    stats = save_tree(conn, tree)
    canonical = canonical_path(topics_dir, tree.profile)
    write_tree_file(canonical, tree)
    if delete_source and source.resolve() != canonical.resolve() and source.exists():
        try:
            source.unlink()
        except OSError as exc:
            paperdb._warn("cannot remove pending file %s: %s" % (source, exc))
    return tree, stats


# ---------------------------------------------------------------------------
# auto tags
# ---------------------------------------------------------------------------

def _clean_label(label: str) -> str:
    return str(label or "").strip().strip("()").strip()


def auto_tags_for_question(question: Dict[str, Any], paper: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return positional tags such as Paper 2 / Question 1 / (a) / Section A."""
    paper = paper or question.get("paper") or {}
    tags: List[str] = []
    paper_code = str(paper.get("paper_code") or "").strip()
    match = re.search(r"\d+", paper_code)
    if match:
        tags.append("Paper %s" % match.group(0))
    section = str(question.get("section") or "").strip()
    if section:
        if section.lower().startswith(("section ", "paper ")):
            tags.append(section)
        else:
            tags.append("Section %s" % section)
    number = str(question.get("number") or "").strip()
    if number and number != "?":
        tags.append("Question %s" % number)
    sub_label = _clean_label(question.get("sub_label") or "")
    if sub_label:
        tags.append("(%s)" % sub_label)
    seen: Set[str] = set()
    ordered: List[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def _question_rows_for_auto_tags(
    conn: sqlite3.Connection, pdf_ids: Optional[Sequence[str]] = None
) -> List[sqlite3.Row]:
    sql = [paperdb.QUESTION_SELECT, "WHERE q.kind IN ('main', 'sub')"]
    params: List[Any] = []
    if pdf_ids:
        placeholders = ", ".join("?" for _ in pdf_ids)
        sql.append("AND q.pdf_id IN (%s)" % placeholders)
        params.extend(pdf_ids)
    sql.append("ORDER BY p.path, q.order_index")
    return list(conn.execute(" ".join(sql), params))


def refresh_auto_tags(
    conn: sqlite3.Connection, pdf_ids: Optional[Sequence[str]] = None
) -> int:
    """Recompute the positional ``source='auto'`` tags for questions."""
    rows = _question_rows_for_auto_tags(conn, pdf_ids)
    now = paperdb.now_utc()
    touched = 0
    for row in rows:
        question = paperdb.row_to_question(row)
        tags = auto_tags_for_question(question, question.get("paper") or {})
        conn.execute(
            "DELETE FROM question_tags WHERE question_id = ? AND source = 'auto'",
            (question["question_id"],),
        )
        for tag in tags:
            conn.execute(
                "INSERT OR IGNORE INTO question_tags (question_id, tag, source, created_at) "
                "VALUES (?, ?, 'auto', ?)",
                (question["question_id"], tag, now),
            )
        touched += 1
    conn.commit()
    return touched


# ---------------------------------------------------------------------------
# tag retrieval (parent questions inherit descendant tags)
# ---------------------------------------------------------------------------

def _question_descendants(conn: sqlite3.Connection, question_id: str) -> List[str]:
    rows = conn.execute(
        """
        WITH RECURSIVE descendants(question_id) AS (
            SELECT question_id FROM questions WHERE question_id = ?
            UNION ALL
            SELECT q.question_id FROM questions q
            JOIN descendants d ON q.parent_id = d.question_id
        )
        SELECT question_id FROM descendants
        """,
        (question_id,),
    ).fetchall()
    return [row["question_id"] for row in rows]


def _question_ancestors(conn: sqlite3.Connection, question_ids: Sequence[str]) -> Set[str]:
    ids = [qid for qid in question_ids if qid]
    if not ids:
        return set()
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        """
        WITH RECURSIVE ancestors(question_id, parent_id) AS (
            SELECT question_id, parent_id FROM questions
            WHERE question_id IN (%s)
            UNION
            SELECT q.question_id, q.parent_id FROM questions q
            JOIN ancestors a ON q.question_id = a.parent_id
        )
        SELECT question_id FROM ancestors
        """ % placeholders,
        ids,
    ).fetchall()
    result = {row["question_id"] for row in rows}
    if not result:
        return result
    keep = {
        row["question_id"]
        for row in conn.execute(
            "SELECT question_id FROM questions WHERE question_id IN (%s) AND kind IN ('main', 'sub')"
            % ", ".join("?" for _ in result),
            tuple(sorted(result)),
        )
    }
    return keep


def _query_questions_by_ids(
    conn: sqlite3.Connection, question_ids: Sequence[str], limit: int = 200
) -> List[Dict[str, Any]]:
    ids = list(dict.fromkeys(qid for qid in question_ids if qid))
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        paperdb.QUESTION_SELECT
        + " WHERE q.question_id IN (%s) ORDER BY p.path, q.order_index LIMIT ?" % placeholders,
        tuple(ids) + (limit,),
    ).fetchall()
    return [paperdb.row_to_question(row) for row in rows]


def get_effective_tags(conn: sqlite3.Connection, question_id: str) -> Dict[str, Any]:
    """Return direct + inherited topic/auto tags for *question_id*.

    A main question inherits the topic and auto tags of all its descendants,
    which is the behaviour required by the history sub-question model.
    """
    descendants = _question_descendants(conn, question_id)
    direct = {question_id}
    if not descendants:
        return {"topics": [], "auto_tags": []}
    placeholders = ", ".join("?" for _ in descendants)

    topic_rows = conn.execute(
        """
        SELECT qt.question_id, qt.topic_id, qt.source, qt.confidence,
               t.name, t.path, t.intro
        FROM question_topics qt
        JOIN topics t ON qt.topic_id = t.topic_id
        WHERE qt.question_id IN (%s)
        ORDER BY t.position
        """ % placeholders,
        tuple(descendants),
    ).fetchall()
    tag_rows = conn.execute(
        """
        SELECT question_id, tag, source
        FROM question_tags
        WHERE question_id IN (%s)
        ORDER BY tag
        """ % placeholders,
        tuple(descendants),
    ).fetchall()

    topic_map: Dict[str, Dict[str, Any]] = {}
    for row in topic_rows:
        entry = topic_map.setdefault(
            row["topic_id"],
            {
                "topic_id": row["topic_id"],
                "name": row["name"],
                "path": row["path"],
                "intro": row["intro"] or "",
                "sources": set(),
                "confidence": None,
                "question_ids": set(),
                "inherited": True,
            },
        )
        entry["sources"].add(row["source"])
        if row["confidence"] is not None:
            entry["confidence"] = max(entry["confidence"] or 0.0, float(row["confidence"]))
        entry["question_ids"].add(row["question_id"])
        if row["question_id"] in direct:
            entry["inherited"] = False

    auto_map: Dict[str, Dict[str, Any]] = {}
    for row in tag_rows:
        entry = auto_map.setdefault(
            row["tag"],
            {
                "tag": row["tag"],
                "sources": set(),
                "question_ids": set(),
                "inherited": True,
            },
        )
        entry["sources"].add(row["source"])
        entry["question_ids"].add(row["question_id"])
        if row["question_id"] in direct:
            entry["inherited"] = False

    for entry in topic_map.values():
        entry["sources"] = sorted(entry["sources"])
        entry["question_ids"] = sorted(entry["question_ids"])
    for entry in auto_map.values():
        entry["sources"] = sorted(entry["sources"])
        entry["question_ids"] = sorted(entry["question_ids"])
    return {
        "topics": sorted(topic_map.values(), key=lambda item: item["path"]),
        "auto_tags": sorted(auto_map.values(), key=lambda item: item["tag"]),
        "question_ids": descendants,
    }


def _topic_descendants(conn: sqlite3.Connection, topic_id: str) -> List[str]:
    rows = conn.execute(
        """
        WITH RECURSIVE descendants(topic_id) AS (
            SELECT topic_id FROM topics WHERE topic_id = ?
            UNION ALL
            SELECT t.topic_id FROM topics t
            JOIN descendants d ON t.parent_id = d.topic_id
        )
        SELECT topic_id FROM descendants
        """,
        (topic_id,),
    ).fetchall()
    return [row["topic_id"] for row in rows]


def find_questions_by_topic(
    conn: sqlite3.Connection,
    topic: str,
    *,
    profile: Optional[str] = None,
    include_descendants: bool = True,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Find questions tagged with *topic* (id or name/path substring)."""
    sql = [
        "SELECT topic_id, name, path FROM topics WHERE (topic_id = ? OR name LIKE ? OR path LIKE ?)"
    ]
    needle = "%" + topic + "%"
    params: List[Any] = [topic, needle, needle]
    if profile:
        sql.append("AND tree_id = ?")
        params.append(_tree_id(profile))
    rows = conn.execute(" ".join(sql), params).fetchall()
    if not rows:
        return []
    topic_ids: List[str] = []
    for row in rows:
        topic_ids.extend(_topic_descendants(conn, row["topic_id"]) if include_descendants else [row["topic_id"]])
    topic_ids = list(dict.fromkeys(topic_ids))
    placeholders = ", ".join("?" for _ in topic_ids)
    direct = [
        row["question_id"]
        for row in conn.execute(
            "SELECT question_id FROM question_topics WHERE topic_id IN (%s)" % placeholders,
            tuple(topic_ids),
        )
    ]
    effective = _question_ancestors(conn, direct)
    return _query_questions_by_ids(conn, sorted(effective), limit=limit)


def find_questions_by_tag(
    conn: sqlite3.Connection,
    tag: str,
    *,
    include_parents: bool = True,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Find questions carrying an auto tag, plus their ancestors."""
    direct = [
        row["question_id"]
        for row in conn.execute(
            "SELECT question_id FROM question_tags WHERE tag = ?", (tag,)
        )
    ]
    effective = _question_ancestors(conn, direct) if include_parents else set(direct)
    return _query_questions_by_ids(conn, sorted(effective), limit=limit)


# ---------------------------------------------------------------------------
# profile resolution
# ---------------------------------------------------------------------------

def resolve_profile_for_tree(
    conn: sqlite3.Connection,
    profile: str,
    papers: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    pdf_paths: Optional[Sequence[Path]] = None,
) -> str:
    """Resolve ``auto`` to history/cs when possible; otherwise echo *profile*."""
    profile = str(profile or "").strip().lower()
    if profile and profile != "auto":
        return profile

    subjects: Set[str] = set()
    syllabi: Set[str] = set()
    for paper in papers or []:
        if not paper:
            continue
        if paper.get("subject"):
            subjects.add(str(paper["subject"]).strip().lower())
        if paper.get("syllabus"):
            syllabi.add(str(paper["syllabus"]).strip())
    if not subjects and not syllabi and pdf_paths:
        for path in pdf_paths:
            meta = paperdb.derive_paper_meta(Path(path), None)
            if meta.get("subject"):
                subjects.add(str(meta["subject"]).strip().lower())
            if meta.get("syllabus"):
                syllabi.add(str(meta["syllabus"]).strip())

    def _is_cs() -> bool:
        return "9618" in syllabi or any(
            "computer" in subject or subject == "cs" for subject in subjects
        )

    def _is_history() -> bool:
        return "9489" in syllabi or any("history" in subject for subject in subjects)

    if _is_cs() and not _is_history():
        return "cs"
    if _is_history() and not _is_cs():
        return "history"
    raise TopicTreeError(
        "cannot infer a topic-tree profile for profile=auto; "
        "use `--profile history` or `--profile cs`"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_tree(tree: TopicTree) -> None:
    if tree.title:
        sys.stdout.write("Title: %s\n" % tree.title)
    if tree.intro:
        sys.stdout.write("Intro: %s\n" % tree.intro)
    sys.stdout.write("Profile: %s  Nodes: %d\n" % (tree.profile, tree.node_count()))
    text = render_tree_text(tree)
    if text:
        sys.stdout.write("\n" + text)
    else:
        sys.stdout.write("(empty tree)\n")


def _connect_cli(args: argparse.Namespace) -> sqlite3.Connection:
    conn = paperdb.connect(Path(args.db))
    paperdb.init_schema(conn)
    return conn


def cmd_view(args: argparse.Namespace) -> int:
    conn = _connect_cli(args)
    try:
        tree = load_tree(conn, args.profile, Path(args.topics_dir))
        if tree is None:
            paperdb._warn("no tree for profile %r" % args.profile)
            return 2
        _print_tree(tree)
    finally:
        conn.close()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    conn = _connect_cli(args)
    try:
        rows = list_trees(conn)
        if not rows:
            sys.stdout.write("(no trees)\n")
        for row in rows:
            sys.stdout.write(
                "%s\t%s\t%s\t%d nodes\n"
                % (
                    row.get("profile") or "",
                    row.get("title") or "",
                    row.get("intro") or "",
                    row.get("node_count") or 0,
                )
            )
    finally:
        conn.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    conn = _connect_cli(args)
    try:
        tree = load_tree(conn, args.profile, Path(args.topics_dir))
        if tree is None:
            paperdb._warn("no tree for profile %r" % args.profile)
            return 2
        text = render_topic_markdown(tree)
        if args.out == "-":
            sys.stdout.write(text)
        else:
            out = Path(args.out) if args.out else canonical_path(Path(args.topics_dir), tree.profile)
            write_tree_file(out, tree)
            sys.stdout.write(str(out) + "\n")
    finally:
        conn.close()
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    conn = _connect_cli(args)
    topics_dir = Path(args.topics_dir)
    imported = 0
    try:
        sources: List[Path] = []
        if args.target:
            candidate = Path(args.target)
            if candidate.exists():
                sources = [candidate]
            else:
                profile = args.target
                pending = pending_path(topics_dir, profile)
                canonical = canonical_path(topics_dir, profile)
                if pending.exists():
                    sources = [pending]
                elif canonical.exists():
                    sources = [canonical]
                else:
                    paperdb._warn("no pending or canonical tree for profile %r" % profile)
                    return 2
        else:
            pending = pending_files(topics_dir)
            if pending:
                sources = pending
            else:
                profile = args.profile or "history"
                canonical = canonical_path(topics_dir, profile)
                if canonical.exists():
                    sources = [canonical]
                else:
                    paperdb._warn("no pending tree files and no canonical tree for %r" % profile)
                    return 2
        for source in sources:
            is_pending = source.parent.name == PENDING_DIRNAME
            try:
                tree, stats = import_tree_file(
                    conn,
                    source,
                    topics_dir=topics_dir,
                    default_profile=args.profile or "",
                    delete_source=is_pending or args.delete_source,
                )
            except TopicTreeError as exc:
                paperdb._warn("%s" % exc)
                continue
            imported += 1
            sys.stdout.write(
                "%s -> profile=%s nodes=%d removed=%d\n"
                % (source, tree.profile, stats["nodes"], stats["removed"])
            )
    finally:
        conn.close()
    return 0 if imported else 2


def cmd_pending(args: argparse.Namespace) -> int:
    files = pending_files(Path(args.topics_dir))
    if not files:
        sys.stdout.write("(no pending tree edits)\n")
        return 0
    for path in files:
        sys.stdout.write("%s\n" % path)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    conn = _connect_cli(args)
    try:
        stats = sync_topic_trees(conn, Path(args.topics_dir))
        tagged = refresh_auto_tags(conn)
        sys.stdout.write(
            "topic files=%d nodes=%d auto_tag_questions=%d\n"
            % (stats["files"], stats["nodes"], tagged)
        )
    finally:
        conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topic_tree.py",
        description="Manage profile topic trees and question tags.",
    )
    parser.add_argument("--db", default=paperdb.DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--topics-dir", default=DEFAULT_TOPICS_DIR, help="topic tree directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_view = sub.add_parser("view", help="show a profile tree")
    p_view.add_argument("profile", nargs="?", default="history")
    p_view.set_defaults(func=cmd_view)

    p_list = sub.add_parser("list", help="list trees")
    p_list.set_defaults(func=cmd_list)

    p_export = sub.add_parser("export", help="write a tree as editable Markdown")
    p_export.add_argument("profile", nargs="?", default="history")
    p_export.add_argument("--out", help="output path, or - for stdout")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import", help="import a tree file or pending edit")
    p_import.add_argument("target", nargs="?", help="file path or profile name")
    p_import.add_argument("--profile", help="fallback profile when the YAML lacks one")
    p_import.add_argument("--delete-source", action="store_true", help="delete a non-pending source file after import")
    p_import.set_defaults(func=cmd_import)

    p_pending = sub.add_parser("pending", help="list pending tree edits")
    p_pending.set_defaults(func=cmd_pending)

    p_sync = sub.add_parser("sync", help="import canonical tree files and refresh auto tags")
    p_sync.set_defaults(func=cmd_sync)

    for subparser in sub.choices.values():
        subparser.add_argument("--db", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        subparser.add_argument(
            "--topics-dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (TopicTreeError, sqlite3.Error) as exc:
        paperdb._warn("%s" % exc)
        return 1
    except KeyboardInterrupt:
        paperdb._warn("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
