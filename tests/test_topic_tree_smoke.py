#!/usr/bin/env python3
"""Smoke test for topic_tree.py.  Uses only the standard library.

Run from the repository root:

    python3 tests/test_topic_tree_smoke.py
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import paperdb  # noqa: E402
import topic_tree  # noqa: E402


TREE_MD = """---
profile: "history"
title: "AS History 9489"
intro: "Thematic tree for AS History."
---

- Peace treaties: 1919-1923 settlements.
  - Treaty of Versailles: The 1919 treaty with Germany.
  - Treaty of Saint-Germain: The 1919 treaty with Austria.
- Cold War: 1945-1991 superpower rivalry.
  - Cuban Missile Crisis: 1962 confrontation.
"""


def make_db(root):
    db = root / "papers.db"
    conn = paperdb.connect(db)
    paperdb.init_schema(conn)
    conn.execute(
        "INSERT INTO papers (pdf_id, sha256, path, subject, syllabus, paper_code, "
        "session, variant, kind) VALUES (?,?,?,?,?,?,?,?,?)",
        ("p1", "sha", "/p1.pdf", "history", "9489", "2", "June 2021", "v1", "QP"),
    )
    conn.execute(
        "INSERT INTO documents (md_id, pdf_id, stem, md_path) VALUES (?,?,?,?)",
        ("m1", "p1", "stem", "/m1.md"),
    )
    rows = [
        ("p1:g1:__stimulus__", "stimulus", None, "", "", "__stimulus__", 0, "Source A..."),
        ("p1:g1:1", "main", "p1:g1:__stimulus__", "1", "", "1", 1, "Explain the aims."),
        ("p1:g1:1-01-a", "sub", "p1:g1:1", "1", "(a)", "1-01-a", 2, "Describe two aims."),
        ("p1:g1:1-02-b", "sub", "p1:g1:1", "1", "(b)", "1-02-b", 3, "Explain why it failed."),
    ]
    for qid, kind, parent, number, sub, flat, order, stem in rows:
        conn.execute(
            "INSERT INTO questions (question_id, md_id, pdf_id, parent_id, kind, "
            "section, group_id, number, sub_label, flat_id, order_index, stem) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                qid,
                "m1",
                "p1",
                parent,
                kind,
                "Section A",
                "g1",
                number,
                sub,
                flat,
                order,
                stem,
            ),
        )
    conn.commit()
    return conn


def main():
    with tempfile.TemporaryDirectory(prefix="topic-tree-test-") as tmp:
        root = Path(tmp)
        conn = make_db(root)

        tree = topic_tree.parse_topic_markdown(TREE_MD, default_profile="history")
        assert tree.profile == "history", tree
        assert tree.title == "AS History 9489", tree
        assert tree.node_count() == 5, tree.node_count()
        assert tree.nodes[0].children[0].name == "Treaty of Versailles", tree

        errors = topic_tree.validate_tree(tree)
        assert not errors, errors

        stats = topic_tree.save_tree(conn, tree)
        assert stats["nodes"] == 5, stats
        loaded = topic_tree.get_tree(conn, "history")
        assert loaded is not None and loaded.node_count() == 5
        ids_before = sorted(node.topic_id for node in loaded.walk_nodes())

        # Reordering roots must preserve topic ids (name-path matching).
        rendered = topic_tree.render_topic_markdown(loaded)
        reparsed = topic_tree.parse_topic_markdown(rendered, default_profile="history")
        reparsed.nodes.reverse()
        topic_tree.save_tree(conn, reparsed)
        loaded2 = topic_tree.get_tree(conn, "history")
        ids_after = sorted(node.topic_id for node in loaded2.walk_nodes())
        assert ids_before == ids_after, (ids_before, ids_after)

        # Auto positional tags.
        count = topic_tree.refresh_auto_tags(conn)
        assert count == 3, count
        tags = {}
        for row in conn.execute("SELECT question_id, tag FROM question_tags"):
            tags.setdefault(row["question_id"], set()).add(row["tag"])
        assert tags["p1:g1:1-01-a"] == {
            "Paper 2",
            "Question 1",
            "Section A",
            "(a)",
        }, tags
        assert tags["p1:g1:1"] == {"Paper 2", "Question 1", "Section A"}, tags

        # Parent question inherits child topic tags.
        by_name = {node.name: node.topic_id for node in loaded2.walk_nodes()}
        conn.execute(
            "INSERT INTO question_topics (question_id, topic_id, source, confidence, created_at) "
            "VALUES (?,?,?,?,?)",
            (
                "p1:g1:1-01-a",
                by_name["Treaty of Versailles"],
                "jev",
                0.91,
                paperdb.now_utc(),
            ),
        )
        conn.commit()
        effective = topic_tree.get_effective_tags(conn, "p1:g1:1")
        topics = {item["name"]: item for item in effective["topics"]}
        assert "Treaty of Versailles" in topics, topics
        assert topics["Treaty of Versailles"]["inherited"] is True, topics
        assert topics["Treaty of Versailles"]["question_ids"] == ["p1:g1:1-01-a"]

        matches = topic_tree.find_questions_by_topic(conn, "Versailles", profile="history")
        assert [q["question_id"] for q in matches] == ["p1:g1:1", "p1:g1:1-01-a"], matches

        matches = topic_tree.find_questions_by_tag(conn, "(a)")
        assert [q["question_id"] for q in matches] == ["p1:g1:1", "p1:g1:1-01-a"], matches

        # Pending-file import writes canonical and removes the pending source.
        topics_dir = root / "topics"
        pending = topic_tree.pending_path(topics_dir, "history")
        topic_tree.write_tree_file(pending, loaded2)
        tree2, stats2 = topic_tree.import_tree_file(
            conn,
            pending,
            topics_dir=topics_dir,
            default_profile="history",
            delete_source=True,
        )
        assert tree2.profile == "history"
        assert not pending.exists(), pending
        assert topic_tree.canonical_path(topics_dir, "history").exists()

        # Validation limits.
        bad = topic_tree.TopicTree(
            profile="history",
            title="x",
            intro="x" * 101,
            nodes=[topic_tree.TopicNode(name="A", intro="y" * 41)],
        )
        bad_errors = topic_tree.validate_tree(bad)
        assert len(bad_errors) == 2, bad_errors

        conn.close()

    print("topic_tree smoke test: OK")


if __name__ == "__main__":
    main()
