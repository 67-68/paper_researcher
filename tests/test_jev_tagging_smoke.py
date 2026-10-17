#!/usr/bin/env python3
"""Smoke test for tag_questions.py using a local mock Jev endpoint.

Run from the repository root:

    python3 tests/test_jev_tagging_smoke.py

The test uses only the standard library and never touches the real API.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

# Make sure the local mock is not routed through the macOS system proxy.
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import paperdb  # noqa: E402
import tag_questions  # noqa: E402


class MockJevHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        criteria = payload.get("questions", {}).get("topic", {}).get("criteria", {})
        choice = next(iter(criteria)) if criteria else "missing"
        body = json.dumps(
            {
                "model": "jev-mock",
                "answers": {
                    "topic": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.88,
                        "probabilities": {choice: 1.0},
                    }
                },
                "usage": {"input_tokens": 12, "output_tokens": 6},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_db(root):
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
        ("p1:g1:__stimulus__", "stimulus", None, "", "", "__stimulus__", 0, "Source A"),
        ("p1:g1:1", "main", "p1:g1:__stimulus__", "1", "", "1", 1, "Explain the Treaty."),
        ("p1:g1:1-01-a", "sub", "p1:g1:1", "1", "(a)", "1-01-a", 2, "Describe two terms."),
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
    conn.close()
    return db


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockJevHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="jev-tag-test-") as tmp:
            root = Path(tmp)
            topics_dir = root / "topics"
            topics_dir.mkdir()
            (topics_dir / "history.tree.md").write_text(
                """---
profile: "history"
title: "History test"
intro: "Test tree."
---

- Peace: peace topics.
  - Versailles: 1919 treaty.
- Cold War: superpower rivalry.
""",
                encoding="utf-8",
            )
            db = build_db(root)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = tag_questions.main(
                    [
                        "--db",
                        str(db),
                        "--topics-dir",
                        str(topics_dir),
                        "--profile",
                        "history",
                        "--paper",
                        "p1",
                        "--api-key",
                        "test-key",
                        "--base-url",
                        "http://127.0.0.1:%d" % port,
                    ]
                )
            assert rc == 0, (rc, stdout.getvalue(), stderr.getvalue())

            conn = paperdb.connect(db)
            try:
                tagged = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT question_id, topic_id, source, confidence "
                        "FROM question_topics ORDER BY question_id"
                    )
                ]
                assert len(tagged) == 2, tagged
                assert all(row["source"] == "jev" for row in tagged), tagged
                assert all(row["confidence"] == 0.88 for row in tagged), tagged

                tags = {
                    row["question_id"]: set()
                    for row in conn.execute(
                        "SELECT DISTINCT question_id FROM question_tags"
                    )
                }
                for row in conn.execute("SELECT question_id, tag FROM question_tags"):
                    tags.setdefault(row["question_id"], set()).add(row["tag"])
                assert tags["p1:g1:1-01-a"] == {
                    "Paper 2",
                    "Question 1",
                    "Section A",
                    "(a)",
                }, tags
                assert tags["p1:g1:1"] == {
                    "Paper 2",
                    "Question 1",
                    "Section A",
                }, tags
            finally:
                conn.close()
    finally:
        server.shutdown()
        server.server_close()

    print("jev tagging smoke test: OK")


if __name__ == "__main__":
    main()
