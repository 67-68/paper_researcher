#!/usr/bin/env python3
"""Smoke test for paperdb.py.  Uses only the standard library.

Run from the repository root:

    python3 tests/test_paperdb_smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paperdb  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_fixture(root):
    papers_dir = root / "papers"
    md_dir = root / "md"
    questions_dir = root / "questions"
    papers_dir.mkdir()
    md_dir.mkdir()
    questions_dir.mkdir()

    pdf_path = papers_dir / "June 2021 (v1) QP.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% synthetic test pdf\n")
    sha = paperdb.sha256_file(pdf_path)
    pdf_id = sha[:16]

    manifest = {
        "url": (
            "https://pmt.example/download/History/A-level/Past-Papers/CAIE/"
            "Paper-1/QP/June 2021 (v1) QP.pdf"
        ),
        "parent": None,
        "depth": 1,
        "kind": "pdf",
        "status": "downloaded",
        "file": pdf_path.name,
        "content_type": "application/pdf",
        "bytes": pdf_path.stat().st_size,
        "sha256": sha,
        "error": None,
    }
    (papers_dir / "manifest.jsonl").write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    stem = "June 2021 (v1) QP"
    md_path = md_dir / (stem + ".md")
    md_path.write_text(
        "# June 2021 (v1) QP\n\n"
        "<!-- table:tbl-0001 -->\n\n"
        "| Term | Meaning |\n| --- | --- |\n| League | ... |\n",
        encoding="utf-8",
    )
    meta = {
        "source_pdf": str(pdf_path.resolve()),
        "stem": stem,
        "generator": "pdf_to_md.py",
        "image_link_style": "markdown",
        "table_format": "md",
        "pages": 1,
        "tables": [
            {
                "id": "tbl-0001",
                "format": "md",
                "markdown": "| Term | Meaning |\n| --- | --- |\n| League | ... |",
                "data": [["Term", "Meaning"], ["League", "..."]],
            }
        ],
        "images": [],
        "blocks": [],
    }
    write_json(md_dir / (stem + ".meta.json"), meta)

    groups = [
        {
            "group_id": "g0001",
            "section": "Paper 1",
            "shared_stem": "Source A: a speech about the League of Nations.",
            "tables": [],
            "attachments": [
                {
                    "alt": "fig-0001",
                    "path": "%s.assets/fig-0001.png" % stem,
                }
            ],
            "questions": [
                {
                    "question_no": "1",
                    "section": "Paper 1",
                    "group_id": "g0001",
                    "stem": "Explain the aims of the League.",
                    "options": {},
                    "marks": 10,
                    "tables": [
                        {
                            "id": "tbl-0001",
                            "format": "md",
                            "markdown": "| Term | Meaning |\n| --- | --- |\n| League | ... |",
                            "data": [["Term", "Meaning"], ["League", "..."]],
                        }
                    ],
                    "attachments": [],
                    "warnings": [],
                    "sub_questions": [
                        {
                            "label": "(a)",
                            "stem": "Describe two aims.",
                            "marks": 4,
                            "tables": [],
                            "attachments": [],
                            "warnings": [],
                        },
                        {
                            "label": "(b)",
                            "stem": "Explain why it failed.",
                            "marks": 6,
                            "tables": [],
                            "attachments": [],
                            "warnings": [],
                        },
                    ],
                },
                {
                    "question_no": "2",
                    "section": "Paper 1",
                    "group_id": "g0001",
                    "stem": "Assess the impact of the Treaty.",
                    "options": {},
                    "marks": 15,
                    "tables": [],
                    "attachments": [],
                    "warnings": [],
                    "sub_questions": [],
                },
            ],
        }
    ]
    write_json(questions_dir / (stem + ".questions.json"), groups)

    return {
        "root": root,
        "papers_dir": papers_dir,
        "md_dir": md_dir,
        "questions_dir": questions_dir,
        "db": root / "papers.db",
        "pdf_id": pdf_id,
        "stem": stem,
    }


def main():
    with tempfile.TemporaryDirectory(prefix="paperdb-test-") as tmp:
        root = Path(tmp)
        fixture = build_fixture(root)

        rc = paperdb.main(
            [
                "--db",
                str(fixture["db"]),
                "sync",
                "--papers-dir",
                str(fixture["papers_dir"]),
                "--md",
                str(fixture["md_dir"]),
                "--questions",
                str(fixture["questions_dir"]),
            ]
        )
        assert rc == 0, rc

        conn = paperdb.connect(fixture["db"])
        try:
            papers = paperdb.query_papers(conn)
            assert len(papers) == 1, len(papers)
            paper = paperdb.row_to_paper(papers[0], papers[0]["question_count"])
            assert paper["subject"] == "history", paper
            assert paper["syllabus"] == "9489", paper
            assert paper["kind"] == "QP", paper
            assert paper["session"] == "June 2021", paper
            assert paper["variant"] == "v1", paper
            assert paper["question_count"] == 4, paper

            questions = [
                paperdb.row_to_question(row)
                for row in paperdb.query_questions(
                    conn, paper=paper["pdf_id"], include_stimulus=True
                )
            ]
            kinds = [q["kind"] for q in questions]
            assert kinds == ["stimulus", "main", "sub", "sub", "main"], kinds
            main = questions[1]
            assert main["flat_id"] == "1", main
            assert main["stem"] == "Explain the aims of the League.", main
            assert main["marks"] == 10, main
            assert main["options"] == {}, main
            assert main["tables"][0]["id"] == "tbl-0001", main
            assert main["attachments"] == [], main

            sub_a = questions[2]
            assert sub_a["sub_label"] == "(a)", sub_a
            assert sub_a["parent_id"] == main["question_id"], sub_a
            assert sub_a["stem"] == "Describe two aims.", sub_a
            assert sub_a["marks"] == 4, sub_a
            assert sub_a["parent_stem"] == main["stem"], sub_a
            assert sub_a["shared_stem"] == questions[0]["stem"], sub_a

            sub_b = questions[3]
            assert sub_b["sub_label"] == "(b)", sub_b
            assert sub_b["marks"] == 6, sub_b

            main2 = questions[4]
            assert main2["flat_id"] == "2", main2
            assert main2["parent_id"] == questions[0]["question_id"], main2
            assert main2["marks"] == 15, main2

            # default query excludes stimulus
            default_rows = paperdb.query_questions(conn, paper=paper["pdf_id"])
            assert [r["kind"] for r in default_rows] == ["main", "sub", "sub", "main"]

            # question id lookup
            found = conn.execute(
                paperdb.QUESTION_SELECT + " WHERE q.question_id = ?",
                (sub_a["question_id"],),
            ).fetchone()
            assert found is not None
            assert paperdb.row_to_question(found)["stem"] == "Describe two aims."
        finally:
            conn.close()

    print("paperdb smoke test: OK")


if __name__ == "__main__":
    main()
