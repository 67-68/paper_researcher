#!/usr/bin/env python3
"""paper_tui.py - a small Rich command REPL for the paper corpus.

It is a thin command layer on top of ``paperdb.py``:

  * list and open papers / questions / Markdown documents
  * inspect a question together with its parent, children, tables, attachments
  * jump through the graph by id or by index
  * run the PDF -> Markdown -> questions -> database pipeline

Only ``rich`` is required for the TUI itself.  The conversion steps are
executed as subprocesses, using ``uv`` when PyMuPDF / pdfplumber / requests are
needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.markup import escape
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
except ImportError:  # pragma: no cover - exercised only without rich
    sys.stderr.write(
        "paper_tui.py requires rich.\n"
        "Install it with: python3 -m pip install rich\n"
    )
    raise SystemExit(2)

import paper_common  # noqa: E402
import paperdb  # noqa: E402


DEFAULT_DB = "data/papers.db"
DEFAULT_PAPERS_DIR = "data/papers"
DEFAULT_MD_DIR = "data/md"
DEFAULT_QUESTIONS_DIR = "data/questions"

SCRIPT_DEPS = {
    "pdf_to_md.py": ["pymupdf", "pdfplumber"],
    "crawl_pdfs.py": ["requests"],
}


def _short(value: Any, length: int = 90) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    if len(text) > length:
        return text[: length - 3] + "..."
    return text


class PaperTUI:
    def __init__(
        self,
        db_path: Path,
        papers_dir: Path,
        md_dir: Path,
        questions_dir: Path,
        assume_yes: bool = False,
    ) -> None:
        self.console = Console()
        self.db_path = db_path
        self.papers_dir = papers_dir
        self.md_dir = md_dir
        self.questions_dir = questions_dir
        self.assume_yes = assume_yes

        self.conn = None  # type: Any
        self.current_kind = ""  # "paper" | "question"
        self.current_id = ""  # type: str
        self.history = []  # type: List[Tuple[str, str]]
        self.last_paper_ids = []  # type: List[str]
        self.last_question_ids = []  # type: List[str]
        self.uv = shutil.which("uv")

    # ------------------------------------------------------------------
    # database / lookup
    # ------------------------------------------------------------------

    def open_db(self) -> None:
        self.conn = paperdb.connect(self.db_path)
        paperdb.init_schema(self.conn)

    def close_db(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def fetch_paper(self, pdf_id: str) -> Optional[Dict[str, Any]]:
        return paperdb.get_paper(self.conn, pdf_id)

    def fetch_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        return paperdb.get_question(self.conn, question_id)

    def fetch_documents(self, pdf_id: str) -> List[Dict[str, Any]]:
        return paperdb.get_documents(self.conn, pdf_id)

    def fetch_paper_questions(
        self, pdf_id: str, include_stimulus: bool = True
    ) -> List[Dict[str, Any]]:
        rows = paperdb.query_questions(
            self.conn, paper=pdf_id, include_stimulus=include_stimulus
        )
        return [paperdb.row_to_question(row) for row in rows]

    def fetch_children(self, question_id: str) -> List[Dict[str, Any]]:
        return paperdb.get_children(self.conn, question_id)

    def search(self, text: str, limit: int = 200) -> List[Dict[str, Any]]:
        return paperdb.search_questions(self.conn, text, limit=limit)

    # ------------------------------------------------------------------
    # rendering primitives
    # ------------------------------------------------------------------

    def render_paper(self, pdf_id: str) -> None:
        paper = self.fetch_paper(pdf_id)
        if paper is None:
            self.console.print("[red]paper not found:[/red] %s" % escape(pdf_id))
            return
        self.current_kind = "paper"
        self.current_id = pdf_id

        body = Text()
        rows = [
            ("pdf_id", paper.get("pdf_id")),
            ("path", paper.get("path")),
            ("subject", paper.get("subject")),
            ("syllabus", paper.get("syllabus")),
            ("paper_code", paper.get("paper_code")),
            ("session", paper.get("session")),
            ("variant", paper.get("variant")),
            ("kind", paper.get("kind")),
            ("bytes", paper.get("bytes")),
            ("source_url", paper.get("source_url")),
            ("question_count", paper.get("question_count")),
        ]
        for index, (key, value) in enumerate(rows):
            body.append("%s: " % key, style="bold")
            body.append("" if value is None else str(value))
            if index != len(rows) - 1:
                body.append("\n")
        self.console.print(Panel(body, title="Paper", border_style="cyan"))

        documents = self.fetch_documents(pdf_id)
        if documents:
            self.render_documents(documents)
        else:
            self.console.print(
                "[yellow]No Markdown document linked to this paper yet.[/yellow]"
            )

        questions = self.fetch_paper_questions(pdf_id, include_stimulus=True)
        self.last_question_ids = [q["question_id"] for q in questions]
        self.render_question_table(
            questions, title="Questions (stimulus + main + sub)"
        )
        if not questions:
            self.console.print(
                "[yellow]No questions indexed for this paper yet.[/yellow]"
            )
            self.console.print(
                "[dim]Run: pipeline %s[/dim]" % escape(pdf_id)
            )
        self.console.print(
            "[dim]question <n> | tree | md | pipeline <pdf_id> | back | home[/dim]"
        )

    def render_question(self, question_id: str) -> None:
        question = self.fetch_question(question_id)
        if question is None:
            self.console.print(
                "[red]question not found:[/red] %s" % escape(question_id)
            )
            return
        self.current_kind = "question"
        self.current_id = question_id

        body = Text()
        fields = [
            ("question_id", question.get("question_id")),
            ("kind", question.get("kind")),
            ("number", question.get("number")),
            ("sub_label", question.get("sub_label")),
            ("marks", question.get("marks")),
            ("section", question.get("section")),
            ("group_id", question.get("group_id")),
            ("stem", question.get("stem")),
            ("parent_stem", question.get("parent_stem")),
            ("shared_stem", question.get("shared_stem")),
            ("parent_id", question.get("parent_id")),
            ("paper_id", question.get("pdf_id")),
            ("md_path", question.get("md_path")),
            ("assets_dir", question.get("assets_dir")),
        ]
        for index, (key, value) in enumerate(fields):
            if key in ("stem", "parent_stem", "shared_stem") and value:
                value = _short(value, 500)
                if len(str(fields[index][1])) > 500:
                    value += "  (truncated; use raw for full)"
            body.append("%s: " % key, style="bold")
            body.append("" if value is None else str(value))
            if index != len(fields) - 1:
                body.append("\n")
        self.console.print(Panel(body, title="Question", border_style="green"))

        paper = self.fetch_paper(question["pdf_id"])
        if paper:
            self.console.print(
                "[dim]paper:[/dim] %s  [dim]subject:[/dim] %s  [dim]session:[/dim] %s"
                % (
                    escape(paper.get("pdf_id") or ""),
                    escape(paper.get("subject") or ""),
                    escape(paper.get("session") or ""),
                )
            )

        parent_id = question.get("parent_id")
        if parent_id:
            parent = self.fetch_question(parent_id)
            if parent:
                self.console.print(
                    Panel(
                        Text(_short(parent.get("stem"), 220)),
                        title="Parent %s" % parent_id,
                        border_style="yellow",
                    )
                )
        children = self.fetch_children(question_id)
        if children:
            self.render_question_table(children, title="Children")
            self.last_question_ids = [child["question_id"] for child in children]

        self.render_tables(question)
        self.render_attachments(question)

        warnings = question.get("warnings") or []
        if warnings:
            self.console.print(
                Panel(
                    Text("\n".join(str(w) for w in warnings)),
                    title="Warnings",
                    border_style="red",
                )
            )
        self.console.print(
            "[dim]parent | children | next | prev | tables | attachments | raw | "
            "paper | md | back[/dim]"
        )

    def render_documents(self, documents: List[Dict[str, Any]]) -> None:
        table = Table(title="Documents", show_lines=False)
        table.add_column("md_id", style="cyan", overflow="fold")
        table.add_column("stem")
        table.add_column("md_path", overflow="fold")
        table.add_column("meta_path", overflow="fold")
        table.add_column("assets_dir", overflow="fold")
        for doc in documents:
            table.add_row(
                Text(doc.get("md_id") or ""),
                Text(doc.get("stem") or ""),
                Text(doc.get("md_path") or ""),
                Text(doc.get("meta_path") or ""),
                Text(doc.get("assets_dir") or ""),
            )
        self.console.print(table)

    def render_question_table(
        self, questions: Sequence[Dict[str, Any]], title: str
    ) -> None:
        table = Table(title=title, show_lines=False)
        table.add_column("#", justify="right", style="bold")
        table.add_column("question_id", style="cyan", overflow="fold")
        table.add_column("kind")
        table.add_column("no")
        table.add_column("sub")
        table.add_column("marks", justify="right")
        table.add_column("stem", overflow="fold")
        for index, question in enumerate(questions, 1):
            table.add_row(
                str(index),
                Text(question.get("question_id") or ""),
                Text(question.get("kind") or ""),
                Text(question.get("number") or ""),
                Text(question.get("sub_label") or ""),
                Text(
                    "" if question.get("marks") is None else str(question.get("marks"))
                ),
                Text(_short(question.get("stem") or "", 120)),
            )
        self.console.print(table)

    def render_tables(self, question: Dict[str, Any]) -> None:
        tables = question.get("tables") or []
        if not tables:
            return
        for table in tables:
            markdown = table.get("markdown")
            if not markdown:
                markdown = paper_common.table_to_markdown(table.get("data") or [])
            title = "Table %s" % (table.get("id") or "")
            self.console.print(Panel(Markdown(markdown), title=title))

    def render_attachments(self, question: Dict[str, Any]) -> None:
        attachments = question.get("attachments") or []
        if not attachments:
            return
        table = Table(title="Attachments", show_lines=False)
        table.add_column("alt", style="cyan")
        table.add_column("path", overflow="fold")
        table.add_column("exists")
        for attachment in attachments:
            rel = attachment.get("path") or ""
            candidate = self.md_dir / rel if rel else None
            exists = bool(candidate and candidate.exists())
            table.add_row(
                Text(attachment.get("alt") or ""),
                Text(rel),
                Text("yes" if exists else "missing"),
            )
        self.console.print(table)

    def render_md(self, ident: str) -> None:
        paper = self.fetch_paper(ident)
        md_path = None
        title = ident
        if paper:
            docs = self.fetch_documents(paper["pdf_id"])
            if docs:
                md_path = docs[0].get("md_path")
                title = docs[0].get("md_id") or paper["pdf_id"]
        else:
            row = self.conn.execute(
                "SELECT * FROM documents WHERE md_id = ?", (ident,)
            ).fetchone()
            if row is not None:
                md_path = row["md_path"]
                title = row["md_id"]
        if not md_path:
            self.console.print("[red]markdown not found for:[/red] %s" % escape(ident))
            return
        path = Path(md_path)
        if not path.exists():
            self.console.print("[red]markdown file is missing:[/red] %s" % escape(str(path)))
            return
        self.console.print(
            Panel(Markdown(path.read_text(encoding="utf-8")), title=str(title))
        )

    def render_raw(self, ident: str) -> None:
        question = self.fetch_question(ident)
        if question is not None:
            payload = question
        else:
            payload = self.fetch_paper(ident)
        if payload is None:
            self.console.print("[red]not found:[/red] %s" % escape(ident))
            return
        self.console.print(
            Syntax(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                "json",
                word_wrap=True,
            )
        )

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------

    def navigate(self, kind: str, ident: str) -> None:
        if self.current_kind and self.current_id:
            self.history.append((self.current_kind, self.current_id))
        if kind == "paper":
            self.render_paper(ident)
        else:
            self.render_question(ident)

    def resolve_index(self, token: str, values: Sequence[str]) -> Optional[str]:
        if not token:
            return None
        if token.isdigit():
            index = int(token) - 1
            if 0 <= index < len(values):
                return values[index]
            return None
        return token

    def current_paper_id(self) -> Optional[str]:
        if self.current_kind == "paper":
            return self.current_id
        if self.current_kind == "question":
            question = self.fetch_question(self.current_id)
            if question:
                return question["pdf_id"]
        return None

    def current_question_id(self) -> Optional[str]:
        if self.current_kind == "question":
            return self.current_id
        return None

    def _question_order(self, pdf_id: str) -> List[Dict[str, Any]]:
        return self.fetch_paper_questions(pdf_id, include_stimulus=False)

    def cmd_back(self, args: List[str]) -> None:
        if not self.history:
            self.console.print("[yellow]history is empty[/yellow]")
            return
        kind, ident = self.history.pop()
        if kind == "paper":
            self.render_paper(ident)
        else:
            self.render_question(ident)

    def cmd_home(self, args: List[str]) -> None:
        self.cmd_papers([])

    def cmd_parent(self, args: List[str]) -> None:
        question_id = self.current_question_id()
        if not question_id:
            self.console.print("[yellow]open a question first[/yellow]")
            return
        question = self.fetch_question(question_id)
        if not question or not question.get("parent_id"):
            self.console.print("[yellow]this question has no parent[/yellow]")
            return
        self.navigate("question", question["parent_id"])

    def cmd_children(self, args: List[str]) -> None:
        question_id = self.current_question_id()
        if not question_id:
            self.console.print("[yellow]open a question first[/yellow]")
            return
        children = self.fetch_children(question_id)
        if not children:
            self.console.print("[yellow]no children[/yellow]")
            return
        self.last_question_ids = [child["question_id"] for child in children]
        self.render_question_table(children, title="Children")
        self.console.print("[dim]question <n>[/dim]")

    def cmd_next(self, args: List[str]) -> None:
        paper_id = self.current_paper_id()
        if not paper_id:
            self.console.print("[yellow]open a paper or question first[/yellow]")
            return
        order = self._question_order(paper_id)
        if not order:
            self.console.print("[yellow]no questions in this paper[/yellow]")
            return
        if self.current_kind == "question":
            current_ids = [q["question_id"] for q in order]
            try:
                index = current_ids.index(self.current_id)
            except ValueError:
                index = -1
            if index + 1 >= len(order):
                self.console.print("[yellow]already at the last question[/yellow]")
                return
            self.navigate("question", order[index + 1]["question_id"])
        else:
            self.navigate("question", order[0]["question_id"])

    def cmd_prev(self, args: List[str]) -> None:
        paper_id = self.current_paper_id()
        if not paper_id:
            self.console.print("[yellow]open a paper or question first[/yellow]")
            return
        order = self._question_order(paper_id)
        if not order:
            self.console.print("[yellow]no questions in this paper[/yellow]")
            return
        if self.current_kind != "question":
            self.console.print("[yellow]open a question first[/yellow]")
            return
        current_ids = [q["question_id"] for q in order]
        try:
            index = current_ids.index(self.current_id)
        except ValueError:
            index = 0
        if index <= 0:
            self.console.print("[yellow]already at the first question[/yellow]")
            return
        self.navigate("question", order[index - 1]["question_id"])

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------

    def cmd_help(self, args: List[str]) -> None:
        table = Table(title="paper_tui commands", show_lines=False)
        table.add_column("command", style="cyan", no_wrap=True)
        table.add_column("what it does")
        help_rows = [
            ("help / ?", "show this help"),
            ("stats", "show database counts"),
            ("papers [--subject S] [--kind QP] [--session S]", "list papers"),
            ("open <n|pdf_id>", "open a paper (n = index from the last list)"),
            ("paper [pdf_id]", "jump to the current question's paper"),
            ("question <n|question_id>", "open a question"),
            ("jump <id>", "open a paper, question or Markdown document"),
            ("tree [pdf_id]", "show the question tree"),
            ("md [pdf_id|md_id]", "render the Markdown document"),
            ("tables [question_id]", "show tables for a question"),
            ("attachments [question_id]", "show attachments for a question"),
            ("parent", "jump to the parent question"),
            ("children", "list child questions"),
            ("next / prev", "move through questions of the current paper"),
            ("back", "go back in navigation history"),
            ("home", "list papers"),
            ("search <text>", "search main/sub question stems"),
            ("raw [id]", "print raw JSON for a question or paper"),
            ("sync", "run paperdb.py sync"),
            ("convert [--all] [pdf ...]", "run pdf_to_md.py"),
            ("split [--profile history|cs|auto] [md ...]", "run split_questions.py"),
            ("pipeline [--profile ...] [--all]", "convert + split + sync"),
            ("run <script.py> [args ...]", "run a repository script directly"),
            ("clear", "clear the screen"),
            ("quit / exit", "leave the TUI"),
        ]
        for command, description in help_rows:
            table.add_row(Text(command), Text(description))
        self.console.print(table)

    def cmd_stats(self, args: List[str]) -> None:
        stats = {
            "papers": self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "documents": self.conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0],
            "questions": self.conn.execute(
                "SELECT COUNT(*) FROM questions"
            ).fetchone()[0],
            "main_or_sub": self.conn.execute(
                "SELECT COUNT(*) FROM questions WHERE kind IN ('main', 'sub')"
            ).fetchone()[0],
        }
        table = Table(title="Stats")
        table.add_column("metric", style="cyan")
        table.add_column("value", justify="right")
        for key, value in stats.items():
            table.add_row(Text(key), Text(str(value)))
        self.console.print(table)

    def cmd_papers(self, args: List[str]) -> None:
        parser = argparse.ArgumentParser(prog="papers", add_help=False)
        parser.add_argument("--subject")
        parser.add_argument("--kind")
        parser.add_argument("--session")
        parser.add_argument("--syllabus")
        parser.add_argument("--paper-code")
        namespace = self._parse(parser, args)
        if namespace is None:
            return
        rows = paperdb.query_papers(
            self.conn,
            subject=namespace.subject,
            kind=namespace.kind,
            session=namespace.session,
            syllabus=namespace.syllabus,
            paper_code=namespace.paper_code,
        )
        papers = [paperdb.row_to_paper(row, row["question_count"]) for row in rows]
        self.last_paper_ids = [p["pdf_id"] for p in papers]

        table = Table(title="Papers", show_lines=False)
        table.add_column("#", justify="right", style="bold")
        table.add_column("pdf_id", style="cyan", no_wrap=True)
        table.add_column("subject")
        table.add_column("kind")
        table.add_column("session")
        table.add_column("var")
        table.add_column("questions", justify="right")
        table.add_column("path", overflow="fold")
        for index, paper in enumerate(papers, 1):
            table.add_row(
                str(index),
                Text(paper.get("pdf_id") or ""),
                Text(paper.get("subject") or ""),
                Text(paper.get("kind") or ""),
                Text(paper.get("session") or ""),
                Text(paper.get("variant") or ""),
                Text(str(paper.get("question_count") or 0)),
                Text(paper.get("path") or ""),
            )
        self.console.print(table)
        self.console.print("[dim]open <n|pdf_id>[/dim]")

    def cmd_open(self, args: List[str]) -> None:
        if not args:
            self.console.print("[yellow]usage: open <n|pdf_id>[/yellow]")
            return
        token = self.resolve_index(args[0], self.last_paper_ids)
        if token and self.fetch_paper(token):
            self.navigate("paper", token)
            return
        self.console.print("[red]paper not found:[/red] %s" % escape(args[0]))

    def cmd_question(self, args: List[str]) -> None:
        if not args:
            question_id = self.current_question_id()
            if question_id:
                self.render_question(question_id)
                return
            self.console.print("[yellow]usage: question <n|question_id>[/yellow]")
            return
        token = self.resolve_index(args[0], self.last_question_ids)
        if token and self.fetch_question(token):
            self.navigate("question", token)
            return
        self.console.print("[red]question not found:[/red] %s" % escape(args[0]))

    def cmd_jump(self, args: List[str]) -> None:
        if not args:
            self.console.print("[yellow]usage: jump <id>[/yellow]")
            return
        token = args[0]
        if token.isdigit():
            if self.current_kind == "question" and self.last_question_ids:
                candidate = self.resolve_index(token, self.last_question_ids)
                if candidate and self.fetch_question(candidate):
                    self.navigate("question", candidate)
                    return
            if self.last_paper_ids:
                candidate = self.resolve_index(token, self.last_paper_ids)
                if candidate and self.fetch_paper(candidate):
                    self.navigate("paper", candidate)
                    return
            if self.last_question_ids:
                candidate = self.resolve_index(token, self.last_question_ids)
                if candidate and self.fetch_question(candidate):
                    self.navigate("question", candidate)
                    return
        if self.fetch_paper(token):
            self.navigate("paper", token)
        elif self.fetch_question(token):
            self.navigate("question", token)
        else:
            row = self.conn.execute(
                "SELECT md_id FROM documents WHERE md_id = ?", (token,)
            ).fetchone()
            if row is not None:
                self.render_md(token)
            else:
                self.console.print("[red]not found:[/red] %s" % escape(token))

    def cmd_paper(self, args: List[str]) -> None:
        pdf_id = args[0] if args else self.current_paper_id()
        if not pdf_id:
            self.console.print("[yellow]usage: paper <pdf_id>[/yellow]")
            return
        if not self.fetch_paper(pdf_id):
            self.console.print("[red]paper not found:[/red] %s" % escape(pdf_id))
            return
        self.navigate("paper", pdf_id)

    def cmd_tree(self, args: List[str]) -> None:
        pdf_id = args[0] if args else self.current_paper_id()
        if not pdf_id:
            self.console.print("[yellow]usage: tree <pdf_id>[/yellow]")
            return
        paper = self.fetch_paper(pdf_id)
        if not paper:
            self.console.print("[red]paper not found:[/red] %s" % escape(pdf_id))
            return
        questions = self.fetch_paper_questions(pdf_id, include_stimulus=True)
        self.last_question_ids = [q["question_id"] for q in questions]
        by_parent = {}  # type: Dict[Optional[str], List[Dict[str, Any]]]
        for question in questions:
            by_parent.setdefault(question.get("parent_id"), []).append(question)

        root_label = Text("Paper %s" % paper["pdf_id"])
        root = Tree(root_label)

        def add_children(node: Any, parent_id: str) -> None:
            for child in by_parent.get(parent_id, []):
                label = self._question_label(child)
                child_node = node.add(Text(label))
                add_children(child_node, child["question_id"])

        for question in by_parent.get(None, []):
            node = root.add(Text(self._question_label(question)))
            add_children(node, question["question_id"])
        self.console.print(root)

    def _question_label(self, question: Dict[str, Any]) -> str:
        number = question.get("number") or ""
        sub = question.get("sub_label") or ""
        kind = question.get("kind") or ""
        marks = question.get("marks")
        mark_text = "" if marks is None else " [%s]" % marks
        stem = _short(question.get("stem") or "", 70)
        return "%s %s%s%s  %s" % (kind, number, sub, mark_text, stem)

    def cmd_md(self, args: List[str]) -> None:
        ident = args[0] if args else self.current_paper_id()
        if not ident:
            self.console.print("[yellow]usage: md <pdf_id|md_id>[/yellow]")
            return
        self.render_md(ident)

    def _resolve_question_arg(self, args: List[str]) -> Optional[str]:
        if args:
            return self.resolve_index(args[0], self.last_question_ids)
        return self.current_question_id()

    def cmd_tables(self, args: List[str]) -> None:
        question_id = self._resolve_question_arg(args)
        if not question_id:
            self.console.print("[yellow]usage: tables <question_id>[/yellow]")
            return
        question = self.fetch_question(question_id)
        if not question:
            self.console.print("[red]question not found:[/red] %s" % escape(question_id))
            return
        if not question.get("tables"):
            self.console.print("[yellow]no tables[/yellow]")
            return
        self.render_tables(question)

    def cmd_attachments(self, args: List[str]) -> None:
        question_id = self._resolve_question_arg(args)
        if not question_id:
            self.console.print("[yellow]usage: attachments <question_id>[/yellow]")
            return
        question = self.fetch_question(question_id)
        if not question:
            self.console.print("[red]question not found:[/red] %s" % escape(question_id))
            return
        if not question.get("attachments"):
            self.console.print("[yellow]no attachments[/yellow]")
            return
        self.render_attachments(question)

    def cmd_search(self, args: List[str]) -> None:
        text = " ".join(args).strip()
        if not text:
            self.console.print("[yellow]usage: search <text>[/yellow]")
            return
        questions = self.search(text)
        self.last_question_ids = [q["question_id"] for q in questions]
        if not questions:
            self.console.print("[yellow]no matches[/yellow]")
            return
        self.render_question_table(questions, title="Search results")
        self.console.print("[dim]question <n>[/dim]")

    def cmd_raw(self, args: List[str]) -> None:
        ident = args[0] if args else (self.current_id or "")
        if not ident:
            self.console.print("[yellow]usage: raw <id>[/yellow]")
            return
        self.render_raw(ident)

    def cmd_clear(self, args: List[str]) -> None:
        self.console.clear()

    # ------------------------------------------------------------------
    # script execution
    # ------------------------------------------------------------------

    def _parse(
        self, parser: argparse.ArgumentParser, args: Sequence[str]
    ) -> Optional[argparse.Namespace]:
        try:
            return parser.parse_args(list(args))
        except SystemExit:
            return None

    def _run_script(
        self, script_name: str, args: Sequence[str], label: str = ""
    ) -> int:
        script_path = REPO_ROOT / script_name
        if not script_path.exists():
            self.console.print("[red]script not found:[/red] %s" % script_name)
            return 2
        deps = SCRIPT_DEPS.get(script_name, [])
        if deps and self.uv:
            cmd = ["uv", "run"]
            for dep in deps:
                cmd.extend(["--with", dep])
            cmd.extend(["python3", str(script_path)])
            cmd.extend(str(arg) for arg in args)
            env = os.environ.copy()
            env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache-paper")
        else:
            cmd = [sys.executable, str(script_path)]
            cmd.extend(str(arg) for arg in args)
            env = os.environ.copy()

        self.console.print(
            Text("$ " + " ".join(shlex.quote(part) for part in cmd), style="dim")
        )
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        except OSError as exc:
            self.console.print("[red]failed to start:[/red] %s" % exc)
            return 1
        assert process.stdout is not None
        for line in process.stdout:
            self.console.print(line.rstrip("\n"))
        return_code = process.wait()
        if return_code != 0:
            self.console.print("[red]exit code %d[/red]" % return_code)
        elif label:
            self.console.print("[green]%s done[/green]" % label)
        return return_code

    def _all_pdfs(self) -> List[Path]:
        return sorted(self.papers_dir.glob("*.pdf"))

    def _qp_pdfs(self) -> List[Path]:
        return [
            path
            for path in self._all_pdfs()
            if "QP" in path.name.upper()
        ]

    def _all_mds(self) -> List[Path]:
        return sorted(self.md_dir.glob("*.md"))

    def _qp_mds(self) -> List[Path]:
        return [
            path
            for path in self._all_mds()
            if "QP" in path.name.upper()
        ]

    def _resolve_pdf_targets(self, targets: Sequence[str]) -> List[Path]:
        """Accept pdf_id, absolute/relative PDF path, or bare filename."""
        pdfs = []  # type: List[Path]
        seen = set()  # type: set
        for target in targets:
            candidates = []
            path = Path(target)
            if path.exists() and path.suffix.lower() == ".pdf":
                candidates.append(path)
            paper = self.fetch_paper(target)
            if paper and paper.get("path"):
                candidates.append(Path(paper["path"]))
            bare = self.papers_dir / target
            if bare.exists():
                candidates.append(bare)
            if not candidates:
                self.console.print(
                    "[yellow]not a PDF path or known pdf_id:[/yellow] %s"
                    % escape(target)
                )
                continue
            candidate = candidates[0]
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                pdfs.append(candidate)
        return pdfs

    def _mds_for_pdfs(self, pdfs: Sequence[Path]) -> List[Path]:
        wanted = set()  # type: set
        for pdf in pdfs:
            try:
                wanted.add(str(pdf.resolve()))
            except OSError:
                pass
        result = []  # type: List[Path]
        if not self.md_dir.exists():
            return result
        for meta_path in sorted(self.md_dir.glob("*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            source = meta.get("source_pdf")
            if not source:
                continue
            try:
                source_resolved = str(Path(source).resolve())
            except OSError:
                continue
            if source_resolved not in wanted:
                continue
            stem = str(meta.get("stem") or meta_path.name[: -len(".meta.json")])
            md_path = self.md_dir / ("%s.md" % stem)
            if md_path.exists():
                result.append(md_path)
        return result

    def _confirm(self, message: str) -> bool:
        if self.assume_yes:
            return True
        try:
            answer = Prompt.ask(message, choices=["y", "n"], default="n")
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return False
        return answer == "y"

    def cmd_sync(self, args: List[str]) -> None:
        code = self._run_script(
            "paperdb.py",
            [
                "--db",
                str(self.db_path),
                "sync",
                "--papers-dir",
                str(self.papers_dir),
                "--md",
                str(self.md_dir),
                "--questions",
                str(self.questions_dir),
            ],
            label="sync",
        )
        if code == 0:
            self.cmd_stats([])

    def cmd_convert(self, args: List[str]) -> None:
        parser = argparse.ArgumentParser(prog="convert", add_help=False)
        parser.add_argument("--all", action="store_true")
        parser.add_argument("paths", nargs="*")
        namespace = self._parse(parser, args)
        if namespace is None:
            return
        if namespace.paths:
            pdfs = [Path(path) for path in namespace.paths]
        else:
            pdfs = self._all_pdfs() if namespace.all else self._qp_pdfs()
        if not pdfs:
            self.console.print("[yellow]no PDFs found[/yellow]")
            return
        if not namespace.paths and not self._confirm(
            "Convert %d PDF(s) to Markdown?" % len(pdfs)
        ):
            return
        self._run_script(
            "pdf_to_md.py",
            ["--out", str(self.md_dir)] + [str(path) for path in pdfs],
            label="convert",
        )

    def cmd_split(self, args: List[str]) -> None:
        parser = argparse.ArgumentParser(prog="split", add_help=False)
        parser.add_argument("--profile", default="history")
        parser.add_argument("paths", nargs="*")
        namespace = self._parse(parser, args)
        if namespace is None:
            return
        if namespace.paths:
            mds = [Path(path) for path in namespace.paths]
        else:
            mds = self._qp_mds()
        if not mds:
            self.console.print("[yellow]no Markdown files found[/yellow]")
            return
        if not namespace.paths and not self._confirm(
            "Split %d Markdown file(s) with profile %s?"
            % (len(mds), namespace.profile)
        ):
            return
        self._run_script(
            "split_questions.py",
            [
                "--profile",
                namespace.profile,
                "--out",
                str(self.questions_dir),
            ]
            + [str(path) for path in mds],
            label="split",
        )

    def cmd_pipeline(self, args: List[str]) -> None:
        parser = argparse.ArgumentParser(prog="pipeline", add_help=False)
        parser.add_argument("--profile", default="history")
        parser.add_argument("--all", action="store_true")
        namespace = self._parse(parser, args)
        if namespace is None:
            return
        pdfs = self._all_pdfs() if namespace.all else self._qp_pdfs()
        if not pdfs:
            self.console.print("[yellow]no PDFs found[/yellow]")
            return
        if not self._confirm(
            "Run full pipeline on %d PDF(s) with profile %s?"
            % (len(pdfs), namespace.profile)
        ):
            return

        code = self._run_script(
            "pdf_to_md.py",
            ["--out", str(self.md_dir)] + [str(path) for path in pdfs],
            label="convert",
        )
        if code != 0:
            return

        mds = self._all_mds() if namespace.all else self._qp_mds()
        if not mds:
            self.console.print("[yellow]no Markdown files produced[/yellow]")
            return
        code = self._run_script(
            "split_questions.py",
            [
                "--profile",
                namespace.profile,
                "--out",
                str(self.questions_dir),
            ]
            + [str(path) for path in mds],
            label="split",
        )
        if code != 0:
            return

        code = self._run_script(
            "paperdb.py",
            [
                "--db",
                str(self.db_path),
                "sync",
                "--papers-dir",
                str(self.papers_dir),
                "--md",
                str(self.md_dir),
                "--questions",
                str(self.questions_dir),
            ],
            label="sync",
        )
        if code == 0:
            self.console.print("[green]pipeline complete[/green]")
            self.cmd_stats([])

    def cmd_run(self, args: List[str]) -> None:
        if not args:
            self.console.print(
                "[yellow]usage: run <script.py> [args ...][/yellow]"
            )
            return
        script_name = args[0]
        if not script_name.endswith(".py"):
            script_name += ".py"
        self._run_script(script_name, args[1:], label=script_name)

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def print_welcome(self) -> None:
        self.console.print(
            Panel(
                Text(
                    "paper_tui\n\n"
                    "This is a command REPL over data/papers.db.\n"
                    "Type 'help' for commands, 'home' for the paper list, "
                    "'quit' to exit."
                ),
                title="paper corpus",
                border_style="cyan",
            )
        )

    def execute(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.console.print("[red]cannot parse command:[/red] %s" % exc)
            return
        if not parts:
            return
        command = parts[0]
        args = parts[1:]
        commands = {
            "help": self.cmd_help,
            "?": self.cmd_help,
            "stats": self.cmd_stats,
            "papers": self.cmd_papers,
            "open": self.cmd_open,
            "paper": self.cmd_paper,
            "question": self.cmd_question,
            "q": self.cmd_question,
            "jump": self.cmd_jump,
            "tree": self.cmd_tree,
            "md": self.cmd_md,
            "tables": self.cmd_tables,
            "attachments": self.cmd_attachments,
            "parent": self.cmd_parent,
            "children": self.cmd_children,
            "next": self.cmd_next,
            "prev": self.cmd_prev,
            "back": self.cmd_back,
            "home": self.cmd_home,
            "search": self.cmd_search,
            "raw": self.cmd_raw,
            "sync": self.cmd_sync,
            "convert": self.cmd_convert,
            "split": self.cmd_split,
            "pipeline": self.cmd_pipeline,
            "run": self.cmd_run,
            "clear": self.cmd_clear,
        }
        if command in ("quit", "exit"):
            raise SystemExit(0)
        handler = commands.get(command)  # type: Optional[Callable[[List[str]], None]]
        if handler is None:
            self.console.print("[red]unknown command:[/red] %s" % escape(command))
            self.console.print("[dim]type 'help'[/dim]")
            return
        handler(args)

    def run_interactive(self) -> int:
        self.open_db()
        self.print_welcome()
        try:
            while True:
                try:
                    line = Prompt.ask("[bold cyan]paper_tui[/bold cyan]")
                except (EOFError, KeyboardInterrupt):
                    self.console.print()
                    break
                if line.strip() in ("quit", "exit", "q!"):
                    break
                try:
                    self.execute(line)
                except SystemExit:
                    break
        finally:
            self.close_db()
        return 0

    def run_commands(self, commands: Sequence[str]) -> int:
        self.open_db()
        try:
            for command in commands:
                for line in command.splitlines():
                    self.execute(line)
        except SystemExit:
            return 0
        finally:
            self.close_db()
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper_tui.py",
        description="Rich command REPL for the exam paper corpus.",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument(
        "--papers-dir", default=DEFAULT_PAPERS_DIR, help="PDF corpus directory"
    )
    parser.add_argument("--md", default=DEFAULT_MD_DIR, help="Markdown directory")
    parser.add_argument(
        "--questions", default=DEFAULT_QUESTIONS_DIR, help="questions JSON directory"
    )
    parser.add_argument(
        "--yes", action="store_true", help="do not ask for confirmation"
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        default=[],
        help="run one command and exit; may be repeated",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args=argv)
    tui = PaperTUI(
        db_path=Path(args.db),
        papers_dir=Path(args.papers_dir),
        md_dir=Path(args.md),
        questions_dir=Path(args.questions),
        assume_yes=args.yes,
    )
    if args.command:
        return tui.run_commands(args.command)
    return tui.run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
