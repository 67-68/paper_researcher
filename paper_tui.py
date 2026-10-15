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

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory

    PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover - optional dependency
    PromptSession = None  # type: ignore
    Completer = object  # type: ignore
    Completion = None  # type: ignore
    FormattedText = None  # type: ignore
    InMemoryHistory = None  # type: ignore
    PROMPT_TOOLKIT = False

import paper_common  # noqa: E402
import paperdb  # noqa: E402


DEFAULT_DB = "data/papers.db"
DEFAULT_PAPERS_DIR = "data/papers"
DEFAULT_MD_DIR = "data/md"
DEFAULT_QUESTIONS_DIR = "data/questions"

HELP_SECTIONS = [
    (
        "通用 General",
        [
            ("help / ?", "显示命令总览"),
            ("help -d / help --definition", "显示概念、数据流、profile 说明"),
            ("help -e / help --examples", "用具体卷子演示查询与 DB 对应关系"),
            ("help <command>", "显示某条命令的说明"),
            ("stats", "显示 papers / documents / questions 计数"),
            ("clear", "清屏"),
            ("quit / exit", "退出 TUI"),
        ],
    ),
    (
        "卷子与题目 Browse",
        [
            (
                "papers [--subject S] [--kind QP|MS] [--session S]",
                "列出卷子；记住序号供 open 使用",
            ),
            (
                "open <n|id>",
                "统一入口：自动识别 paper / question / md；数字用最近列表",
            ),
            (
                "question / q / jump",
                "兼容别名，等同于 open",
            ),
            ("paper [pdf_id]", "跳到当前题所属的卷子"),
            ("tree [pdf_id]", "显示 stimulus -> main -> sub 树"),
            ("md [pdf_id|md_id]", "渲染 Markdown 全文"),
            ("tables [question_id]", "显示题目关联的表格"),
            ("attachments [question_id]", "显示题目关联的图片附件"),
            ("stimulus / sources", "查看当前题目的 shared_stem / Source 文本"),
            ("parent", "跳到父题或 stimulus"),
            ("children", "列出子题"),
            ("next / prev", "在当前卷子内前后移动"),
            ("search <text>", "搜索主题干 / 题干 / 共享材料"),
            ("raw [id]", "显示完整 JSON"),
            ("back", "返回上一个节点"),
            ("home", "回到卷子列表"),
        ],
    ),
    (
        "Paper 处理 Pipeline",
        [
            (
                "pipeline [--profile P] [--yes] [--all] [pdf_id|pdf ...]",
                "convert + split + sync；默认 QP；--all 会连 MS 一起转换/入库，但 MS 不切题",
            ),
            ("process [pdf_id]", "处理当前卷子或指定 pdf_id"),
            ("convert [--yes] [--all] [pdf_id|pdf ...]", "PDF -> Markdown"),
            ("split [--profile P] [--yes] [md ...]", "Markdown -> questions.json"),
            ("sync", "Markdown + questions.json -> papers.db"),
        ],
    ),
    (
        "脚本调用 Run scripts",
        [
            ("run <script.py> [args ...]", "直接运行仓库脚本"),
            ("run pdf_to_md.py --help", "查看 PDF -> Markdown 参数"),
            ("run split_questions.py --help", "查看切题参数"),
            ("run paperdb.py --help", "查看数据库查询参数"),
        ],
    ),
]

COMMAND_NAMES = [
    "help",
    "?",
    "stats",
    "papers",
    "open",
    "paper",
    "question",
    "q",
    "jump",
    "tree",
    "md",
    "tables",
    "attachments",
    "stimulus",
    "sources",
    "parent",
    "children",
    "next",
    "prev",
    "back",
    "home",
    "search",
    "raw",
    "sync",
    "convert",
    "split",
    "pipeline",
    "process",
    "run",
    "clear",
    "quit",
    "exit",
]

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
        self.last_list = ""  # "papers" | "questions"
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
        self.last_list = "questions"
        self.render_question_table(
            questions, title="Questions (stimulus + main + sub)"
        )
        if not questions:
            if (paper.get("kind") or "").upper() == "MS":
                self.console.print(
                    "[yellow]This is a mark scheme (MS).[/yellow]"
                )
                self.console.print(
                    "[dim]split_questions.py currently supports QP only; "
                    "MS is converted/stored but not split into questions.[/dim]"
                )
                self.console.print(
                    "[dim]Use `md %s` to inspect the Markdown.[/dim]"
                    % escape(pdf_id)
                )
            else:
                self.console.print(
                    "[yellow]No questions indexed for this paper yet.[/yellow]"
                )
                self.console.print(
                    "[dim]Run: pipeline %s[/dim]" % escape(pdf_id)
                )
        self.console.print(
            "[dim]question <n> | tree | md | pipeline <pdf_id> | back | home[/dim]"
        )

    def render_question(self, question_id: str, full: bool = False) -> None:
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
            if (
                not full
                and key in ("stem", "parent_stem", "shared_stem")
                and value
            ):
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
            self.last_list = "questions"

        self.render_tables(question)
        self.render_attachments(question)

        if question.get("shared_stem") and not full:
            self.console.print(
                "[dim]shared_stem is the Source / Passage text; "
                "use `stimulus` to read it in full.[/dim]"
            )

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
            "[dim]parent | stimulus | children | next | prev | tables | "
            "attachments | raw | paper | md | back[/dim]"
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

    def _document_id(self, md_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT md_id FROM documents WHERE md_id = ?", (md_id,)
        ).fetchone()
        return row["md_id"] if row is not None else None

    def resolve_open_ident(self, token: str) -> Optional[Tuple[str, str]]:
        """Resolve a numeric index or id to (paper|question|md, id)."""
        if token.isdigit():
            candidates = []  # type: List[Tuple[str, Sequence[str]]]
            if self.last_list == "questions" and self.last_question_ids:
                candidates.append(("question", self.last_question_ids))
                candidates.append(("paper", self.last_paper_ids))
            elif self.last_list == "papers" and self.last_paper_ids:
                candidates.append(("paper", self.last_paper_ids))
                candidates.append(("question", self.last_question_ids))
            elif self.current_kind in ("paper", "question") and self.last_question_ids:
                candidates.append(("question", self.last_question_ids))
                candidates.append(("paper", self.last_paper_ids))
            else:
                candidates.append(("paper", self.last_paper_ids))
                candidates.append(("question", self.last_question_ids))
            for kind, values in candidates:
                ident = self.resolve_index(token, values)
                if not ident:
                    continue
                if kind == "paper" and self.fetch_paper(ident):
                    return ("paper", ident)
                if kind == "question" and self.fetch_question(ident):
                    return ("question", ident)
            return None

        if self.fetch_paper(token):
            return ("paper", token)
        if self.fetch_question(token):
            return ("question", token)
        if self._document_id(token):
            return ("md", token)
        return None

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

    def cmd_stimulus(self, args: List[str]) -> None:
        """Open the stimulus node and render shared_stem in full."""
        target = self.current_question_id()
        if args:
            target = self.resolve_index(args[0], self.last_question_ids) or args[0]
        if not target:
            self.console.print(
                "[yellow]open a question first, or use: stimulus <question_id>[/yellow]"
            )
            return
        question = self.fetch_question(target)
        if not question:
            self.console.print(
                "[red]question not found:[/red] %s" % escape(target)
            )
            return

        if question.get("kind") == "stimulus":
            stimulus_id = question["question_id"]
        else:
            parent_id = question.get("parent_id")
            if not parent_id:
                if question.get("shared_stem"):
                    # No separate stimulus row; render this question in full
                    # so shared_stem is visible.
                    if self.current_kind and self.current_id:
                        self.history.append((self.current_kind, self.current_id))
                    self.render_question(target, full=True)
                    return
                self.console.print("[yellow]this question has no shared_stem[/yellow]")
                return
            parent = self.fetch_question(parent_id)
            if not parent:
                self.console.print("[red]parent not found[/red]")
                return
            stimulus_id = parent_id

        stimulus = self.fetch_question(stimulus_id)
        if not stimulus:
            self.console.print("[red]stimulus not found[/red]")
            return
        if not stimulus.get("shared_stem"):
            self.console.print("[yellow]this stimulus has no shared_stem[/yellow]")
            return
        if self.current_kind and self.current_id:
            self.history.append((self.current_kind, self.current_id))
        self.render_question(stimulus_id, full=True)

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
        self.last_list = "questions"
        self.render_question_table(children, title="Children")
        self.console.print("[dim]open <n>[/dim]")

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
        if args and args[0] in ("-d", "--definition"):
            self.print_definition()
            return
        if args and args[0] in ("-e", "--examples"):
            self.print_examples()
            return

        if args:
            needle = args[0].lstrip("-")
            matches = []  # type: List[Tuple[str, str, str]]
            for section, items in HELP_SECTIONS:
                for command, description in items:
                    if needle in command or needle in command.split()[0]:
                        matches.append((section, command, description))
            if not matches:
                self.console.print(
                    "[yellow]no help for:[/yellow] %s" % escape(args[0])
                )
                self.console.print("[dim]try: help -d[/dim]")
                return
            table = Table(title="help %s" % escape(args[0]), show_lines=False)
            table.add_column("section", style="magenta")
            table.add_column("command", style="cyan", no_wrap=True)
            table.add_column("what it does")
            for section, command, description in matches:
                table.add_row(Text(section), Text(command), Text(description))
            self.console.print(table)
            return

        for section, items in HELP_SECTIONS:
            table = Table(title=section, show_lines=False, title_justify="left")
            table.add_column("command", style="cyan", no_wrap=True)
            table.add_column("what it does")
            for command, description in items:
                table.add_row(Text(command), Text(description))
            self.console.print(table)
        self.console.print(
            "[dim]help -d : 基本概念、处理流程、profile、导航索引[/dim]\n"
            "[dim]help -e : 具体卷子示例与 DB 字段对应[/dim]"
        )

    def print_definition(self) -> None:
        concepts = Text()
        concepts.append("PDF\n", style="bold cyan")
        concepts.append(
            "  不可变源头。pdf_id = 文件 sha256 的前 16 位。\n"
            "  文件名会重复，所以用内容哈希而不是文件名当 ID。\n\n"
        )
        concepts.append("Markdown\n", style="bold cyan")
        concepts.append(
            "  由 pdf_to_md.py 生成：\n"
            "    <stem>.md           正文，图片以 Markdown 链接出现\n"
            "    <stem>.assets/      图片附件\n"
            "    <stem>.meta.json    表格、图片、版面块等元数据\n\n"
        )
        concepts.append("questions\n", style="bold cyan")
        concepts.append(
            "  由 split_questions.py 生成。层级是：\n"
            "    stimulus -> main -> sub\n"
            "  stimulus 是 Source / Passage 共享材料，正文存在 shared_stem；\n"
            "  main 是题号 1、2、3；\n"
            "  sub 是 (a)、(b)、(i)、(ii) 等小题。\n"
            "  每道题可以有题干、分值、选项、表格、attachments、warnings。\n"
            "  注意：attachments 只表示图片/图形；History 的 Source 文本在\n"
            "  shared_stem 里，用 `stimulus` 或 `sources` 查看。\n\n"
        )
        concepts.append("QP / MS\n", style="bold cyan")
        concepts.append(
            "  QP = question paper，用于切题；\n"
            "  MS = mark scheme，目前只做 PDF -> Markdown 和入库，\n"
            "       不参与 split_questions.py 切题。\n\n"
        )
        concepts.append("papers.db\n", style="bold cyan")
        concepts.append(
            "  paperdb.py 建立的派生索引，不是 source of truth。\n"
            "  源头始终是磁盘上的 PDF / Markdown / questions.json；\n"
            "  数据库删除后重新 sync 就可以重建。"
        )
        self.console.print(
            Panel(concepts, title="基本概念", border_style="cyan")
        )

        flow = Text()
        flow.append("PDF\n", style="bold")
        flow.append("  -> pdf_to_md.py\n")
        flow.append("<stem>.md + <stem>.assets/ + <stem>.meta.json\n", style="bold")
        flow.append("  -> split_questions.py --profile history\n")
        flow.append("<stem>.questions.json / <stem>.questions.jsonl\n", style="bold")
        flow.append("  -> paperdb.py sync\n")
        flow.append("data/papers.db\n", style="bold")
        flow.append("  papers -> documents -> questions")
        self.console.print(
            Panel(flow, title="一次 paper 的处理流程", border_style="green")
        )

        relations = Text()
        relations.append("papers\n", style="bold cyan")
        relations.append(
            "  pdf_id (主键)\n"
            "  sha256, path, subject, syllabus, session, variant, kind\n"
        )
        relations.append("    | 1 : N\n", style="dim")
        relations.append("    v\n", style="dim")
        relations.append("documents\n", style="bold cyan")
        relations.append(
            "  md_id (主键)\n"
            "  pdf_id -> papers.pdf_id\n"
            "  stem, md_path, meta_path, assets_dir\n"
        )
        relations.append("    | 1 : N\n", style="dim")
        relations.append("    v\n", style="dim")
        relations.append("questions\n", style="bold cyan")
        relations.append(
            "  question_id (主键)\n"
            "  md_id -> documents.md_id\n"
            "  pdf_id -> papers.pdf_id\n"
            "  parent_id -> questions.question_id\n"
            "  kind = stimulus / main / sub\n"
            "  stem, parent_stem, shared_stem, marks, tables, attachments"
        )
        self.console.print(
            Panel(relations, title="数据库关系", border_style="green")
        )

        profiles = Text()
        profiles.append(
            "profile 是 split_questions.py 的切题规则集，\n"
            "只影响 Markdown -> 题目 这一步，不影响 PDF 解析。\n\n"
        )
        profiles.append("auto     ", style="bold cyan")
        profiles.append(
            "通用规则。先试它；如果题号或小题切得不对，再指定专用 profile。\n"
        )
        profiles.append("history  ", style="bold cyan")
        profiles.append(
            "AS History 9489。识别 Section A/B、Source A-D、\n"
            "         (a)/(b)、[15]/[25] 等分值。\n"
        )
        profiles.append("cs       ", style="bold cyan")
        profiles.append(
            "AS Computer Science 9618。识别编号题、(i)/(ii)、\n"
            "         表格题、图形填空/逻辑门等附件。\n\n"
        )
        profiles.append("用法：\n", style="bold")
        profiles.append(
            "  split --profile history\n"
            "  pipeline --profile history <pdf_id>\n"
            "  pipeline --profile cs <pdf_id>"
        )
        self.console.print(
            Panel(profiles, title="profile 是什么", border_style="yellow")
        )

        nav = Text()
        nav.append(
            "open 是统一入口，会自动识别 paper / question / md。\n"
            "如果给的是数字，就使用“最近一次列表”的序号：\n\n"
        )
        nav.append("papers ...     ", style="bold")
        nav.append("记住 last_paper_ids\n")
        nav.append("open 2         ", style="bold")
        nav.append("打开第 2 份卷子\n")
        nav.append("open <pdf_id>  ", style="bold")
        nav.append("直接打开指定卷子\n\n")
        nav.append("打开卷子后     ", style="bold")
        nav.append("题目表写入 last_question_ids\n")
        nav.append("open 3         ", style="bold")
        nav.append("打开当前卷子题目表里的第 3 条\n")
        nav.append("open <qid>     ", style="bold")
        nav.append("直接打开指定题目\n\n")
        nav.append("question / q / jump  ", style="bold")
        nav.append("兼容别名，等同于 open\n")
        nav.append("stimulus / sources   ", style="bold")
        nav.append("查看当前题目的 shared_stem\n")
        nav.append("back           ", style="bold")
        nav.append("回到上一个节点\n")
        nav.append("home           ", style="bold")
        nav.append("回到卷子列表\n")
        nav.append("tree           ", style="bold")
        nav.append("查看 stimulus -> main -> sub 树\n")
        nav.append("raw            ", style="bold")
        nav.append("查看完整 JSON\n\n")
        nav.append(
            "输入框下方会实时显示匹配命令和候选参数（需要 prompt_toolkit）。"
        )
        self.console.print(
            Panel(nav, title="导航与索引", border_style="magenta")
        )

        flows = Text()
        flows.append("全量处理\n", style="bold cyan")
        flows.append("  pipeline\n  pipeline --all\n  pipeline --profile history --yes\n\n")
        flows.append("单份卷子\n", style="bold cyan")
        flows.append("  open <pdf_id>\n  process\n  # 或\n  pipeline <pdf_id>\n\n")
        flows.append("MS（mark scheme）\n", style="bold cyan")
        flows.append(
            "  convert <ms_pdf>     # 只转 Markdown\n"
            "  pipeline --all       # 转换所有 PDF，但只切 QP\n"
            "  # pipeline <ms_id> 会提示 MS 不切题并停止\n\n"
        )
        flows.append("查题\n", style="bold cyan")
        flows.append(
            "  papers --kind QP\n"
            "  open <n|pdf_id>\n"
            "  open <n|question_id>\n"
            "  stimulus\n"
            "  tree\n"
            "  tables\n"
            "  attachments\n\n"
        )
        flows.append("直接跑脚本\n", style="bold cyan")
        flows.append(
            "  run pdf_to_md.py --help\n"
            "  run split_questions.py --help\n"
            "  run paperdb.py --help"
        )
        self.console.print(
            Panel(flows, title="常用工作流", border_style="blue")
        )

    def print_examples(self) -> None:
        demo = Text()
        demo.append("用 June 2017 (v1) QP 举例：\n\n", style="bold")
        demo.append("1. 找卷子\n", style="bold cyan")
        demo.append(
            '   papers --session "June 2017" --kind QP\n'
            "   # 输出里会看到：\n"
            "   #   7cc8cf1a98c4b37f  history  QP  June 2017  v1\n\n"
        )
        demo.append("2. 打开卷子\n", style="bold cyan")
        demo.append(
            "   open 7cc8cf1a98c4b37f\n"
            "   open 1                 # 也可以用上一次 papers 列表的序号\n\n"
        )
        demo.append("3. 打开主题 / 小题\n", style="bold cyan")
        demo.append(
            "   open 2                 # 当前卷子题目表第 2 条 = main Q1\n"
            "   open 7cc8cf1a98c4b37f:g0006:1        # main Q1\n"
            "   open 7cc8cf1a98c4b37f:g0006:1-01-a   # Q1(a)\n\n"
        )
        demo.append("4. 看题干和 Source\n", style="bold cyan")
        demo.append(
            "   raw\n"
            "   # main Q1: stem = \"Read the sources and then answer both parts...\"\n"
            "   stimulus\n"
            "   # 打开 stimulus 节点，完整显示 shared_stem（Source A/B/C/D）\n"
            "   sources\n"
            "   # 同上，stimulus 的别名\n\n"
        )
        demo.append("5. 查关联和全文\n", style="bold cyan")
        demo.append(
            "   parent\n"
            "   children\n"
            "   paper\n"
            "   md\n"
            "   search Napoleon\n"
            "   tables\n"
            "   attachments\n"
        )
        self.console.print(
            Panel(demo, title="示例：从卷子到题目", border_style="cyan")
        )

        mapping = Text()
        mapping.append(
            "试卷里的不同部分，在数据库里对应：\n\n", style="bold"
        )
        mapping.append("PDF 文件\n", style="bold cyan")
        mapping.append(
            "  papers.pdf_id        sha256 前 16 位\n"
            "  papers.path          PDF 路径\n"
            "  papers.kind          QP / MS\n"
            "  papers.session       June 2017\n"
            "  papers.variant       v1\n\n"
        )
        mapping.append("整份 Markdown\n", style="bold cyan")
        mapping.append(
            "  documents.md_id      <pdf_id>:md:<stem>\n"
            "  documents.md_path    .md 文件\n"
            "  documents.meta_path  .meta.json\n"
            "  documents.assets_dir 图片目录\n\n"
        )
        mapping.append("Section A / B\n", style="bold cyan")
        mapping.append("  questions.section\n\n")
        mapping.append("Source A/B/C/D 文本\n", style="bold cyan")
        mapping.append(
            "  questions.shared_stem\n"
            "  对应 kind = stimulus 的行，也在 main/sub 里冗余保存\n\n"
        )
        mapping.append("主题干\n", style="bold cyan")
        mapping.append(
            "  questions.stem\n"
            "  对应 kind = main\n"
            "  数字题号 1/2/3 存在 questions.number\n\n"
        )
        mapping.append("小题 (a)/(b)/(i)/(ii)\n", style="bold cyan")
        mapping.append(
            "  questions.stem\n"
            "  对应 kind = sub\n"
            "  questions.sub_label = (a)\n"
            "  questions.parent_id -> 主题的 question_id\n\n"
        )
        mapping.append("分值 [15] / [25]\n", style="bold cyan")
        mapping.append("  questions.marks\n\n")
        mapping.append("图片 / 图形题\n", style="bold cyan")
        mapping.append("  questions.attachments\n\n")
        mapping.append("表格\n", style="bold cyan")
        mapping.append("  questions.tables")
        self.console.print(
            Panel(mapping, title="试卷结构 -> DB 字段", border_style="green")
        )

        ids = Text()
        ids.append("question_id 的组成：\n\n", style="bold")
        ids.append("<pdf_id>:<group_id>:<题目路径>\n\n", style="bold cyan")
        ids.append(
            "  7cc8cf1a98c4b37f:g0006:1\n"
            "  7cc8cf1a98c4b37f:g0006:1-01-a\n"
            "  7cc8cf1a98c4b37f:g0006:__stimulus__\n\n"
            "  g0006        第 6 个解析组，通常对应一套 Source + 主题\n"
            "  1            main question 1\n"
            "  1-01-a       main 1 的第 1 个小题 (a)\n"
            "  __stimulus__ 该组的共享材料节点\n\n"
        )
        ids.append("数字索引规则：\n", style="bold")
        ids.append(
            "  papers 之后：open <n> 打开第 n 份卷子\n"
            "  打开卷子后：open <n> 打开题目表第 n 条\n"
            "  search 之后：open <n> 打开搜索结果第 n 条"
        )
        self.console.print(
            Panel(ids, title="ID 与数字索引", border_style="yellow")
        )

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
        self.last_list = "papers"

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
            if self.current_kind == "paper" and self.current_id:
                self.render_paper(self.current_id)
                return
            if self.current_kind == "question" and self.current_id:
                self.render_question(self.current_id)
                return
            self.console.print(
                "[yellow]usage: open <n|pdf_id|question_id|md_id>[/yellow]"
            )
            return
        resolved = self.resolve_open_ident(args[0])
        if resolved is None:
            self.console.print("[red]not found:[/red] %s" % escape(args[0]))
            return
        kind, ident = resolved
        if kind == "paper":
            self.navigate("paper", ident)
        elif kind == "question":
            self.navigate("question", ident)
        else:
            self.render_md(ident)

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
        """Alias for open; kept for compatibility."""
        self.cmd_open(args)

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
        self.last_list = "questions"
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

        attachments = question.get("attachments") or []
        if not attachments and question.get("parent_id"):
            parent = self.fetch_question(question["parent_id"])
            if parent and parent.get("attachments"):
                self.console.print(
                    "[dim]No direct attachments; showing parent stimulus attachments.[/dim]"
                )
                attachments = parent["attachments"]

        if not attachments:
            self.console.print("[yellow]no image attachments[/yellow]")
            if question.get("shared_stem"):
                self.console.print(
                    "[dim]This question has shared_stem text (Source / Passage). "
                    "In History QP the sources are text, not image attachments.[/dim]"
                )
                self.console.print(
                    "[dim]Use `stimulus` to read shared_stem in full, "
                    "or `md` for the whole paper.[/dim]"
                )
            return
        self.render_attachments({"attachments": attachments})

    def cmd_search(self, args: List[str]) -> None:
        text = " ".join(args).strip()
        if not text:
            self.console.print("[yellow]usage: search <text>[/yellow]")
            return
        questions = self.search(text)
        self.last_question_ids = [q["question_id"] for q in questions]
        self.last_list = "questions"
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

    def _paper_for_path(self, pdf: Path) -> Optional[Dict[str, Any]]:
        resolved = str(pdf.resolve())
        row = self.conn.execute(
            "SELECT pdf_id FROM papers WHERE path = ? OR path = ?",
            (resolved, str(pdf)),
        ).fetchone()
        if row is None:
            return None
        return self.fetch_paper(row["pdf_id"])

    def _pdf_kind(self, pdf: Path) -> str:
        """Return QP / MS for a PDF, preferring the indexed metadata."""
        paper = self._paper_for_path(pdf)
        if paper and paper.get("kind"):
            return str(paper["kind"]).upper()
        match = paperdb.KIND_RE.search(pdf.name)
        if match:
            return match.group(1).upper()
        return ""

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

    def _confirm(self, message: str, force_yes: bool = False) -> bool:
        if force_yes or self.assume_yes:
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
        parser.add_argument("--yes", action="store_true")
        parser.add_argument("paths", nargs="*")
        namespace = self._parse(parser, args)
        if namespace is None:
            return
        explicit = bool(namespace.paths)
        if explicit:
            pdfs = self._resolve_pdf_targets(namespace.paths)
        else:
            pdfs = self._all_pdfs() if namespace.all else self._qp_pdfs()
        if not pdfs:
            self.console.print("[yellow]no PDFs found[/yellow]")
            return
        if not explicit and not self._confirm(
            "Convert %d PDF(s) to Markdown?" % len(pdfs),
            force_yes=namespace.yes,
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
        parser.add_argument("--yes", action="store_true")
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
            % (len(mds), namespace.profile),
            force_yes=namespace.yes,
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
        parser.add_argument("--yes", action="store_true")
        parser.add_argument(
            "targets",
            nargs="*",
            help="pdf_id, PDF path, or bare PDF filename; default all QP PDFs",
        )
        namespace = self._parse(parser, args)
        if namespace is None:
            return
        explicit = bool(namespace.targets)
        if explicit:
            pdfs = self._resolve_pdf_targets(namespace.targets)
        else:
            pdfs = self._all_pdfs() if namespace.all else self._qp_pdfs()
        if not pdfs:
            self.console.print("[yellow]no PDFs found[/yellow]")
            return

        # split_questions.py is QP-oriented.  MS files can be converted and
        # stored, but they must not be fed to the question splitter.
        qp_pdfs = [path for path in pdfs if self._pdf_kind(path) != "MS"]
        ms_pdfs = [path for path in pdfs if self._pdf_kind(path) == "MS"]

        if explicit and ms_pdfs:
            self.console.print(
                "[yellow]MS detected: %s[/yellow]"
                % ", ".join(path.name for path in ms_pdfs)
            )
            self.console.print(
                "[yellow]split_questions.py currently supports QP only; "
                "MS will be converted and synced but not split.[/yellow]"
            )
            if not qp_pdfs:
                self.console.print(
                    "[red]No QP selected; nothing to split.[/red]"
                )
                self.console.print(
                    "[dim]Use the QP PDF, or `convert <ms_pdf>` if you only "
                    "want Markdown and metadata.[/dim]"
                )
                return
            self.console.print(
                "[dim]QP will be split; MS will only be converted and synced.[/dim]"
            )

        if ms_pdfs:
            confirm_message = (
                "Run pipeline on %d PDF(s): %d QP, %d MS; MS will only be "
                "converted/synced, not split. Profile %s?"
                % (len(pdfs), len(qp_pdfs), len(ms_pdfs), namespace.profile)
            )
        else:
            confirm_message = (
                "Run full pipeline on %d PDF(s) with profile %s?"
                % (len(pdfs), namespace.profile)
            )
        if not self._confirm(confirm_message, force_yes=namespace.yes):
            return

        code = self._run_script(
            "pdf_to_md.py",
            ["--out", str(self.md_dir)] + [str(path) for path in pdfs],
            label="convert",
        )
        if code != 0:
            return

        mds = self._mds_for_pdfs(qp_pdfs) if qp_pdfs else []
        if not mds:
            self.console.print(
                "[yellow]no QP Markdown produced for the selected PDF(s); "
                "skipping split[/yellow]"
            )
        else:
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
            if explicit and self.current_kind == "paper":
                current_paper = self.fetch_paper(self.current_id)
                if current_paper and current_paper.get("path"):
                    current_path = str(Path(current_paper["path"]).resolve())
                    wanted = {
                        str(pdf.resolve())
                        for pdf in pdfs
                        if pdf.exists()
                    }
                    if current_path in wanted:
                        self.render_paper(self.current_id)

    def cmd_process(self, args: List[str]) -> None:
        """Pipeline a single paper; defaults to the currently open paper."""
        if not args:
            pdf_id = self.current_paper_id()
            if not pdf_id:
                self.console.print(
                    "[yellow]usage: process <pdf_id> (or open a paper first)[/yellow]"
                )
                return
            args = [pdf_id]
        self.cmd_pipeline(args)

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
                    "这是一个操作 data/papers.db 的命令 REPL。\n\n"
                    "  help       命令总览（按通用 / 查询 / 处理 / 脚本分区）\n"
                    "  help -d    基本概念、处理流程、profile 和导航说明\n"
                    "  help -e    用具体卷子演示查询与 DB 对应关系\n"
                    "  open <n|id> 统一入口：自动识别 paper / question / md\n"
                    "  home       卷子列表\n"
                    "  quit       退出\n\n"
                    "输入时会实时显示匹配命令和候选参数。"
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
            "stimulus": self.cmd_stimulus,
            "sources": self.cmd_stimulus,
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
            "process": self.cmd_process,
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
        try:
            if PROMPT_TOOLKIT:
                return self._run_prompt_toolkit()
            return self._run_basic_repl()
        finally:
            self.close_db()

    def _run_basic_repl(self) -> int:
        self.print_welcome()
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
        return 0

    def _run_prompt_toolkit(self) -> int:
        """REPL with live command/arg hints when prompt_toolkit is available."""
        tui = self

        class TuiCompleter(Completer):  # type: ignore[misc, valid-type]
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                word = document.get_word_before_cursor()
                for candidate in tui.completion_candidates(text):
                    if candidate.startswith(word):
                        yield Completion(candidate, start_position=-len(word))

        holder = {}  # type: Dict[str, Any]

        def toolbar():
            session = holder.get("session")
            text = session.default_buffer.text if session is not None else ""
            return FormattedText(
                [("class:bottom-toolbar", tui.suggest_line(text))]
            )

        session = PromptSession(
            completer=TuiCompleter(),
            complete_while_typing=True,
            bottom_toolbar=toolbar,
            history=InMemoryHistory(),
        )
        holder["session"] = session
        self.print_welcome()
        while True:
            try:
                line = session.prompt("paper_tui> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            if line.strip() in ("quit", "exit", "q!"):
                break
            try:
                self.execute(line)
            except SystemExit:
                break
        return 0

    def _option_list(self, command: str) -> List[str]:
        options = {
            "papers": [
                "--subject",
                "--kind",
                "--session",
                "--syllabus",
                "--paper-code",
            ],
            "pipeline": ["--profile", "--yes", "--all"],
            "convert": ["--yes", "--all"],
            "split": ["--profile", "--yes"],
            "help": ["-d", "--definition"],
        }
        return options.get(command, [])

    def completion_candidates(self, text: str) -> List[str]:
        if not text:
            return list(COMMAND_NAMES)
        parts = text.split()
        if text.endswith(" "):
            word = ""
            prior = parts
        else:
            word = parts[-1]
            prior = parts[:-1]

        if not prior:
            return [c for c in COMMAND_NAMES if c.startswith(word)]

        command = prior[0]
        if prior and prior[-1] == "--profile":
            return [v for v in ("history", "cs", "auto") if v.startswith(word)]
        if prior and prior[-1] == "--kind":
            return [v for v in ("QP", "MS") if v.upper().startswith(word.upper())]

        options = self._option_list(command)
        candidates = []  # type: List[str]
        if word.startswith("-"):
            candidates.extend(options)
        else:
            if command in (
                "open",
                "question",
                "q",
                "jump",
                "paper",
                "tree",
                "md",
                "tables",
                "attachments",
                "stimulus",
                "sources",
                "raw",
                "process",
                "pipeline",
                "convert",
            ):
                candidates.extend(self.last_paper_ids)
                candidates.extend(self.last_question_ids)
            candidates.extend(options)
        return [c for c in candidates if c.startswith(word)][:20]

    def suggest_line(self, text: str) -> str:
        raw = text or ""
        stripped = raw.strip()
        trailing = raw.endswith(" ")
        if not stripped:
            return (
                "commands: help  papers  open  stimulus  tree  md  attachments  "
                "search  sync  pipeline  process  convert  split  run  quit"
            )

        parts = stripped.split()
        command = parts[0]
        if len(parts) == 1 and not trailing:
            matches = [c for c in COMMAND_NAMES if c.startswith(command)]
            if matches:
                return "commands: " + "  ".join(matches[:12])
            return "unknown command; type help"

        if command == "papers":
            if "--kind" in parts:
                return "values: QP  MS"
            if "--subject" in parts:
                return "values: history  cs"
            if "--session" in parts:
                return "values: June 2021  November 2017  Specimen (2015) ..."
            return "options: --subject --kind --session --syllabus --paper-code"
        if command in ("pipeline", "convert"):
            if "--profile" in parts:
                return "values: history  cs  auto"
            return (
                "options: --profile --yes --all  "
                "targets: pdf_id/path; recent papers: "
                + self._sample_ids(self.last_paper_ids)
            )
        if command == "split":
            if "--profile" in parts:
                return "values: history  cs  auto"
            return "options: --profile --yes  paths: .md"
        if command == "run":
            return (
                "scripts: pdf_to_md.py  split_questions.py  paperdb.py  "
                "crawl_pdfs.py"
            )
        if command in ("help", "?"):
            return "args: -d/--definition  <command>"
        if command in (
            "open",
            "question",
            "q",
            "jump",
            "paper",
            "tree",
            "md",
            "tables",
            "attachments",
            "stimulus",
            "sources",
            "raw",
            "process",
        ):
            if self.last_list == "questions" and self.last_question_ids:
                return (
                    "args: <n|id>; recent questions: "
                    + self._sample_ids(self.last_question_ids)
                )
            if self.last_paper_ids:
                return (
                    "args: <n|id>; recent papers: "
                    + self._sample_ids(self.last_paper_ids)
                )
            return "args: <n|id>"
        return "type help for commands; help -d for concepts"

    def _sample_ids(self, values: Sequence[str], limit: int = 3) -> str:
        if not values:
            return "(none yet)"
        return "  ".join(
            "%d=%s" % (index + 1, value)
            for index, value in enumerate(values[:limit])
        )

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
