#!/usr/bin/env python3
"""jev.py - call the OpenCode Zen Jev (System One) API.

Jev is not a text-generation model.  One request has:

    state       the text (or JSON object) to evaluate
    questions   a map of typed questions (noul | choice | score)

and returns one structured answer per question::

    state + questions  ->  answers

The OpenCode Zen API root is ``https://opencode.ai/zen/v1``.  Jev is served
from ``<root>/systemone``, so this script defaults to::

    https://opencode.ai/zen/v1/systemone

Examples
--------

Generic: questions come from a JSON file / string::

    printf 'The paper is due tomorrow.' | python3 jev.py \
        --questions questions.json

    python3 jev.py --state-file paper.md \
        --questions-json '{"is_urgent": {"type": "noul",
                         "instructions": "Does this convey urgency?"}}'

Yes/no::

    python3 jev.py --state-file paper.md \
        --noul "Does this convey urgency?"

Multiple choice::

    python3 jev.py --state-file paper.md \
        --choice "Which team should handle this?" \
        --option billing="Payments and invoices" \
        --option technical="Bugs and integrations"

Rubric score::

    python3 jev.py --state-file paper.md \
        --score "How frustrated is the writer?" \
        --level Calm --level Frustrated --level "Very angry"

List models::

    python3 jev.py --list-models

API key
-------

Pass ``--api-key`` or ``--api-key-file``, or set one of
``OPENCODE_API_KEY``, ``OPENCODE_ZEN_API_KEY`` or ``TYPESAFE_API_KEY``.
``OPENCODE_API_KEY_FILE`` is also honoured.

Output is JSON on stdout.  Diagnostics go to stderr.  Exit codes:
0 success, 1 HTTP/network error, 2 usage/validation error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Sequence, Tuple

__version__ = "0.1.0"

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_MODEL = "jev-1.13"
DEFAULT_TIMEOUT = 120.0
USER_AGENT = "paper_researcher-jev/%s" % __version__

ENV_API_KEYS = (
    "OPENCODE_API_KEY",
    "OPENCODE_ZEN_API_KEY",
    "TYPESAFE_API_KEY",
)
ENV_ORG_ID = "OPENCODE_ORG_ID"

QUESTION_TYPES = ("noul", "choice", "score")


class JevError(Exception):
    """Base class for script errors."""


class JevHTTPError(JevError):
    """A non-2xx HTTP response from the API."""

    def __init__(self, status: int, message: str, body: bytes = b"") -> None:
        super().__init__("HTTP %s: %s" % (status, message))
        self.status = status
        self.message = message
        self.body = body


def _warn(message: str) -> None:
    sys.stderr.write("jev: %s\n" % message)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# API key / URL helpers
# ---------------------------------------------------------------------------

def _read_api_key_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError as exc:
        raise JevError("--api-key-file: %s" % exc) from exc
    return value or None


def resolve_api_key(
    explicit: Optional[str] = None,
    key_file: Optional[str] = None,
) -> Optional[str]:
    if explicit is not None:
        explicit = explicit.strip()
        if explicit:
            return explicit
    key_file = key_file or os.environ.get("OPENCODE_API_KEY_FILE")
    if key_file:
        from_file = _read_api_key_file(key_file)
        if from_file:
            return from_file
    for name in ENV_API_KEYS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def systemone_url(base_url: str) -> str:
    """Return the Jev evaluation endpoint for *base_url*."""
    base = base_url.rstrip("/")
    if base.endswith("/systemone"):
        return base
    return base + "/systemone"


def models_url(base_url: str) -> str:
    """Return the model-list endpoint for *base_url*."""
    base = base_url.rstrip("/")
    if base.endswith("/models"):
        return base
    if base.endswith("/systemone"):
        base = base[: -len("/systemone")]
    return base + "/models"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _read_text_source(value: str, option: str) -> str:
    if value == "-":
        return sys.stdin.read()
    try:
        with open(value, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise JevError("%s: %s" % (option, exc)) from exc


def _parse_json_text(text: str, option: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise JevError("%s: invalid JSON: %s" % (option, exc)) from exc


def _state_uses_stdin(args: argparse.Namespace) -> bool:
    if args.state_file == "-" or args.state_json_file == "-" or args.state_json == "-":
        return True
    return not any(
        value is not None
        for value in (args.state, args.state_file, args.state_json, args.state_json_file)
    )


def _questions_use_stdin(args: argparse.Namespace) -> bool:
    return args.questions == "-" or args.questions_json == "-"


def load_state(args: argparse.Namespace) -> Any:
    if args.state is not None:
        return args.state
    if args.state_json is not None:
        text = sys.stdin.read() if args.state_json == "-" else args.state_json
        return _parse_json_text(text, "--state-json")
    if args.state_file is not None:
        return _read_text_source(args.state_file, "--state-file")
    if args.state_json_file is not None:
        text = _read_text_source(args.state_json_file, "--state-json-file")
        return _parse_json_text(text, "--state-json-file")
    return sys.stdin.read()


def _question_mode(args: argparse.Namespace) -> str:
    if args.questions is not None or args.questions_json is not None:
        return "generic"
    if args.noul is not None:
        return "noul"
    if args.choice is not None:
        return "choice"
    if args.score is not None:
        return "score"
    raise JevError(
        "no questions given: use --questions/--questions-json, "
        "--noul, --choice or --score"
    )


def _parse_option(raw: str) -> Tuple[str, Optional[str]]:
    key, separator, description = raw.partition("=")
    key = key.strip()
    if not key:
        raise JevError("--option needs KEY=DESCRIPTION: %r" % raw)
    if not separator:
        return key, None
    description = description.strip()
    return key, (description if description else None)


def build_questions(args: argparse.Namespace) -> Dict[str, Any]:
    mode = _question_mode(args)

    if mode == "generic":
        if args.id != "answer":
            raise JevError("--id is only valid with --noul / --choice / --score")
        for option, value in (
            ("--true", args.true_text),
            ("--false", args.false_text),
            ("--option", args.option),
            ("--level", args.level),
        ):
            if value is not None:
                raise JevError("%s is only valid with a shorthand question type" % option)
        if args.questions is not None:
            text = _read_text_source(args.questions, "--questions")
        else:
            text = (
                sys.stdin.read()
                if args.questions_json == "-"
                else args.questions_json
            )
        return _parse_json_text(text, "--questions")

    if mode == "noul":
        if args.option or args.level:
            raise JevError("--option/--level cannot be used with --noul")
        question: Dict[str, Any] = {"type": "noul", "instructions": args.noul}
        criteria: Dict[str, Any] = {}
        if args.true_text is not None:
            criteria["true"] = args.true_text
        if args.false_text is not None:
            criteria["false"] = args.false_text
        if criteria:
            question["criteria"] = criteria
        return {args.id: question}

    if mode == "choice":
        if args.true_text is not None or args.false_text is not None:
            raise JevError("--true/--false cannot be used with --choice")
        if args.level:
            raise JevError("--level cannot be used with --choice")
        if not args.option:
            raise JevError("--choice needs at least one --option KEY=DESCRIPTION")
        criteria_choice: Dict[str, Optional[str]] = {}
        for raw in args.option:
            key, description = _parse_option(raw)
            if key in criteria_choice:
                raise JevError("duplicate --option key: %r" % key)
            criteria_choice[key] = description
        return {
            args.id: {
                "type": "choice",
                "instructions": args.choice,
                "criteria": criteria_choice,
            }
        }

    if mode == "score":
        if args.true_text is not None or args.false_text is not None:
            raise JevError("--true/--false cannot be used with --score")
        if args.option:
            raise JevError("--option cannot be used with --score")
        levels = list(args.level or [])
        if len(levels) < 2:
            raise JevError("--score needs at least two --level values")
        return {
            args.id: {
                "type": "score",
                "instructions": args.score,
                "criteria": levels,
            }
        }

    raise JevError("unsupported question mode: %s" % mode)


def validate_questions(questions: Any, source: str = "questions") -> None:
    if not isinstance(questions, dict):
        raise JevError("%s must be a JSON object mapping ids to questions" % source)
    if not questions:
        raise JevError("%s must contain at least one question" % source)
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise JevError("%s: question ids must be non-empty strings" % source)
        if not isinstance(question, dict):
            raise JevError("%s: question %r must be an object" % (source, question_id))
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise JevError(
                "%s: question %r has unsupported type %r (expected one of %s)"
                % (source, question_id, question_type, ", ".join(QUESTION_TYPES))
            )
        if "instructions" not in question:
            raise JevError("%s: question %r is missing instructions" % (source, question_id))
        if question_type == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise JevError(
                    "%s: choice question %r needs a non-empty criteria object"
                    % (source, question_id)
                )
        elif question_type == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise JevError(
                    "%s: score question %r needs at least two criteria levels"
                    % (source, question_id)
                )
        elif question_type == "noul":
            criteria = question.get("criteria")
            if criteria is not None and not isinstance(criteria, dict):
                raise JevError(
                    "%s: noul question %r criteria must be an object"
                    % (source, question_id)
                )


# ---------------------------------------------------------------------------
# Convenience Python API
# ---------------------------------------------------------------------------

def evaluate(
    state: Any,
    questions: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    org_id: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Evaluate *state* against *questions* and return the API response.

    This is the importable equivalent of the CLI's normal request path.
    """
    validate_questions(questions)
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        raise JevError(
            "missing API key: pass api_key or set OPENCODE_API_KEY "
            "(or OPENCODE_ZEN_API_KEY / TYPESAFE_API_KEY)"
        )
    payload = {"state": state, "model": model, "questions": questions}
    return http_json(
        "POST",
        systemone_url(base_url),
        payload=payload,
        api_key=resolved_key,
        org_id=org_id,
        timeout=timeout,
    )


def fetch_models(
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    models_url_override: Optional[str] = None,
    org_id: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Return the model-list response from ``<base_url>/models``."""
    url = models_url_override or models_url(base_url)
    return http_json(
        "GET",
        url,
        api_key=resolve_api_key(api_key),
        org_id=org_id,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer <redacted>"
    return redacted


def _format_error_body(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return "(empty response body)"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
            if message:
                return str(message)
        if isinstance(error, str) and error:
            return error
        detail = payload.get("detail")
        if detail is not None:
            if isinstance(detail, str):
                return detail
            return json.dumps(detail, ensure_ascii=False)
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
    return text


def http_json(
    method: str,
    url: str,
    *,
    payload: Optional[Any] = None,
    api_key: Optional[str] = None,
    org_id: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> Any:
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    if org_id:
        headers["X-Opencode-Org-Id"] = org_id

    if dry_run:
        return {
            "dry_run": True,
            "method": method,
            "url": url,
            "headers": _redact_headers(headers),
            "body": payload,
            "api_key_present": bool(api_key),
        }

    data: Optional[bytes] = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise JevHTTPError(exc.code, _format_error_body(raw), raw) from exc
    except urllib.error.URLError as exc:
        raise JevError("connection failed: %s" % (exc.reason,)) from exc
    except OSError as exc:
        raise JevError("connection failed: %s" % exc) from exc

    if not raw:
        return {}
    try:
        return json.loads(raw.decode(charset, "replace"))
    except json.JSONDecodeError as exc:
        snippet = raw[:300].decode("utf-8", "replace")
        raise JevError("server returned invalid JSON: %s (%s)" % (exc, snippet)) from exc


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def output_json(value: Any, *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    sys.stdout.write(text + "\n")


def _primary_value(answer: Any) -> Any:
    if not isinstance(answer, dict):
        return answer
    answer_type = answer.get("type")
    if answer_type == "noul":
        return answer.get("noul")
    if answer_type == "choice":
        return answer.get("choice")
    if answer_type == "score":
        return answer.get("score")
    return answer


def output_answers(response: Any, args: argparse.Namespace) -> None:
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        answers = {}

    if args.value:
        if len(answers) != 1:
            raise JevError("--value needs exactly one answer (got %d)" % len(answers))
        value = _primary_value(next(iter(answers.values())))
        if isinstance(value, str):
            sys.stdout.write(value + "\n")
        else:
            sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
        return

    if args.answers_only:
        output_json(answers, compact=args.compact)
    else:
        output_json(response, compact=args.compact)


def output_models(response: Any, args: argparse.Namespace) -> None:
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list):
        output_json(response, compact=args.compact)
        return
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            sys.stdout.write(item["id"] + "\n")
        else:
            sys.stdout.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev.py",
        description="Call OpenCode Zen's Jev (System One) API.",
        epilog=(
            "Jev returns structured answers, not generated text. "
            "Use --questions for arbitrary typed questions, or the "
            "--noul/--choice/--score shortcuts for a single question."
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenCode API key (default: OPENCODE_API_KEY, OPENCODE_ZEN_API_KEY or TYPESAFE_API_KEY)",
    )
    parser.add_argument(
        "--api-key-file",
        default=os.environ.get("OPENCODE_API_KEY_FILE"),
        help="read the API key from a file (default: OPENCODE_API_KEY_FILE)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API root (default: %(default)s)",
    )
    parser.add_argument(
        "--models-url",
        default=None,
        help="exact model-list URL (default: <base-url>/models)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id (default: %(default)s)")
    parser.add_argument("--org-id", default=os.environ.get(ENV_ORG_ID), help="X-Opencode-Org-Id header")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")

    parser.add_argument("--compact", action="store_true", help="one-line JSON output")
    parser.add_argument("--answers-only", action="store_true", help="print only the answers map")
    parser.add_argument(
        "--value",
        action="store_true",
        help="print only the scalar value of a single answer (noul/score/choice)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the request without sending it")
    parser.add_argument("--list-models", action="store_true", help="list available model ids and exit")

    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument("--state", default=None, help="state text")
    state_group.add_argument("--state-file", default=None, help="read state text from a file, or '-' for stdin")
    state_group.add_argument("--state-json", default=None, help="state as an inline JSON value, or '-' for stdin")
    state_group.add_argument(
        "--state-json-file",
        default=None,
        help="state as JSON in a file, or '-' for stdin",
    )

    question_group = parser.add_mutually_exclusive_group()
    question_group.add_argument(
        "--questions",
        default=None,
        help="JSON file mapping ids to typed questions, or '-' for stdin",
    )
    question_group.add_argument(
        "--questions-json",
        default=None,
        help="inline JSON object of typed questions, or '-' for stdin",
    )
    question_group.add_argument("--noul", default=None, metavar="INSTRUCTIONS", help="single yes/no question")
    question_group.add_argument(
        "--choice",
        default=None,
        metavar="INSTRUCTIONS",
        help="single multiple-choice question (needs --option)",
    )
    question_group.add_argument(
        "--score",
        default=None,
        metavar="INSTRUCTIONS",
        help="single rubric question (needs --level)",
    )

    parser.add_argument("--id", default="answer", help="question id for shorthand modes (default: %(default)s)")
    parser.add_argument("--true", dest="true_text", default=None, help="noul: what 'true' means")
    parser.add_argument("--false", dest="false_text", default=None, help="noul: what 'false' means")
    parser.add_argument(
        "--option",
        action="append",
        default=None,
        metavar="KEY=DESCRIPTION",
        help="choice: add an option (repeatable)",
    )
    parser.add_argument(
        "--level",
        action="append",
        default=None,
        metavar="LEVEL",
        help="score: add a rubric level, lowest first (repeatable)",
    )
    return parser


def _validate_stdin_usage(args: argparse.Namespace) -> None:
    if _state_uses_stdin(args) and _questions_use_stdin(args):
        raise JevError(
            "cannot read both state and questions from stdin; "
            "use --state-file/--state or a --questions path"
        )


def _run(args: argparse.Namespace) -> int:
    api_key = resolve_api_key(args.api_key, args.api_key_file)

    if args.list_models:
        url = args.models_url or models_url(args.base_url)
        if args.dry_run:
            output_json(
                http_json(
                    "GET",
                    url,
                    api_key=api_key,
                    org_id=args.org_id,
                    timeout=args.timeout,
                    dry_run=True,
                ),
                compact=args.compact,
            )
            return 0
        response = fetch_models(
            api_key=api_key,
            base_url=args.base_url,
            models_url_override=args.models_url,
            org_id=args.org_id,
            timeout=args.timeout,
        )
        output_models(response, args)
        return 0

    _validate_stdin_usage(args)
    questions = build_questions(args)
    validate_questions(questions)
    state = load_state(args)

    if not api_key and not args.dry_run:
        raise JevError(
            "missing API key: pass --api-key or set OPENCODE_API_KEY "
            "(or OPENCODE_ZEN_API_KEY / TYPESAFE_API_KEY)"
        )

    url = systemone_url(args.base_url)
    payload = {
        "state": state,
        "model": args.model,
        "questions": questions,
    }

    if args.dry_run:
        output_json(
            http_json(
                "POST",
                url,
                payload=payload,
                api_key=api_key,
                org_id=args.org_id,
                timeout=args.timeout,
                dry_run=True,
            ),
            compact=args.compact,
        )
        return 0

    response = evaluate(
        state,
        questions,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        org_id=args.org_id,
        timeout=args.timeout,
    )
    output_answers(response, args)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except JevHTTPError as exc:
        _warn(str(exc))
        return 1
    except JevError as exc:
        _warn(str(exc))
        return 2
    except KeyboardInterrupt:
        _warn("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
