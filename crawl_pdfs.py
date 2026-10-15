#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_pdfs.py

Recursively crawl HTML pages and download linked PDFs.

Design goals:
  - one job, one file, one output directory
  - respect robots.txt by default
  - deterministic output: PDFs + manifest.jsonl
  - no HTML/PDF parsing beyond link discovery; downstream tools handle the content

Example:
  python3 crawl_pdfs.py \
    --out out/papers \
    --max-depth 3 \
    --delay 1.0 \
    "https://example.com/pastpapers/"

The manifest is append-only JSONL:
  {"url": "...", "file": "papers/foo.pdf", "sha256": "...", ...}

Downstream:
  pdf_to_md.py       reads PDFs and produces Markdown + attachments
  split_questions.py reads Markdown and produces question JSON
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import (
    parse_qsl,
    unquote,
    urlencode,
    urldefrag,
    urljoin,
    urlparse,
    urlunparse,
)
from urllib.robotparser import RobotFileParser

import requests

DEFAULT_UA = "pdf-crawler/2.0 (+https://example.invalid; contact: you@example.com)"
PDF_MAGIC = b"%PDF-"
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}


@dataclass
class Record:
    url: str
    parent: Optional[str]
    depth: int
    kind: str
    status: str
    file: Optional[str] = None
    content_type: Optional[str] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    error: Optional[str] = None


class LinkExtractor(HTMLParser):
    """Extract href/src links from HTML using only the standard library."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []
        self.base: Optional[str] = None

    def handle_starttag(self, tag: str, attrs) -> None:
        d = dict(attrs)
        if tag in ("a", "area"):
            href = d.get("href")
            if href:
                self.links.append(href)
        elif tag == "base" and d.get("href"):
            self.base = d["href"]


class Crawler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out).expanduser().resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out / "manifest.jsonl"

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": args.user_agent})
        for header in args.header:
            if ":" in header:
                k, v = header.split(":", 1)
                self.session.headers[k.strip()] = v.strip()
        if args.cookie:
            self.session.headers["Cookie"] = args.cookie

        self.seen: Set[str] = set()
        self.downloaded_urls: Set[str] = set()
        self.robots: Dict[str, Optional[RobotFileParser]] = {}
        self.allowed_hosts: Set[str] = set(getattr(args, "allow_host", []) or [])
        self.start_urls: Set[str] = set()

        self.pages = 0
        self.pdfs = 0
        self.records: List[Record] = []
        self._load_resume()

    # ---------- setup / resume ----------

    def _load_resume(self) -> None:
        if not (self.args.resume and self.manifest_path.exists()):
            return
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("status") == "downloaded" and rec.get("url"):
                    self.downloaded_urls.add(rec["url"])
                    self.seen.add(rec["url"])

    # ---------- logging ----------

    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def debug(self, msg: str) -> None:
        if self.args.verbose:
            self.log(msg)

    # ---------- URL helpers ----------

    @staticmethod
    def normalize(url: str) -> Optional[str]:
        try:
            url, _ = urldefrag(url)
            p = urlparse(url)
        except ValueError:
            return None
        if p.scheme not in ("http", "https"):
            return None
        query = [
            (k, v)
            for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        path = p.path or "/"
        return urlunparse((p.scheme, p.netloc, path, p.params, urlencode(query), ""))

    @staticmethod
    def looks_like_pdf(url: str) -> bool:
        return Path(unquote(urlparse(url).path)).suffix.lower() == ".pdf"

    def in_scope(self, url: str) -> bool:
        p = urlparse(url)
        if not self.args.any_domain and p.netloc not in self.allowed_hosts:
            return False
        if self.args.path_prefix:
            prefixes = [x.rstrip("/") + "/" for x in self.args.path_prefix]
            if not any(p.path.startswith(prefix) for prefix in prefixes):
                return False
        if url in self.start_urls:
            return True
        if self.args.exclude and any(re.search(pattern, url) for pattern in self.args.exclude):
            return False
        if self.args.include and not any(re.search(pattern, url) for pattern in self.args.include):
            return False
        return True

    # ---------- robots ----------

    def allowed_by_robots(self, url: str) -> bool:
        if self.args.no_robots:
            return True
        p = urlparse(url)
        base = "%s://%s" % (p.scheme, p.netloc)
        if base not in self.robots:
            robots_url = urljoin(base, "/robots.txt")
            rp: Optional[RobotFileParser] = None
            try:
                # Use our own session/UA instead of RobotFileParser.read():
                # read() uses urllib's default UA and can silently parse an
                # empty/403 response as "block everything".
                resp = self.session.get(
                    robots_url,
                    timeout=self.args.timeout,
                    allow_redirects=True,
                )
                if resp.status_code != 404:
                    resp.raise_for_status()
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.parse(resp.text.splitlines())
            except Exception:
                # No robots.txt, or fetch failed: be permissive by default.
                rp = None
            self.robots[base] = rp
        rp = self.robots[base]
        return rp is None or rp.can_fetch(self.args.user_agent, url)

    # ---------- HTTP ----------

    def request(self, url: str) -> requests.Response:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.args.retries + 1):
            try:
                resp = self.session.get(
                    url,
                    stream=True,
                    timeout=self.args.timeout,
                    allow_redirects=True,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    status = resp.status_code
                    resp.close()
                    raise requests.HTTPError("HTTP %s" % status)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.args.retries:
                    delay = min(
                        60.0,
                        self.args.delay * (2 ** (attempt - 1)),
                    ) + random.uniform(0, 0.3)
                    self.log(
                        "retry %d/%d %s: %s (sleep %.1fs)"
                        % (attempt, self.args.retries, url, exc, delay)
                    )
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def polite_sleep(self) -> None:
        if self.args.delay > 0:
            time.sleep(self.args.delay + random.uniform(0, self.args.delay * 0.2))

    # ---------- files ----------

    @staticmethod
    def safe_name(name: str) -> str:
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
        return name or "download"

    @staticmethod
    def unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        i = 1
        while True:
            candidate = path.with_name("%s-%d%s" % (stem, i, suffix))
            if not candidate.exists():
                return candidate
            i += 1

    def filename_from_response(self, resp: requests.Response, url: str) -> str:
        cd = resp.headers.get("Content-Disposition", "")
        candidates = [
            r"filename\*\s*=\s*[^']*'[^']*'([^;]+)",
            r'filename\s*=\s*"([^"]+)"',
            r"filename\s*=\s*([^;]+)",
        ]
        for pattern in candidates:
            m = re.search(pattern, cd, re.I)
            if m:
                return self.safe_name(unquote(m.group(1).strip().strip('"\'')))
        name = Path(unquote(urlparse(url).path)).name
        if not name:
            name = "download_%s.pdf" % hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        return self.safe_name(name)

    def target_path(self, resp: requests.Response, url: str) -> Path:
        name = self.filename_from_response(resp, url)
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        if self.args.layout == "flat":
            return self.out / name

        p = urlparse(url)
        base = self.out / self.safe_name(p.netloc)
        parts = [self.safe_name(x) for x in Path(unquote(p.path)).parts if x not in ("/", "\\", "")]
        for part in parts[:-1]:
            base = base / part
        return base / name

    def save_pdf(
        self,
        resp: requests.Response,
        url: str,
        parent: Optional[str],
        depth: int,
    ) -> Record:
        ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        chunks = resp.iter_content(chunk_size=64 * 1024)
        first = b""
        for chunk in chunks:
            if chunk:
                first = chunk
                break
        if not first.startswith(PDF_MAGIC):
            resp.close()
            return Record(
                url=url,
                parent=parent,
                depth=depth,
                kind="other",
                status="skipped",
                content_type=ct,
                error="not a PDF",
            )

        path = self.target_path(resp, url)
        if path.exists() and not self.args.overwrite:
            path = self.unique_path(path)
        if self.args.overwrite and path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")

        digest = hashlib.sha256()
        total = 0
        try:
            with tmp.open("wb") as f:
                for chunk in itertools.chain([first], chunks):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if self.args.max_size_mb and total > self.args.max_size_mb * 1024 * 1024:
                        raise ValueError("exceeds --max-size-mb")
                    digest.update(chunk)
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        finally:
            resp.close()

        self.pdfs += 1
        if self.args.print_paths or self.args.print0:
            if self.args.print0:
                sys.stdout.buffer.write(str(path.resolve()).encode("utf-8") + b"\0")
                sys.stdout.buffer.flush()
            else:
                print(str(path.resolve()), flush=True)
        return Record(
            url=url,
            parent=parent,
            depth=depth,
            kind="pdf",
            status="downloaded",
            file=str(path.relative_to(self.out)),
            content_type=ct,
            bytes=total,
            sha256=digest.hexdigest(),
        )

    # ---------- HTML ----------

    def extract_links(self, resp: requests.Response) -> List[str]:
        content = resp.content
        if self.args.max_html_mb and len(content) > self.args.max_html_mb * 1024 * 1024:
            raise ValueError("HTML exceeds --max-html-mb")
        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        html = content.decode(encoding, errors="replace")
        parser = LinkExtractor()
        try:
            parser.feed(html)
        except Exception:
            pass
        base_url = urljoin(resp.url, parser.base) if parser.base else resp.url
        links: List[str] = []
        for href in parser.links:
            href = href.strip()
            if not href:
                continue
            if href.lower().startswith(("javascript:", "mailto:", "tel:", "data:", "blob:")):
                continue
            link = self.normalize(urljoin(base_url, href))
            if link:
                links.append(link)
        return links

    # ---------- bookkeeping ----------

    def record(self, rec: Record) -> None:
        self.records.append(rec)
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def summary(self) -> None:
        html_pages = sum(
            1 for r in self.records if r.kind == "html" and r.status == "fetched"
        )
        pdfs = sum(1 for r in self.records if r.status == "downloaded")
        errors = [r for r in self.records if r.status == "error"]
        self.log("")
        self.log("done: %d html pages, %d pdfs, %d errors" % (html_pages, pdfs, len(errors)))
        for r in errors[:20]:
            self.log("  error: %s -> %s" % (r.url, r.error))
        self.log("manifest: %s" % self.manifest_path)

    # ---------- main loop ----------

    def run(self, start_urls: List[str]) -> None:
        q: Deque[Tuple[str, Optional[str], int, bool]] = deque()
        for raw in start_urls:
            url = self.normalize(raw)
            if not url:
                self.log("skip invalid start URL: %s" % raw)
                continue
            self.allowed_hosts.add(urlparse(url).netloc)
            self.start_urls.add(url)
            q.append((url, None, 0, True))

        while q:
            url, parent, depth, allow_follow = q.popleft()
            if url in self.seen:
                continue
            self.seen.add(url)

            if not self.in_scope(url):
                self.record(Record(url, parent, depth, "skip", "out_of_scope"))
                continue

            if not self.allowed_by_robots(url):
                self.debug("robots blocked: %s" % url)
                self.record(Record(url, parent, depth, "skip", "blocked_robots"))
                continue

            if self.args.max_pdfs and self.pdfs >= self.args.max_pdfs:
                self.log("max-pdfs reached: %d" % self.args.max_pdfs)
                break
            if self.args.max_pages and self.pages >= self.args.max_pages:
                self.log("max-pages reached: %d" % self.args.max_pages)
                break

            self.debug("fetch depth=%d %s" % (depth, url))

            if self.args.dry_run:
                self.record(Record(url, parent, depth, "unknown", "dry_run"))
                continue

            try:
                resp = self.request(url)
            except Exception as exc:
                self.record(Record(url, parent, depth, "error", "error", error=str(exc)))
                continue

            ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            is_html = "html" in ct or "xhtml" in ct

            if is_html:
                if not allow_follow:
                    resp.close()
                    self.record(Record(url, parent, depth, "html", "fetched", content_type=ct))
                    continue
                try:
                    links = self.extract_links(resp)
                except Exception as exc:
                    resp.close()
                    self.record(
                        Record(url, parent, depth, "html", "error", content_type=ct, error=str(exc))
                    )
                    continue
                resp.close()
                self.pages += 1
                self.record(Record(url, parent, depth, "html", "fetched", content_type=ct))
                child_allow_follow = self.args.max_depth < 0 or depth < self.args.max_depth
                for link in links:
                    q.append((link, url, depth + 1, child_allow_follow))
                self.polite_sleep()
                continue

            # non-HTML: try to treat as a PDF, regardless of URL suffix
            try:
                rec = self.save_pdf(resp, url, parent, depth)
            except Exception as exc:
                try:
                    resp.close()
                except Exception:
                    pass
                self.record(
                    Record(url, parent, depth, "pdf", "error", content_type=ct, error=str(exc))
                )
                continue
            self.record(rec)
            if rec.status == "downloaded":
                self.polite_sleep()

        self.summary()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recursively crawl HTML pages and download linked PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("urls", nargs="+", help="start URL(s)")
    p.add_argument("--out", "-o", default="downloads", help="output directory")
    p.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="additional host to crawl, e.g. --allow-host pmt.example.com; repeatable",
    )
    p.add_argument("--max-depth", type=int, default=3, help="max HTML recursion depth; -1 = unlimited")
    p.add_argument("--max-pages", type=int, default=200, help="max HTML pages to fetch")
    p.add_argument("--max-pdfs", type=int, default=0, help="max PDFs to download; 0 = unlimited")
    p.add_argument("--max-html-mb", type=float, default=10.0, help="max HTML size in MB; 0 = unlimited")
    p.add_argument("--max-size-mb", type=float, default=200.0, help="max PDF size in MB; 0 = unlimited")
    p.add_argument("--delay", type=float, default=1.0, help="base delay between requests in seconds")
    p.add_argument("--timeout", type=float, default=30.0, help="request timeout in seconds")
    p.add_argument("--retries", type=int, default=3, help="retry count")
    p.add_argument("--user-agent", default=DEFAULT_UA)
    p.add_argument("--cookie", default=None, help="Cookie header for sites that need a session")
    p.add_argument("--header", action="append", default=[], help="extra header, e.g. 'X-Token: abc'; repeatable")
    p.add_argument("--layout", choices=("flat", "mirror"), default="flat", help="flat filenames or mirror URL path")
    p.add_argument(
        "--any-domain",
        action="store_true",
        help="allow every domain; by default only start hosts plus --allow-host are followed",
    )
    p.add_argument("--path-prefix", action="append", default=[], help="only crawl URLs under this path prefix; repeatable")
    p.add_argument("--exclude", action="append", default=[], help="regex of URLs to exclude; repeatable")
    p.add_argument("--include", action="append", default=[], help="regex of URLs to include; repeatable")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    p.add_argument("--resume", action="store_true", help="skip URLs already downloaded in manifest.jsonl")
    p.add_argument("--no-robots", action="store_true", help="do not respect robots.txt")
    p.add_argument("--dry-run", action="store_true", help="discover URLs but do not download")
    p.add_argument("--verbose", action="store_true", help="verbose fetch logging")
    p.add_argument(
        "--print-paths",
        action="store_true",
        help="print downloaded PDF paths to stdout, one per line; logs stay on stderr",
    )
    p.add_argument(
        "--print0",
        action="store_true",
        help="like --print-paths but NUL-delimited for xargs -0/parallel",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    crawler = Crawler(args)
    try:
        crawler.run(args.urls)
    except KeyboardInterrupt:
        crawler.log("interrupted")
        crawler.summary()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
