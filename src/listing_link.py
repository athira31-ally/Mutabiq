"""Check a listing straight from its Bayut / Property Finder link.

Two things happen with a pasted link:

1. **Always:** the listing ID is read from the URL itself (no network). It is later compared with the
   listing ID inside the permit QR code, which catches a permit QR copied from another ad.
2. **One polite fetch:** the page is requested once, only if robots.txt allows it, with an honest
   User-Agent. If the portal allows it, the photos (from the page's structured data) and the page text
   are used for the checks. If the portal blocks automated requests - Bayut answers 401 - we stop and
   ask for the page saved as a PDF instead. We never try to get around a block (no disguised browser,
   no CAPTCHA solving, no proxy rotation).

Only Bayut and Property Finder pages are fetched, and images only from those portals' own domains, so the
server can't be pointed at arbitrary URLs.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

USER_AGENT = "TrakheesiComplianceDemo/1.0 (+https://github.com/athira31-ally/Trakheesi)"
TIMEOUT = 10
MAX_PAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES = 12

PORTALS = {  # host suffix -> (name, listing-ID pattern in the URL path)
    "bayut.com": ("Bayut", re.compile(r"details-(\d+)")),
    "propertyfinder.ae": ("Property Finder", re.compile(r"-(\d{6,})\.html")),
}
IMAGE_HOST_SUFFIXES = ("bayut.com", "propertyfinder.ae")
BLOCK_MARKERS = ("captcha", "security check", "are you a robot", "access denied", "challenge-platform")


@dataclass
class ListingLink:
    url: str
    portal: str
    listing_ref: str | None


@dataclass
class FetchResult:
    ok: bool
    reason: str = ""                   # why it failed, in plain words
    http_status: int | None = None
    page_text: str = ""
    image_paths: list[str] = field(default_factory=list)


def parse_listing_url(url: str) -> ListingLink:
    """Validate the link and pull the listing ID out of it. Raises ValueError for anything else."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Paste a full listing link starting with https://")
    for suffix, (name, pattern) in PORTALS.items():
        if host == suffix or host.endswith("." + suffix):
            m = pattern.search(parsed.path)
            return ListingLink(url=url.strip(), portal=name, listing_ref=m.group(1) if m else None)
    raise ValueError("Only Bayut and Property Finder listing links are supported.")


def _http_get(url: str, limit: int) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read(limit + 1)
    except urllib.error.HTTPError as e:
        return e.code, b""


def _robots_allows(url: str) -> bool:
    parsed = urlparse(url)
    try:
        status, body = _http_get(f"{parsed.scheme}://{parsed.netloc}/robots.txt", 512 * 1024)
    except (urllib.error.URLError, TimeoutError, OSError):
        return True  # robots.txt unreachable: the page request itself will tell us if we're unwelcome
    if status != 200:
        return True
    rp = RobotFileParser()
    rp.parse(body.decode("utf-8", "ignore").splitlines())
    return rp.can_fetch(USER_AGENT, url)


def _image_urls(page: str, base: str) -> list[str]:
    """Listing photos from the page's structured data (JSON-LD `image`) and Open Graph tags."""
    urls: list[str] = []

    def add(u):
        if isinstance(u, dict):
            u = u.get("url") or u.get("contentUrl")
        if isinstance(u, str) and u not in urls:
            urls.append(urljoin(base, html.unescape(u)))

    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page, re.S | re.I):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop(0)
            if isinstance(item, dict):
                imgs = item.get("image")
                for u in imgs if isinstance(imgs, list) else [imgs]:
                    add(u)
                stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
            elif isinstance(item, list):
                stack.extend(item)
    for u in re.findall(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', page, re.I):
        add(u)
    return [u for u in urls if (urlparse(u).hostname or "").endswith(IMAGE_HOST_SUFFIXES)][:MAX_IMAGES]


def _page_text(page: str) -> str:
    page = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", page))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def fetch_listing(url: str, out_dir: str | Path) -> FetchResult:
    """One polite attempt at reading the listing page. Never raises for network/portal problems."""
    link = parse_listing_url(url)
    try:
        if not _robots_allows(link.url):
            return FetchResult(False, f"{link.portal}'s robots.txt doesn't allow automated access to this page.")
        status, body = _http_get(link.url, MAX_PAGE_BYTES)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return FetchResult(False, f"Couldn't reach {link.portal} ({type(e).__name__}).")

    page = body.decode("utf-8", "ignore")
    if status in (401, 403, 429, 503) or (status == 200 and any(m in page.lower() for m in BLOCK_MARKERS)):
        return FetchResult(False, f"{link.portal} blocks automated requests (HTTP {status}).", http_status=status)
    if status != 200 or not page:
        return FetchResult(False, f"{link.portal} returned HTTP {status}.", http_status=status)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img_url in enumerate(_image_urls(page, link.url)):
        try:
            img_status, data = _http_get(img_url, MAX_IMAGE_BYTES)
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        if img_status == 200 and 0 < len(data) <= MAX_IMAGE_BYTES:
            path = out / f"photo_{i}.jpg"
            path.write_bytes(data)
            paths.append(str(path))
    return FetchResult(True, http_status=status, page_text=_page_text(page), image_paths=paths)
