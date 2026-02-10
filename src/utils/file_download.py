"""Helpers for detecting and downloading file URLs (PDF, CSV, etc.)."""

import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# MIME types that indicate a downloadable file (not a web page)
FILE_CONTENT_TYPES = {
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/octet-stream",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-excel",  # .xls
    "application/msword",  # .doc
    "text/csv",
}

# URL path extensions that indicate a file
FILE_EXTENSIONS = {
    ".pdf", ".zip", ".gz", ".tar", ".7z", ".rar",
    ".csv", ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt",
    ".json", ".xml", ".txt",
}

# Download directory at project root
DOWNLOADS_DIR = Path(__file__).resolve().parent.parent.parent / "downloads"


def url_has_file_extension(url: str) -> bool:
    """Check if the URL path ends with a known file extension."""
    try:
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in FILE_EXTENSIONS)
    except Exception:
        return False


def is_file_content_type(content_type: str) -> bool:
    """Check if a Content-Type header value matches a known file MIME type."""
    # Content-Type may include charset, e.g. "text/csv; charset=utf-8"
    mime = content_type.split(";")[0].strip().lower()
    return mime in FILE_CONTENT_TYPES


def extract_filename(url: str, content_disposition: str = "") -> str:
    """Extract a safe filename from Content-Disposition header or URL path."""
    # Try Content-Disposition first
    if content_disposition:
        match = re.search(r'filename[*]?=["\']?([^"\';]+)', content_disposition)
        if match:
            name = unquote(match.group(1)).strip()
            if name:
                return _sanitize_filename(name)

    # Fall back to URL path
    path = urlparse(url).path
    basename = os.path.basename(unquote(path))
    if basename:
        return _sanitize_filename(basename)

    return "download"


def _sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe for filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ")


def check_url_is_file(url: str) -> tuple:
    """HEAD-request the URL to check Content-Type. Returns (is_file, content_type).

    Falls back to extension check if the HEAD request fails.
    """
    # Fast path: check extension first
    has_ext = url_has_file_extension(url)

    try:
        req = Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urlopen(req, timeout=5) as resp:
            ct = resp.headers.get("Content-Type", "")
            if is_file_content_type(ct):
                return True, ct.split(";")[0].strip()
            # If HEAD says text/html, trust that even if the extension looks like a file
            if "text/html" in ct.lower():
                return False, ct.split(";")[0].strip()
    except (URLError, OSError, ValueError) as e:
        logger.debug(f"HEAD request failed for {url}: {e}")

    # HEAD failed or returned ambiguous type — fall back to extension
    if has_ext:
        return True, "unknown (extension match)"
    return False, ""


def download_via_context(context_request, url: str) -> str | None:
    """Download a file using Playwright's context.request (shares browser cookies).

    Returns the saved filepath on success, None on failure.
    """
    try:
        resp = context_request.get(url)
        if resp.status >= 400:
            logger.warning(f"Download failed: HTTP {resp.status} for {url}")
            return None

        content_disposition = resp.headers.get("content-disposition", "")
        content_type = resp.headers.get("content-type", "")
        filename = extract_filename(url, content_disposition)

        # Ensure downloads directory exists
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # Deduplicate filename if it already exists
        filepath = DOWNLOADS_DIR / filename
        if filepath.exists():
            stem = filepath.stem
            suffix = filepath.suffix
            counter = 1
            while filepath.exists():
                filepath = DOWNLOADS_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        filepath.write_bytes(resp.body())
        logger.info(f"Downloaded {len(resp.body())} bytes -> {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return None


def page_looks_empty(dom_text: str) -> bool:
    """Conservative check: True if DOM has no interactive elements AND page text is very short.

    Used to detect file/binary pages that render as blank in the browser.
    """
    if not dom_text:
        return True
    has_no_elements = dom_text.startswith("(no interactive elements found)")
    # Check if PAGE TEXT section exists and is very short
    if "PAGE TEXT:" in dom_text:
        page_text = dom_text.split("PAGE TEXT:", 1)[1].strip()
        return has_no_elements and len(page_text) < 20
    # No PAGE TEXT section at all
    return has_no_elements
