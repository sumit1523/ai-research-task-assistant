"""Load user-selected sources into plain text for the research workflow."""

from io import BytesIO
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf.errors import DependencyError, FileNotDecryptedError
from pypdf import PdfReader


def load_url(url: str) -> tuple[str, str]:
    """Fetch readable webpage text. Only public HTTP(S) URLs are accepted."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Use a complete public URL beginning with https:// or http://.")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Local URLs are not allowed as research sources.")
    response = requests.get(url, timeout=12, headers={"User-Agent": "ResearchTaskAssistant/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    if len(text) < 100:
        raise ValueError("This page did not contain enough readable text.")
    return url, text[:20_000]


def load_upload(upload) -> tuple[str, str]:
    """Read a user-uploaded .txt, .md, or PDF file without saving it to disk."""
    name = upload.name
    raw = upload.getvalue()
    if name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError(f"{name} is password-protected. Upload an unlocked copy to use it as a source.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except DependencyError as error:
            raise ValueError(
                f"{name} uses PDF encryption that needs the cryptography package. Please restart the app after updating dependencies."
            ) from error
        except FileNotDecryptedError as error:
            raise ValueError(f"{name} is password-protected. Upload an unlocked copy to use it as a source.") from error
    else:
        text = raw.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        raise ValueError(f"{name} did not contain readable text.")
    return name, text[:20_000]
