from pathlib import Path
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader


def load_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n\n[PAGE {i + 1}]\n{text}")
    return "\n".join(pages)


def load_docx(path: str | Path) -> str:
    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_html(path: str | Path) -> str:
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n")


def load_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md", ".csv", ".json", ".py", ".js", ".ts"]:
        return load_txt(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in [".html", ".htm"]:
        return load_html(path)
    raise ValueError(f"Unsupported file type: {suffix}")
