import hashlib
import re
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    metadata: dict


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fixed_size_chunking(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Chunk]:
    text = clean_text(text)
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(chunk_text, {"chunk_index": idx, "chunk_strategy": "fixed", "char_start": start, "char_end": end}))
            idx += 1
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)

    return chunks


def recursive_chunking(text: str, chunk_size: int = 900, chunk_overlap: int = 150, separators: list[str] | None = None) -> list[Chunk]:
    text = clean_text(text)
    separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def split_recursively(segment: str, sep_index: int = 0) -> list[str]:
        if len(segment) <= chunk_size:
            return [segment]
        if sep_index >= len(separators):
            step = max(1, chunk_size - chunk_overlap)
            return [segment[i:i + chunk_size] for i in range(0, len(segment), step)]
        sep = separators[sep_index]
        if sep == "":
            step = max(1, chunk_size - chunk_overlap)
            return [segment[i:i + chunk_size] for i in range(0, len(segment), step)]
        parts = segment.split(sep)
        if len(parts) == 1:
            return split_recursively(segment, sep_index + 1)
        output: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + sep + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    output.extend(split_recursively(current, sep_index + 1))
                current = part
        if current:
            output.extend(split_recursively(current, sep_index + 1))
        return output

    raw_chunks = split_recursively(text)
    chunks: list[Chunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        if chunk_overlap > 0 and idx > 0:
            previous_tail = raw_chunks[idx - 1][-chunk_overlap:]
            chunk_text = (previous_tail + " " + chunk_text).strip()
        chunks.append(Chunk(chunk_text, {"chunk_index": idx, "chunk_strategy": "recursive"}))
    return chunks


def sentence_chunking(text: str, chunk_size: int = 900, chunk_overlap_sentences: int = 1) -> list[Chunk]:
    text = clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[Chunk] = []
    current: list[str] = []
    idx = 0
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if len(candidate) <= chunk_size:
            current.append(sentence)
        else:
            if current:
                chunks.append(Chunk(" ".join(current).strip(), {"chunk_index": idx, "chunk_strategy": "sentence"}))
                idx += 1
            overlap = current[-chunk_overlap_sentences:] if chunk_overlap_sentences else []
            current = overlap + [sentence]
    if current:
        chunks.append(Chunk(" ".join(current).strip(), {"chunk_index": idx, "chunk_strategy": "sentence"}))
    return chunks


def markdown_header_chunking(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Chunk]:
    text = clean_text(text)
    lines = text.splitlines()
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            if current_lines:
                sections.append((heading_stack.copy(), "\n".join(current_lines).strip()))
                current_lines = []
            level = len(match.group(1))
            heading = match.group(2).strip()
            heading_stack = heading_stack[:level - 1]
            heading_stack.append(heading)
        current_lines.append(line)

    if current_lines:
        sections.append((heading_stack.copy(), "\n".join(current_lines).strip()))

    chunks: list[Chunk] = []
    idx = 0
    for headings, section_text in sections:
        for sub in recursive_chunking(section_text, chunk_size, chunk_overlap):
            metadata = dict(sub.metadata)
            metadata.update({"chunk_index": idx, "chunk_strategy": "markdown", "heading_path": " > ".join(headings)})
            chunks.append(Chunk(sub.text, metadata))
            idx += 1
    return chunks


def code_chunking(text: str, chunk_size: int = 1400, chunk_overlap: int = 100) -> list[Chunk]:
    text = clean_text(text)
    blocks = re.split(r"(?=^\s*(class|def|async def|function|const|let|var)\s+)", text, flags=re.MULTILINE)
    rebuilt: list[str] = []
    if blocks:
        if blocks[0].strip():
            rebuilt.append(blocks[0])
        i = 1
        while i < len(blocks):
            keyword = blocks[i]
            body = blocks[i + 1] if i + 1 < len(blocks) else ""
            rebuilt.append(keyword + " " + body)
            i += 2
    if not rebuilt:
        rebuilt = [text]

    chunks: list[Chunk] = []
    idx = 0
    for block in rebuilt:
        if len(block) <= chunk_size:
            chunks.append(Chunk(block.strip(), {"chunk_index": idx, "chunk_strategy": "code"}))
            idx += 1
        else:
            for sub in recursive_chunking(block, chunk_size, chunk_overlap):
                metadata = dict(sub.metadata)
                metadata.update({"chunk_index": idx, "chunk_strategy": "code"})
                chunks.append(Chunk(sub.text, metadata))
                idx += 1
    return chunks


def table_chunking(text: str, max_rows_per_chunk: int = 25) -> list[Chunk]:
    text = clean_text(text)
    lines = [line for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line or "," in line]
    if not table_lines:
        return recursive_chunking(text)
    header = table_lines[0]
    rows = table_lines[1:]
    chunks: list[Chunk] = []
    idx = 0
    for i in range(0, len(rows), max_rows_per_chunk):
        group = rows[i:i + max_rows_per_chunk]
        chunk_text = "\n".join([header] + group)
        chunks.append(Chunk(chunk_text, {"chunk_index": idx, "chunk_strategy": "table", "row_start": i, "row_end": i + len(group)}))
        idx += 1
    return chunks


def semantic_chunking(text: str, embed_fn, similarity_threshold: float = 0.72, max_chunk_size: int = 1200) -> list[Chunk]:
    text = clean_text(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return [Chunk(text, {"chunk_index": 0, "chunk_strategy": "semantic"})]
    embeddings = np.array(embed_fn(sentences))
    chunks: list[Chunk] = []
    current: list[str] = [sentences[0]]
    idx = 0
    for i in range(1, len(sentences)):
        sim = float(cosine_similarity(embeddings[i - 1].reshape(1, -1), embeddings[i].reshape(1, -1))[0][0])
        candidate = " ".join(current + [sentences[i]])
        if sim < similarity_threshold or len(candidate) > max_chunk_size:
            chunks.append(Chunk(" ".join(current).strip(), {"chunk_index": idx, "chunk_strategy": "semantic"}))
            idx += 1
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(Chunk(" ".join(current).strip(), {"chunk_index": idx, "chunk_strategy": "semantic"}))
    return chunks


def parent_child_chunking(text: str, parent_chunk_size: int = 2200, child_chunk_size: int = 500, child_overlap: int = 80) -> list[Chunk]:
    parents = recursive_chunking(text, chunk_size=parent_chunk_size, chunk_overlap=200)
    child_chunks: list[Chunk] = []
    child_idx = 0
    for parent_idx, parent in enumerate(parents):
        parent_id = stable_hash(parent.text)
        children = recursive_chunking(parent.text, chunk_size=child_chunk_size, chunk_overlap=child_overlap)
        for child in children:
            metadata = dict(child.metadata)
            metadata.update({
                "chunk_index": child_idx,
                "chunk_strategy": "parent_child",
                "parent_id": parent_id,
                "parent_index": parent_idx,
                "parent_text": parent.text,
            })
            child_chunks.append(Chunk(child.text, metadata))
            child_idx += 1
    return child_chunks


def chunk_document(text: str, strategy: str = "recursive", chunk_size: int = 900, chunk_overlap: int = 150, embed_fn=None) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_size_chunking(text, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return recursive_chunking(text, chunk_size, chunk_overlap)
    if strategy == "sentence":
        return sentence_chunking(text, chunk_size)
    if strategy == "markdown":
        return markdown_header_chunking(text, chunk_size, chunk_overlap)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires embed_fn")
        return semantic_chunking(text, embed_fn=embed_fn)
    if strategy == "parent_child":
        return parent_child_chunking(text)
    if strategy == "code":
        return code_chunking(text, chunk_size, chunk_overlap)
    if strategy == "table":
        return table_chunking(text)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
