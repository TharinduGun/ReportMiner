import uuid
from typing import List, Dict, Any, Tuple
from django.conf import settings
from .extractor import RawDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import re

# Configurable thresholds for grouping large tables
ROW_EMBED_THRESHOLD = getattr(settings, "INGESTION_ROW_EMBED_THRESHOLD", 200)
ROW_GROUP_SIZE = getattr(settings, "INGESTION_ROW_GROUP_SIZE", 50)

# Use cl100k_base encoding (used by OpenAI embedding models) for precise token counting
encoding = tiktoken.get_encoding("cl100k_base")

HEADING_REGEX = re.compile(r'^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9\s\-]{5,}$')

def extract_headings(text: str) -> List[Tuple[int, str]]:
    """
    Heuristic: scan each line for either:
     - Numeric outline headings (e.g. "2.4.2 Movement Recognition")
     - All-caps titles longer than 10 chars
    Returns list of (char_offset, heading_text).
    """
    headings: List[Tuple[int,str]] = []
    char_offset = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip().rstrip('.')
        # 1) Numeric outline (e.g. "1. Introduction", "2.5 Control Strategies")
        if re.match(r'^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9\s\-]{3,}$', stripped):
            headings.append((char_offset, stripped))
        # 2) All-caps long line (e.g. a section title)
        elif len(stripped) > 10 and stripped.upper() == stripped:
            headings.append((char_offset, stripped))
        char_offset += len(line)

    return headings

def filter_heading_list(headings):
    """
    Remove boilerplate and bogus headings:
     - “GROUP NO.: …”
     - pure student-ID lines like “200171C 200736N”
     - super-short headings (<2 words)
    """
    clean = []
    for pos, title in headings:
        # 1) Skip the “GROUP NO.” boilerplate
        if title.startswith("GROUP NO"):
            continue

        # 2) Skip pure student codes (e.g. “200171C 200736N”)
        toks = title.split()
        if all(re.fullmatch(r"\d+[A-Z]", t) for t in toks):
            continue

        # 3) Skip too-short
        if len(toks) < 2:
            continue

        clean.append((pos, title))
    return clean


def normalize_headings(headings: List[Tuple[int,str]]) -> List[Tuple[int,str]]:
    normalized = []
    skip = False
    for i, (pos, title) in enumerate(headings):
        if skip:
            skip = False
            continue
        # if the very next heading is within, say, 50 chars, stitch them
        if i+1 < len(headings) and headings[i+1][0] - pos < 50:
            _, next_title = headings[i+1]
            combined = f"{title} {next_title}"
            normalized.append((pos, combined))
            skip = True
        else:
            normalized.append((pos, title))
    return normalized


def count_tokens(text: str) -> int:
    """Return the number of tokens in `text` using cl100k_base encoding."""
    return len(encoding.encode(text))


def split_text(raw: RawDocument) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    # ── Token-based splitting via character splitter ──────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # target ~500 tokens
        chunk_overlap=100,       # 100 token overlap
        length_function=count_tokens  # measure length by tokens
    )

    for page in raw.pages:
        full_text = page["text"]
        source = page['metadata'].get('source')
        page_num = page['metadata'].get('page')
        # 1) Find all headings in this page
        raw_headings = extract_headings(full_text)
        headings = normalize_headings(filter_heading_list(raw_headings))

        # 2) Split into chunks
        sub_texts = splitter.split_text(full_text)

        # 3) As you enumerate sub_texts, compute the approximate
        #    character offset of each chunk to pick the right heading.
        char_offset = 0
        for idx, sub in enumerate(sub_texts):
            # Find the last heading whose index <= char_offset
            section_title = None
            for pos, title in headings:
                if pos <= char_offset:
                    section_title = title
                else:
                    break

            metadata = {
                "source": source,
                "page": page_num,
                "chunk_index": idx,
                "section_title": section_title,  # newly added
            }
            token_count = count_tokens(sub)
            chunks.append({
                "text": sub,
                "metadata": metadata,
                "token_count": token_count
            })

            # Advance the offset for next iteration
            char_offset += len(sub)

    # ── Structured tables (no change) ────────────────────────────
    for table in raw.tables:
        sheet = table['sheet_name']
        df = table['dataframe']
        # header chunk
        columns = df.columns.tolist()
        header_text = "Columns: " + ", ".join(columns)
        header_metadata = {
            'source': sheet,
            'type': 'header',
            'columns': ", ".join(columns)
        }
        header_token_count = count_tokens(header_text)
        chunks.append({
            'text': header_text,
            'metadata': header_metadata,
            'token_count': header_token_count
        })
        # row chunks
        for row_idx, row in df.iterrows():
            row_text = "; ".join(f"{col}: {row[col]}" for col in columns)
            metadata = {
                'source': sheet,
                'type': 'row',
                'row_index': int(row_idx),
                'columns': ", ".join(columns)
            }
            token_count = count_tokens(row_text)
            chunks.append({
                'text': row_text,
                'metadata': metadata,
                'token_count': token_count
            })

    return chunks










