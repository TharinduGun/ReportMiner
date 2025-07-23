import uuid
from typing import List, Dict, Any, Tuple
from django.conf import settings
from .extractor import RawDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import re
import pandas as pd


# # Configurable thresholds for grouping large tables
# ROW_EMBED_THRESHOLD = getattr(settings, "INGESTION_ROW_EMBED_THRESHOLD", 200)
# ROW_GROUP_SIZE = getattr(settings, "INGESTION_ROW_GROUP_SIZE", 50)                  REMOVED AS NO ROW GROUPING NOW


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
        chunk_size=400,          # target ~500 tokens
        chunk_overlap=100,       # 100 token overlap
        length_function=count_tokens  # measure length by tokens
    )


    for page in raw.pages:
        full_text = page["text"]
        source = page['metadata'].get('source')
        page_num = page['metadata'].get('page')

        # 1) Find/normalize headings to slice page into sections
        raw_headings = extract_headings(full_text)
        headings     = normalize_headings(filter_heading_list(raw_headings))

        # 2) Build (title, text) segments
        segments = []
        if headings:
            for i, (pos, title) in enumerate(headings):
                start = pos
                end   = headings[i+1][0] if i+1 < len(headings) else len(full_text)
                segments.append((title.strip(), full_text[start:end].strip()))
        else:
            # no headings → one big section
            segments = [("", full_text)]

        # 3) Within each section, sub-split if large
        for sec_idx, (section_title, section_text) in enumerate(segments):
            sub_texts = splitter.split_text(section_text)
            for idx, sub in enumerate(sub_texts):
                metadata = {
                    "source":       source,
                    "page":         page_num,
                    "section":      section_title or "Introduction",
                    "section_idx":  sec_idx,
                    "chunk_idx":    idx,
                }
                token_count = count_tokens(sub)
                chunks.append({
                    "text":        sub,
                    "metadata":    metadata,
                    "token_count": token_count
                })

    # ── Structured tables with smart chunking ─────────────
    for table in raw.tables:
        sheet = table['sheet_name']
        df    = table['dataframe']

        # Create base description for all chunks
        base_description = f"This is a {sheet} dataset with the following columns: {', '.join(df.columns.tolist())}\n\n"
        
        # Check if table is too large for single chunk
        test_json = df.to_json(orient="records")
        estimated_tokens = count_tokens(base_description + test_json)
        
        # If table is small enough, keep as single chunk
        if estimated_tokens <= 15000:  # Safe limit for single chunk
            # Add sample data for better semantic understanding
            if len(df) > 0:
                base_description += "Sample records:\n"
                sample_size = min(3, len(df))
                for i in range(sample_size):
                    row = df.iloc[i]
                    row_desc = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    base_description += f"- {row_desc}\n"
                base_description += "\n"
            
            full_content = base_description + f"Complete dataset:\n{test_json}"
            
            metadata = {
                "source":     sheet,
                "type":       "table",
                "row_count":  len(df),
                "columns":    ", ".join(df.columns.tolist()),
                "chunk_part": "complete"
            }
            token_count = count_tokens(full_content)
            chunks.append({
                "text":        full_content,
                "metadata":    metadata,
                "token_count": token_count
            })
        else:
            # Split large table into smaller chunks
            rows_per_chunk = 50  # Adjust based on data complexity
            total_chunks = (len(df) + rows_per_chunk - 1) // rows_per_chunk
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * rows_per_chunk
                end_idx = min(start_idx + rows_per_chunk, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                # Create chunk-specific description
                chunk_description = base_description
                chunk_description += f"This is part {chunk_idx + 1} of {total_chunks}, containing rows {start_idx + 1}-{end_idx} ({len(chunk_df)} records):\n\n"
                
                # Add chunk data
                chunk_json = chunk_df.to_json(orient="records")
                full_content = chunk_description + chunk_json
                
                metadata = {
                    "source":     sheet,
                    "type":       "table",
                    "row_count":  len(chunk_df),
                    "columns":    ", ".join(df.columns.tolist()),
                    "chunk_part": f"{chunk_idx + 1}/{total_chunks}",
                    "start_row":  start_idx + 1,
                    "end_row":    end_idx
                }
                token_count = count_tokens(full_content)
                chunks.append({
                    "text":        full_content,
                    "metadata":    metadata,
                    "token_count": token_count
                })

    return chunks


