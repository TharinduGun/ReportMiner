import uuid
from typing import List, Dict, Any
from .extractor import RawDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Use cl100k_base encoding (used by OpenAI embedding models) for precise token counting
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Return the number of tokens in `text` using cl100k_base encoding."""
    return len(encoding.encode(text))


def split_text(raw: RawDocument) -> List[Dict[str, Any]]:
    """
    Split a RawDocument into chunk-level dicts ready for embedding.

    - Unstructured pages are split into overlapping text chunks
    - Structured tables (DataFrames) are serialized row-by-row,
      with a header chunk to preserve column context

    Returns:
        List of dicts with keys:
          - 'text': chunk text
          - 'metadata': dict including source, page or sheet and indices
          - 'token_count': precise token count
    """
    chunks: List[Dict[str, Any]] = []

    # 1. Unstructured pages: split into text chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    for page in raw.pages:
        source = page['metadata'].get('source')
        page_num = page['metadata'].get('page')
        # split text of this page
        sub_texts = splitter.split_text(page['text'])
        for idx, sub in enumerate(sub_texts):
            metadata = {
                'source': source,
                'page': page_num,
                'chunk_index': idx
            }
            token_count = count_tokens(sub)
            chunks.append({
                'text': sub,
                'metadata': metadata,
                'token_count': token_count
            })

    # 2. Structured tables: add header chunk then row chunks
    for table in raw.tables:
        sheet = table.get('sheet_name')
        df = table.get('dataframe')
        # Header chunk to preserve column context
        columns = df.columns.tolist()
        header_text = "Columns: " + ", ".join(columns)
        header_metadata = {
            'source': sheet,
            'type': 'header',
            'columns': columns
        }
        header_token_count = count_tokens(header_text)
        chunks.append({
            'text': header_text,
            'metadata': header_metadata,
            'token_count': header_token_count
        })
        # Row chunks
        for row_idx, row in df.iterrows():
            # Serialize each row into text with column names
            text = "; ".join(f"{col}: {row[col]}" for col in columns)
            metadata = {
                'source': sheet,
                'type': 'row',
                'row_index': int(row_idx),
                'columns': columns
            }
            token_count = count_tokens(text)
            chunks.append({
                'text': text,
                'metadata': metadata,
                'token_count': token_count
            })

    return chunks
