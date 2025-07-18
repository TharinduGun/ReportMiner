import uuid
from typing import List, Dict, Any
from django.conf import settings
from .extractor import RawDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Configurable thresholds for grouping large tables
ROW_EMBED_THRESHOLD = getattr(settings, "INGESTION_ROW_EMBED_THRESHOLD", 200)
ROW_GROUP_SIZE = getattr(settings, "INGESTION_ROW_GROUP_SIZE", 50)

# Use cl100k_base encoding (used by OpenAI embedding models) for precise token counting
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Return the number of tokens in `text` using cl100k_base encoding."""
    return len(encoding.encode(text))


def split_text(raw: RawDocument) -> List[Dict[str, Any]]:
    """
    Split a RawDocument into chunk-level dicts ready for embedding, 
    grouping table rows when too many for individual embeddings.
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
        sub_texts = splitter.split_text(page['text'])
        for idx, sub in enumerate(sub_texts):
            metadata = {
                'source': source,
                'page': page_num,
                'chunk_index': idx
            }
            token_count = count_tokens(sub)
            chunks.append({'text': sub, 'metadata': metadata, 'token_count': token_count})

    # 2. Structured tables: add header chunk then row or grouped row chunks
    for table in raw.tables:
        sheet = table.get('sheet_name')
        df = table.get('dataframe')

        # Header chunk to preserve column context
        columns = df.columns.tolist()
        header_text = "Columns: " + ", ".join(columns)
        header_metadata = {
            'source': sheet,
            'type': 'header',
            'columns': ", ".join(columns)
        }
        header_token_count = count_tokens(header_text)
        chunks.append({'text': header_text, 'metadata': header_metadata, 'token_count': header_token_count})

        n_rows = len(df)
        # If too many rows, group them into batches to reduce embedding calls
        if n_rows > ROW_EMBED_THRESHOLD:
            for start in range(0, n_rows, ROW_GROUP_SIZE):
                group = df.iloc[start:start + ROW_GROUP_SIZE]
                # Serialize multiple rows into one chunk
                texts = []
                for row_idx, row in group.iterrows():
                    row_text = "; ".join(f"{col}: {row[col]}" for col in columns)
                    texts.append(row_text)
                combined_text = "\n".join(texts)

                metadata = {
                    'source': sheet,
                    'type': 'grouped_rows',
                    'row_range': f"{start}-{start + len(group) - 1}",
                    'columns': ", ".join(columns)
                }
                token_count = count_tokens(combined_text)
                chunks.append({'text': combined_text, 'metadata': metadata, 'token_count': token_count})
        else:
            # Few rows: embed individually
            for row_idx, row in df.iterrows():
                text = "; ".join(f"{col}: {row[col]}" for col in columns)
                metadata = {
                    'source': sheet,
                    'type': 'row',
                    'row_index': int(row_idx),
                    'columns': ", ".join(columns)
                }
                token_count = count_tokens(text)
                chunks.append({'text': text, 'metadata': metadata, 'token_count': token_count})

    return chunks
