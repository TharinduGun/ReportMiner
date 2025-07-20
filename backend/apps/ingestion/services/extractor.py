import os
import pdfplumber
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from django.conf import settings
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.document_loaders import PyPDFLoader

@dataclass
class RawDocument:
    """
    RawDocument holds the extracted pages and tables from an input file.

    pages: list of dicts with 'text' and 'metadata'
    tables: list of dicts with 'sheet_name' and 'dataframe'
    """
    pages: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]


def extract_raw(file_path: str) -> RawDocument:
    """
    Load and parse the file into raw text pages and DataFrame tables.

    Supports PDF table extraction via pdfplumber, CSV fallback encoding,
    chunked CSV reading for large files, with Unicode errors replaced to avoid crashes.
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    if ext == '.pdf':
        # 1) Extract narrative text pages with PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
        for idx, doc in enumerate(docs):
            md = doc.metadata.copy()
            md.update({'source': file_path, 'page': idx + 1, 'chunk_type': 'text'})
            pages.append({'text': doc.page_content, 'metadata': md})

        # 2) Extract tables from PDF using pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                extracted = page.extract_tables()
                for table_idx, table in enumerate(extracted, start=1):
                    # Convert to DataFrame (first row as header)
                    if not table or len(table) < 2:
                        continue  # skip empty or header-only tables
                    df = pd.DataFrame(table[1:], columns=table[0])
                    sheet_name = f"page{page_num}_table{table_idx}"
                    # Add metadata for precise retrieval
                    tables.append({
                        'sheet_name': sheet_name,
                        'dataframe': df,
                        'metadata': {
                            'source': file_path,
                            'page': page_num,
                            'table_index': table_idx,
                            'chunk_type': 'table',
                            'columns': df.columns.tolist()
                        }
                    })

    elif ext == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
        for idx, doc in enumerate(docs):
            md = doc.metadata.copy()
            md.update({'source': file_path, 'page': idx + 1, 'chunk_type': 'text'})
            pages.append({'text': doc.page_content, 'metadata': md})

    elif ext in {'.xlsx', '.xls'}:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            # Full-sheet ingest: one chunk per sheet
            tables.append({
                'sheet_name': sheet_name,
                'dataframe': df,
                'metadata': {
                    'source': file_path,
                    'chunk_type': 'csv_sheet',
                    'sheet_name': sheet_name,
                    'row_count': len(df),
                    'columns': df.columns.tolist()
                }
            })

    elif ext == '.csv':
        # Attempt to read CSV in chunks, catching Unicode errors
        chunksize = getattr(settings, 'CSV_CHUNKSIZE', 50000)
        try:
            reader = pd.read_csv(file_path, iterator=True, chunksize=chunksize)
            for i, chunk_df in enumerate(reader):
                part_name = f"{os.path.basename(file_path)}_part{i+1}"
                tables.append({
                    'sheet_name': part_name,
                    'dataframe': chunk_df,
                    'metadata': {
                        'source': file_path,
                        'chunk_type': 'csv_sheet',
                        'sheet_name': part_name,
                        'row_count': len(chunk_df),
                        'columns': chunk_df.columns.tolist()
                    }
                })
        except UnicodeDecodeError:
            # Fallback: replace invalid bytes then re-read
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = pd.read_csv(f, iterator=True, chunksize=chunksize)
                for i, chunk_df in enumerate(reader):
                    part_name = f"{os.path.basename(file_path)}_part{i+1}"
                    tables.append({
                        'sheet_name': part_name,
                        'dataframe': chunk_df,
                        'metadata': {
                            'source': file_path,
                            'chunk_type': 'csv_sheet',
                            'sheet_name': part_name,
                            'row_count': len(chunk_df),
                            'columns': chunk_df.columns.tolist()
                        }
                    })
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return RawDocument(pages=pages, tables=tables)
