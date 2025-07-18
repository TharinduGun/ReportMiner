import os
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
@dataclass
class RawDocument:
    """
    RawDocument holds the extracted pages and tables from an input file.
    pages: list of dicts with keys 'text' and 'metadata' (including page number)
    tables: list of dicts with keys 'sheet_name' and 'dataframe' for structured data
    """
    pages: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]


def extract_raw(file_path: str) -> RawDocument:
    """
    Load and parse the file into raw text pages and DataFrame tables.

    - PDF & DOCX use LangChain's Unstructured loaders for page-level extraction
    - XLSX/CSV use pandas, each sheet or file becomes a table

    Returns:
        RawDocument(pages, tables)
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    if ext == '.pdf':
        loader = UnstructuredPDFLoader(file_path)
        docs = loader.load()
        for idx, doc in enumerate(docs):
            md = doc.metadata.copy()
            md['source'] = file_path
            md['page'] = idx + 1
            pages.append({'text': doc.page_content, 'metadata': md})

    elif ext == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
        for idx, doc in enumerate(docs):
            md = doc.metadata.copy()
            md['source'] = file_path
            md['page'] = idx + 1
            pages.append({'text': doc.page_content, 'metadata': md})

    elif ext in {'.xlsx', '.xls'}:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            tables.append({'sheet_name': sheet_name, 'dataframe': df})

    elif ext == '.csv':
        df = pd.read_csv(file_path)
        tables.append({'sheet_name': os.path.basename(file_path), 'dataframe': df})

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return RawDocument(pages=pages, tables=tables)
