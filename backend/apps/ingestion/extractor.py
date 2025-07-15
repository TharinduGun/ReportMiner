import PyPDF2
import pdfplumber
import docx
import pandas as pd
from pdf2image import convert_from_path    #fixed
import pytesseract

from .utils import is_empty_text

# Main dispatcher: given a file path + type, extract text
def extract_text_from_file(file_path, file_type):
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
        if is_empty_text(text):
            text = extract_text_from_pdf_advanced(file_path)
        if is_empty_text(text):
            text = extract_text_from_scanned_pdf(file_path)
        return text

    elif file_type == 'docx':
        return extract_text_from_docx(file_path)

    elif file_type == 'xlsx':
        return extract_text_from_excel(file_path)
    
    elif file_type == 'csv':
        return extract_text_from_csv(file_path)

    else:
        return "Unsupported file type."

# Basic PDF text layer extraction using PyPDF2
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        text = f"[PyPDF2 Error] {e}"
    return text

# Advanced layout-aware PDF text extraction using pdfplumber
def extract_text_from_pdf_advanced(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        text = f"[pdfplumber Error] {e}"
    return text

# OCR fallback for scanned PDFs using pdf2image + pytesseract
def extract_text_from_scanned_pdf(file_path):
    text = ""
    try:
        images = convert_from_path(file_path)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        text = f"[OCR Error] {e}"
    return text

# Extract text from Word documents (.docx)
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        text = f"[DOCX Error] {e}"
    return text

# Extract text from Excel sheets (.xlsx)
def extract_text_from_excel(file_path):
    text = ""
    try:
        # Read all sheets as dataframes
        sheets = pd.read_excel(file_path, sheet_name=None)
        for sheet, df in sheets.items():
            text += f"Sheet: {sheet}\n"
            text += df.to_string(index=False)
            text += "\n\n"
    except Exception as e:
        text = f"[Excel Error] {e}"
    return text

# Extract text from CSV files (.csv)
def extract_text_from_csv(file_path):
    text = ""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Add header information
        text += f"CSV File: {file_path}\n"
        text += f"Columns: {', '.join(df.columns)}\n"
        text += f"Rows: {len(df)}\n\n"
        
        # Convert dataframe to string
        text += df.to_string(index=False)
        
    except Exception as e:
        text = f"[CSV Error] {e}"
    return text
