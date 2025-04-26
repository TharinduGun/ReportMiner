import os

# return file extension (e.g., '.pdf', '.docx')
def get_file_extension(file_path):
    return os.path.splitext(file_path)[1].lower()

# Check if extracted text is empty or only whitespace
def is_empty_text(text):
    return not text.strip()

# Return a preview (first 1000 characters) of extracted text
def preview_text(text, max_len=1000):
    return text[:max_len] + "..." if len(text) > max_len else text
