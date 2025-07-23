# import uuid
# from celery import shared_task
# from django.utils import timezone
# from .models import Document
# from .services.extractor import extract_raw
# from .services.splitter import split_text
# from .services.embedder import embed_texts
# from .services.vector_store import add_vectors

# from celery.utils.log import get_task_logger

# logger = get_task_logger(__name__)

# @shared_task(bind=True)
# def process_document(self, document_id):
#     """
#     Orchestrates the ingestion pipeline:
#       1. Load raw content (text or tables)
#       2. Split into chunks
#       3. Embed chunks
#       4. Store vectors in ChromaDB
#       5. Update Document status and metrics
#     """
#     try:
#         # 1. Retrieve and mark processing
#         doc = Document.objects.get(id=document_id)
#         doc.mark_processing()

#         # 2. Extract raw content
#         raw_doc = extract_raw(doc.file.path)

#         # 3. Build ingestion‐ready chunks:
#         chunks = []

#         # 3a) pages from PDF/DOCX → one chunk per page
#         for page in raw_doc.pages:
#             # page already has {'text': ..., 'metadata': {...}}
#             chunks.append(page)

#         # 3b) tables from XLSX/CSV → one chunk per row
#         for table in raw_doc.tables:
#             df = table["dataframe"]
#             sheet = table.get("sheet_name", "")
#             for _, row in df.iterrows():
#                 row_meta = row.to_dict()  # {col: primitive}
#                 # you can include sheet name too:
#                 row_meta["sheet_name"] = sheet
#                 # text could be JSON or just CSV‐stringified
#                 row_text = row.to_json()
#                 chunks.append({"text": row_text, "metadata": row_meta})

#         # 4. Embed texts
#         texts = [c['text'] for c in chunks]
#         embeddings = embed_texts(texts)

#     # ——>>> add this:
#         logger.info(f"Generated {len(embeddings)} embeddings for Document {document_id}")


#         # 5. Add vectors to ChromaDB
#         add_vectors(chunks, embeddings)

#         # 6. Finalize success
#         total_tokens = sum(c.get('token_count', 0) for c in chunks)
#         doc.mark_success(chunk_count=len(chunks), total_tokens=total_tokens)
#     except Exception as e:
#         # Log error and mark document failed
#         doc = Document.objects.filter(id=document_id).first()
#         if doc:
#             doc.mark_error(str(e))
#         raise
import uuid
from celery import shared_task
from django.utils import timezone
from .models import Document
from .services.extractor import extract_raw
from .services.splitter import split_text
from .services.vector_store import add_vectors

from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@shared_task(bind=True)
def process_document(self, document_id):
    """
    Orchestrates the ingestion pipeline:
      1. Load raw content (text or tables)
      2. Split into chunks
      3. Embed chunks
      4. Store vectors in ChromaDB
      5. Update Document status and metrics
    """
    try:
        # 1. Retrieve and mark processing
        doc = Document.objects.get(id=document_id)
        doc.mark_processing()

        # 2. Extract raw content
        raw_doc = extract_raw(doc.file.path)

        # 3. Use unified chunking strategy for both text and tables
        chunks = split_text(raw_doc)

        logger.info(f"Processing {len(chunks)} chunks for Document {document_id}")

        # 4. Add vectors to ChromaDB (LangChain Chroma handles embeddings automatically)
        add_vectors(chunks, [])

        # 5. Finalize success
        total_tokens = sum(c.get('token_count', 0) for c in chunks)
        doc.mark_success(chunk_count=len(chunks), total_tokens=total_tokens)
    except Exception as e:
        # Log error and mark document failed
        doc = Document.objects.filter(id=document_id).first()
        if doc:
            doc.mark_error(str(e))
        raise