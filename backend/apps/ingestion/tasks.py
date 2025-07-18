import uuid
from celery import shared_task
from django.utils import timezone
from .models import Document
from .services.extractor import extract_raw
from .services.splitter import split_text
from .services.embedder import embed_texts
from .services.vector_store import add_vectors

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
        raw = extract_raw(doc.file.path)

        # 3. Split into chunks
        chunks = split_text(raw)

        # 4. Embed texts
        texts = [c['text'] for c in chunks]
        embeddings = embed_texts(texts)

        # 5. Add vectors to ChromaDB
        add_vectors(chunks, embeddings)

        # 6. Finalize success
        total_tokens = sum(c.get('token_count', 0) for c in chunks)
        doc.mark_success(chunk_count=len(chunks), total_tokens=total_tokens)
    except Exception as e:
        # Log error and mark document failed
        doc = Document.objects.filter(id=document_id).first()
        if doc:
            doc.mark_error(str(e))
        raise
