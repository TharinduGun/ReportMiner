-- =================================================================
-- REPORTMINER POSTGRESQL SCHEMA FOR WINDOWS
-- =================================================================

-- Enable extensions (may need to install pgvector separately????????????)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";
CREATE EXTENSION IF NOT EXISTS "vector";  -- Uncomment when pgvector is ready

-- =================================================================
-- 1. DOCUMENTS TABLE
-- =================================================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path TEXT,
    file_type VARCHAR(10) NOT NULL CHECK (file_type IN ('pdf', 'docx', 'xlsx', 'csv', 'txt')),
    file_size BIGINT,
    mime_type VARCHAR(100),
    
    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'pending' 
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed', 'requires_review')),
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    processing_error TEXT,
    
    -- Document metadata (JSON)
    metadata TEXT DEFAULT '{}',  -- Will upgrade to JSONB later
    extraction_summary TEXT DEFAULT '{}',
    
    -- Document classification
    document_type VARCHAR(50),
    language VARCHAR(10) DEFAULT 'en',
    page_count INTEGER,
    
    -- Audit fields
    uploaded_by VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =================================================================
-- 2. TEXT SEGMENTS TABLE
-- =================================================================

CREATE TABLE document_text_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Content organization
    sequence_number INTEGER NOT NULL,
    page_number INTEGER,
    section_id VARCHAR(100),
    section_title VARCHAR(500),
    
    -- Content details
    content TEXT NOT NULL,
    content_cleaned TEXT,
    content_length INTEGER,
    word_count INTEGER,
    
    -- Segment classification
    segment_type VARCHAR(50) DEFAULT 'paragraph' 
        CHECK (segment_type IN ('paragraph', 'heading', 'title', 'table_caption', 'list_item', 'footer', 'header', 'quote')),
    
    -- AI embeddings (when pgvector is available)
    -- embedding VECTOR(1536),  -- Uncomment when pgvector installed
    embedding_text TEXT,  -- Temporary text storage for embeddings
    embedding_model VARCHAR(50) DEFAULT 'text-embedding-ada-002',
    
    -- Extracted entities
    extracted_entities TEXT DEFAULT '{}',  -- JSON as text for now
    
    -- Position info
    bbox_json TEXT,  -- JSON as text
    font_info_json TEXT,  -- JSON as text
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_doc_sequence UNIQUE (document_id, sequence_number)
);

-- =================================================================
-- 3. DOCUMENT TABLES
-- =================================================================

CREATE TABLE document_tables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Table identification
    table_name VARCHAR(255),
    table_index INTEGER,
    sheet_name VARCHAR(255),
    page_number INTEGER,
    
    -- Table structure
    row_count INTEGER,
    column_count INTEGER,
    has_header BOOLEAN DEFAULT TRUE,
    
    -- Table content as JSON text
    table_data_json TEXT NOT NULL,
    column_definitions_json TEXT,
    table_summary_json TEXT DEFAULT '{}',
    
    -- Position info
    bbox_json TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =================================================================
-- 4. STRUCTURED DATA CELLS
-- =================================================================

CREATE TABLE document_structured_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_id UUID REFERENCES document_tables(id) ON DELETE CASCADE,
    
    -- Cell position
    row_number INTEGER NOT NULL,
    column_number INTEGER NOT NULL,
    column_name VARCHAR(255),
    
    -- Cell content
    cell_value TEXT,
    display_value TEXT,
    
    -- Typed values
    text_value TEXT,
    numeric_value DECIMAL(20,6),
    integer_value BIGINT,
    date_value DATE,
    datetime_value TIMESTAMP,
    boolean_value BOOLEAN,
    
    -- Data classification
    data_type VARCHAR(20) DEFAULT 'text' 
        CHECK (data_type IN ('text', 'number', 'integer', 'decimal', 'date', 'datetime', 'boolean', 'currency', 'percentage')),
    
    -- Metadata
    cell_metadata_json TEXT DEFAULT '{}',
    confidence_score DECIMAL(3,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_cell_position UNIQUE (table_id, row_number, column_number)
);

-- =================================================================
-- 5. KEY-VALUE PAIRS
-- =================================================================

CREATE TABLE document_key_values (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Key-value details
    key_name VARCHAR(255) NOT NULL,
    key_category VARCHAR(100),
    value_text TEXT,
    
    -- Typed values
    value_numeric DECIMAL(20,6),
    value_date DATE,
    value_boolean BOOLEAN,
    
    -- Source location
    page_number INTEGER,
    section_title VARCHAR(255),
    extraction_method VARCHAR(50),
    
    -- Validation
    confidence_score DECIMAL(3,2),
    is_verified BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =================================================================
-- 6. DOCUMENT SUMMARIES
-- =================================================================

CREATE TABLE document_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Summary details
    summary_type VARCHAR(50) NOT NULL,
    summary_text TEXT NOT NULL,
    summary_length INTEGER,
    
    -- AI model info
    model_used VARCHAR(100),
    model_version VARCHAR(50),
    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Summary embeddings (when available)
    summary_embedding_text TEXT,
    
    -- Quality metrics
    quality_score DECIMAL(3,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =================================================================
-- 7. PROCESSING LOG
-- =================================================================

CREATE TABLE document_processing_log (
    id SERIAL PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    processing_stage VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    processing_details_json TEXT DEFAULT '{}'
);

-- =================================================================
-- 8. BASIC INDEXES
-- =================================================================

-- Documents indexes
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_type ON documents(file_type);
CREATE INDEX idx_documents_uploaded ON documents(uploaded_at);

-- Text segments indexes
CREATE INDEX idx_text_segments_doc_id ON document_text_segments(document_id);
CREATE INDEX idx_text_segments_sequence ON document_text_segments(document_id, sequence_number);
CREATE INDEX idx_text_segments_type ON document_text_segments(segment_type);

-- Full-text search on content (PostgreSQL built-in)
CREATE INDEX idx_text_segments_content_gin ON document_text_segments USING gin(to_tsvector('english', content));

-- Structured data indexes
CREATE INDEX idx_structured_doc_id ON document_structured_data(document_id);
CREATE INDEX idx_structured_table_id ON document_structured_data(table_id);
CREATE INDEX idx_structured_column ON document_structured_data(column_name);
CREATE INDEX idx_structured_numeric ON document_structured_data(numeric_value) WHERE numeric_value IS NOT NULL;

-- Tables indexes
CREATE INDEX idx_tables_doc_id ON document_tables(document_id);

-- Key-value indexes
CREATE INDEX idx_key_values_doc_id ON document_key_values(document_id);
CREATE INDEX idx_key_values_key ON document_key_values(key_name);

-- =================================================================
-- 9. FUNCTIONS
-- =================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Auto-calculate text stats
CREATE OR REPLACE FUNCTION calculate_text_stats()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_length = LENGTH(NEW.content);
    NEW.word_count = array_length(string_to_array(trim(NEW.content), ' '), 1);
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_text_segment_stats 
    BEFORE INSERT OR UPDATE ON document_text_segments 
    FOR EACH ROW EXECUTE FUNCTION calculate_text_stats();

-- =================================================================
-- 10. BASIC SEARCH FUNCTION
-- =================================================================

-- Text search function (works without pgvector)
CREATE OR REPLACE FUNCTION search_documents_by_text(
    search_query TEXT,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    document_id UUID,
    filename VARCHAR(255),
    segment_id UUID,
    content TEXT,
    rank_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.filename,
        ts.id,
        ts.content,
        ts_rank(to_tsvector('english', ts.content), plainto_tsquery('english', search_query)) as rank_score
    FROM documents d
    JOIN document_text_segments ts ON d.id = ts.document_id
    WHERE to_tsvector('english', ts.content) @@ plainto_tsquery('english', search_query)
    ORDER BY rank_score DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- =================================================================
-- 11. SAMPLE DATA
-- =================================================================

-- Insert sample document for testing
INSERT INTO documents (
    filename, 
    original_filename, 
    file_type, 
    processing_status, 
    document_type,
    metadata
) VALUES (
    'sample_test.pdf', 
    'Sample Test Document.pdf', 
    'pdf', 
    'completed', 
    'test_document',
    '{"created_by": "system", "test": true}'
);

-- Insert sample text segment
INSERT INTO document_text_segments (
    document_id,
    sequence_number,
    content,
    segment_type
) VALUES (
    (SELECT id FROM documents WHERE filename = 'sample_test.pdf'),
    1,
    'This is a sample text segment for testing the ReportMiner database schema.',
    'paragraph'
);

-- Success message
SELECT 'ReportMiner database schema created successfully!' as status;