-- Voice Conversational Agentic AI - Supabase Database Schema
-- This script sets up the database schema for the hackathon project

-- Enable the pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the conversations table for storing chat history
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for efficient session-based queries
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);

-- Create the documents table for storing RAG source document metadata
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_status TEXT NOT NULL DEFAULT 'pending' CHECK (processed_status IN ('pending', 'processing', 'completed', 'error')),
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for document queries
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processed_status);
CREATE INDEX IF NOT EXISTS idx_documents_upload_timestamp ON documents(upload_timestamp);

-- Create the document_chunks table for storing RAG text chunks and embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    embedding vector(384), -- 384 dimensions for all-MiniLM-L6-v2 model
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient RAG queries
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_order ON document_chunks(chunk_order);

-- Create a vector similarity index for efficient semantic search
-- Using HNSW (Hierarchical Navigable Small World) algorithm for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to automatically update the updated_at column
CREATE TRIGGER trigger_update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for easy conversation history retrieval
CREATE OR REPLACE VIEW conversation_history AS
SELECT 
    c.id,
    c.session_id,
    c.role,
    c.content,
    c.timestamp,
    ROW_NUMBER() OVER (PARTITION BY c.session_id ORDER BY c.timestamp) as message_order
FROM conversations c
ORDER BY c.session_id, c.timestamp;

-- Create a view for document statistics
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    d.id,
    d.filename,
    d.file_type,
    d.file_size,
    d.upload_timestamp,
    d.processed_status,
    d.chunk_count,
    COUNT(dc.id) as actual_chunk_count,
    AVG(LENGTH(dc.chunk_text)) as avg_chunk_length
FROM documents d
LEFT JOIN document_chunks dc ON d.id = dc.document_id
GROUP BY d.id, d.filename, d.file_type, d.file_size, d.upload_timestamp, d.processed_status, d.chunk_count;

-- Function to perform semantic search on document chunks
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(384),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    chunk_text TEXT,
    chunk_order INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id,
        dc.document_id,
        dc.chunk_text,
        dc.chunk_order,
        1 - (dc.embedding <=> query_embedding) as similarity
    FROM document_chunks dc
    WHERE 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get conversation history for a session
CREATE OR REPLACE FUNCTION get_conversation_history(
    p_session_id TEXT,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    role TEXT,
    content TEXT,
    timestamp TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.role,
        c.content,
        c.timestamp
    FROM conversations c
    WHERE c.session_id = p_session_id
    ORDER BY c.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to clear conversation history for a session
CREATE OR REPLACE FUNCTION clear_conversation_history(p_session_id TEXT)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM conversations WHERE session_id = p_session_id;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create RLS (Row Level Security) policies if needed
-- Note: For hackathon purposes, we'll keep it simple without RLS
-- But in production, you'd want to enable RLS and create appropriate policies

-- Insert some sample data for testing (optional)
-- INSERT INTO conversations (session_id, role, content) VALUES
-- ('test-session-1', 'user', 'Hello, how can you help me?'),
-- ('test-session-1', 'assistant', 'I can help you with various tasks. What would you like to know?');

-- Create a function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    table_name TEXT,
    row_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'conversations'::TEXT as table_name,
        COUNT(*)::BIGINT as row_count
    FROM conversations
    UNION ALL
    SELECT 
        'documents'::TEXT as table_name,
        COUNT(*)::BIGINT as row_count
    FROM documents
    UNION ALL
    SELECT 
        'document_chunks'::TEXT as table_name,
        COUNT(*)::BIGINT as row_count
    FROM document_chunks;
END;
$$ LANGUAGE plpgsql;

-- Comments for setup instructions
/*
SETUP INSTRUCTIONS:

1. Go to your Supabase project dashboard
2. Navigate to the SQL Editor
3. Run this entire script to create the database schema
4. Make sure the pgvector extension is enabled (it should be after running this script)
5. Verify the tables are created by checking the Table Editor
6. Update your .env file with your Supabase project URL and anon key

IMPORTANT NOTES:
- The embedding vector size is set to 384 dimensions for the all-MiniLM-L6-v2 model
- If you use a different embedding model, update the vector size accordingly
- The HNSW index on embeddings provides fast approximate nearest neighbor search
- All tables use UUIDs as primary keys for better scalability
- Timestamps are stored with timezone information
*/ 