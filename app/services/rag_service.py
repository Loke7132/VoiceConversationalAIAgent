from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import io
import json
import pandas as pd
from pypdf import PdfReader
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import settings
from ..services.supabase_service import SupabaseService
from ..models import DocumentChunk

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval Augmented Generation (RAG) operations."""
    
    def __init__(self, supabase_service: SupabaseService):
        self.supabase_service = supabase_service
        self.embedding_model = None
        self.text_splitter = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info("RAG service initialized")
    
    async def initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            def _load_model():
                return SentenceTransformer(settings.embedding_model)
            
            self.embedding_model = await self._run_sync(_load_model)
            logger.info(f"Embedding model {settings.embedding_model} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self.embedding_model:
            await self.initialize_embedding_model()
        
        def _encode():
            return self.embedding_model.encode(texts).tolist()
        
        return await self._run_sync(_encode)
    
    async def _extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF file."""
        try:
            def _extract():
                pdf_file = io.BytesIO(pdf_data)
                reader = PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            
            text = await self._run_sync(_extract)
            logger.info(f"Extracted text from PDF: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    async def _extract_text_from_csv(self, csv_data: bytes) -> str:
        """Extract text from CSV file with property-specific chunking."""
        try:
            def _extract():
                csv_file = io.StringIO(csv_data.decode('utf-8'))
                df = pd.read_csv(csv_file)
                
                # Create individual property chunks instead of one large text
                chunks = []
                for index, row in df.iterrows():
                    # Create a comprehensive text representation for each property
                    property_text = f"PROPERTY_UNIT_{index + 1}:\n"
                    
                    # Add all property details with clean formatting
                    for col, val in row.items():
                        # Clean up column names and format values
                        clean_col = col.replace('_', ' ').replace('-', ' ').title()
                        clean_val = str(val).strip()
                        
                        # Remove formatting markers and clean up the value
                        clean_val = clean_val.replace('**', '').replace('*', '')
                        clean_val = clean_val.replace('###', '').replace('#', '')
                        clean_val = clean_val.replace('||', '').replace('|', '')
                        clean_val = clean_val.replace('--', '-').replace('- -', '-')
                        clean_val = ' '.join(clean_val.split())  # Remove extra whitespace
                        
                        if clean_val and clean_val != 'nan':
                            property_text += f"{clean_col}: {clean_val}\n"
                    
                    # Add specific searchable patterns for better matching
                    address = row.get('Property Address', '')
                    floor = row.get('Floor', '')
                    suite = row.get('Suite', '')
                    
                    if address and floor and suite:
                        property_text += f"\nFULL_LOCATION: {address} Floor {floor} Suite {suite}\n"
                        property_text += f"ADDRESS_FLOOR_SUITE: {address}|{floor}|{suite}\n"
                    
                    property_text += "\n" + "="*50 + "\n"
                    chunks.append(property_text)
                
                # Join all property chunks with clear separators
                return "\n".join(chunks)
            
            text = await self._run_sync(_extract)
            logger.info(f"Extracted text from CSV: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from CSV: {str(e)}")
            raise
    
    async def _extract_text_from_json(self, json_data: bytes) -> str:
        """Extract text from JSON file."""
        try:
            def _extract():
                json_obj = json.loads(json_data.decode('utf-8'))
                
                def flatten_json(obj, prefix=""):
                    """Flatten JSON object to text."""
                    items = []
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            new_key = f"{prefix}.{key}" if prefix else key
                            if isinstance(value, (dict, list)):
                                items.extend(flatten_json(value, new_key))
                            else:
                                items.append(f"{new_key}: {value}")
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                            if isinstance(item, (dict, list)):
                                items.extend(flatten_json(item, new_key))
                            else:
                                items.append(f"{new_key}: {item}")
                    return items
                
                text_items = flatten_json(json_obj)
                return "\n".join(text_items)
            
            text = await self._run_sync(_extract)
            logger.info(f"Extracted text from JSON: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from JSON: {str(e)}")
            raise
    
    async def _extract_text_from_txt(self, txt_data: bytes) -> str:
        """Extract text from TXT file."""
        try:
            text = txt_data.decode('utf-8')
            logger.info(f"Extracted text from TXT: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {str(e)}")
            raise
    
    async def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove formatting markers first
        text = text.replace('**', '').replace('*', '')
        text = text.replace('###', '').replace('#', '')
        text = text.replace('||', '').replace('|', '')
        text = text.replace('--', '-').replace('- -', '-')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation and currency symbols
        text = re.sub(r'[^\w\s\.,!?;:()\-$@]', '', text)
        # Strip whitespace
        text = text.strip()
        return text
    
    async def process_document(
        self, 
        filename: str, 
        file_data: bytes, 
        file_type: str
    ) -> str:
        """
        Process a document and store it in the RAG knowledge base.
        
        Args:
            filename: Name of the file
            file_data: Raw file data
            file_type: Type of file (pdf, txt, csv, json)
            
        Returns:
            Document ID
        """
        try:
            # Save document metadata
            document_id = await self.supabase_service.save_document_metadata(
                filename, file_type, len(file_data)
            )
            
            # Update status to processing
            await self.supabase_service.update_document_status(
                document_id, "processing"
            )
            
            # Extract text based on file type
            if file_type.lower() == 'pdf':
                text = await self._extract_text_from_pdf(file_data)
            elif file_type.lower() == 'csv':
                text = await self._extract_text_from_csv(file_data)
            elif file_type.lower() == 'json':
                text = await self._extract_text_from_json(file_data)
            elif file_type.lower() == 'txt':
                text = await self._extract_text_from_txt(file_data)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Clean the text
            text = await self._clean_text(text)
            
            if not text.strip():
                raise ValueError("No text content found in document")
            
            # Use specialized chunking for CSV files
            if file_type.lower() == 'csv':
                chunks = await self._chunk_csv_by_properties(text)
            else:
                # Use standard text splitter for other file types
                chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            # Generate embeddings for all chunks
            embeddings = await self._generate_embeddings(chunks)
            
            # Save chunks and embeddings to database
            chunk_ids = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = await self.supabase_service.save_document_chunk(
                    document_id, chunk, i, embedding
                )
                chunk_ids.append(chunk_id)
            
            # Update document status to completed
            await self.supabase_service.update_document_status(
                document_id, "completed", len(chunks)
            )
            
            logger.info(f"Processed document {filename}: {len(chunks)} chunks created")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            
            # Update document status to error
            try:
                await self.supabase_service.update_document_status(
                    document_id, "error", error_message=str(e)
                )
            except:
                pass  # Don't fail if we can't update status
            
            raise
    
    async def _chunk_csv_by_properties(self, text: str) -> List[str]:
        """
        Split CSV text by individual property units instead of character count.
        
        Args:
            text: The processed CSV text
            
        Returns:
            List of chunks, one per property
        """
        try:
            # Split by the property unit markers we created
            property_chunks = text.split("PROPERTY_UNIT_")
            
            chunks = []
            for chunk in property_chunks:
                if chunk.strip() and not chunk.startswith("PROPERTY_UNIT_"):
                    # Add back the marker and clean up
                    clean_chunk = f"PROPERTY_UNIT_{chunk}".strip()
                    
                    # Remove the separator line at the end
                    clean_chunk = clean_chunk.replace("="*50, "").strip()
                    
                    if len(clean_chunk) > 50:  # Only include substantial chunks
                        chunks.append(clean_chunk)
            
            logger.info(f"Created {len(chunks)} property-specific chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in CSV chunking: {str(e)}")
            # Fallback to standard text splitting
            return self.text_splitter.split_text(text)
    
    async def get_relevant_context(
        self, 
        query: str, 
        max_chunks: int = None,
        similarity_threshold: float = 0.001  # Much lower threshold for address matching
    ) -> str:
        """
        Get relevant context for a query using similarity search.
        
        Args:
            query: Query text to search for
            max_chunks: Maximum number of chunks to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Combined context text from relevant chunks
        """
        try:
            if max_chunks is None:
                max_chunks = settings.top_k_results
            
            # Detect superlative queries that need more data for numerical comparisons
            superlative_patterns = [
                r'\b(?:maximum|minimum|max|min|highest|lowest|largest|smallest|biggest|tiniest)\b',
                r'\b(?:most\s+expensive|cheapest|most\s+spacious|most\s+compact)\b',
                r'\b(?:best|worst|top|bottom|first|last)\b',
                r'\b(?:greater\s+than|less\s+than|more\s+than|above|below|over|under)\b',
                r'\b(?:compare|comparison|sort|order|rank|list\s+all)\b',
                r'\btop\s+\d+\b',
                r'\bbottom\s+\d+\b'
            ]
            
            is_superlative_query = any(
                re.search(pattern, query, re.IGNORECASE) for pattern in superlative_patterns
            )
            
            # Increase search scope for superlative queries to get more data for comparison
            if is_superlative_query:
                max_chunks = min(max_chunks * 10, 225)  # Increase by 10x and cap at 225 (total CSV rows)
                similarity_threshold = 0.0001  # Use very low threshold for broader search
                logger.info(f"Detected superlative query, increasing search scope to {max_chunks} chunks with threshold {similarity_threshold}")
            
            # Generate embedding for the query (use original query without normalization)
            query_embedding = await self._generate_embeddings([query])
            query_embedding = query_embedding[0]
            
            # Always try hybrid search first (combines keyword + similarity)
            results = await self.supabase_service.hybrid_search(
                query, query_embedding, match_threshold=similarity_threshold, match_count=max_chunks
            )
            
            # For superlative queries, try additional search strategies to get better coverage
            if is_superlative_query and len(results) < max_chunks:
                logger.info(f"Superlative query returned {len(results)} chunks, trying additional search strategies")
                
                # Try broader keyword search with common property terms
                additional_terms = ['rent', 'monthly', 'size', 'square', 'feet', 'rate', 'annual', 'property']
                for term in additional_terms:
                    if term.lower() in query.lower():
                        additional_results = await self.supabase_service.search_similar_chunks(
                            query_embedding, match_threshold=0.0001, match_count=max_chunks
                        )
                        
                        # Merge results, avoiding duplicates
                        existing_ids = {chunk.id for chunk, _ in results}
                        for chunk, similarity in additional_results:
                            if chunk.id not in existing_ids:
                                results.append((chunk, similarity))
                                existing_ids.add(chunk.id)
                        
                        if len(results) >= max_chunks:
                            break
                
                logger.info(f"After additional search strategies: {len(results)} chunks")
            
            # If no results from hybrid search, try with progressively lower thresholds
            if not results and similarity_threshold > 0.001:
                logger.info(f"No hybrid results, trying similarity search with threshold {similarity_threshold}")
                results = await self.supabase_service.search_similar_chunks(
                    query_embedding, match_threshold=similarity_threshold, match_count=max_chunks
                )
            
            # If still no results, try with an extremely low threshold
            if not results:
                logger.info("Still no results, trying with threshold 0.0001")
                results = await self.supabase_service.search_similar_chunks(
                    query_embedding, match_threshold=0.0001, match_count=max_chunks
                )
            
            if not results:
                logger.info("No relevant context found for query")
                return ""
            
            # Combine context from relevant chunks
            context_parts = []
            for chunk, similarity in results:
                # Add chunk text with similarity score for debugging
                context_parts.append(f"[Similarity: {similarity:.3f}] {chunk.chunk_text}")
            
            context = "\n\n".join(context_parts)
            
            logger.info(f"Retrieved {len(results)} relevant chunks for query '{query}'")
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {str(e)}")
            return ""
    
    def _normalize_address_query(self, query: str) -> str:
        """
        Normalize address queries for better matching.
        
        Args:
            query: Original query
            
        Returns:
            Normalized query
        """
        # Convert to lowercase for consistent matching
        normalized = query.lower()
        
        # Common address abbreviations
        abbreviations = {
            'st': 'street',
            'ave': 'avenue',
            'blvd': 'boulevard',
            'rd': 'road',
            'dr': 'drive',
            'ln': 'lane',
            'ct': 'court',
            'pl': 'place',
            'w': 'west',
            'e': 'east',
            'n': 'north',
            's': 'south'
        }
        
        # Expand abbreviations
        words = normalized.split()
        expanded_words = []
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?;:')
            expanded_words.append(abbreviations.get(clean_word, clean_word))
        
        # Also include the original query to catch exact matches
        expanded_query = ' '.join(expanded_words)
        
        # Return both original and expanded for better matching
        return f"{query} {expanded_query}"
    
    async def search_documents(
        self, 
        query: str, 
        max_results: int = 10,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to a query.
        
        Args:
            query: Query text to search for
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results with metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = await self._generate_embeddings([query])
            query_embedding = query_embedding[0]
            
            # Search for similar chunks
            results = await self.supabase_service.search_similar_chunks(
                query_embedding, match_threshold=similarity_threshold, match_count=max_results
            )
            
            # Get document metadata for each result
            search_results = []
            for chunk, similarity in results:
                doc_metadata = await self.supabase_service.get_document_metadata(
                    chunk.document_id
                )
                
                search_results.append({
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": doc_metadata.filename if doc_metadata else "Unknown",
                    "file_type": doc_metadata.file_type if doc_metadata else "Unknown",
                    "chunk_text": chunk.chunk_text,
                    "chunk_order": chunk.chunk_order,
                    "similarity": similarity
                })
            
            logger.info(f"Found {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    async def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """
        Get a summary of a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document summary
        """
        try:
            # Get document metadata
            doc_metadata = await self.supabase_service.get_document_metadata(document_id)
            
            if not doc_metadata:
                return {"error": "Document not found"}
            
            # Get first few chunks as preview
            results = await self.supabase_service.search_similar_chunks(
                [0.0] * 384,  # Dummy embedding to get all chunks
                match_threshold=0.0,
                match_count=3
            )
            
            preview_chunks = [chunk.chunk_text for chunk, _ in results 
                            if chunk.document_id == document_id][:3]
            
            return {
                "document_id": document_id,
                "filename": doc_metadata.filename,
                "file_type": doc_metadata.file_type,
                "upload_timestamp": doc_metadata.upload_timestamp,
                "processed_status": doc_metadata.processed_status,
                "chunk_count": doc_metadata.chunk_count,
                "preview_chunks": preview_chunks
            }
            
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            return {"error": str(e)}
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # The database cascade delete will handle chunk deletion
            # We just need to update the document status
            await self.supabase_service.update_document_status(
                document_id, "deleted"
            )
            
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    async def health_check(self) -> bool:
        """
        Check if the RAG service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test embedding generation
            test_embedding = await self._generate_embeddings(["test"])
            
            # Test text splitting
            test_chunks = self.text_splitter.split_text("This is a test document.")
            
            return len(test_embedding) > 0 and len(test_chunks) > 0
            
        except Exception as e:
            logger.error(f"RAG service health check failed: {str(e)}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information.
        
        Returns:
            Service information dictionary
        """
        return {
            "service": "RAG",
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k_results": settings.top_k_results,
            "model_loaded": self.embedding_model is not None
        } 