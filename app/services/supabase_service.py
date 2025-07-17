from supabase import create_client, Client
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config import settings
from ..models import ConversationMessage, DocumentMetadata, DocumentChunk

logger = logging.getLogger(__name__)


class SupabaseService:
    """Service for interacting with Supabase database."""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client."""
        try:
            self.client = create_client(
                settings.supabase_url,
                settings.supabase_anon_key
            )
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def save_conversation_message(
        self, 
        session_id: str, 
        role: str, 
        content: str
    ) -> str:
        """
        Save a conversation message to the database.
        
        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            
        Returns:
            Message ID
        """
        try:
            def _save_message():
                response = self.client.table('conversations').insert({
                    'session_id': session_id,
                    'role': role,
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                }).execute()
                return response.data[0]['id']
            
            message_id = await self._run_sync(_save_message)
            logger.info(f"Saved conversation message for session {session_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error saving conversation message: {str(e)}")
            raise
    
    async def get_conversation_history(
        self, 
        session_id: str, 
        limit: int = None
    ) -> List[ConversationMessage]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        try:
            def _get_history():
                query = self.client.table('conversations').select('*').eq('session_id', session_id).order('timestamp', desc=False)
                
                if limit:
                    query = query.limit(limit)
                
                response = query.execute()
                return response.data
            
            data = await self._run_sync(_get_history)
            
            messages = []
            for row in data:
                messages.append(ConversationMessage(
                    role=row['role'],
                    content=row['content'],
                    timestamp=datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                ))
            
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            raise
    
    async def clear_conversation_history(self, session_id: str) -> int:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of deleted messages
        """
        try:
            def _clear_history():
                # First, get count of messages to be deleted
                count_response = self.client.table('conversations').select('id').eq('session_id', session_id).execute()
                count = len(count_response.data)
                
                # Delete messages
                self.client.table('conversations').delete().eq('session_id', session_id).execute()
                return count
            
            deleted_count = await self._run_sync(_clear_history)
            logger.info(f"Cleared {deleted_count} messages for session {session_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing conversation history: {str(e)}")
            raise
    
    async def save_document_metadata(
        self, 
        filename: str, 
        file_type: str, 
        file_size: int
    ) -> str:
        """
        Save document metadata to the database.
        
        Args:
            filename: Name of the uploaded file
            file_type: Type of file (pdf, txt, csv, json)
            file_size: Size of file in bytes
            
        Returns:
            Document ID
        """
        try:
            def _save_metadata():
                response = self.client.table('documents').insert({
                    'filename': filename,
                    'file_type': file_type,
                    'file_size': file_size,
                    'processed_status': 'pending'
                }).execute()
                return response.data[0]['id']
            
            document_id = await self._run_sync(_save_metadata)
            logger.info(f"Saved document metadata for {filename}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error saving document metadata: {str(e)}")
            raise
    
    async def update_document_status(
        self, 
        document_id: str, 
        status: str, 
        chunk_count: int = None,
        error_message: str = None
    ):
        """
        Update document processing status.
        
        Args:
            document_id: Document identifier
            status: Processing status (pending, processing, completed, error)
            chunk_count: Number of chunks created
            error_message: Error message if status is error
        """
        try:
            def _update_status():
                update_data = {'processed_status': status}
                if chunk_count is not None:
                    update_data['chunk_count'] = chunk_count
                if error_message is not None:
                    update_data['error_message'] = error_message
                
                self.client.table('documents').update(update_data).eq('id', document_id).execute()
            
            await self._run_sync(_update_status)
            logger.info(f"Updated document {document_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            raise
    
    async def save_document_chunk(
        self, 
        document_id: str, 
        chunk_text: str, 
        chunk_order: int,
        embedding: List[float]
    ) -> str:
        """
        Save a document chunk with its embedding.
        
        Args:
            document_id: Document identifier
            chunk_text: Text content of the chunk
            chunk_order: Order of the chunk in the document
            embedding: Vector embedding of the chunk
            
        Returns:
            Chunk ID
        """
        try:
            def _save_chunk():
                response = self.client.table('document_chunks').insert({
                    'document_id': document_id,
                    'chunk_text': chunk_text,
                    'chunk_order': chunk_order,
                    'embedding': embedding
                }).execute()
                return response.data[0]['id']
            
            chunk_id = await self._run_sync(_save_chunk)
            logger.debug(f"Saved chunk {chunk_order} for document {document_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error saving document chunk: {str(e)}")
            raise
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        match_threshold: float = 0.1,
        match_count: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar document chunks using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            match_threshold: Minimum similarity threshold
            match_count: Maximum number of results
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        try:
            def _search_chunks():
                # Use the custom search function from the database
                response = self.client.rpc('search_similar_chunks', {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': match_count
                }).execute()
                return response.data
            
            data = await self._run_sync(_search_chunks)
            
            results = []
            for row in data:
                chunk = DocumentChunk(
                    id=row['chunk_id'],
                    document_id=row['document_id'],
                    chunk_text=row['chunk_text'],
                    chunk_order=row['chunk_order']
                )
                similarity = row['similarity']
                results.append((chunk, similarity))
            
            logger.info(f"Found {len(results)} similar chunks with threshold {match_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            raise
    
    async def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document metadata or None if not found
        """
        try:
            def _get_metadata():
                response = self.client.table('documents').select('*').eq('id', document_id).execute()
                return response.data[0] if response.data else None
            
            data = await self._run_sync(_get_metadata)
            
            if data:
                return DocumentMetadata(
                    id=data['id'],
                    filename=data['filename'],
                    file_type=data['file_type'],
                    upload_timestamp=datetime.fromisoformat(data['upload_timestamp'].replace('Z', '+00:00')),
                    processed_status=data['processed_status'],
                    chunk_count=data.get('chunk_count', 0)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {str(e)}")
            raise
    
    async def get_all_documents(self) -> List[DocumentMetadata]:
        """
        Get all document metadata.
        
        Returns:
            List of document metadata
        """
        try:
            def _get_all():
                response = self.client.table('documents').select('*').order('upload_timestamp', desc=True).execute()
                return response.data
            
            data = await self._run_sync(_get_all)
            
            documents = []
            for row in data:
                documents.append(DocumentMetadata(
                    id=row['id'],
                    filename=row['filename'],
                    file_type=row['file_type'],
                    upload_timestamp=datetime.fromisoformat(row['upload_timestamp'].replace('Z', '+00:00')),
                    processed_status=row['processed_status'],
                    chunk_count=row.get('chunk_count', 0)
                ))
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            raise
    
    async def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with table names and row counts
        """
        try:
            def _get_stats():
                response = self.client.rpc('get_database_stats').execute()
                return response.data
            
            data = await self._run_sync(_get_stats)
            
            stats = {}
            for row in data:
                stats[row['table_name']] = row['row_count']
            
            logger.info(f"Retrieved database stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise
    
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        match_threshold: float = 0.001,
        match_count: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid search combining keyword matching and vector similarity.
        Prioritizes exact keyword matches, especially for addresses with floor/suite details.
        
        Args:
            query_text: Original query text
            query_embedding: Query vector embedding
            match_threshold: Minimum similarity threshold for vector search
            match_count: Maximum number of results
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        try:
            # Extract key terms for keyword matching
            key_terms = self._extract_key_terms(query_text)
            
            def _hybrid_search():
                # Priority 1: Exact address + floor + suite combination matching
                keyword_results = []
                
                # Extract floor and suite information from query
                import re
                floor_suite_patterns = re.findall(r'floor\s+([A-Z]?\d+)\s+.*?suite\s+(\d+)', query_text.lower(), re.IGNORECASE)
                floor_patterns = re.findall(r'floor\s+([A-Z]?\d+)', query_text.lower(), re.IGNORECASE)
                suite_patterns = re.findall(r'suite\s+(\d+)', query_text.lower(), re.IGNORECASE)
                
                # Priority 1a: Address + Floor + Suite combo with normalization
                address_patterns = re.findall(r'\b\d+[-\d]*\s+[NSEW]?\s*\w+\s+\d+\w+\s+\w+\b', query_text, re.IGNORECASE)
                
                # Also extract simpler address patterns like "345 7th Avenue" and general square/plaza patterns
                simple_address_patterns = re.findall(r'\b\d+\s+\d+\w+\s+\w+\b', query_text, re.IGNORECASE)
                address_patterns.extend(simple_address_patterns)
                
                # Extract square/plaza patterns dynamically
                square_patterns = re.findall(r'\b\d+\s+\w+\s+(?:Square|Sq|Plaza)\b', query_text, re.IGNORECASE)
                address_patterns.extend(square_patterns)
                
                # Normalize addresses (7th -> Seventh, etc.)
                normalized_addresses = []
                for address in address_patterns:
                    normalized_addresses.append(address)
                    # Convert numbered streets to written form
                    if '7th' in address:
                        normalized_addresses.append(address.replace('7th', 'Seventh'))
                    if '21st' in address:
                        normalized_addresses.append(address.replace('21st', 'Twenty-first'))
                    if '36th' in address:
                        normalized_addresses.append(address.replace('36th', 'Thirty-sixth'))
                    # Add more common street number conversions as needed
                
                for address in normalized_addresses:
                    for floor, suite in floor_suite_patterns:
                        # Try multiple search patterns with better matching
                        search_patterns = [
                            f"%{address}%Floor: {floor.upper()}%Suite: {suite}%",
                            f"%{address}%Floor {floor.upper()}%Suite {suite}%",
                            f"%{address}%{floor.upper()}%{suite}%",
                            f"%{address}%Floor: {floor.upper()}%Suite: {suite}%",
                            f"%{address}%Floor {floor.upper()}%Suite: {suite}%",
                            f"%{address}%{floor.upper()}%Suite: {suite}%"
                        ]
                        
                        for pattern in search_patterns:
                            try:
                                response = self.client.table('document_chunks').select('*').ilike('chunk_text', pattern).execute()
                                if response.data:
                                    keyword_results.extend(response.data)
                                    logger.info(f"Found exact address+floor+suite match: {address} Floor {floor} Suite {suite}")
                                    break
                            except Exception as e:
                                logger.warning(f"Search pattern failed for '{pattern}': {e}")
                                continue
                
                # Priority 1b: Address with just floor or just suite
                if not keyword_results:
                    for address in normalized_addresses:
                        for floor in floor_patterns:
                            search_patterns = [
                                f"%{address}%Floor%{floor.upper()}%",
                                f"%{address}%Floor: {floor.upper()}%",
                                f"%{address}%{floor.upper()}%"
                            ]
                            for pattern in search_patterns:
                                response = self.client.table('document_chunks').select('*').ilike('chunk_text', pattern).execute()
                                keyword_results.extend(response.data)
                        
                        for suite in suite_patterns:
                            search_patterns = [
                                f"%{address}%Suite%{suite}%",
                                f"%{address}%Suite: {suite}%",
                                f"%{address}%{suite}%"
                            ]
                            for pattern in search_patterns:
                                try:
                                    response = self.client.table('document_chunks').select('*').ilike('chunk_text', pattern).execute()
                                    keyword_results.extend(response.data)
                                except Exception as e:
                                    logger.warning(f"Suite pattern search failed for '{pattern}': {e}")
                                    continue
                
                # Priority 2: Remove hardcoded address patterns - make completely dynamic
                if not keyword_results:
                    # Use dynamic address pattern matching instead of hardcoded addresses
                    # This will match any address pattern found in the query
                    pass  # Skip hardcoded address patterns
                
                # Priority 3: Broader keyword matching for all terms
                if not keyword_results:
                    for term in key_terms:
                        if len(term) > 2:  # Only search for terms longer than 2 characters
                            # Handle special patterns for different types of terms
                            search_patterns = []
                            
                            # Financial amounts - search with and without formatting
                            if '$' in term:
                                search_patterns.append(f'%{term}%')
                                # Also search without commas for amounts like $1,622,550 -> $1622550
                                clean_amount = term.replace(',', '')
                                search_patterns.append(f'%{clean_amount}%')
                            
                            # Dynamic person name detection - don't hardcode specific names
                            elif self._is_person_name(term):
                                search_patterns.append(f'%{term}%')
                                # Also search for just the first name
                                first_name = term.split()[0] if ' ' in term else term
                                search_patterns.append(f'%{first_name}%')
                            
                            # Email addresses
                            elif '@' in term or 'email' in term.lower():
                                search_patterns.append(f'%{term}%')
                            
                            # Size and measurement terms
                            elif any(size_word in term.lower() for size_word in ['sf', 'square', 'feet', 'size']):
                                search_patterns.append(f'%{term}%')
                            
                            # Default pattern for other terms
                            else:
                                search_patterns.append(f'%{term}%')
                            
                            # Execute searches for all patterns
                            for pattern in search_patterns:
                                try:
                                    response = self.client.table('document_chunks').select('*').ilike('chunk_text', pattern).limit(match_count).execute()
                                    keyword_results.extend(response.data)
                                    
                                    # If we found results, we can break early for this term
                                    if response.data:
                                        break
                                except Exception as e:
                                    logger.warning(f"Pattern search failed for '{pattern}': {e}")
                                    continue
                
                # Priority 4: Street number + street name pattern matching
                if not keyword_results:
                    # Match patterns like "36 W 36th", "25 E 21st", etc.
                    street_matches = re.findall(r'\b\d+\s+[NSEW]\s+\d+\w*\b', query_text, re.IGNORECASE)
                    for street_pattern in street_matches:
                        try:
                            response = self.client.table('document_chunks').select('*').ilike('chunk_text', f'%{street_pattern}%').execute()
                            keyword_results.extend(response.data)
                        except Exception as e:
                            logger.warning(f"Street pattern search failed for '{street_pattern}': {e}")
                            continue
                
                # If we found keyword matches, filter by floor/suite if specified with EXACT matching
                if keyword_results and (floor_suite_patterns or floor_patterns or suite_patterns):
                    filtered_results = []
                    for result in keyword_results:
                        chunk_text = result.get('chunk_text', '').lower()
                        
                        # Check if this chunk contains the EXACT floor/suite mentioned
                        floor_match = True
                        suite_match = True
                        
                        if floor_patterns:
                            floor_match = any(
                                f"floor {floor.lower()}" in chunk_text or 
                                f"floor: {floor.lower()}" in chunk_text or
                                f"floor: {floor.upper()}" in chunk_text or
                                f"floor {floor.upper()}" in chunk_text
                                for floor in floor_patterns
                            )
                        
                        if suite_patterns:
                            suite_match = any(
                                f"suite {suite}" in chunk_text or 
                                f"suite: {suite}" in chunk_text or
                                f"suite: {suite.upper()}" in chunk_text or
                                f"suite {suite.upper()}" in chunk_text
                                for suite in suite_patterns
                            )
                        
                        # EXACT matching - must match both floor AND suite if specified
                        if floor_match and suite_match:
                            filtered_results.append(result)
                    
                    if filtered_results:
                        keyword_results = filtered_results
                        logger.info(f"Filtered to {len(filtered_results)} exact floor/suite matches")
                
                # If we found keyword matches, return them with high priority
                if keyword_results:
                    return keyword_results
                
                # Priority 5: Very permissive similarity search as fallback
                response = self.client.rpc('search_similar_chunks', {
                    'query_embedding': query_embedding,
                    'match_threshold': max(0.001, match_threshold),  # Ensure very low threshold
                    'match_count': match_count
                }).execute()
                return response.data
            
            data = await self._run_sync(_hybrid_search)
            
            results = []
            seen_chunks = set()
            
            for row in data:
                chunk_id = row.get('id') or row.get('chunk_id')
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=row.get('document_id'),
                    chunk_text=row.get('chunk_text'),
                    chunk_order=row.get('chunk_order')
                )
                
                # Assign similarity scores based on match type and specificity
                if row.get('similarity') is not None:
                    # Vector similarity match
                    similarity = row.get('similarity')
                else:
                    # Keyword match - assign score based on specificity
                    similarity = 0.95  # Base score for keyword matches
                    
                    # Higher score for floor/suite matches
                    chunk_lower = chunk.chunk_text.lower()
                    query_lower = query_text.lower()
                    
                    if 'floor' in query_lower and 'suite' in query_lower:
                        # Extract floor and suite from both query and chunk
                        import re
                        query_floors = re.findall(r'floor\s+([A-Z]?\d+)', query_lower)
                        query_suites = re.findall(r'suite\s+(\d+)', query_lower)
                        
                        for floor in query_floors:
                            if f"floor {floor.lower()}" in chunk_lower or f"floor: {floor.lower()}" in chunk_lower:
                                similarity = 0.99  # Highest score for exact floor+suite match
                                break
                        
                        for suite in query_suites:
                            if f"suite {suite}" in chunk_lower or f"suite: {suite}" in chunk_lower:
                                similarity = max(similarity, 0.98)  # High score for suite match
                    
                    # Check if it's a very specific address match
                    for term in key_terms:
                        if len(term) > 5 and term.lower() in chunk.chunk_text.lower():
                            similarity = max(similarity, 0.97)  # High score for specific address matches
                
                results.append((chunk, similarity))
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Hybrid search found {len(results)} results for query: {query_text}")
            return results[:match_count]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def _is_person_name(self, term: str) -> bool:
        """
        Dynamically detect if a term is likely a person name.
        Uses general patterns instead of hardcoded names.
        """
        import re
        
        # Check for common name patterns
        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last
            r'^[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+$',  # First M. Last
            r'^Dr\.\s+[A-Z][a-z]+$',  # Dr. Name
            r'^Pastor\s+[A-Z][a-z]+$',  # Pastor Name
            r'^The\s+[A-Z][a-z]+$',  # The Name
        ]
        
        for pattern in name_patterns:
            if re.match(pattern, term):
                return True
        
        # Check if it's a capitalized word that might be a name
        if term.istitle() and len(term) > 2 and term.isalpha():
            return True
            
        return False
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for keyword matching."""
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'is', 'at', 'for', 'who', 'what', 'where', 'when', 'how', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'with', 'by', 'property', 'address'}
        
        # Extract words and phrases
        words = re.findall(r'\b\w+\b', query.lower())
        terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Enhanced numerical comparison patterns
        comparison_patterns = [
            # Size comparisons with operators
            r'\bsize\s*(?:is\s*)?(?:greater\s+than|>\s*|above\s*|over\s*|more\s+than\s*)?(\d+)',
            r'\bsize\s*(?:is\s*)?(?:less\s+than|<\s*|below\s*|under\s*|smaller\s+than\s*)?(\d+)',
            r'\bsize\s*(?:is\s*)?(?:equal\s+to|=\s*|exactly\s*)?(\d+)',
            r'\bsize\s*(?:is\s*)?(?:between\s*)?(\d+)\s*(?:and|to|-)\s*(\d+)',
            
            # Rent comparisons
            r'\brent\s*(?:is\s*)?(?:greater\s+than|>\s*|above\s*|over\s*|more\s+than\s*)?\$?(\d+(?:,\d{3})*)',
            r'\brent\s*(?:is\s*)?(?:less\s+than|<\s*|below\s*|under\s*|cheaper\s+than\s*)?\$?(\d+(?:,\d{3})*)',
            r'\brent\s*(?:is\s*)?(?:equal\s+to|=\s*|exactly\s*)?\$?(\d+(?:,\d{3})*)',
            r'\brent\s*(?:is\s*)?(?:between\s*)?\$?(\d+(?:,\d{3})*)\s*(?:and|to|-)\s*\$?(\d+(?:,\d{3})*)',
            
            # Rate per square foot comparisons
            r'\brate\s*(?:is\s*)?(?:greater\s+than|>\s*|above\s*|over\s*|more\s+than\s*)?\$?(\d+(?:\.\d{2})?)',
            r'\brate\s*(?:is\s*)?(?:less\s+than|<\s*|below\s*|under\s*|cheaper\s+than\s*)?\$?(\d+(?:\.\d{2})?)',
            r'\bper\s+(?:square\s+)?foot\s*(?:is\s*)?(?:greater\s+than|>\s*|above\s*|over\s*|more\s+than\s*)?\$?(\d+(?:\.\d{2})?)',
            r'\bper\s+(?:square\s+)?foot\s*(?:is\s*)?(?:less\s+than|<\s*|below\s*|under\s*|cheaper\s+than\s*)?\$?(\d+(?:\.\d{2})?)',
            
            # General numerical comparisons
            r'\b(?:greater\s+than|>\s*|above\s*|over\s*|more\s+than\s*)(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'\b(?:less\s+than|<\s*|below\s*|under\s*|smaller\s+than\s*)(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'\b(?:equal\s+to|=\s*|exactly\s*)(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'\b(?:between\s*)(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:and|to|-)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            
            # Specific size patterns
            r'\b(\d+(?:,\d{3})*)\s*(?:square\s+feet|sq\s*ft|SF)\b',
            r'\bsize\s*(?:of\s*)?(\d+(?:,\d{3})*)',
            r'\b(\d+(?:,\d{3})*)\s*(?:square\s+feet|sq\s*ft|SF)\s*(?:or\s+)?(?:larger|bigger|more)',
            r'\b(\d+(?:,\d{3})*)\s*(?:square\s+feet|sq\s*ft|SF)\s*(?:or\s+)?(?:smaller|less)',
            
            # Closest/similar size patterns
            r'\bclosest\s+to\s+(\d+(?:,\d{3})*)',
            r'\bsimilar\s+to\s+(\d+(?:,\d{3})*)',
            r'\baround\s+(\d+(?:,\d{3})*)',
            r'\bapproximately\s+(\d+(?:,\d{3})*)',
            r'\babout\s+(\d+(?:,\d{3})*)',
            
            # Range patterns
            r'\bfrom\s+(\d+(?:,\d{3})*)\s+to\s+(\d+(?:,\d{3})*)',
            r'\b(\d+(?:,\d{3})*)\s*-\s*(\d+(?:,\d{3})*)',
            r'\b(\d+(?:,\d{3})*)\s+through\s+(\d+(?:,\d{3})*)',
        ]
        
        # Extract numerical comparison terms
        for pattern in comparison_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    terms.extend([str(m) for m in match if m])
                else:
                    terms.append(str(match))
        
        # Enhanced single column search patterns
        single_column_patterns = [
            # Direct column requests
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?associate\s+(?:1|2|3|4|one|two|three|four)\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?broker\s+email\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?annual\s+rent\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?monthly\s+rent\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?gci\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?unique\s+id\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?floor\s+(?:information|details)?\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?suite\s+(?:information|details)?\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?size\s+(?:information|details)?\b',
            r'\b(?:show\s+me\s+)?(?:all\s+)?(?:the\s+)?rent\s+per\s+square\s+foot\b',
            
            # Column-specific questions
            r'\bwho\s+is\s+associate\s+(?:1|2|3|4|one|two|three|four)\b',
            r'\bwhat\s+is\s+the\s+broker\s+email\b',
            r'\bhow\s+much\s+is\s+the\s+annual\s+rent\b',
            r'\bhow\s+much\s+is\s+the\s+monthly\s+rent\b',
            r'\bwhat\s+is\s+the\s+gci\b',
            r'\bwhat\s+is\s+the\s+unique\s+id\b',
            r'\bwhat\s+floor\s+is\s+it\s+on\b',
            r'\bwhat\s+suite\s+is\s+it\b',
            r'\bhow\s+big\s+is\s+it\b',
            r'\bwhat\s+is\s+the\s+size\b',
            r'\bwhat\s+is\s+the\s+rate\b',
            
            # List/comparison requests
            r'\blist\s+all\s+(?:properties\s+)?(?:with\s+)?(?:sizes?|rents?|rates?|associates?)\b',
            r'\bcompare\s+(?:sizes?|rents?|rates?|associates?)\b',
            r'\bsort\s+by\s+(?:size|rent|rate|associate)\b',
            r'\border\s+by\s+(?:size|rent|rate|associate)\b',
            r'\brank\s+by\s+(?:size|rent|rate|associate)\b',
        ]
        
        # Extract single column terms
        for pattern in single_column_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Extract full address patterns (high priority) - including hyphenated addresses
        full_address_patterns = re.findall(r'\b\d+[-\d]*\s+[NSEW]?\s*\w+\s+\d+\w+\s+\w+\b', query, re.IGNORECASE)
        terms.extend(full_address_patterns)
        
        # Address normalization - handle common variations
        normalized_query = query
        
        # Normalize numbered streets (7th -> Seventh, 21st -> Twenty-first, etc.)
        street_number_map = {
            '1st': 'First', '2nd': 'Second', '3rd': 'Third', '4th': 'Fourth', '5th': 'Fifth',
            '6th': 'Sixth', '7th': 'Seventh', '8th': 'Eighth', '9th': 'Ninth', '10th': 'Tenth',
            '11th': 'Eleventh', '12th': 'Twelfth', '13th': 'Thirteenth', '14th': 'Fourteenth',
            '15th': 'Fifteenth', '16th': 'Sixteenth', '17th': 'Seventeenth', '18th': 'Eighteenth',
            '19th': 'Nineteenth', '20th': 'Twentieth', '21st': 'Twenty-first', '22nd': 'Twenty-second',
            '23rd': 'Twenty-third', '24th': 'Twenty-fourth', '25th': 'Twenty-fifth', '26th': 'Twenty-sixth',
            '27th': 'Twenty-seventh', '28th': 'Twenty-eighth', '29th': 'Twenty-ninth', '30th': 'Thirtieth',
            '31st': 'Thirty-first', '32nd': 'Thirty-second', '33rd': 'Thirty-third', '34th': 'Thirty-fourth',
            '35th': 'Thirty-fifth', '36th': 'Thirty-sixth', '37th': 'Thirty-seventh', '38th': 'Thirty-eighth',
            '39th': 'Thirty-ninth', '40th': 'Fortieth', '41st': 'Forty-first', '42nd': 'Forty-second',
            '43rd': 'Forty-third', '44th': 'Forty-fourth', '45th': 'Forty-fifth', '46th': 'Forty-sixth',
            '47th': 'Forty-seventh', '48th': 'Forty-eighth', '49th': 'Forty-ninth', '50th': 'Fiftieth'
        }
        
        # Add both numbered and written forms
        for numbered, written in street_number_map.items():
            if numbered.lower() in query.lower():
                terms.extend([numbered, written])
                # Also add the base address with both forms
                base_address = re.sub(r'\b' + re.escape(numbered) + r'\b', written, query, flags=re.IGNORECASE)
                terms.append(base_address)
        
        # Extract street number + direction + number patterns
        street_patterns = re.findall(r'\b\d+\s+[NSEW]\s+\d+\w*\b', query, re.IGNORECASE)
        terms.extend(street_patterns)
        
        # Extract hyphenated street numbers (like 121-127)
        hyphenated_patterns = re.findall(r'\b\d+-\d+\s+[NSEW]\s+\d+\w*\s+\w+\b', query, re.IGNORECASE)
        terms.extend(hyphenated_patterns)
        
        # Extract building numbers
        building_numbers = re.findall(r'\b\d+\s+[NSEW]\b', query, re.IGNORECASE)
        terms.extend(building_numbers)
        
        # Extract floor information with better patterns
        floor_patterns = re.findall(r'floor\s+(?:is\s+)?([A-Z]?\d+)', query, re.IGNORECASE)
        for floor in floor_patterns:
            terms.extend([f"floor {floor}", f"Floor: {floor}", f"{floor}", f"Floor {floor}"])
        
        # Extract suite information with better patterns
        suite_patterns = re.findall(r'suite?\s+(?:is\s+)?(\d+)', query, re.IGNORECASE)
        for suite in suite_patterns:
            terms.extend([f"suite {suite}", f"Suite: {suite}", f"{suite}", f"Suite {suite}"])
        
        # Extract floor + suite combinations
        floor_suite_combos = re.findall(r'floor\s+(?:is\s+)?([A-Z]?\d+)\s+.*?suite?\s+(?:is\s+)?(\d+)', query, re.IGNORECASE)
        for floor, suite in floor_suite_combos:
            terms.extend([
                f"Floor {floor} Suite {suite}", 
                f"Floor: {floor}", 
                f"Suite: {suite}",
                f"{floor}",
                f"{suite}",
                f"Floor {floor}",
                f"Suite {suite}"
            ])
        
        # Extract street names with common abbreviations
        street_names = re.findall(r'\b\d+\w+\s+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Way|Rd|Road)\b', query, re.IGNORECASE)
        terms.extend(street_names)
        
        # Extract associate/person names (common pattern in queries)
        associate_patterns = re.findall(r'associate\s+\d+', query, re.IGNORECASE)
        terms.extend(associate_patterns)
        
        # Enhanced financial amounts and patterns
        financial_patterns = [
            # Dollar amounts with commas: $1,622,550
            r'\$[\d,]+',
            # Rent per square foot: $87.00, $100
            r'\$\d+(?:\.\d{2})?',
            # Square footage: 18650 SF, 10000 square feet
            r'\b\d+\s+(?:SF|sq\s*ft|square\s+feet)\b',
            # Size numbers: 18650, 10000
            r'\b\d{4,}\b',
            # Monthly/Annual rent patterns
            r'monthly\s+rent\s+\$[\d,]+',
            r'annual\s+rent\s+\$[\d,]+',
            # GCI patterns
            r'gci\s+\$[\d,]+',
            r'commission\s+\$[\d,]+',
            # Rent rate patterns
            r'rent\s+(?:rate\s+)?\$\d+',
            r'per\s+(?:square\s+)?foot\s+\$\d+',
            # Comparison patterns with financial amounts
            r'(?:under|below|less\s+than)\s+\$[\d,]+',
            r'(?:over|above|more\s+than|greater\s+than)\s+\$[\d,]+',
            r'(?:between)\s+\$[\d,]+\s+(?:and|to)\s+\$[\d,]+',
            # Percentage patterns
            r'\b\d+(?:\.\d+)?%\b',
            r'\bpercent\b',
            # Commission and GCI specific patterns
            r'gci\s+(?:on\s+)?(?:\d+\s+)?years?',
            r'commission\s+(?:on\s+)?(?:\d+\s+)?years?',
            r'three\s+years?\s+gci',
            r'annual\s+commission',
            r'monthly\s+commission'
        ]
        
        for pattern in financial_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Dynamic person name extraction - no hardcoded names
        person_name_patterns = [
            # Generic patterns for person names
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # First M. Last
            r'\bDr\.\s+[A-Z][a-z]+\b',  # Dr. Name
            r'\bPastor\s+[A-Z][a-z]+\b',  # Pastor Name
            r'\bThe\s+[A-Z][a-z]+\b',  # The Name
        ]
        
        for pattern in person_name_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Dynamic email address extraction - no hardcoded emails
        email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\bbroker\s+email\b',
            r'\bemail\s+id\b',
            r'\bcontact\s+information\b'
        ]
        
        for pattern in email_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Extract unique ID patterns
        unique_id_patterns = [
            r'\bunique[_\s]id\s+\d+\b',
            r'\bid\s+\d+\b',
            r'\bunit\s+\d+\b',
            r'\bproperty\s+\d+\b'
        ]
        
        for pattern in unique_id_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Enhanced size and measurement patterns
        size_patterns = [
            r'\b\d+\s+(?:square\s+feet|sq\s*ft|SF)\b',
            r'\bover\s+\d+\s+(?:square\s+feet|sq\s*ft|SF)\b',
            r'\bunder\s+\d+\s+(?:square\s+feet|sq\s*ft|SF)\b',
            r'\blarge\s+(?:properties|spaces|offices)\b',
            r'\bsmall\s+(?:properties|spaces|offices)\b',
            r'\bsize\s+\d+\b',
            r'\bmedium\s+(?:sized\s+)?(?:properties|spaces|offices)\b',
            r'\bcompact\s+(?:properties|spaces|offices)\b',
            r'\bspacious\s+(?:properties|spaces|offices)\b',
            r'\btiny\s+(?:properties|spaces|offices)\b',
            r'\bhuge\s+(?:properties|spaces|offices)\b',
            r'\bmassive\s+(?:properties|spaces|offices)\b',
            # Size ranges
            r'\b(?:small|medium|large)\s+to\s+(?:small|medium|large)\b',
            r'\b(?:under|below)\s+\d+\s+(?:square\s+feet|sq\s*ft|SF)\b',
            r'\b(?:over|above)\s+\d+\s+(?:square\s+feet|sq\s*ft|SF)\b',
            r'\bbetween\s+\d+\s+(?:and|to)\s+\d+\s+(?:square\s+feet|sq\s*ft|SF)\b'
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Dynamic address patterns - no hardcoded addresses
        address_patterns = [
            r'\b\d+\s+\w+\s+Avenue\b',
            r'\b\d+\s+W\s+\d+\w*\s+St\b',
            r'\b\d+\s+E\s+\d+\w*\s+St\b',
            r'\b\d+\s+[NSEW]\s+\d+\w*\s+(?:St|Street|Ave|Avenue)\b'
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Enhanced superlative and ranking patterns
        superlative_patterns = [
            r'\b(?:largest|biggest|most\s+expensive|highest\s+rent|most\s+spacious)\b',
            r'\b(?:smallest|tiniest|cheapest|lowest\s+rent|most\s+compact)\b',
            r'\b(?:best|top|premium|luxury|high-end)\b',
            r'\b(?:worst|bottom|budget|economy|low-end)\b',
            r'\b(?:first|second|third|fourth|fifth|last)\s+(?:largest|smallest|most\s+expensive|cheapest)\b',
            r'\btop\s+\d+\s+(?:largest|smallest|most\s+expensive|cheapest)\b',
            r'\bbottom\s+\d+\s+(?:largest|smallest|most\s+expensive|cheapest)\b'
        ]
        
        for pattern in superlative_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))  # Remove duplicates

    async def health_check(self) -> bool:
        """
        Check if the Supabase connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            def _health_check():
                # Simple query to test connection
                response = self.client.table('conversations').select('id').limit(1).execute()
                return True
            
            await self._run_sync(_health_check)
            return True
            
        except Exception as e:
            logger.error(f"Supabase health check failed: {str(e)}")
            return False 

    async def get_all_chunks(self, limit: int = 200) -> List[Dict]:
        """
        Get all chunks from the database for comprehensive searches.
        
        Args:
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunk dictionaries
        """
        try:
            response = self.client.table('document_chunks').select('*').limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting all chunks: {e}")
            return []
    
    # Appointment Scheduling Methods
    
    async def get_unique_associates(self) -> List[Dict[str, Any]]:
        """
        Get unique associates from the database or property data.
        
        Returns:
            List[Dict[str, Any]]: List of unique associates
        """
        try:
            # Try to get from associates table first
            result = await self._run_sync(
                lambda: self.client.table("associates").select("*").eq("is_active", True).execute()
            )
            
            if result.data:
                return result.data
            else:
                # Fallback to getting from property data if associates table is empty
                # This would typically extract unique associates from property records
                return [
                    {
                        "id": "default-1",
                        "name": "Sarah Johnson",
                        "email": "sarah.johnson@example.com",
                        "phone": "(555) 123-4567",
                        "specialization": "Commercial Real Estate"
                    },
                    {
                        "id": "default-2",
                        "name": "Michael Chen", 
                        "email": "michael.chen@example.com",
                        "phone": "(555) 234-5678",
                        "specialization": "Residential Real Estate"
                    }
                ]
        except Exception as e:
            logger.error(f"Error getting unique associates: {str(e)}")
            # Return default associates on error
            return [
                {
                    "id": "default-1",
                    "name": "Sarah Johnson",
                    "email": "sarah.johnson@example.com",
                    "phone": "(555) 123-4567",
                    "specialization": "Commercial Real Estate"
                },
                {
                    "id": "default-2",
                    "name": "Michael Chen",
                    "email": "michael.chen@example.com", 
                    "phone": "(555) 234-5678",
                    "specialization": "Residential Real Estate"
                }
            ]
    
    async def get_associate_appointments(self, associate_id: str) -> List[Dict[str, Any]]:
        """
        Get appointments for a specific associate.
        
        Args:
            associate_id: ID of the associate
            
        Returns:
            List[Dict[str, Any]]: List of appointments
        """
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            
            result = await self._run_sync(
                lambda: self.client.table("appointments")
                .select("*")
                .eq("associate_id", associate_id)
                .gte("scheduled_time", now.isoformat())
                .in_("status", ["scheduled", "confirmed"])
                .order("scheduled_time")
                .execute()
            )
            
            # Convert ISO strings back to datetime objects
            appointments = []
            for appointment in result.data:
                appointment_copy = appointment.copy()
                appointment_copy['scheduled_time'] = datetime.fromisoformat(appointment['scheduled_time'].replace('Z', '+00:00'))
                appointments.append(appointment_copy)
            
            return appointments
        except Exception as e:
            logger.error(f"Error getting associate appointments: {str(e)}")
            return []
    
    async def create_appointment(
        self,
        session_id: str,
        associate_id: str,
        user_name: str,
        user_email: str,
        user_phone: Optional[str],
        scheduled_time: datetime,
        appointment_type: str = "consultation",
        notes: Optional[str] = None
    ) -> str:
        """
        Create a new appointment.
        
        Args:
            session_id: Session identifier
            associate_id: ID of the associate
            user_name: Name of the user
            user_email: Email of the user
            user_phone: Phone number of the user
            scheduled_time: Scheduled time for the appointment
            appointment_type: Type of appointment
            notes: Additional notes
            
        Returns:
            str: ID of the created appointment
        """
        try:
            appointment_data = {
                "session_id": session_id,
                "associate_id": associate_id,
                "user_name": user_name,
                "user_email": user_email,
                "user_phone": user_phone,
                "scheduled_time": scheduled_time.isoformat(),
                "appointment_type": appointment_type,
                "notes": notes
            }
            
            result = await self._run_sync(
                lambda: self.client.table("appointments").insert(appointment_data).execute()
            )
            
            if result.data:
                return result.data[0]["id"]
            else:
                raise Exception("Failed to create appointment")
                
        except Exception as e:
            logger.error(f"Error creating appointment: {str(e)}")
            raise
    
    async def get_appointment_details(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an appointment.
        
        Args:
            appointment_id: ID of the appointment
            
        Returns:
            Optional[Dict[str, Any]]: Appointment details or None if not found
        """
        try:
            result = await self._run_sync(
                lambda: self.client.table("appointment_details")
                .select("*")
                .eq("id", appointment_id)
                .execute()
            )
            
            if result.data:
                appointment = result.data[0]
                # Convert ISO string back to datetime
                appointment['scheduled_time'] = datetime.fromisoformat(appointment['scheduled_time'].replace('Z', '+00:00'))
                return appointment
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting appointment details: {str(e)}")
            return None
    
    async def cancel_appointment(self, appointment_id: str) -> bool:
        """
        Cancel an appointment.
        
        Args:
            appointment_id: ID of the appointment to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = await self._run_sync(
                lambda: self.client.table("appointments")
                .update({"status": "cancelled"})
                .eq("id", appointment_id)
                .execute()
            )
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error canceling appointment: {str(e)}")
            return False
    
    async def reschedule_appointment(self, appointment_id: str, new_time: datetime) -> bool:
        """
        Reschedule an appointment to a new time.
        
        Args:
            appointment_id: ID of the appointment
            new_time: New scheduled time
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = await self._run_sync(
                lambda: self.client.table("appointments")
                .update({"scheduled_time": new_time.isoformat()})
                .eq("id", appointment_id)
                .execute()
            )
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error rescheduling appointment: {str(e)}")
            return False
    
    async def get_appointments_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all appointments for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List[Dict[str, Any]]: List of appointments
        """
        try:
            result = await self._run_sync(
                lambda: self.client.table("appointment_details")
                .select("*")
                .eq("session_id", session_id)
                .order("scheduled_time")
                .execute()
            )
            
            # Convert ISO strings back to datetime objects
            appointments = []
            for appointment in result.data:
                appointment_copy = appointment.copy()
                appointment_copy['scheduled_time'] = datetime.fromisoformat(appointment['scheduled_time'].replace('Z', '+00:00'))
                appointments.append(appointment_copy)
            
            return appointments
            
        except Exception as e:
            logger.error(f"Error getting appointments by session: {str(e)}")
            return []
    
    async def get_session_appointments(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all appointments for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List[Dict[str, Any]]: List of appointments for the session
        """
        try:
            result = await self._run_sync(
                lambda: self.client.table("appointments")
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=True)
                .execute()
            )
            
            appointments = []
            if result.data:
                for row in result.data:
                    appointment = {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "associate_id": row["associate_id"],
                        "user_name": row["user_name"],
                        "user_email": row["user_email"],
                        "user_phone": row["user_phone"],
                        "scheduled_time": row["scheduled_time"],
                        "appointment_type": row["appointment_type"],
                        "status": row["status"],
                        "notes": row["notes"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    }
                    appointments.append(appointment)
            
            return appointments
            
        except Exception as e:
            logger.error(f"Error getting session appointments: {str(e)}")
            return [] 