#!/usr/bin/env python3
"""
ReAct (Reasoning and Acting) Service for enhanced query processing.
This service implements the ReAct pattern to provide better reasoning and action-taking
capabilities for complex queries, especially numerical comparisons.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .rag_service import RAGService
from .supabase_service import SupabaseService
from .gemini_service import GeminiService

logger = logging.getLogger(__name__)

@dataclass
class ReActStep:
    """Represents a single step in the ReAct process."""
    step_type: str  # "thought", "action", "observation"
    content: str
    action_type: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None

class ReActService:
    """
    ReAct (Reasoning and Acting) Service for enhanced query processing.
    
    This service implements the ReAct pattern where the AI:
    1. Thinks about the problem
    2. Takes actions (searches, calculations)
    3. Observes the results
    4. Iterates until finding the correct answer
    """
    
    def __init__(self, rag_service: RAGService, supabase_service: SupabaseService, gemini_service: GeminiService):
        self.rag_service = rag_service
        self.supabase_service = supabase_service
        self.gemini_service = gemini_service
        self.max_iterations = 5
        
    async def process_query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[ReActStep]]:
        """
        Process a query using the ReAct pattern.
        
        Args:
            query: User query to process
            session_id: Session ID for conversation tracking (optional)
            
        Returns:
            Tuple of (final_answer, list_of_react_steps)
        """
        logger.info(f"Processing query with ReAct: {query}")
        
        # Load conversation history if session_id is provided
        conversation_history = []
        if session_id:
            try:
                conversation_history = await self.supabase_service.get_conversation_history(session_id)
                logger.info(f"Loaded {len(conversation_history)} conversation messages for ReAct")
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
        
        # Enhanced query processing for contextual references
        enhanced_query = query
        
        # If the query contains contextual references, enhance it with conversation history
        if conversation_history:
            contextual_references = ['that address', 'this property', 'the same location', 'that property', 'this address', 'the property']
            if any(ref in query.lower() for ref in contextual_references):
                # Extract property addresses from recent conversation history
                recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
                for msg in reversed(recent_history):
                    if msg.role == 'assistant':
                        # Look for property addresses in assistant responses
                        import re
                        address_patterns = [
                            r'\b\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street)\b',
                            r'\b\d+\s+\w+\s+(?:Avenue|Ave)\b',
                            r'\b\d+\s+\w+\s+(?:Square|Sq|Plaza|Blvd|Boulevard|Way|Road|Rd)\b',
                            r'Address[:\s]+([^,\n]+)',
                            r'Property[:\s]+([^,\n]+)',
                            r'(\d+\s+\w+\s+(?:Square|Sq|Plaza),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                            r'(\d+\s+[NSEW]?\s*\w+\s+\d+\w+\s+(?:St|Street),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)',
                            r'(\d+\s+\w+\s+(?:Avenue|Ave),?\s+Floor\s+[A-Z]?\d+,?\s+Suite\s+\d+[A-Z]?)'
                        ]
                        
                        for pattern in address_patterns:
                            matches = re.findall(pattern, msg.content, re.IGNORECASE)
                            if matches:
                                # Use the first found address to enhance the query
                                found_address = matches[0]
                                enhanced_query = f"{query} {found_address}"
                                logger.info(f"Enhanced contextual query for ReAct: {enhanced_query}")
                                break
                        
                        if enhanced_query != query:
                            break
        
        steps = []
        all_property_data = []
        
        # Initial reasoning (now with conversation context)
        initial_thought = await self._generate_initial_thought(enhanced_query, conversation_history)
        steps.append(ReActStep("thought", initial_thought))
        
        # For superlative queries, we need comprehensive data
        if self._is_superlative_query(enhanced_query):
            # Action: Get ALL property data for numerical comparison
            action_step = ReActStep(
                "action", 
                "Getting comprehensive property data for numerical comparison",
                action_type="get_all_properties",
                action_input={"query": enhanced_query}
            )
            steps.append(action_step)
            
            # Execute comprehensive data retrieval
            observation = await self._get_all_property_data(enhanced_query)
            steps.append(ReActStep("observation", observation.content, result=observation.result))
            
            if observation.result:
                all_property_data = observation.result
                
        else:
            # For non-superlative queries, use regular search
            action_step = ReActStep(
                "action",
                f"Searching for relevant properties: {enhanced_query}",
                action_type="search",
                action_input={"query": enhanced_query}
            )
            steps.append(action_step)
            
            observation = await self._execute_search(action_step)
            steps.append(ReActStep("observation", observation.content, result=observation.result))
            
            if observation.result:
                all_property_data = observation.result
        
        # Generate final answer with all collected data and conversation context
        final_answer = await self._generate_final_answer(enhanced_query, steps, all_property_data, conversation_history)
        
        # Save conversation to database if session_id is provided
        if session_id:
            try:
                await self.supabase_service.save_conversation_message(session_id, "user", query)
                await self.supabase_service.save_conversation_message(session_id, "assistant", final_answer)
                logger.info(f"Saved ReAct conversation to database for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to save ReAct conversation: {e}")
        
        return final_answer, steps
    
    async def _generate_initial_thought(self, query: str, conversation_history: List = None) -> str:
        """Generate initial reasoning about the query with conversation context."""
        
        context_info = ""
        if conversation_history:
            context_info = f" I have access to {len(conversation_history)} previous conversation messages for context."
        
        if self._is_superlative_query(query):
            superlative_type = self._extract_superlative_type(query)
            return f"This is a superlative query asking for '{superlative_type}'. I need to get ALL property data and perform comprehensive numerical comparison to find the correct answer.{context_info}"
        elif self._is_comparison_query(query):
            return f"This is a comparison query. I need to find properties matching the criteria and compare their values.{context_info}"
        elif self._is_specific_property_query(query):
            return f"This is a specific property query. I need to search for the exact property mentioned.{context_info}"
        else:
            return f"This is a general query about properties. I need to search for relevant information.{context_info}"
    
    async def _get_all_property_data(self, query: str) -> ReActStep:
        """Get comprehensive property data for superlative queries."""
        
        try:
            all_data = []
            
            # Strategy 1: Get all chunks from database
            logger.info("Fetching all chunks from database...")
            all_chunks = await self.supabase_service.get_all_chunks(limit=300)
            
            # Extract and filter only property chunks
            property_chunks = []
            for chunk in all_chunks:
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('chunk_text', '')
                    if chunk_text and 'PROPERTY_UNIT_' in chunk_text:
                        property_chunks.append(chunk_text)
                elif isinstance(chunk, str) and 'PROPERTY_UNIT_' in chunk:
                    property_chunks.append(chunk)
            
            logger.info(f"Found {len(property_chunks)} property chunks")
            
            # Strategy 2: Enhanced RAG search with maximum scope
            logger.info("Performing enhanced RAG search...")
            rag_context = await self.rag_service.get_relevant_context(
                "monthly rent property address floor suite size", 
                max_chunks=250
            )
            
            # Combine all data sources
            all_data.extend(property_chunks)
            if rag_context:
                all_data.append(rag_context)
            
            # Strategy 3: Multiple targeted searches for comprehensive coverage
            search_terms = [
                "monthly rent", 
                "annual rent", 
                "property address", 
                "floor suite", 
                "size square feet",
                "rent per square foot"
            ]
            
            for term in search_terms:
                context = await self.rag_service.get_relevant_context(term, max_chunks=50)
                if context:
                    all_data.append(context)
            
            observation_text = f"Retrieved {len(all_data)} comprehensive data sources covering all properties for numerical analysis."
            
            return ReActStep("observation", observation_text, result=all_data)
            
        except Exception as e:
            logger.error(f"Error getting comprehensive property data: {e}")
            return ReActStep("observation", f"Failed to get comprehensive data: {str(e)}", result=None)
    
    async def _execute_search(self, action_step: ReActStep) -> ReActStep:
        """Execute a regular search."""
        
        search_input = action_step.action_input
        query = search_input["query"]
        
        try:
            context = await self.rag_service.get_relevant_context(query, max_chunks=50)
            
            if context:
                observation_text = f"Found relevant property data for query: {query}"
                return ReActStep("observation", observation_text, result=[context])
            else:
                return ReActStep("observation", "No relevant data found", result=None)
                
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return ReActStep("observation", f"Search failed: {str(e)}", result=None)
    
    def _extract_property_summaries(self, all_data: List[Any]) -> str:
        """Extract and summarize property data for LLM analysis without overwhelming detail."""
        
        property_summaries = []
        seen_properties = set()
        
        for data in all_data:
            if not data:
                continue
                
            data_str = str(data)
            
            # Extract individual property units
            property_units = data_str.split('PROPERTY_UNIT_')
            
            for unit in property_units:
                if not unit.strip():
                    continue
                    
                # The property data is all on one line, so we need to parse it differently
                unit_text = unit.strip()
                
                # Extract key information using regex patterns since it's all on one line
                property_info = {}
                
                # Extract Property Address
                address_match = re.search(r'Property Address:\s*([^:]+?)(?:\s+Floor:|$)', unit_text)
                if address_match:
                    property_info['Property Address'] = address_match.group(1).strip()
                
                # Extract Floor
                floor_match = re.search(r'Floor:\s*(\S+)', unit_text)
                if floor_match:
                    property_info['Floor'] = floor_match.group(1).strip()
                
                # Extract Suite
                suite_match = re.search(r'Suite:\s*(\S+)', unit_text)
                if suite_match:
                    property_info['Suite'] = suite_match.group(1).strip()
                
                # Extract Monthly Rent
                monthly_rent_match = re.search(r'Monthly Rent:\s*(\$[\d,]+)', unit_text)
                if monthly_rent_match:
                    property_info['Monthly Rent'] = monthly_rent_match.group(1).strip()
                
                # Extract Size
                size_match = re.search(r'Size \(Sf\):\s*(\d+)', unit_text)
                if size_match:
                    property_info['Size (Sf)'] = size_match.group(1).strip()
                
                # Extract Rent Rate
                rent_rate_match = re.search(r'RentSfYear:\s*(\$[\d.]+)', unit_text)
                if rent_rate_match:
                    property_info['RentSfYear'] = rent_rate_match.group(1).strip()
                
                # Extract essential fields
                address = property_info.get('Property Address', '')
                floor = property_info.get('Floor', '')
                suite = property_info.get('Suite', '')
                monthly_rent = property_info.get('Monthly Rent', '')
                size = property_info.get('Size (Sf)', '')
                rent_rate = property_info.get('RentSfYear', '')
                
                if address:
                    property_id = f"{address}_{floor}_{suite}"
                    
                    if property_id not in seen_properties:
                        seen_properties.add(property_id)
                        
                        # Format as clean summary
                        summary = f"Property: {address}"
                        if floor:
                            summary += f", Floor {floor}"
                        if suite:
                            summary += f", Suite {suite}"
                        if monthly_rent:
                            summary += f" | Monthly Rent: {monthly_rent}"
                        if size:
                            summary += f" | Size: {size} sq ft"
                        if rent_rate:
                            summary += f" | Rate/sqft: {rent_rate}"
                        
                        property_summaries.append(summary)
        
        return '\n'.join(property_summaries[:100])  # Limit to 100 properties to avoid overwhelming the LLM
    
    async def _generate_final_answer(self, query: str, steps: List[ReActStep], all_data: List[Any], conversation_history: List = None) -> str:
        """Generate the final answer based on all gathered information."""
        
        # Extract and summarize property data instead of dumping everything
        property_summaries = self._extract_property_summaries(all_data)
        
        # Build conversation context if available
        conversation_context = ""
        if conversation_history:
            recent_messages = conversation_history[-6:]  # Last 6 messages for context
            context_parts = []
            for msg in recent_messages:
                context_parts.append(f"{msg.role}: {msg.content}")
            conversation_context = f"""
CONVERSATION CONTEXT:
{chr(10).join(context_parts)}

"""
        
        # Create enhanced prompt for numerical analysis
        if self._is_superlative_query(query):
            prompt = f"""
You are a real estate assistant analyzing property data to answer: "{query}"

{conversation_context}PROPERTY DATA SUMMARY:
{property_summaries}

INSTRUCTIONS:
1. Analyze the property data above to find the answer to the user's query
2. For superlative queries (highest/lowest/maximum/minimum):
   - Extract monthly rent values from all properties
   - Convert rent values to numbers (remove $ and commas)
   - Find the TRUE highest/lowest value
   - Return ONLY the specific property details for the answer

3. CONTEXT UNDERSTANDING:
   - If the query references "that address", "this property", etc., use the conversation context above
   - Extract the specific property being referenced from previous messages
   - Provide information about that exact property

4. RESPONSE FORMAT:
   - Provide ONLY the final answer in bullet points
   - Include: Address, Floor, Suite, Size, Monthly Rent, Rate per sq ft
   - Do NOT show your analysis process or list all properties
   - Do NOT include raw data or lengthy explanations

5. ANSWER ONLY: {query}
"""
        else:
            prompt = f"""
Based on the property data below, provide a concise answer to: "{query}"

{conversation_context}PROPERTY DATA:
{property_summaries}

INSTRUCTIONS:
- Provide ONLY the final answer in bullet points
- Include relevant property details
- Do NOT show analysis process or raw data
- Keep response focused and concise
- If the query references previous conversation, use the conversation context above
"""
        
        try:
            # Convert prompt to message format expected by Gemini service
            messages = [{"role": "user", "content": prompt}]
            response = await self.gemini_service.generate_response(messages)
            
            # Format the final answer to ensure it's clean and properly structured
            formatted_response = self._format_final_answer(response, query)
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return f"I found the property data but encountered an error generating the final answer: {str(e)}"
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean the LLM response to ensure it's concise and doesn't include raw data."""
        
        # First, remove formatting markers like asterisks
        response = response.replace('**', '').replace('*', '')
        response = response.replace('###', '').replace('#', '')
        response = response.replace('||', '').replace('|', '')
        response = response.replace('--', '-').replace('- -', '-')
        
        # Remove any lines that look like raw data dumps
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip lines that look like raw data or analysis process
            if any(skip_phrase in line.lower() for skip_phrase in [
                'property_unit_',
                'analyzing',
                'extracting',
                'converting',
                'step by step',
                'analysis:',
                'process:',
                'here are all',
                'listing all',
                'property data summary',
                'instructions:',
                'response format:',
                'answer only:',
                'based on the property data'
            ]):
                continue
            
            # Skip empty lines and lines with just separators
            if not line or line in ['---', '***', '===', '...']:
                continue
            
            # Keep lines that are actual answer content
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or 
                        line.startswith('1.') or line.startswith('2.') or
                        any(word in line.lower() for word in ['address:', 'floor:', 'suite:', 'rent:', 'size:', 'rate:']) or
                        'property' in line.lower() or
                        '$' in line or
                        'sq' in line.lower() or
                        'square' in line.lower()):
                cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines)
        
        # If cleaning removed too much, return a more conservative clean
        if len(cleaned_response.strip()) < 50:
            # Try a more conservative approach - just remove obvious raw data
            conservative_lines = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not any(skip in line.lower() for skip in ['property_unit_', 'analyzing', 'extracting']):
                    conservative_lines.append(line)
            
            conservative_response = '\n'.join(conservative_lines)
            if len(conservative_response.strip()) >= 50:
                return conservative_response
            else:
                # Last resort - return truncated original but still remove asterisks
                truncated = response[:500] + "..." if len(response) > 500 else response
                return truncated.replace('**', '').replace('*', '')
        
        return cleaned_response
    
    def _format_final_answer(self, raw_response: str, query: str) -> str:
        """Format the final answer to ensure it's properly structured."""
        
        # Clean the response first
        cleaned = self._clean_llm_response(raw_response)
        
        # If it's a superlative query, ensure we have a clear structure
        if self._is_superlative_query(query):
            lines = cleaned.split('\n')
            formatted_lines = []
            
            # Look for the actual answer in the cleaned response
            for line in lines:
                line = line.strip()
                if line and (
                    'highest' in line.lower() or 'lowest' in line.lower() or 
                    'maximum' in line.lower() or 'minimum' in line.lower() or
                    line.startswith('•') or line.startswith('-') or line.startswith('*') or
                    any(word in line.lower() for word in ['address:', 'floor:', 'suite:', 'rent:', 'size:'])
                ):
                    formatted_lines.append(line)
            
            if formatted_lines:
                final_result = '\n'.join(formatted_lines)
                # Ensure no asterisks remain
                return final_result.replace('**', '').replace('*', '')
        
        # Ensure no asterisks remain in the final output
        return cleaned.replace('**', '').replace('*', '')
    
    def _is_superlative_query(self, query: str) -> bool:
        """Check if the query is asking for superlative (max/min) values."""
        superlative_patterns = [
            r'\b(maximum|minimum|highest|lowest|largest|smallest|most|least)\b',
            r'\b(max|min)\b',
            r'\b(top|bottom)\s+\d+\b',
            r'\b(best|worst)\b',
            r'\b(which\s+property\s+has\s+(?:highest|lowest|maximum|minimum))\b'
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in superlative_patterns)
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if the query involves comparisons."""
        comparison_patterns = [
            r'\b(greater|less|more|fewer)\s+than\b',
            r'\b(above|below|over|under)\b',
            r'\b(between|from)\s+\d+\s+(and|to)\s+\d+\b',
            r'\b(compare|comparison)\b'
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in comparison_patterns)
    
    def _is_specific_property_query(self, query: str) -> bool:
        """Check if the query is asking about a specific property."""
        # Check for address patterns
        address_patterns = [
            r'\b\d+\s+[NSEW]\s+\d+\w*\s+(St|Street|Ave|Avenue|Rd|Road)\b',
            r'\b\d+\s+\w+\s+(St|Street|Ave|Avenue|Rd|Road)\b'
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in address_patterns)
    
    def _extract_superlative_type(self, query: str) -> str:
        """Extract the type of superlative being asked for."""
        query_lower = query.lower()
        
        if 'rent' in query_lower:
            if any(word in query_lower for word in ['maximum', 'highest', 'most', 'max']):
                return "highest monthly rent"
            elif any(word in query_lower for word in ['minimum', 'lowest', 'least', 'min']):
                return "lowest monthly rent"
        
        if 'size' in query_lower:
            if any(word in query_lower for word in ['maximum', 'highest', 'largest', 'biggest']):
                return "largest size"
            elif any(word in query_lower for word in ['minimum', 'lowest', 'smallest']):
                return "smallest size"
        
        return "superlative value" 