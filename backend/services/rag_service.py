import os
import logging
from typing import List, Dict, Any, Iterator, Generator, Callable
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from config.settings import settings
from services.embeddings import VectorStoreService
from services.reranker import RerankerService
from services.document_processor import DocumentProcessor
from sqlalchemy.orm import Session
from models.database import Document

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-based question answering"""
    
    def __init__(
        self,
        vector_store: VectorStoreService,
        reranker: RerankerService,
        doc_processor: DocumentProcessor
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.doc_processor = doc_processor
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.llm_api_key,
            openai_api_base=settings.llm_api_base,
            temperature=0.7,
            streaming=True
        )
        # Non-streaming LLM for query generation
        self.llm_sync = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.llm_api_key,
            openai_api_base=settings.llm_api_base,
            temperature=0.7,
            streaming=False
        )

    def generate_query_variations(self, original_query: str) -> List[str]:
        """Generate 3 different query variations using LLM."""
        messages = [
            SystemMessage(content=(
                "You are a query expansion assistant. Given a user question, generate exactly 3 different "
                "variations of the question that might help find relevant information. Each variation should "
                "approach the question from a different angle or use different keywords.\n\n"
                "Return ONLY the 3 queries, one per line, without numbering or bullets."
            )),
            HumanMessage(content=f"Original question: {original_query}")
        ]
        
        response = self.llm_sync.invoke(messages)
        variations = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        # Return up to 3 variations, or pad with original if needed
        result = variations[:3]
        while len(result) < 3:
            result.append(original_query)
        return result

    def retrieve_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve chunks for a single query."""
        return self.vector_store.search(query, top_k=settings.top_k_retrieval)

    def _inject_metadata_chunks(
        self,
        chunks: List[Dict[str, Any]],
        seen_chunk_keys: set,
        emit_thinking: Callable = None
    ) -> List[Dict[str, Any]]:
        """
        Inject metadata chunks for all documents found in the search results.
        This ensures metadata (author, title, etc.) is always considered in reranking.
        """
        # Get unique doc_ids from search results
        doc_ids = list(set(chunk.get('doc_id') for chunk in chunks if chunk.get('doc_id')))
        
        if not doc_ids:
            return chunks
        
        # Track docs that already have explicit metadata chunks present
        docs_with_metadata = {
            chunk.get('doc_id')
            for chunk in chunks
            if chunk.get('section') == 'Document Metadata'
        }
        
        # Fetch metadata chunks for these documents
        metadata_chunks = self.vector_store.get_metadata_chunks_for_docs(doc_ids)
        
        if not metadata_chunks:
            return chunks
        
        # Add metadata chunks that aren't already in results
        injected_count = 0
        for meta_chunk in metadata_chunks:
            doc_id = meta_chunk.get('doc_id')
            if doc_id in docs_with_metadata:
                continue  # Already have metadata chunk for this doc
            chunk_key = f"meta_{doc_id}_{meta_chunk.get('chunk_id')}"
            if chunk_key not in seen_chunk_keys:
                meta_chunk['metadata_priority'] = True
                seen_chunk_keys.add(chunk_key)
                chunks.append(meta_chunk)
                docs_with_metadata.add(doc_id)
                injected_count += 1
        
        if emit_thinking and injected_count > 0:
            emit_thinking("metadata_injection", f"Injected {injected_count} metadata chunks for {len(doc_ids)} documents")
        
        return chunks

    def _search_with_queries(
        self,
        queries: List[str],
        seen_chunk_keys: set,
        emit_thinking: Callable,
        round_name: str = ""
    ) -> tuple[List[Dict[str, Any]], set]:
        """Execute search for multiple queries and collect unique chunks."""
        all_chunks: List[Dict[str, Any]] = []
        
        for i, query in enumerate(queries):
            prefix = f"{round_name} " if round_name else ""
            display_query = f"\"{query[:80]}...\"" if len(query) > 80 else f"\"{query}\""
            emit_thinking("searching", f"{prefix}Query {i+1}: {display_query}")
            
            chunks = self.retrieve_for_query(query)
            new_chunks = 0
            for chunk in chunks:
                chunk_key = f"{chunk.get('doc_id')}_{chunk.get('chunk_id')}"
                if chunk_key not in seen_chunk_keys:
                    seen_chunk_keys.add(chunk_key)
                    all_chunks.append(chunk)
                    new_chunks += 1
            emit_thinking("search_complete", f"{prefix}Query {i+1}: {len(chunks)} results, {new_chunks} new unique chunks")
        
        return all_chunks, seen_chunk_keys

    def multi_query_retrieve_and_rerank(
        self, 
        original_query: str, 
        db: Session,
        on_thinking: Callable[[str], None] | None = None
    ) -> tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Multi-query retrieval with iterative retry for poor results.
        
        Strategy:
        - Round 1: 3 query variations from original
        - Round 2: If quality < 0.4, try 3 alternative formulations
        - Round 3: Trigger when Round 2 improves the score but still leaves quality < 0.5
        
        Returns:
            Tuple of (parent_contexts, source_descriptions, thinking_steps)
        """
        thinking_steps: List[Dict[str, Any]] = []
        seen_chunk_keys: set = set()
        all_accumulated_chunks: List[Dict[str, Any]] = []
        
        # Quality thresholds
        MIN_ACCEPTABLE_SCORE = 0.4  # Below this, retry with new queries
        GOOD_SCORE = 0.5  # Above this, no more retries needed
        
        def emit_thinking(step_type: str, message: str, details: Any = None):
            step = {"type": step_type, "message": message, "details": details}
            thinking_steps.append(step)
            if on_thinking:
                on_thinking(step)
        
        emit_thinking("start", "Starting iterative multi-query retrieval...")
        
        # ==================== ROUND 1 ====================
        emit_thinking("round1_start", "📍 Round 1: Generating 3 query variations...")
        query_variations = self.generate_query_variations(original_query)
        emit_thinking("queries_generated", "Generated queries", query_variations)
        
        round1_chunks, seen_chunk_keys = self._search_with_queries(
            query_variations, seen_chunk_keys, emit_thinking, "Round 1"
        )
        all_accumulated_chunks.extend(round1_chunks)
        
        # Inject metadata chunks for found documents (author, title, etc.)
        all_accumulated_chunks = self._inject_metadata_chunks(
            all_accumulated_chunks, seen_chunk_keys, emit_thinking
        )
        
        emit_thinking("round1_dedup", f"Round 1 total: {len(all_accumulated_chunks)} chunks (incl. metadata)")
        
        if not all_accumulated_chunks:
            emit_thinking("round1_no_results", "No results in Round 1, proceeding to Round 2...")
            round1_best_score = 0.0
        else:
            # Rerank Round 1 results
            emit_thinking("round1_reranking", f"Reranking {len(all_accumulated_chunks)} chunks...")
            reranked = self.reranker.rerank(original_query, all_accumulated_chunks, top_k=settings.top_k_rerank)
            round1_best_score = reranked[0].get('rerank_score', 0) if reranked else 0
            
            emit_thinking("round1_score", f"Round 1 best score: {round1_best_score:.3f}",
                         [{"text": c['text'][:80], "score": c.get('rerank_score', 0)} for c in reranked[:3]])
            
            # If good enough, return early
            if round1_best_score >= GOOD_SCORE:
                emit_thinking("round1_success", f"✅ Good quality results (score: {round1_best_score:.3f}), skipping additional rounds")
                parent_contexts, sources = self._load_parents_from_chunks(reranked, db)
                emit_thinking("complete", f"Retrieved {len(parent_contexts)} contexts")
                return parent_contexts, sources, thinking_steps
        
        # ==================== ROUND 2 ====================
        if round1_best_score < MIN_ACCEPTABLE_SCORE:
            emit_thinking("round2_start", f"📍 Round 2: Score {round1_best_score:.3f} < {MIN_ACCEPTABLE_SCORE}, trying alternative formulations...")
            
            # Generate completely different queries
            messages = [
                SystemMessage(content=(
                    "The previous search did not find good results. Generate 3 COMPLETELY DIFFERENT "
                    "formulations of the question. Try:\n"
                    "1. Using synonyms and related terms\n"
                    "2. Breaking down into sub-questions\n"
                    "3. Asking from a different perspective\n\n"
                    "Return ONLY the 3 queries, one per line, without numbering or bullets."
                )),
                HumanMessage(content=f"Original question: {original_query}")
            ]
            
            response = self.llm_sync.invoke(messages)
            round2_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()][:3]
            emit_thinking("round2_queries", "Generated alternative queries", round2_queries)
            
            round2_chunks, seen_chunk_keys = self._search_with_queries(
                round2_queries, seen_chunk_keys, emit_thinking, "Round 2"
            )
            all_accumulated_chunks.extend(round2_chunks)
            
            # Inject metadata chunks for any new documents found
            all_accumulated_chunks = self._inject_metadata_chunks(
                all_accumulated_chunks, seen_chunk_keys, emit_thinking
            )
            
            emit_thinking("round2_dedup", f"Round 2 total: {len(all_accumulated_chunks)} chunks (incl. metadata)")
            
            if all_accumulated_chunks:
                emit_thinking("round2_reranking", f"Reranking all {len(all_accumulated_chunks)} accumulated chunks...")
                reranked = self.reranker.rerank(original_query, all_accumulated_chunks, top_k=settings.top_k_rerank)
                round2_best_score = reranked[0].get('rerank_score', 0) if reranked else 0
                
                improvement = round2_best_score - round1_best_score
                emit_thinking("round2_score", 
                             f"Round 2 best score: {round2_best_score:.3f} (improvement: +{improvement:.3f})",
                             [{"text": c['text'][:80], "score": c.get('rerank_score', 0)} for c in reranked[:3]])
                
                # ==================== ROUND 3 (optional) ====================
                # Only if: improved but still not great, and we have some context to work with
                if (round2_best_score >= GOOD_SCORE):
                    emit_thinking("round2_success", f"✅ Good quality achieved (score: {round2_best_score:.3f})")
                elif (improvement > 0 and round2_best_score < GOOD_SCORE):
                    emit_thinking("round3_start", 
                                 f"📍 Round 3: Improvement detected (+{improvement:.3f}), refining based on best results...")
                    
                    # Use the best result to generate refined queries
                    best_context = reranked[0]['text'][:500] if reranked else ""
                    
                    messages = [
                        SystemMessage(content=(
                            "Based on a partially relevant result, generate 3 more specific queries that might "
                            "find better information. The queries should be related to what was found but more targeted.\n\n"
                            "Return ONLY the 3 queries, one per line, without numbering or bullets."
                        )),
                        HumanMessage(content=f"Original question: {original_query}\n\nPartially relevant content found:\n{best_context}")
                    ]
                    
                    response = self.llm_sync.invoke(messages)
                    round3_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()][:3]
                    emit_thinking("round3_queries", "Generated refined queries", round3_queries)
                    
                    round3_chunks, seen_chunk_keys = self._search_with_queries(
                        round3_queries, seen_chunk_keys, emit_thinking, "Round 3"
                    )
                    all_accumulated_chunks.extend(round3_chunks)
                    
                    # Inject metadata chunks for any new documents found
                    all_accumulated_chunks = self._inject_metadata_chunks(
                        all_accumulated_chunks, seen_chunk_keys, emit_thinking
                    )
                    
                    emit_thinking("round3_dedup", f"Round 3 total: {len(all_accumulated_chunks)} chunks (incl. metadata)")
                    
                    # Final rerank with all chunks
                    emit_thinking("round3_reranking", f"Final reranking of all {len(all_accumulated_chunks)} chunks...")
                    reranked = self.reranker.rerank(original_query, all_accumulated_chunks, top_k=settings.top_k_rerank)
                    round3_best_score = reranked[0].get('rerank_score', 0) if reranked else 0
                    
                    emit_thinking("round3_score", f"Final best score: {round3_best_score:.3f}",
                                 [{"text": c['text'][:80], "score": c.get('rerank_score', 0)} for c in reranked[:3]])
                else:
                    emit_thinking("round2_final", f"No improvement after Round 2, using best available results")
            else:
                emit_thinking("no_results_final", "No results found after 6 queries (Round 1 + Round 2)")
                return [], [], thinking_steps
        else:
            # Round 1 was acceptable but not great
            emit_thinking("round1_acceptable", f"Acceptable quality (score: {round1_best_score:.3f}), no retry needed")
            reranked = self.reranker.rerank(original_query, all_accumulated_chunks, top_k=settings.top_k_rerank)
        
        # Load final results
        if not reranked:
            emit_thinking("no_results", "No results to return")
            return [], [], thinking_steps
        
        emit_thinking("loading_parents", "Loading parent documents...")
        parent_contexts, sources = self._load_parents_from_chunks(reranked, db)
        
        total_queries = len(query_variations) + len(thinking_steps)  # Approximate
        emit_thinking("complete", f"✅ Completed with {len(parent_contexts)} contexts")
        
        return parent_contexts, sources, thinking_steps

    def _load_parents_from_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        db: Session
    ) -> tuple[List[str], List[Dict[str, str]]]:
        """Load parent documents from reranked chunks with enhanced metadata."""
        parent_contexts: List[str] = []
        sources: List[Dict[str, str]] = []
        seen_parents = set()
        
        for chunk in chunks:
            doc_id = chunk['doc_id']
            parent_id = chunk.get('parent_id')
            
            parent_key = f"{doc_id}_{parent_id}"
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)
            
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document or not document.pickle_path:
                continue
            
            parent_text = self.doc_processor.load_parent_document(
                document.pickle_path, 
                parent_id
            )
            
            if parent_text:
                parent_contexts.append(parent_text)
                
                # Build enhanced source info with metadata
                doc_name = chunk.get('document_name') or document.filename
                section = chunk.get('section', '')
                rerank_score = chunk.get('rerank_score', 0)
                
                # Create descriptive label
                label_parts = [doc_name]
                if section and section != "Unknown" and section != "Introduction":
                    label_parts.append(f"§ {section}")
                label_parts.append(f"(Relevanz: {rerank_score:.0%})")
                
                sources.append({
                    "label": " - ".join(label_parts),
                    "content": parent_text.strip(),
                    "document": doc_name,
                    "section": section,
                    "score": f"{rerank_score:.3f}"
                })
        
        return parent_contexts, sources
    
    def retrieve_and_rerank(self, query: str, db: Session) -> tuple[List[str], List[Dict[str, str]]]:
        """
        Retrieve relevant chunks, rerank them, and return parent documents
        
        Returns:
            Tuple of (parent_contexts, source_descriptions)
        """
        # Step 1: Retrieve child chunks from vector store (hybrid search)
        retrieved_chunks = self.vector_store.search(query, top_k=settings.top_k_retrieval)
        
        if not retrieved_chunks:
            return [], []
        
        # Step 2: Rerank chunks with dynamic threshold
        reranked_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=settings.top_k_rerank)
        
        # Step 3: Load parent documents with enhanced metadata
        return self._load_parents_from_chunks(reranked_chunks, db)
    
    def _build_messages(self, query: str, contexts: List[str], chat_history: List[Dict[str, str]] | None = None) -> List[Any]:
        """Build messages for the LLM including context and history."""
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        messages: List[Any] = [
            SystemMessage(content=(
                "You are a helpful assistant that answers questions based on the provided context. "
                "Use the context to answer the question accurately. If the context doesn't contain "
                "enough information to answer the question, say so."
            ))
        ]

        if chat_history:
            for msg in chat_history[-5:]:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))

        user_message = f"Context:\n{context_str}\n\nQuestion: {query}"
        messages.append(HumanMessage(content=user_message))
        return messages

    def generate_answer(self, query: str, contexts: List[str], chat_history: List[Dict[str, str]] = None) -> str:
        """
        Generate answer using LLM based on retrieved contexts
        
        Args:
            query: User question
            contexts: List of relevant context strings
            chat_history: Optional list of previous messages
            
        Returns:
            Generated answer
        """
        messages = self._build_messages(query, contexts, chat_history)
        response = self.llm.invoke(messages)
        return response.content

    def generate_answer_stream(
        self,
        query: str,
        contexts: List[str],
        chat_history: List[Dict[str, str]] | None = None
    ) -> Iterator[str]:
        """Stream answer tokens from the LLM."""
        messages = self._build_messages(query, contexts, chat_history)
        for chunk in self.llm.stream(messages):
            text = ""
            if hasattr(chunk, "message") and getattr(chunk.message, "content", None):
                text = chunk.message.content
            elif hasattr(chunk, "content") and chunk.content:
                text = chunk.content
            elif hasattr(chunk, "delta") and isinstance(chunk.delta, dict):
                text = chunk.delta.get("content", "")

            if text:
                yield text
    
    def query(self, query: str, db: Session, chat_history: List[Dict[str, str]] = None) -> tuple[str, List[str]]:
        """
        Main RAG query method
        
        Returns:
            Tuple of (answer, sources)
        """
        # Retrieve and rerank
        contexts, sources = self.retrieve_and_rerank(query, db)
        
        if not contexts:
            return "I couldn't find relevant information in the documents to answer your question.", []
        
        # Generate answer
        answer = self.generate_answer(query, contexts, chat_history)
        
        return answer, sources
