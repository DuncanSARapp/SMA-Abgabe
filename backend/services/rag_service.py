from __future__ import annotations

import logging
from typing import List, Dict, Any, Iterator, Callable, Optional, Tuple, Set

from cachetools import TTLCache
from langchain.schema import HumanMessage, SystemMessage, AIMessage  # type: ignore[import-not-found]
from sqlalchemy.orm import Session

from sqlalchemy.orm.attributes import InstrumentedAttribute

from .settings import settings
from .document_processor import DocumentProcessor, load_parent_document
from .llm_factory import create_llm
from .reranker import RerankerService
from .vector_store import VectorStoreService
from db.models import Document

logger = logging.getLogger(__name__)

QUERY_EXPANSION_PROMPT = """You are a query expansion assistant. Given a user question, generate exactly 3 different variations of the question that might help find relevant information. Each variation should approach the question from a different angle or use different keywords.

Return ONLY the 3 queries, one per line, without numbering or bullets."""

ALTERNATIVE_QUERIES_PROMPT = """The previous search did not find good results. Generate 3 COMPLETELY DIFFERENT formulations of the question. Try:
1. Using synonyms and related terms
2. Breaking down into sub-questions
3. Asking from a different perspective

Return ONLY the 3 queries, one per line, without numbering or bullets."""

REFINED_QUERIES_PROMPT = """Based on a partially relevant result, generate 3 more specific queries that might find better information. The queries should be related to what was found but more targeted.

Return ONLY the 3 queries, one per line, without numbering or bullets."""

ANSWER_GENERATION_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context. Use the context to answer the question accurately. If the context doesn't contain enough information to answer the question, say so."""

MIN_ACCEPTABLE_SCORE = 0.4
GOOD_SCORE = 0.5
MAX_CHAT_HISTORY = 5


def _build_messages(
        query: str,
        contexts: List[str],
        chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Any]:
    context_str = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])

    messages: List[Any] = [
        SystemMessage(content=ANSWER_GENERATION_SYSTEM_PROMPT)
    ]

    if chat_history:
        for msg in chat_history[-MAX_CHAT_HISTORY:]:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))

    user_message = f"Context:\n{context_str}\n\nQuestion: {query}"
    messages.append(HumanMessage(content=user_message))

    return messages


def _expand_parent_neighbors(
        base_entries: List[Dict[str, Any]],
        doc_cache: Dict[int, Optional[Document]],
        doc_order_map: Dict[int, int],
        seen_parents: Set[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    limit = max(settings.top_k_rerank, 1)

    if not settings.enable_neighbor_expansion or settings.neighbor_expansion_window <= 0:
        return base_entries[:limit]

    if len(base_entries) >= limit:
        return base_entries[:limit]

    expanded = list(base_entries)
    window = settings.neighbor_expansion_window
    base_snapshot = list(base_entries)

    for entry in base_snapshot:
        if len(expanded) >= limit:
            break

        document = entry.get('document') or doc_cache.get(entry['doc_id'])
        if not document or not document.pickle_path:
            continue

        if len(expanded) < limit:
            prev_key = (entry['doc_id'], entry['parent_id'] - 1)
            if prev_key[1] >= 0 and prev_key not in seen_parents:
                prev_text = load_parent_document(document.pickle_path, prev_key[1])  # type: ignore[arg-type]
                if prev_text:
                    expanded.append({
                        'doc_id': entry['doc_id'],
                        'parent_id': prev_key[1],
                        'document': document,
                        'document_name': entry.get('document_name'),
                        'section': entry.get('section', ''),
                        'position': entry.get('position', ''),
                        'score': max(entry.get('score', 0.0) * 0.95, 0.0),
                        'text': prev_text,
                        'is_neighbor': True,
                        'neighbor_direction': -1,
                        'doc_order': doc_order_map.setdefault(entry['doc_id'], len(doc_order_map)),
                    })
                    seen_parents.add(prev_key)

        for offset in range(1, window + 1):
            if len(expanded) >= limit:
                break

            next_key = (entry['doc_id'], entry['parent_id'] + offset)
            if next_key in seen_parents:
                continue

            next_text = load_parent_document(document.pickle_path, next_key[1])  # type: ignore[arg-type]
            if not next_text:
                continue

            expanded.append({
                'doc_id': entry['doc_id'],
                'parent_id': next_key[1],
                'document': document,
                'document_name': entry.get('document_name'),
                'section': entry.get('section', ''),
                'position': entry.get('position', ''),
                'score': max(entry.get('score', 0.0) * 0.98, 0.0),
                'text': next_text,
                'is_neighbor': True,
                'neighbor_direction': 1,
                'doc_order': doc_order_map.setdefault(entry['doc_id'], len(doc_order_map)),
            })
            seen_parents.add(next_key)

    reorder = any(entry.get('is_neighbor') for entry in expanded)
    if reorder:
        expanded.sort(key=lambda item: (
            item.get('doc_order', 0),
            item.get('parent_id', 0)
        ))
    else:
        expanded.sort(key=lambda item: item.get('score', 0.0), reverse=True)

    return expanded[:limit]


def _load_parents_from_chunks(
        chunks: List[Dict[str, Any]],
        db: Session
) -> Tuple[List[str], List[Dict[str, str]]]:
    entries: List[Dict[str, Any]] = []
    seen_parents: Set[Tuple[int, int]] = set()
    doc_cache: Dict[int, Optional[Document]] = {}
    doc_order_map: Dict[int, int] = {}
    limit = max(settings.top_k_rerank, 1)

    for chunk in chunks:
        doc_id = chunk.get('doc_id')
        parent_id = chunk.get('parent_id')
        if doc_id is None or parent_id is None:
            continue

        parent_key = (doc_id, parent_id)
        if parent_key in seen_parents:
            continue

        document = doc_cache.get(doc_id)
        if document is None:
            document = db.query(Document).filter(Document.id == doc_id).first()  # type: ignore[arg-type]
            doc_cache[doc_id] = document

        if not document or not document.pickle_path:
            continue

        parent_text = load_parent_document(
            document.pickle_path,  # type: ignore[arg-type]
            parent_id,
        )
        if not parent_text:
            continue

        entry = {
            'doc_id': doc_id,
            'parent_id': parent_id,
            'document': document,
            'document_name': chunk.get('document_name') or document.filename,
            'section': chunk.get('section', ''),
            'position': chunk.get('position', ''),
            'score': chunk.get('rerank_score', 0.0),
            'text': parent_text,
            'is_neighbor': False,
            'neighbor_direction': 0,
            'doc_order': doc_order_map.setdefault(doc_id, len(doc_order_map)),
        }

        entries.append(entry)
        seen_parents.add(parent_key)

        if len(entries) >= limit:
            break

    final_entries = _expand_parent_neighbors(entries, doc_cache, doc_order_map, seen_parents)

    parent_contexts = [entry['text'] for entry in final_entries]
    sources: List[Dict[str, str]] = []

    for entry in final_entries:
        section = entry.get('section', '')
        label_parts = [entry.get('document_name', 'Document')]
        if section and section not in ("Unknown", "Introduction"):
            label_parts.append(f"§ {section}")

        if entry.get('is_neighbor'):
            direction = entry.get('neighbor_direction', 0)
            if direction > 0:
                label_parts.append("Folgeabschnitt")
            elif direction < 0:
                label_parts.append("Vorabschnitt")
            else:
                label_parts.append("Nachbarabschnitt")

        score = entry.get('score', 0.0)
        label_parts.append(f"(Relevanz: {score:.0%})")

        sources.append({
            "label": " - ".join(label_parts),
            "content": entry['text'].strip(),
            "document": entry.get('document_name', 'Document'),
            "section": section,
            "score": f"{score:.3f}"
        })

    return parent_contexts, sources


class RAGService:
    def __init__(
            self,
            vector_store: VectorStoreService,
            reranker: RerankerService,
            doc_processor: DocumentProcessor
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.doc_processor = doc_processor

        self.llm = create_llm(streaming=True, max_tokens=4096)
        self.llm_sync = create_llm(streaming=False, max_tokens=1024)

        self.query_expansion_cache = TTLCache(
            maxsize=settings.query_expansion_cache_size,
            ttl=settings.query_expansion_cache_ttl
        )
        logger.info(f"Query expansion cache enabled: {settings.query_expansion_cache_size} entries, "
                   f"TTL={settings.query_expansion_cache_ttl}s")

    def _generate_queries_from_llm(
        self,
        messages: List[Any],
        original_query: str,
        round_name: str
    ) -> List[str]:
        """Helper method to generate queries from LLM with error handling."""
        try:
            response = self.llm_sync.invoke(messages)
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()][:3]
            return queries if queries else [original_query]
        except Exception as exc:
            logger.warning(f"{round_name} query generation failed: {exc}")
            return [original_query]

    def generate_query_variations(self, original_query: str) -> List[str]:
        if original_query in self.query_expansion_cache:
            logger.debug(f"Query expansion cache hit for: {original_query[:50]}...")
            return self.query_expansion_cache[original_query]

        messages = [
            SystemMessage(content=QUERY_EXPANSION_PROMPT),
            HumanMessage(content=f"Original question: {original_query}")
        ]

        variations = self._generate_queries_from_llm(messages, original_query, "Query expansion")
        result = variations[:3]

        while len(result) < 3:
            result.append(original_query)

        # Cache the result
        self.query_expansion_cache[original_query] = result
        return result

    def retrieve_for_query(
            self,
            query: str,
            doc_collection_map: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        return self.vector_store.search(
            query,
            doc_collection_map,
            top_k=settings.top_k_retrieval
        )

    def _inject_metadata_chunks(
            self,
            chunks: List[Dict[str, Any]],
            seen_chunk_keys: Set[str],
            emit_thinking: Optional[Callable] = None,
            doc_collection_map: Optional[Dict[int, str]] = None
    ) -> List[Dict[str, Any]]:
        doc_ids = list(set(chunk.get('doc_id') for chunk in chunks if chunk.get('doc_id')))

        if not doc_ids or not doc_collection_map:
            return chunks

        docs_with_metadata = {
            chunk.get('doc_id')
            for chunk in chunks
            if chunk.get('section') == 'Document Metadata'
        }

        subset = {
            doc_id: doc_collection_map[doc_id]
            for doc_id in doc_ids
            if doc_id in doc_collection_map
        }
        metadata_chunks = self.vector_store.get_metadata_chunks_for_docs(subset)

        if not metadata_chunks:
            return chunks

        injected_count = 0
        for meta_chunk in metadata_chunks:
            doc_id = meta_chunk.get('doc_id')
            if doc_id in docs_with_metadata:
                continue

            chunk_key = f"meta_{doc_id}_{meta_chunk.get('chunk_id')}"
            if chunk_key not in seen_chunk_keys:
                meta_chunk['metadata_priority'] = True
                seen_chunk_keys.add(chunk_key)
                chunks.append(meta_chunk)
                docs_with_metadata.add(doc_id)
                injected_count += 1

        if emit_thinking and injected_count > 0:
            emit_thinking(
                "metadata_injection",
                f"Injected {injected_count} metadata chunks for {len(doc_ids)} documents"
            )

        return chunks

    def _search_with_queries(
            self,
            queries: List[str],
            seen_chunk_keys: Set[str],
            emit_thinking: Callable,
            round_name: str = "",
            doc_collection_map: Optional[Dict[int, str]] = None
    ) -> Tuple[List[Dict[str, Any]], Set[str]]:
        all_chunks: List[Dict[str, Any]] = []

        for i, query in enumerate(queries):
            prefix = f"{round_name} " if round_name else ""
            display_query = f'"{query[:80]}..."' if len(query) > 80 else f'"{query}"'
            emit_thinking("searching", f"{prefix}Query {i + 1}: {display_query}")

            if not doc_collection_map:
                break

            chunks = self.retrieve_for_query(query, doc_collection_map)
            new_chunks = 0

            for chunk in chunks:
                chunk_key = f"{chunk.get('doc_id')}_{chunk.get('chunk_id')}"
                if chunk_key not in seen_chunk_keys:
                    seen_chunk_keys.add(chunk_key)
                    all_chunks.append(chunk)
                    new_chunks += 1

            emit_thinking(
                "search_complete",
                f"{prefix}Query {i + 1}: {len(chunks)} results, {new_chunks} new unique chunks"
            )

        return all_chunks, seen_chunk_keys

    def multi_query_retrieve_and_rerank(
            self,
            original_query: str,
            db: Session,
            doc_collection_map: Dict[int, str],
            on_thinking: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]]]:
        thinking_steps: List[Dict[str, Any]] = []
        seen_chunk_keys: Set[str] = set()
        accumulated_chunks: List[Dict[str, Any]] = []

        def emit_thinking(step_type: str, message: str, details: Any = None):
            step = {"type": step_type, "message": message, "details": details}
            thinking_steps.append(step)
            if on_thinking:
                on_thinking(step)

        emit_thinking("start", "Starting iterative multi-query retrieval...")
        emit_thinking("round1_start", "Round 1: Generating 3 query variations...")

        query_variations = self.generate_query_variations(original_query)
        emit_thinking("queries_generated", "Generated queries", query_variations)

        if not doc_collection_map:
            emit_thinking("no_documents", "No active document collections selected")
            return [], [], thinking_steps

        round1_chunks, seen_chunk_keys = self._search_with_queries(
            query_variations, seen_chunk_keys, emit_thinking, "Round 1", doc_collection_map
        )
        accumulated_chunks.extend(round1_chunks)

        accumulated_chunks = self._inject_metadata_chunks(
            accumulated_chunks, seen_chunk_keys, emit_thinking, doc_collection_map
        )

        emit_thinking("round1_dedup", f"Round 1 total: {len(accumulated_chunks)} chunks (incl. metadata)")

        if not accumulated_chunks:
            emit_thinking("round1_no_results", "No results in Round 1, proceeding to Round 2...")
            round1_best_score = 0.0
        else:
            emit_thinking("round1_reranking", f"Reranking {len(accumulated_chunks)} chunks...")
            reranked = self.reranker.rerank(
                original_query, accumulated_chunks, top_k=settings.top_k_rerank
            )
            round1_best_score = reranked[0].get('rerank_score', 0) if reranked else 0

            emit_thinking(
                "round1_score",
                f"Round 1 best score: {round1_best_score:.3f}",
                [{"text": c['text'][:80], "score": c.get('rerank_score', 0)} for c in reranked[:3]]
            )

            if round1_best_score >= GOOD_SCORE:
                emit_thinking(
                    "round1_success",
                    f"Good quality results (score: {round1_best_score:.3f}), skipping additional rounds"
                )
                parent_contexts, sources = _load_parents_from_chunks(reranked, db)
                emit_thinking("complete", f"Retrieved {len(parent_contexts)} contexts")
                return parent_contexts, sources, thinking_steps

        if round1_best_score < MIN_ACCEPTABLE_SCORE:
            reranked = self._run_retry_round(
                original_query, accumulated_chunks, seen_chunk_keys,
                emit_thinking, doc_collection_map, round1_best_score
            )

        else:
            emit_thinking(
                "round1_acceptable",
                f"Acceptable quality (score: {round1_best_score:.3f}), no retry needed"
            )
            reranked = self.reranker.rerank(
                original_query, accumulated_chunks, top_k=settings.top_k_rerank
            )

        if not reranked:
            emit_thinking("no_results", "No results to return")
            return [], [], thinking_steps

        emit_thinking("loading_parents", "Loading parent documents...")
        parent_contexts, sources = _load_parents_from_chunks(reranked, db)

        emit_thinking("complete", f"Completed with {len(parent_contexts)} contexts")

        return parent_contexts, sources, thinking_steps

    def _run_retry_round(
            self,
            original_query: str,
            accumulated_chunks: List[Dict[str, Any]],
            seen_chunk_keys: Set[str],
            emit_thinking: Callable,
            doc_collection_map: Dict[int, str],
            round1_best_score: float
    ) -> List[Dict[str, Any]]:
        emit_thinking(
            "round2_start",
            f"Round 2: Score {round1_best_score:.3f} < {MIN_ACCEPTABLE_SCORE}, trying alternative formulations..."
        )

        messages = [
            SystemMessage(content=ALTERNATIVE_QUERIES_PROMPT),
            HumanMessage(content=f"Original question: {original_query}")
        ]

        round2_queries = self._generate_queries_from_llm(messages, original_query, "Round 2")

        emit_thinking("round2_queries", "Generated alternative queries", round2_queries)

        round2_chunks, seen_chunk_keys = self._search_with_queries(
            round2_queries, seen_chunk_keys, emit_thinking, "Round 2", doc_collection_map
        )
        accumulated_chunks.extend(round2_chunks)

        accumulated_chunks = self._inject_metadata_chunks(
            accumulated_chunks, seen_chunk_keys, emit_thinking, doc_collection_map
        )

        emit_thinking("round2_dedup", f"Round 2 total: {len(accumulated_chunks)} chunks (incl. metadata)")

        if not accumulated_chunks:
            emit_thinking("no_results_final", "No results found after 6 queries (Round 1 + Round 2)")
            return []

        emit_thinking("round2_reranking", f"Reranking all {len(accumulated_chunks)} accumulated chunks...")
        reranked = self.reranker.rerank(
            original_query, accumulated_chunks, top_k=settings.top_k_rerank
        )
        round2_best_score = reranked[0].get('rerank_score', 0) if reranked else 0

        improvement = round2_best_score - round1_best_score
        emit_thinking(
            "round2_score",
            f"Round 2 best score: {round2_best_score:.3f} (improvement: +{improvement:.3f})",
            [{"text": c['text'][:80], "score": c.get('rerank_score', 0)} for c in reranked[:3]]
        )

        if round2_best_score >= GOOD_SCORE:
            emit_thinking("round2_success", f"Good quality achieved (score: {round2_best_score:.3f})")
        elif improvement > 0 and round2_best_score < GOOD_SCORE:
            reranked = self._run_refinement_round(
                original_query, accumulated_chunks, reranked, seen_chunk_keys,
                emit_thinking, doc_collection_map, improvement
            )
        else:
            emit_thinking("round2_final", "No improvement after Round 2, using best available results")

        return reranked

    def _run_refinement_round(
            self,
            original_query: str,
            accumulated_chunks: List[Dict[str, Any]],
            reranked: List[Dict[str, Any]],
            seen_chunk_keys: Set[str],
            emit_thinking: Callable,
            doc_collection_map: Dict[int, str],
            improvement: float
    ) -> List[Dict[str, Any]]:
        emit_thinking(
            "round3_start",
            f"Round 3: Improvement detected (+{improvement:.3f}), refining based on best results..."
        )

        best_context = reranked[0]['text'][:500] if reranked else ""

        messages = [
            SystemMessage(content=REFINED_QUERIES_PROMPT),
            HumanMessage(
                content=f"Original question: {original_query}\n\nPartially relevant content found:\n{best_context}"
            )
        ]

        round3_queries = self._generate_queries_from_llm(messages, original_query, "Round 3")

        emit_thinking("round3_queries", "Generated refined queries", round3_queries)

        round3_chunks, seen_chunk_keys = self._search_with_queries(
            round3_queries, seen_chunk_keys, emit_thinking, "Round 3", doc_collection_map
        )
        accumulated_chunks.extend(round3_chunks)

        accumulated_chunks = self._inject_metadata_chunks(
            accumulated_chunks, seen_chunk_keys, emit_thinking, doc_collection_map
        )

        emit_thinking("round3_dedup", f"Round 3 total: {len(accumulated_chunks)} chunks (incl. metadata)")
        emit_thinking("round3_reranking", f"Final reranking of all {len(accumulated_chunks)} chunks...")

        reranked = self.reranker.rerank(
            original_query, accumulated_chunks, top_k=settings.top_k_rerank
        )
        round3_best_score = reranked[0].get('rerank_score', 0) if reranked else 0

        emit_thinking(
            "round3_score",
            f"Final best score: {round3_best_score:.3f}",
            [{"text": c['text'][:80], "score": c.get('rerank_score', 0)} for c in reranked[:3]]
        )

        return reranked

    def retrieve_and_rerank(
            self,
            query: str,
            db: Session,
            doc_collection_map: Dict[int, str]
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        logger.debug(f"retrieve_and_rerank called with query: '{query[:100]}...'")
        logger.debug(f"Document collection map: {doc_collection_map}")

        try:
            retrieved_chunks = self.vector_store.search(
                query,
                doc_collection_map,
                top_k=settings.top_k_retrieval
            )

            if not retrieved_chunks:
                logger.warning(f"No chunks retrieved for query: '{query[:100]}...'")
                return [], []

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks before reranking")

            reranked_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=settings.top_k_rerank)
            logger.info(f"Reranked to {len(reranked_chunks)} chunks")

            result = _load_parents_from_chunks(reranked_chunks, db)
            logger.info(f"Loaded {len(result[0])} parent contexts from chunks")

            return result

        except Exception as exc:
            logger.error(f"retrieve_and_rerank failed: {type(exc).__name__}: {exc}", exc_info=True)
            raise

    def generate_answer(
            self,
            query: str,
            contexts: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        messages = _build_messages(query, contexts, chat_history)
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error(f"Answer generation failed: {exc}")
            raise

    def generate_answer_stream(
            self,
            query: str,
            contexts: List[str],
            chat_history: list[dict[str, InstrumentedAttribute[str]]] = None
    ) -> Iterator[str]:
        messages = _build_messages(query, contexts, chat_history)

        try:
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as exc:
            logger.error(f"Streaming answer generation failed: {exc}")
            raise

    def query(
            self,
            query: str,
            db: Session,
            doc_collection_map: Dict[int, str],
            chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        try:
            logger.info(f"Processing query: '{query[:100]}...' with {len(doc_collection_map)} active documents")
            logger.debug(f"Active document collections: {doc_collection_map}")

            contexts, sources = self.retrieve_and_rerank(query, db, doc_collection_map)

            if not contexts:
                logger.warning(f"No contexts found for query: '{query[:100]}...'")
                return "I couldn't find relevant information in the documents to answer your question.", []

            logger.info(f"Found {len(contexts)} contexts for query")
            answer = self.generate_answer(query, contexts, chat_history)

            return answer, sources

        except Exception as exc:
            logger.error(
                f"Query failed: {type(exc).__name__}: {exc}\n"
                f"Query: '{query[:100]}...'\n"
                f"Document collections: {doc_collection_map}",
                exc_info=True
            )
            raise
