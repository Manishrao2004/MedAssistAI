"""
Semantic Retriever.

Combines the embedding encoder and FAISS vector store to provide
end-to-end semantic search over the MedQuAD knowledge base.
"""

from dataclasses import dataclass, field

from medical_chatbot.embeddings.encoder import EmbeddingEncoder
from medical_chatbot.vector_store.faiss_store import FAISSStore
from medical_chatbot.preprocessing.text_processor import TextProcessor
from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result from semantic search."""

    chunk_text: str
    source: str
    question: str
    focus_area: str
    similarity_score: float
    doc_id: int
    chunk_id: int
    full_answer: str = ""

    @property
    def confidence(self) -> str:
        """Human-readable confidence level based on similarity score."""
        if self.similarity_score >= 0.75:
            return "High"
        elif self.similarity_score >= 0.50:
            return "Medium"
        elif self.similarity_score >= 0.25:
            return "Low"
        return "Very Low"


class SemanticRetriever:
    """
    Semantic search retriever using embeddings + FAISS.

    Orchestrates query preprocessing, embedding generation, and FAISS search
    to retrieve the most relevant document chunks.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        faiss_store: FAISSStore,
        text_processor: TextProcessor,
        min_similarity: float = 0.25,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            encoder: Embedding encoder instance.
            faiss_store: FAISS vector store instance (must be loaded).
            text_processor: Text preprocessor instance.
            min_similarity: Minimum similarity threshold for results.
        """
        self.encoder = encoder
        self.faiss_store = faiss_store
        self.text_processor = text_processor
        self.min_similarity = min_similarity
        logger.info(
            "SemanticRetriever initialized (min_similarity=%.2f, index_size=%d)",
            min_similarity,
            faiss_store.size,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a query.

        Pipeline:
            1. Preprocess the query text.
            2. Generate query embedding.
            3. Search FAISS index.
            4. Filter by minimum similarity.
            5. Return structured results.

        Args:
            query: User's medical question.
            top_k: Number of top results to retrieve.

        Returns:
            List of RetrievalResult objects sorted by similarity (descending).

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Preprocess
        processed_query = self.text_processor.preprocess_query(query)
        logger.info("Processed query: '%s'", processed_query)

        # 2. Generate embedding
        query_embedding = self.encoder.encode_query(processed_query)

        # 3. Search FAISS
        raw_results = self.faiss_store.search(query_embedding, top_k=top_k)

        # 4. Filter and structure results
        results: list[RetrievalResult] = []
        for r in raw_results:
            score = r.get("similarity_score", 0.0)
            if score < self.min_similarity:
                continue

            results.append(
                RetrievalResult(
                    chunk_text=r.get("chunk_text", ""),
                    source=r.get("source", "Unknown"),
                    question=r.get("question", ""),
                    focus_area=r.get("focus_area", ""),
                    similarity_score=score,
                    doc_id=r.get("doc_id", -1),
                    chunk_id=r.get("chunk_id", -1),
                    full_answer=r.get("full_answer", ""),
                )
            )

        logger.info(
            "Retrieved %d results for query (top_k=%d, min_sim=%.2f)",
            len(results),
            top_k,
            self.min_similarity,
        )
        return results

    def retrieve_with_keywords(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[RetrievalResult], list[str]]:
        """
        Retrieve results and also extract keywords from the query.

        Args:
            query: User's medical question.
            top_k: Number of top results.

        Returns:
            Tuple of (retrieval results, extracted keywords).
        """
        keywords = self.text_processor.extract_keywords(query)
        results = self.retrieve(query, top_k=top_k)
        return results, keywords
