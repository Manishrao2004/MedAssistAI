"""
FAISS Vector Store.

Manages a FAISS index for storing and searching document embeddings.
Uses IndexFlatIP (inner product) on L2-normalized vectors for cosine similarity.
"""

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


class FAISSStore:
    """
    FAISS-based vector store for semantic search.

    Stores embeddings alongside metadata (question, answer, source, etc.)
    and supports fast nearest-neighbor retrieval.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize the vector store.

        Args:
            dimension: Embedding vector dimension (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict[str, Any]] = []
        logger.info("FAISSStore initialized (dimension=%d)", dimension)

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """
        Build a FAISS index from embeddings and associated metadata.

        Args:
            embeddings: numpy array of shape (n, dimension).
            metadata: List of metadata dicts, one per embedding.
                      Each dict should contain: question, chunk_text,
                      full_answer, source, focus_area, doc_id, chunk_id.

        Raises:
            ValueError: If embeddings and metadata lengths don't match.
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                f"count mismatch."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"expected dimension {self.dimension}."
            )

        logger.info(
            "Building FAISS index with %d vectors (dim=%d) ...",
            len(embeddings),
            self.dimension,
        )

        # IndexFlatIP = exact inner product search
        # With L2-normalized vectors, inner product == cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

        logger.info("FAISS index built. Total vectors: %d", self.index.ntotal)

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            index_path: Path for the FAISS index file.
            metadata_path: Path for the pickled metadata file.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Build the index first.")

        # Create directories
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)
        logger.info("FAISS index saved to: %s", index_path)

        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Metadata saved to: %s (%d entries)", metadata_path, len(self.metadata))

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load a FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the pickled metadata file.

        Raises:
            FileNotFoundError: If either file doesn't exist.
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(index_path)
        logger.info(
            "FAISS index loaded: %d vectors (dim=%d)",
            self.index.ntotal,
            self.index.d,
        )

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info("Metadata loaded: %d entries", len(self.metadata))

        # Update dimension from loaded index
        self.dimension = self.index.d

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the FAISS index for the most similar vectors.

        Args:
            query_embedding: Query vector of shape (1, dimension).
            top_k: Number of top results to return.

        Returns:
            List of result dicts, each containing metadata fields plus
            'similarity_score' (cosine similarity).

        Raises:
            RuntimeError: If no index is loaded.
        """
        if self.index is None:
            raise RuntimeError(
                "No FAISS index loaded. Run build_index() or load() first."
            )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), top_k
        )

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots
            result = dict(self.metadata[idx])
            result["similarity_score"] = float(score)
            results.append(result)

        logger.debug("Search returned %d results (top_k=%d)", len(results), top_k)
        return results

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index else 0
