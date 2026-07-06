"""
Embedding Encoder using Sentence-Transformers.

Generates dense vector embeddings for text using the all-MiniLM-L6-v2 model.
Supports batch encoding and L2 normalization for cosine similarity via inner product.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingEncoder:
    """
    Generates text embeddings using a Sentence-Transformers model.

    The default model (all-MiniLM-L6-v2) produces 384-dimensional vectors
    that capture semantic meaning for similarity search.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ) -> None:
        """
        Initialize the embedding encoder.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
            normalize: If True, L2-normalize embeddings for cosine similarity
                       via inner product (IndexFlatIP).
        """
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded. Dimension: %d, Normalize: %s",
            self.dimension,
            self.normalize,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts to process in each batch.
            show_progress: If True, shows a progress bar.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        logger.info("Encoding %d texts (batch_size=%d) ...", len(texts), batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        logger.info("Encoding complete. Shape: %s", embeddings.shape)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Args:
            query: The query text.

        Returns:
            numpy array of shape (1, dimension).
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding
