"""
NLP Text Preprocessing Pipeline.

Provides text normalization, tokenization, lemmatization, keyword extraction,
query preprocessing, and sentence-aware chunking for medical text.
"""

import re
import unicodedata

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextProcessor:
    """
    NLP preprocessing pipeline using spaCy.

    Handles normalization, tokenization, lemmatization, keyword extraction,
    and text chunking with sentence boundary awareness.
    """

    def __init__(self, spacy_model: str = "en_core_sci_sm") -> None:
        """
        Initialize the text processor with a spaCy model.

        Args:
            spacy_model: Name of the spaCy/SciSpacy model to load.
        """
        logger.info("Loading spaCy model: %s", spacy_model)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(
                "Model '%s' not found. Falling back to 'en_core_web_sm'.", spacy_model
            )
            self.nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully.")

    def normalize(self, text: str) -> str:
        """
        Normalize text for consistent processing.

        Steps:
            1. Unicode normalization (NFKD → ASCII-safe).
            2. Lowercase.
            3. Collapse multiple whitespace characters.
            4. Strip leading/trailing whitespace.

        Args:
            text: Raw input text.

        Returns:
            Normalized text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Unicode normalize
        text = unicodedata.normalize("NFKD", text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove parenthetical media references (e.g., "(Watch the video ...)")
        text = re.sub(r"\(watch the .*?\)", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)

        # Strip
        text = text.strip()

        return text

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using spaCy.

        Args:
            text: Input text.

        Returns:
            List of token strings (excluding punctuation and whitespace).
        """
        doc = self.nlp(text)
        tokens = [
            token.text
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        return tokens

    def lemmatize(self, text: str) -> list[str]:
        """
        Lemmatize text using spaCy, preserving medical terminology.

        Medical terms (identified as entities) are kept in their original form
        to avoid losing important clinical meaning.

        Args:
            text: Input text.

        Returns:
            List of lemmatized tokens.
        """
        doc = self.nlp(text)

        # Collect entity spans to preserve
        entity_spans = set()
        for ent in doc.ents:
            for token in ent:
                entity_spans.add(token.i)

        lemmas = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            # Preserve medical entities as-is
            if token.i in entity_spans:
                lemmas.append(token.text.lower())
            else:
                lemmas.append(token.lemma_.lower())

        return lemmas

    def extract_keywords(self, text: str) -> list[str]:
        """
        Extract medical keywords using noun chunks and entities.

        Args:
            text: Input text.

        Returns:
            Deduplicated list of keywords.
        """
        doc = self.nlp(text)

        keywords: list[str] = []

        # Add named entities
        for ent in doc.ents:
            kw = ent.text.strip().lower()
            if kw and kw not in keywords:
                keywords.append(kw)

        # Add noun chunks (filter stopwords)
        for chunk in doc.noun_chunks:
            kw = chunk.text.strip().lower()
            if kw and kw not in STOP_WORDS and kw not in keywords:
                keywords.append(kw)

        return keywords

    def preprocess_query(self, query: str) -> str:
        """
        Full query normalization pipeline.

        Applies normalization, then reconstructs a clean query string suitable
        for embedding generation.

        Args:
            query: Raw user query.

        Returns:
            Preprocessed query string.
        """
        if not query or not query.strip():
            return ""

        normalized = self.normalize(query)
        return normalized

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
        min_chunk_size: int = 50,
    ) -> list[str]:
        """
        Split text into overlapping chunks with sentence boundary awareness.

        Tries to break at sentence boundaries to preserve semantic coherence.
        Falls back to character-level splitting if sentences are too long.

        Args:
            text: The text to chunk.
            chunk_size: Target chunk size in characters.
            overlap: Number of overlapping characters between consecutive chunks.
            min_chunk_size: Minimum chunk size; shorter chunks are discarded.

        Returns:
            List of text chunks.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text and len(text) >= min_chunk_size else []

        # Use spaCy for sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If a single sentence exceeds chunk_size, split it by characters
            if sentence_length > chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0

                # Character-level split for oversized sentence
                for i in range(0, sentence_length, chunk_size - overlap):
                    sub = sentence[i : i + chunk_size]
                    if len(sub) >= min_chunk_size:
                        chunks.append(sub)
                continue

            # Check if adding this sentence would exceed chunk_size
            if current_length + sentence_length + 1 > chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(chunk_text)

                # Start new chunk with overlap from previous sentences
                overlap_text = chunk_text[-overlap:] if overlap > 0 else ""
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(" ".join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1

        # Flush remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)

        logger.debug(
            "Chunked text (%d chars) into %d chunks", len(text), len(chunks)
        )
        return chunks
