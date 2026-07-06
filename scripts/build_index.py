"""
FAISS Index Building Script.

Loads the cleaned MedQuAD dataset, chunks answers, generates embeddings,
builds a FAISS index, and saves it with metadata to disk.
"""

import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from medical_chatbot.preprocessing.text_processor import TextProcessor
from medical_chatbot.embeddings.encoder import EmbeddingEncoder
from medical_chatbot.vector_store.faiss_store import FAISSStore
from medical_chatbot.utils.config import load_config
from medical_chatbot.utils.logger import setup_logger

logger = setup_logger("build_index", level="INFO", log_file="logs/build_index.log")


def main() -> None:
    """Run the FAISS index building pipeline."""
    logger.info("=" * 60)
    logger.info("  FAISS INDEX BUILDER")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    cleaned_path = PROJECT_ROOT / config["dataset"]["cleaned_path"]
    chunk_size = config["preprocessing"]["chunk_size"]
    chunk_overlap = config["preprocessing"]["chunk_overlap"]
    min_chunk_size = config["preprocessing"]["min_chunk_size"]
    embedding_model = config["embeddings"]["model_name"]
    embedding_dim = config["embeddings"]["dimension"]
    batch_size = config["embeddings"]["batch_size"]
    index_path = str(PROJECT_ROOT / config["vector_store"]["index_path"])
    metadata_path = str(PROJECT_ROOT / config["vector_store"]["metadata_path"])

    # Step 1: Load cleaned dataset
    logger.info("\n[Step 1/5] Loading cleaned dataset ...")
    if not cleaned_path.exists():
        logger.error(
            "Cleaned dataset not found at: %s. Run prepare_dataset.py first.",
            cleaned_path,
        )
        sys.exit(1)

    df = pd.read_csv(cleaned_path, encoding="utf-8")
    logger.info("Loaded %d QA pairs.", len(df))

    # Step 2: Initialize components
    logger.info("\n[Step 2/5] Initializing text processor and encoder ...")
    text_processor = TextProcessor(config["preprocessing"]["spacy_model"])
    encoder = EmbeddingEncoder(model_name=embedding_model, normalize=True)

    # Step 3: Chunk all answers
    logger.info("\n[Step 3/5] Chunking answers (size=%d, overlap=%d) ...", chunk_size, chunk_overlap)

    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    for doc_id, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        answer = row.get("answer_clean", row.get("answer", ""))
        if not answer or not isinstance(answer, str) or len(answer.strip()) < min_chunk_size:
            continue

        chunks = text_processor.chunk_text(
            text=answer,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

        for chunk_id, chunk_text in enumerate(chunks):
            all_chunks.append(chunk_text)
            all_metadata.append({
                "doc_id": int(doc_id),
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "question": str(row.get("question", "")),
                "full_answer": str(answer),
                "source": str(row.get("source", "Unknown")),
                "focus_area": str(row.get("focus_area", "General")),
            })

    logger.info("Total chunks created: %d (from %d documents)", len(all_chunks), len(df))

    if not all_chunks:
        logger.error("No chunks were created. Check the dataset and chunking parameters.")
        sys.exit(1)

    # Step 4: Generate embeddings
    logger.info("\n[Step 4/5] Generating embeddings (model=%s) ...", embedding_model)
    embeddings = encoder.encode(all_chunks, batch_size=batch_size, show_progress=True)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Step 5: Build and save FAISS index
    logger.info("\n[Step 5/5] Building FAISS index ...")
    faiss_store = FAISSStore(dimension=embedding_dim)
    faiss_store.build_index(embeddings, all_metadata)
    faiss_store.save(index_path, metadata_path)

    logger.info("\n" + "=" * 60)
    logger.info("  INDEX BUILD COMPLETE")
    logger.info("  Total vectors : %d", faiss_store.size)
    logger.info("  Dimension     : %d", embedding_dim)
    logger.info("  Index saved   : %s", index_path)
    logger.info("  Metadata saved: %s", metadata_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
