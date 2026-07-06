"""
Dataset Preparation Script.

Loads the raw MedQuAD CSV, validates it, preprocesses text,
and saves a cleaned version ready for indexing.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medical_chatbot.data.loader import MedQuADLoader
from medical_chatbot.preprocessing.text_processor import TextProcessor
from medical_chatbot.utils.config import load_config
from medical_chatbot.utils.logger import setup_logger

logger = setup_logger("prepare_dataset", level="INFO", log_file="logs/prepare_dataset.log")


def main() -> None:
    """Run the dataset preparation pipeline."""
    logger.info("=" * 60)
    logger.info("  MEDQUAD DATASET PREPARATION")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    raw_path = PROJECT_ROOT / config["dataset"]["raw_path"]
    cleaned_path = PROJECT_ROOT / config["dataset"]["cleaned_path"]

    # Step 1: Load dataset
    logger.info("\n[Step 1/4] Loading dataset ...")
    loader = MedQuADLoader(str(raw_path))
    df = loader.load()

    # Step 2: Validate
    logger.info("\n[Step 2/4] Validating dataset ...")
    df = loader.validate(df)

    # Step 3: Display statistics
    logger.info("\n[Step 3/4] Dataset statistics:")
    stats = MedQuADLoader.get_stats(df)
    for key, value in stats.items():
        if key == "source_distribution":
            logger.info("  %s:", key)
            for source, count in value.items():
                logger.info("    %-30s : %d", source, count)
        else:
            logger.info("  %-25s : %s", key, value)

    # Step 4: Preprocess text
    logger.info("\n[Step 4/4] Preprocessing text ...")
    text_processor = TextProcessor(config["preprocessing"]["spacy_model"])

    df["question_clean"] = df["question"].apply(text_processor.normalize)
    df["answer_clean"] = df["answer"].apply(text_processor.normalize)

    # Save cleaned dataset
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False, encoding="utf-8")
    logger.info("Cleaned dataset saved to: %s", cleaned_path)

    logger.info("\n" + "=" * 60)
    logger.info("  PREPARATION COMPLETE")
    logger.info("  Total QA pairs: %d", len(df))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
