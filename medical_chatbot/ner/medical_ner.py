"""
Medical Named Entity Recognition using SciSpacy.

Extracts and categorizes medical entities (Diseases, Symptoms, Drugs, Anatomy)
from clinical text using the SciSpacy biomedical NLP pipeline.
"""

from collections import defaultdict

import spacy

from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)

# UMLS Semantic Types → Category Mapping
# Reference: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
UMLS_CATEGORY_MAP: dict[str, str] = {
    # Diseases & Disorders
    "T047": "DISEASE",     # Disease or Syndrome
    "T048": "DISEASE",     # Mental or Behavioral Dysfunction
    "T191": "DISEASE",     # Neoplastic Process
    "T190": "DISEASE",     # Anatomical Abnormality
    "T019": "DISEASE",     # Congenital Abnormality
    "T020": "DISEASE",     # Acquired Abnormality
    "T037": "DISEASE",     # Injury or Poisoning
    "T046": "DISEASE",     # Pathologic Function
    "T049": "DISEASE",     # Cell or Molecular Dysfunction
    "T184": "SYMPTOM",     # Sign or Symptom
    "T033": "SYMPTOM",     # Finding
    "T034": "SYMPTOM",     # Laboratory or Test Result
    # Drugs & Chemicals
    "T121": "DRUG",        # Pharmacologic Substance
    "T200": "DRUG",        # Clinical Drug
    "T195": "DRUG",        # Antibiotic
    "T109": "DRUG",        # Organic Chemical
    "T103": "DRUG",        # Chemical
    "T125": "DRUG",        # Hormone
    "T126": "DRUG",        # Enzyme
    "T127": "DRUG",        # Vitamin
    "T129": "DRUG",        # Immunologic Factor
    "T131": "DRUG",        # Hazardous or Poisonous Substance
    # Anatomy
    "T023": "ANATOMY",     # Body Part, Organ, or Organ Component
    "T024": "ANATOMY",     # Tissue
    "T025": "ANATOMY",     # Cell
    "T026": "ANATOMY",     # Cell Component
    "T029": "ANATOMY",     # Body Location or Region
    "T030": "ANATOMY",     # Body Space or Junction
    "T031": "ANATOMY",     # Body Substance
    "T022": "ANATOMY",     # Body System
}

# Fallback keyword lists for when UMLS linking is unavailable
SYMPTOM_KEYWORDS = {
    "pain", "ache", "fever", "cough", "fatigue", "nausea", "vomiting",
    "diarrhea", "headache", "dizziness", "swelling", "rash", "itching",
    "bleeding", "numbness", "weakness", "stiffness", "shortness of breath",
    "sore throat", "chest pain", "abdominal pain", "back pain", "joint pain",
    "muscle pain", "blurred vision", "loss of appetite", "weight loss",
    "insomnia", "anxiety", "depression", "tremor", "seizure", "confusion",
}

ANATOMY_KEYWORDS = {
    "heart", "lung", "liver", "kidney", "brain", "stomach", "bone", "skin",
    "eye", "ear", "nose", "throat", "blood", "muscle", "nerve", "artery",
    "vein", "spine", "joint", "tendon", "ligament", "cartilage", "gland",
    "pancreas", "intestine", "colon", "bladder", "uterus", "ovary", "prostate",
    "thyroid", "adrenal", "pituitary", "retina", "cornea", "optic nerve",
}


class MedicalNER:
    """
    Medical Named Entity Recognition using SciSpacy.

    Extracts entities from text and categorizes them into:
    - DISEASE: Diseases, syndromes, disorders
    - SYMPTOM: Signs, symptoms, findings
    - DRUG: Drugs, chemicals, pharmacologic substances
    - ANATOMY: Body parts, organs, tissues, cells
    """

    def __init__(self, model_name: str = "en_core_sci_sm") -> None:
        """
        Initialize the NER pipeline.

        Args:
            model_name: SciSpacy model name.
        """
        logger.info("Loading NER model: %s", model_name)
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(
                "Model '%s' not found. Falling back to 'en_core_web_sm'.",
                model_name,
            )
            self.nlp = spacy.load("en_core_web_sm")

        # Check if UMLS entity linker is available
        self._has_linker = "scispacy_linker" in self.nlp.pipe_names
        logger.info(
            "NER model loaded. UMLS linker available: %s", self._has_linker
        )

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """
        Extract and categorize medical entities from text.

        Uses UMLS semantic types when available, falls back to
        heuristic keyword matching for categorization.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary mapping category names to lists of unique entity strings.
            Categories: DISEASE, SYMPTOM, DRUG, ANATOMY
        """
        if not text or not text.strip():
            return {"DISEASE": [], "SYMPTOM": [], "DRUG": [], "ANATOMY": []}

        doc = self.nlp(text)
        entities: dict[str, set[str]] = defaultdict(set)

        for ent in doc.ents:
            entity_text = ent.text.strip().lower()
            if len(entity_text) < 2:
                continue

            category = self._categorize_entity(ent, entity_text)
            if category:
                entities[category].add(entity_text.title())

        # Convert sets to sorted lists
        result = {
            "DISEASE": sorted(entities.get("DISEASE", set())),
            "SYMPTOM": sorted(entities.get("SYMPTOM", set())),
            "DRUG": sorted(entities.get("DRUG", set())),
            "ANATOMY": sorted(entities.get("ANATOMY", set())),
        }

        total = sum(len(v) for v in result.values())
        logger.debug("Extracted %d medical entities from text.", total)
        return result

    def _categorize_entity(self, ent: spacy.tokens.Span, text: str) -> str | None:
        """
        Determine the medical category of an entity.

        Attempts UMLS-based categorization first, then falls back to
        keyword matching.

        Args:
            ent: spaCy entity span.
            text: Lowercased entity text.

        Returns:
            Category string or None if uncategorizable.
        """
        # Try UMLS-based categorization if linker is available
        if self._has_linker and hasattr(ent, "_") and hasattr(ent._, "kb_ents"):
            for umls_ent in ent._.kb_ents:
                cui = umls_ent[0] if isinstance(umls_ent, tuple) else umls_ent
                # Look up semantic type from the linker's knowledge base
                linker = self.nlp.get_pipe("scispacy_linker")
                if hasattr(linker, "kb") and cui in linker.kb:
                    concept = linker.kb[cui]
                    for stype in getattr(concept, "types", []):
                        if stype in UMLS_CATEGORY_MAP:
                            return UMLS_CATEGORY_MAP[stype]

        # Fallback: keyword-based categorization
        return self._keyword_categorize(text)

    @staticmethod
    def _keyword_categorize(text: str) -> str | None:
        """
        Categorize entity using keyword heuristics.

        Args:
            text: Lowercased entity text.

        Returns:
            Category string or None.
        """
        text_lower = text.lower()

        # Check symptoms first (more specific)
        for kw in SYMPTOM_KEYWORDS:
            if kw in text_lower:
                return "SYMPTOM"

        # Check anatomy
        for kw in ANATOMY_KEYWORDS:
            if kw in text_lower:
                return "ANATOMY"

        # Disease indicators
        disease_suffixes = (
            "itis", "osis", "emia", "oma", "pathy", "ectomy",
            "plasty", "scopy", "emia", "trophy",
        )
        disease_keywords = {
            "disease", "disorder", "syndrome", "cancer", "tumor",
            "infection", "failure", "deficiency", "condition",
        }

        for suffix in disease_suffixes:
            if text_lower.endswith(suffix):
                return "DISEASE"
        for kw in disease_keywords:
            if kw in text_lower:
                return "DISEASE"

        # Drug indicators
        drug_keywords = {
            "drug", "medication", "medicine", "therapy", "treatment",
            "vaccine", "antibiotic", "steroid", "inhibitor",
        }
        for kw in drug_keywords:
            if kw in text_lower:
                return "DRUG"

        # Default: categorize as DISEASE if entity is recognized by SciSpacy
        # (SciSpacy primarily recognizes biomedical concepts)
        return "DISEASE"

    @staticmethod
    def format_entities(entities: dict[str, list[str]]) -> str:
        """
        Format extracted entities into a human-readable string.

        Args:
            entities: Dictionary of categorized entities.

        Returns:
            Formatted string for display.
        """
        lines: list[str] = []
        category_icons = {
            "DISEASE": "🦠",
            "SYMPTOM": "🤒",
            "DRUG": "💊",
            "ANATOMY": "🫁",
        }

        for category, items in entities.items():
            if items:
                icon = category_icons.get(category, "•")
                lines.append(f"{icon} {category}: {', '.join(items)}")

        return "\n".join(lines) if lines else "No medical entities detected."

    def has_medical_entities(self, text: str) -> bool:
        """
        Quick check whether text contains any medical entities.

        Args:
            text: Input text.

        Returns:
            True if at least one entity is found.
        """
        entities = self.extract_entities(text)
        return any(len(v) > 0 for v in entities.values())
