"""
Retrieval-Augmented Generation (RAG) Pipeline.

Constructs prompts that combine retrieved context with user questions,
enforcing grounded responses from the LLM.
"""

from medical_chatbot.retriever.semantic_retriever import RetrievalResult
from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)

# System prompt for the medical chatbot
SYSTEM_PROMPT = """You are a Medical Information Assistant powered by the MedQuAD knowledge base.

STRICT RULES:
1. Answer ONLY using the provided context below.
2. If the context does not contain sufficient information to answer the question, explicitly state: "I cannot provide a complete answer based on the available medical knowledge base."
3. NEVER make up, infer, or hallucinate information that is not in the context.
4. Cite the source of information when available.
5. Use clear, professional medical language that is also understandable by patients.
6. If the context contains partial information, answer what you can and clearly state what information is missing.
7. Include relevant medical disclaimers when appropriate (e.g., "Consult your healthcare provider for personalized medical advice")."""


class RAGPipeline:
    """
    Constructs RAG prompts from retrieved chunks and user queries.

    Formats the retrieved context, detected entities, and user question
    into a structured prompt that constrains the LLM to ground its
    response in the retrieved evidence.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            system_prompt: Custom system prompt. Uses default if None.
        """
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        logger.info("RAG pipeline initialized.")

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list[RetrievalResult],
        detected_entities: dict[str, list[str]] | None = None,
    ) -> str:
        """
        Build a RAG prompt from query, retrieved context, and entities.

        Args:
            query: User's medical question.
            retrieved_chunks: Top-k retrieved chunks from semantic search.
            detected_entities: Optional NER-detected medical entities.

        Returns:
            Formatted prompt string to send to the LLM.
        """
        # Build context section
        context_parts: list[str] = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Context {i}] (Source: {chunk.source} | "
                f"Topic: {chunk.focus_area} | "
                f"Similarity: {chunk.similarity_score:.3f})\n"
                f"{chunk.chunk_text}"
            )

        context_text = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        # Build entities section
        entities_text = ""
        if detected_entities:
            entity_parts: list[str] = []
            for category, items in detected_entities.items():
                if items:
                    entity_parts.append(f"  - {category}: {', '.join(items)}")
            if entity_parts:
                entities_text = "\nDetected Medical Entities:\n" + "\n".join(entity_parts)

        # Construct full prompt
        prompt = f"""{self.system_prompt}

---

CONTEXT:
{context_text}

---

QUESTION:
{query}
{entities_text}

---

INSTRUCTIONS:
- Answer the question using ONLY the context provided above.
- If the context is insufficient, clearly state that you cannot answer.
- Be precise, professional, and cite sources when possible.
- Include a brief medical disclaimer at the end.

ANSWER:"""

        logger.debug("RAG prompt constructed (%d chars, %d chunks)", len(prompt), len(retrieved_chunks))
        return prompt

    def build_no_context_response(self, query: str) -> str:
        """
        Build a response when no relevant context is found.

        Args:
            query: The user's original question.

        Returns:
            A formatted message indicating insufficient context.
        """
        return (
            f"I was unable to find relevant information in the medical knowledge base "
            f"to answer your question: \"{query}\"\n\n"
            f"This could mean:\n"
            f"• The topic may not be covered in the MedQuAD dataset.\n"
            f"• Try rephrasing your question with different medical terminology.\n\n"
            f"⚠️ For medical concerns, please consult a qualified healthcare professional."
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt being used."""
        return self.system_prompt
