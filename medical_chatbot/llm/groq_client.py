"""
Groq LLM Client.

Interfaces with the Groq API for fast LLM inference using open-weight models.
Uses the official groq Python SDK.
"""

from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


class GroqClient:
    """
    Client for the Groq inference API.

    Provides fast inference using models like Llama 3.3 70B running
    on Groq's LPU hardware.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize the Groq client.

        Args:
            api_key: Groq API key.
            model: Model identifier to use for generation.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in the generated response.

        Raises:
            ValueError: If api_key is empty or None.
        """
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "Invalid Groq API key. Please set the GROQ_API_KEY "
                "environment variable with a valid key from "
                "https://console.groq.com/keys"
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            "GroqClient initialized (model=%s, temp=%.1f, max_tokens=%d)",
            model,
            temperature,
            max_tokens,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user/RAG prompt to send.
            system_prompt: Optional system-level instruction.
            history: Optional list of previous conversation messages [{"role": "...", "content": "..."}]

        Returns:
            The generated text response.

        Raises:
            ConnectionError: If unable to reach the Groq API.
            RuntimeError: If the API returns an error.
        """
        if not prompt or not prompt.strip():
            return "No prompt provided."

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        try:
            logger.info("Sending request to Groq API (model=%s) ...", self.model)

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1.0,
                stream=False,
            )

            response_text = chat_completion.choices[0].message.content

            logger.info(
                "Response received (%d chars, model=%s)",
                len(response_text) if response_text else 0,
                self.model,
            )
            return response_text or "No response generated."

        except APIConnectionError as e:
            logger.error("Failed to connect to Groq API: %s", e)
            raise ConnectionError(
                "Unable to reach the Groq API. Please check your internet "
                "connection and try again."
            ) from e

        except RateLimitError as e:
            logger.error("Groq API rate limit exceeded: %s", e)
            raise RuntimeError(
                "Groq API rate limit exceeded. Please wait a moment and try again."
            ) from e

        except APIStatusError as e:
            logger.error("Groq API error (status=%s): %s", e.status_code, e.message)
            raise RuntimeError(
                f"Groq API error: {e.message}"
            ) from e

        except Exception as e:
            logger.error("Unexpected error during LLM generation: %s", e)
            raise RuntimeError(
                f"An unexpected error occurred: {str(e)}"
            ) from e

    def health_check(self) -> bool:
        """
        Verify that the Groq API is reachable and the model is available.

        Returns:
            True if the API responds successfully.
        """
        try:
            response = self.generate("Say 'OK' in one word.")
            return bool(response and len(response) > 0)
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False
