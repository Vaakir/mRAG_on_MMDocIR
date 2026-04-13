"""
LLM-based answer generator using Ollama with chat API.

This implementation uses the ollama Python library for more robust communication
with the Ollama server, following the pattern from the generative AI course.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

try:
    from ollama import Client, ResponseError
except ImportError:
    raise ImportError(
        "ollama package not found. Install it with: pip install ollama"
    )
# -------------------------------------------------------------------
# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
def normalize_ws(text: str) -> str:
    """Normalize whitespace in text (collapse multiple spaces/newlines)."""
    if not text:
        return text
    return " ".join(text.split())
# -------------------------------------------------------------------
# Baseline prompt template for the system role
SYSTEM_PROMPT = """You are a concise assistant. Answer using ONLY the provided context.

Strict Instructions (NON-NEGOTIABLE):
- Read the whole context and think before answering.
- NEVER add preamble, explanation, or context. ANSWER ONLY.
- DO NOT rephrase, explain, or add context.
- Multiple answers: ONLY output ['answer1', 'answer2', ...]. NOTHING ELSE.
- For yes/no questions, answer only "Yes" or "No".
- Do not rely only on explicit statements. If the answer can be derived from the context through calculation (e.g., growth rate, difference, ratio, count), compute it before answering.

Be direct. No padding. No explanations unless specifically asked."""
# -------------------------------------------------------------------
class BaselineGenerator:
    """LLM-based answer generator using Ollama via the ollama Python library."""
    
    def __init__(
        self,
        base_url: str = "https://ollama.ux.uis.no",
        model: str = "qwen3-vl:8b",
        api_key: str = None
    ):
        """
        Initialize the Ollama client.
        
        Parameters
        ----------
        base_url : str
            Base URL of the Ollama server
        model : str
            Model name to use
        api_key : str, optional
            API key for authentication. If None, attempts to load from OLLAMA_API_KEY env var.
        """
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash
        self.model = model # Model name to use for generation
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('OLLAMA_API_KEY')
        
        if not self.api_key:
            logger.warning("No OLLAMA_API_KEY found. API calls may fail.")
        else:
            logger.info(f"Loaded API key: {self.api_key[:10]}...")
        
        # Initialize the Ollama client with authentication header
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}' # Ollama uses Bearer token authentication
        
        self._client = Client( # Initialize the Ollama client with base URL and headers
            host=self.base_url, # Base URL of the Ollama server
            headers=headers if self.api_key else None  # Include headers only if API key is provided
        )
        logger.info(f"Initialized Ollama client for model: {self.model}")
    #-------------------
    def _pull_model(self, model_name: str) -> None:
        """
        Pull a model from Ollama if it doesn't exist.
        
        Parameters
        ----------
        model_name : str
            The model name to pull
        """
        logger.info(f"Pulling model {model_name}...")
        self._client.pull(model_name) # Pull the model from Ollama server (blocking call)
        logger.info(f"Successfully pulled model {model_name}")
    #-------------------
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send a chat request to the Ollama model and return the generated response.
        
        Parameters
        ----------
        messages : List[Dict[str, str]]
            A list of message dictionaries following the chat format:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                ...
            ]
        **kwargs : dict
            Optional keyword arguments.
            Supported:
            - options (dict): Generation parameters such as:
              {
                  "temperature": float,
                  "top_p": float
              }
              If not provided, defaults to: {"temperature": 0.0, "top_p": 0.1}
        
        Returns
        -------
        str
            The generated text response from the model.
        
        Raises
        ------
        ResponseError
            If the API request fails and cannot be recovered.
        """
        # Default options for generation
        options = {"temperature": 0.0, "top_p": 0.1} # Default to deterministic output for baseline (want these to be low for accurate extraction)
        options.update(kwargs.pop("options", {})) # Allow overriding options via kwargs
        #-------------------
        def _call_chat() -> Any:
            """Internal function to call the chat API."""
            return self._client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=False,
                think=False,  # Disable reasoning/thinking for faster responses
                **kwargs,
            )
        #-------------------
        # Try to call chat, with auto-pull if model not found
        try:
            resp = _call_chat()
        except ResponseError as e:
            # If model not found (404), try to pull it and retry
            if getattr(e, "status_code", None) == 404:
                logger.warning(f"Model {self.model} not found on server, attempting to pull...")
                self._pull_model(self.model)
                resp = _call_chat()
            else:
                logger.error(f"Ollama API error: {e}")
                raise
        
        # Extract content from response
        # The response object has a 'message' attribute with content
        content = getattr(getattr(resp, "message", None), "content", "")
        
        if not content:
            raise ResponseError("Empty response from Ollama chat API")
        
        return normalize_ws(content)
    #-------------------
    def generate(
        self,
        question: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT
    ) -> str:
        """
        Generate an answer given a question and context.
        
        This is the main interface that converts prompt-based format to chat format.
        
        Parameters
        ----------
        question : str
            The question to answer
        context : str
            The context/document snippets/chunks to use for answering
        system_prompt : str
            The system prompt template
        
        Returns
        -------
        str
            The generated answer
        """
        # Construct the user message with context and question
        user_message = f"""Context:
{context}

Question: {question}"""
        
        # Build messages in chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            logger.debug(f"Generating answer for question: {question[:100]}...")
            answer = self.chat(messages) # Call the chat method to get the answer
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
        except ResponseError as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return f"Error generating response: {str(e)}"
    #-------------------
    def generate_batch(
        self,
        questions: list,
        contexts: list
    ) -> list:
        """
        Generate answers for multiple questions.
        
        Parameters
        ----------
        questions : list
            List of questions
        contexts : list
            List of contexts (one per question)
        
        Returns
        -------
        list
            List of generated answers
        """
        answers = []
        for i, (q, c) in enumerate(zip(questions, contexts)):
            try:
                answer = self.generate(q, c)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating answer for question {i}: {e}")
                answers.append(f"Error: {str(e)}")
        return answers
