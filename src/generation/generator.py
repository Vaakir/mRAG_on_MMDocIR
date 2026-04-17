"""
LLM-based answer generator using Ollama with chat API.

This implementation uses the ollama Python library for more robust communication
with the Ollama server, following the pattern from the generative AI course.
"""

import logging
import sqlite3
import json
import hashlib
from typing import Dict, Optional, List
from pathlib import Path

from config.config import CACHE_DB_PATH

try:
    from ollama import Client, ResponseError
except ImportError:
    raise ImportError(
        "ollama package not found. Install it with: pip install ollama"
    )

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx package not found. Install it with: pip install httpx"
    )
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# def normalize_ws(text: str) -> str:
#     """Normalize whitespace in text (collapse multiple spaces/newlines)."""
#     if not text:
#         return text
#     return " ".join(text.split())

# def strip_thinking(text: str) -> str:
#     """Remove <think>...</think> reasoning block that qwen3 models emit."""
#     return normalize_ws(_THINK_RE.sub("", text))
# -------------------------------------------------------------------
class BaselineGenerator:
    """LLM-based answer generator using Ollama via the ollama Python library."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str
    ):
        """
        Initialize the Ollama client.
        
        Parameters
        ----------
        base_url : str
            Base URL of the Ollama server (REQUIRED - no defaults)
        model : str
            Model name to use (REQUIRED - no defaults)
        api_key : str
            API key for authentication (REQUIRED - no defaults)
        
        Raises
        ------
        ValueError
            If base_url, model, or api_key are None or empty strings
        """
        # Validate required parameters - fail loud if missing
        if not base_url:
            raise ValueError(
                "base_url is required and cannot be None or empty. "
                "It must be provided by config."
            )
        if not model:
            raise ValueError(
                "model is required and cannot be None or empty. "
                "It must be provided by config (LLM_MODEL)."
            )
        if not api_key:
            raise ValueError(
                "api_key is required and cannot be None or empty. "
                "It must be provided by config (OLLAMA_API_KEY from .env)."
            )
        
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash
        self.model = model # Model name to use for generation
        self.api_key = api_key
        
        logger.info(f"Loaded API key: {self.api_key[:10]}...")
        
        # Initialize the Ollama client with authentication header
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}' # Ollama uses Bearer token authentication
        
        self._client = Client( # Initialize the Ollama client with base URL and headers
            host=self.base_url, # Base URL of the Ollama server
            headers=headers if self.api_key else None,  # Include headers only if API key is provided
            timeout=300,  # 5 min — image calls + model reload can take a long time
        )
        logger.info(f"Initialized Ollama client for model: {self.model} at {self.base_url}")
        
        # Initialize Cache Database
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize SQLite database for caching generated answers."""
        db_path = Path(CACHE_DB_PATH)
        
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.cache_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.cache_conn.execute('''
            CREATE TABLE IF NOT EXISTS generator_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                messages_json TEXT,
                answer TEXT
            )
        ''')
        self.cache_conn.commit()

    def _get_cache_key(self, messages_json: str) -> str:
        """Generate a unique cache key based on the model and the exact messages."""
        combined = f"{self.model}|{messages_json}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _get_cached_answer(self, cache_key: str) -> Optional[str]:
        """Retrieve a cached answer for an exact key match."""
        cursor = self.cache_conn.cursor()
        cursor.execute("SELECT answer FROM generator_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _cache_answer(self, cache_key: str, messages_json: str, answer: str):
        """Save the exact request messages and the generated answer to the database."""
        try:
            with self.cache_conn:
                self.cache_conn.execute(
                    "INSERT OR REPLACE INTO generator_cache (cache_key, model, messages_json, answer) VALUES (?, ?, ?, ?)",
                    (cache_key, self.model, messages_json, answer)
                )
        except Exception as e:
            logger.warning(f"Failed to cache generated answer: {e}")
            
    #-------------------
    def _call_with_retries(self, func, max_retries=3, retry_delay=10, exponential_backoff=True):
        """
        Generic retry helper for calling functions with exponential backoff.
        
        Parameters
        ----------
        func : callable
            Function to call (should raise Exception on failure)
        max_retries : int
            Maximum number of retry attempts (default: 3)
        retry_delay : int
            Initial delay in seconds between retries (default: 10)
        exponential_backoff : bool
            Whether to use exponential backoff (2^attempt multiplier) (default: True)
        
        Returns
        -------
        Any
            Result from successful func call
        
        Raises
        ------
        Exception
            The last exception if all retries exhausted
        """
        import time as _time
        last_exc = None
        
        for attempt in range(max_retries):
            try:
                return func(attempt)
            except Exception as e:
                last_exc = e
                error_type = type(e).__name__
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed ({error_type}): {str(e)}")
                
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt) if exponential_backoff else retry_delay
                    logger.info(f"Retrying in {delay} seconds...")
                    _time.sleep(delay)
        
        raise last_exc
    #-------------------
    def _pull_model(self, model_name):
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
        options = {"temperature": 0.0, "top_p": 0.1, "num_predict": 1024}
        options.update(kwargs.pop("options", {}))
        
        def _call_chat(attempt):
            """Internal function to call the chat API with retry handling."""
            try:
                resp = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    stream=False,
                    think=False,
                    **kwargs,
                )
                return resp
            except ResponseError as e:
                status = getattr(e, "status_code", None)
                if status == 404:
                    logger.warning(f"Model {self.model} not found, pulling...")
                    self._pull_model(self.model)
                    raise  # Re-raise to trigger retry
                elif status == 500:
                    logger.warning(f"Server error 500, retrying...")
                    raise  # Re-raise to trigger retry
                else:
                    logger.error(f"Ollama API error: {e}")
                    raise
        
        # Call with retries (higher retry_delay for model loading)
        resp = self._call_with_retries(_call_chat, max_retries=4, retry_delay=30, exponential_backoff=False)
        
        # Extract content from response
        content = getattr(getattr(resp, "message", None), "content", "")
        if not content:
            raise ResponseError("Empty response from Ollama chat API")
        
        return content
        #return strip_thinking(content)
    #-------------------
    def generate(
        self,
        question: str,
        context: str,
        system_prompt: str
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
            The system prompt template (REQUIRED - must be provided by prompting strategy)
        
        Returns
        -------
        str
            The generated answer
        
        Raises
        ------
        ValueError
            If system_prompt is None or empty
        """
        if not system_prompt:
            raise ValueError(
                "system_prompt is required and cannot be None or empty. "
                "It must be provided by a prompting strategy."
            )
        # Construct the user message with context and question
        user_message = f"""Context:
{context}

Question: {question}"""
        
        # Build messages in chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Check Cache
        messages_json = json.dumps(messages, ensure_ascii=False)
        cache_key = self._get_cache_key(messages_json)
        cached_answer = self._get_cached_answer(cache_key)
        if cached_answer:
            logger.debug(f"Cache hit for generation with question: {question[:50]}...")
            return cached_answer
        
        try:
            logger.debug(f"Generating answer for question: {question[:100]}...")
            answer = self.chat(messages) # Call the chat method to get the answer
            logger.debug(f"Generated answer: {answer[:100]}...")
            
            # Save into cache
            self._cache_answer(cache_key, messages_json, answer)
            
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
        contexts: list,
        system_prompt: str = None
    ) -> list:
        """
        Generate answers for multiple questions.
        
        Parameters
        ----------
        questions : list
            List of questions
        contexts : list
            List of contexts (one per question)
        system_prompt : str
            The system prompt template (REQUIRED - must be provided by prompting strategy)
        
        Returns
        -------
        list
            List of generated answers
        
        Raises
        ------
        ValueError
            If system_prompt is None or empty
        """
        if not system_prompt:
            raise ValueError(
                "system_prompt is required and cannot be None or empty. "
                "It must be provided by a prompting strategy."
            )
        
        answers = []
        for i, (q, c) in enumerate(zip(questions, contexts)):
            try:
                answer = self.generate(q, c, system_prompt)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating answer for question {i}: {e}")
                answers.append(f"Error: {str(e)}")
        return answers


# -------------------------------------------------------------------
# Image encoding is now handled by preprocessing.image_processor.encode_image()


class VisionGenerator(BaselineGenerator):
    """
    Extends BaselineGenerator with image support for vision-language models
    (e.g. qwen3-vl:8b).  Images are base64-encoded and passed in the
    ollama messages 'images' field.

    qwen3-vl:8b does not support think=True reliably (returns empty responses),
    so this class overrides chat() to always disable thinking.
    """
    
    def __init__(self, base_url: str, model: str, config=None, api_key: str = None):
        """
        Initialize the VisionGenerator.
        
        Parameters
        ----------
        base_url : str
            Base URL of the Ollama server (REQUIRED)
        model : str
            Model name to use (REQUIRED - should be a vision model like qwen3-vl:8b)
        config : object, optional
            Configuration object with VLM_USE_RAW_CHATML setting
        api_key : str
            API key for authentication (REQUIRED)
        
        Raises
        ------
        ValueError
            If base_url, model, or api_key are None or empty strings
        """
        super().__init__(base_url=base_url, model=model, api_key=api_key)
        self.config = config

    @staticmethod
    def _inject_no_think(messages: list) -> list:
        """
        Prepend /no_think to the last user message content.
        This is qwen3's documented soft switch to suppress the <think> block.
        The Ollama think=False parameter is unreliable on some server builds.
        """
        messages = [m.copy() for m in messages]
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = "/no_think\n" + msg.get("content", "")
                break
        return messages

    def chat(self, messages, think=False, **kwargs):
        """
        Chat with VLM, optionally injecting /no_think to suppress reasoning.
        Uses retry helper for resilience.
        
        Parameters
        ----------
        messages : list
            Chat messages
        think : bool
            Whether to enable thinking/reasoning
        **kwargs : dict
            Additional options
        """
        kwargs.pop("think", None)

        if not think:
            messages = self._inject_no_think(messages)

        options = {"temperature": 0.0, "top_p": 0.1}
        if not think:
            options["repeat_penalty"] = 1.1
        options.update(kwargs.pop("options", {}))

        def _call_vlm_chat(attempt):
            """Internal function to call VLM chat with error handling."""
            try:
                return self._client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    stream=False,
                    think=think,
                    **kwargs,
                )
            except ResponseError as e:
                status = getattr(e, "status_code", None)
                if status == 404:
                    logger.warning(f"Model {self.model} not found, pulling...")
                    self._pull_model(self.model)
                    raise  # Let helper retry immediately after pulling
                elif status == 500:
                    raise  # Let helper retry
                else:
                    raise

        resp = self._call_with_retries(_call_vlm_chat, max_retries=3, retry_delay=10, exponential_backoff=True)
        content = getattr(getattr(resp, "message", None), "content", "")
        if not content:
            raise Exception("Empty response from Ollama chat API")
        return content
        # return normalize_ws(content)

    def generate(self, question, context, system_prompt, think=False):
        """
        Text-only generation — enable thinking for better accuracy.
        
        Parameters
        ----------
        question : str
            The question to answer
        context : str
            The context/document snippets/chunks to use for answering
        system_prompt : str
            The system prompt template (REQUIRED - must be provided by prompting strategy)
        think : bool
            Whether to enable thinking/reasoning
        
        Raises
        ------
        ValueError
            If system_prompt is None or empty
        """
        if not system_prompt:
            raise ValueError(
                "system_prompt is required and cannot be None or empty. "
                "It must be provided by a prompting strategy."
            )
        
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            return self.chat(messages, think=think)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def generate_with_images(
        self,
        question: str,
        image_paths: List[str],
        text_context: str = "",
        system_prompt: str = None,
        think: bool = False,
        max_retries: int = 3,
        retry_delay: int = 10,
    ) -> str:
        """
        Generate an answer given a question, one or more image paths, and
        optional supporting text context. Respects VLM_USE_RAW_CHATML config.
        
        If image_paths are provided but cannot be loaded, still proceeds with
        text-only request using the same raw/standard ChatML mode.

        Parameters
        ----------
        question : str
        image_paths : list of absolute file paths to images
        text_context : str
            Any text chunks retrieved alongside the images (may be empty).
        system_prompt : str
            The system prompt template (REQUIRED - must be provided by prompting strategy)
        think : bool
            Whether to enable thinking/reasoning
        max_retries : int
            Number of retry attempts for failed requests (default: 3)
        retry_delay : int
            Initial delay in seconds between retries (uses exponential backoff)
        
        Raises
        ------
        ValueError
            If system_prompt is None or empty
        Exception
            If VLM generation fails after all retries exhausted (does not fall back to text)
        """
        if not system_prompt:
            raise ValueError(
                "system_prompt is required and cannot be None or empty. "
                "It must be provided by a prompting strategy."
            )
        from preprocessing.image_processor import encode_image
        
        # Build user message content
        user_parts = []
        if text_context:
            user_parts.append(f"Context:\n{text_context}\n")
        user_parts.append(f"Question: {question}")
        user_content = "\n".join(user_parts)

        # Encode images (if any valid paths provided)
        encoded_images = []
        if image_paths:
            for path in image_paths:
                try:
                    encoded_images.append(encode_image(path))
                except Exception as e:
                    logger.warning(f"Could not load image {path}: {e}")
            
            if not encoded_images:
                logger.warning(f"No valid images could be encoded from {len(image_paths)} path(s); continuing with text-only request")

        # Check if we should use raw ChatML workaround for qwen3-vl:8b thinking bug
        use_raw_chatml = getattr(self.config, 'VLM_USE_RAW_CHATML', False) if self.config else False

        if use_raw_chatml:
            # Use raw ChatML with /no_think to bypass qwen3-vl thinking bug
            # Format: system prompt + user message with /no_think + prefilled thinking block
            # Must use direct HTTP call with raw=true (not supported by ollama Python client)
            user_message_dict = {"role": "user", "content": user_content + "\n/no_think"}
            if encoded_images:
                user_message_dict["images"] = encoded_images
            
            raw_messages = [
                {"role": "system", "content": system_prompt},
                user_message_dict,
                {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
            ]
            
            def _call_raw_chatml(attempt):
                """Internal function to call raw ChatML with retries."""
                logger.info(f"VLM attempt {attempt + 1}/{max_retries} for question: {question[:60]}... (raw ChatML mode)")
                
                api_url = f"{self.base_url}/api/chat"
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                payload = {
                    "model": self.model,
                    "messages": raw_messages,
                    "stream": False,
                    "raw": True,
                    "options": {"temperature": 0.0, "top_p": 0.1}
                }
                
                with httpx.Client(timeout=300) as client:
                    http_response = client.post(api_url, json=payload, headers=headers)
                    http_response.raise_for_status()
                    resp_json = http_response.json()
                    content = resp_json.get("message", {}).get("content", "")
                    
                    if not content:
                        raise Exception("Empty response from VLM (raw ChatML mode)")
                    return content
                    #return normalize_ws(content)
            
            try:
                return self._call_with_retries(_call_raw_chatml, max_retries=max_retries, retry_delay=retry_delay, exponential_backoff=True)
            except Exception as e:
                logger.error(f"VLM failed after {max_retries} attempts in raw ChatML mode: {e}")
                raise
        
        # Standard mode (use normal chat with think=False)
        user_message_dict = {"role": "user", "content": user_content}
        if encoded_images:
            user_message_dict["images"] = encoded_images
        
        messages = [
            {"role": "system", "content": system_prompt},
            user_message_dict,
        ]

        def _call_standard_chat(attempt):
            """Internal function to call standard chat with images."""
            logger.info(f"VLM attempt {attempt + 1}/{max_retries} for question: {question[:60]}...")
            return self.chat(messages, think)
        
        try:
            return self._call_with_retries(_call_standard_chat, max_retries=max_retries, retry_delay=retry_delay, exponential_backoff=True)
        except Exception as e:
            logger.error(f"VLM failed after {max_retries} attempts in standard mode: {e}")
            raise
