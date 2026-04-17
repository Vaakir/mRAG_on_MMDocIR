"""
LLM-based answer generator using Ollama with chat API.

This implementation uses the ollama Python library for more robust communication
with the Ollama server, following the pattern from the generative AI course.
"""

import os
import re
import base64
import logging
import sqlite3
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

from config.config import CACHE_DB_PATH

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
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def normalize_ws(text: str) -> str:
    """Normalize whitespace in text (collapse multiple spaces/newlines)."""
    if not text:
        return text
    return " ".join(text.split())

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning block that qwen3 models emit."""
    return normalize_ws(_THINK_RE.sub("", text))
# -------------------------------------------------------------------
# Baseline prompt template for the system role
SYSTEM_PROMPT2 = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question using ONLY the provided context.
- If the question requires counting, listing, or comparison:
  - Identify all relevant items in the context.
  - Apply the required condition step by step.
  - Ensure the final count is correct.
  - But also be conscious of not hallucinating items that are not in the context.
  - For counting questions, output only the final number and don't over think it.
- If the answer is partially available, explain what is missing.
- If the answer cannot be found, say: "I cannot find the answer in the provided context."
- Be concise but ensure accuracy."""

SYSTEM_PROMPT = """You are a concise assistant. Answer using ONLY the provided context.

Strict Instructions (NON-NEGOTIABLE):
- Read the whole context and think before answering.
- NEVER add preamble, explanation, or context. ANSWER ONLY.
- DO NOT rephrase, explain, or add context.
- Multiple answers: ONLY output ['answer1', 'answer2', ...]. NOTHING ELSE.
- For yes/no questions, answer only "Yes" or "No".
- Do not rely only on explicit statements. If the answer can be derived from the context through calculation (e.g., growth rate, difference, ratio, count), compute it before answering.

Be direct. No padding. No explanations unless specifically asked."""

VISION_PROMPT = """Answer the question using the image(s) provided.
Output ONLY the direct answer. No explanation, no reasoning, no preamble, no "Based on...".

- Number question → output the number only.
- Yes/no question → output "Yes" or "No" only.
- List question → output ["item1", "item2"] only.
- Single answer → output the answer only."""
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
            headers=headers if self.api_key else None,  # Include headers only if API key is provided
            timeout=300,  # 5 min — image calls + model reload can take a long time
        )
        logger.info(f"Initialized Ollama client for model: {self.model}")
        
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
        options = {"temperature": 0.0, "top_p": 0.1, "num_predict": 1024} # Default to deterministic output for baseline (want these to be low for accurate extraction)
        options.update(kwargs.pop("options", {})) # Allow overriding options via kwargs
        #-------------------
        def _call_chat() -> Any:
            """Internal function to call the chat API."""
            return self._client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=False,
                think=False,
                **kwargs,
            )
        #-------------------
        # Try to call chat, with retry on 500 (server swapping models) and auto-pull on 404
        import time as _time
        max_retries = 4
        retry_delay = 30  # seconds — gives server time to unload previous model
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = _call_chat()
                last_exc = None
                break
            except ResponseError as e:
                status = getattr(e, "status_code", None)
                if status == 404:
                    logger.warning(f"Model {self.model} not found, pulling...")
                    self._pull_model(self.model)
                elif status == 500:
                    logger.warning(f"Server 500 (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                    last_exc = e
                    _time.sleep(retry_delay)
                else:
                    logger.error(f"Ollama API error: {e}")
                    raise
        if last_exc is not None:
            raise last_exc
        
        # Extract content from response
        # The response object has a 'message' attribute with content
        content = getattr(getattr(resp, "message", None), "content", "")
        
        if not content:
            raise ResponseError("Empty response from Ollama chat API")

        return strip_thinking(content)
    #-------------------
    def generate(
        self,
        question: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT2
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


# -------------------------------------------------------------------
from functools import lru_cache

@lru_cache(maxsize=256)
def _encode_image(image_path: str, max_side: int = 1120, quality: int = 65) -> str:
    """
    Read an image, resize so the longest side ≤ max_side (preserving aspect
    ratio), then return base64-encoded JPEG bytes.

    Results are cached (LRU, 256 entries) so the same page image isn't
    re-encoded across different questions.

    1120 px is the native tile size qwen3-vl uses internally; sending larger
    images just bloats the payload without helping quality.
    quality=65 is sufficient for document pages (text, tables, charts).
    """
    from PIL import Image as _Image
    import io as _io

    img = _Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), _Image.LANCZOS)

    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VisionGenerator(BaselineGenerator):
    """
    Extends BaselineGenerator with image support for vision-language models
    (e.g. qwen3-vl:8b).  Images are base64-encoded and passed in the
    ollama messages 'images' field.

    qwen3-vl:8b does not support think=True reliably (returns empty responses),
    so this class overrides chat() to always disable thinking.
    """

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
        think=False  → image call: injects /no_think, num_predict=2048
        think=True   → text call:  thinking enabled,  num_predict=4096
        """
        import time as _time
        from ollama import ResponseError
        kwargs.pop("think", None)  # discard if caller passed it via kwargs

        if not think:
            messages = self._inject_no_think(messages)

        options = {"temperature": 0.0, "top_p": 0.1}#, "num_predict": 7200}
        if not think:
            options["repeat_penalty"] = 1.1
        options.update(kwargs.pop("options", {}))

        max_retries = 3
        retry_delay = 10
        last_exc = None
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    stream=False,
                    think=think,
                    **kwargs,
                )
                last_exc = None
                break
            except ResponseError as e:
                status = getattr(e, "status_code", None)
                if status == 404:
                    self._pull_model(self.model)
                elif status == 500:
                    logger.warning(f"VLM server 500 (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                    last_exc = e
                    _time.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                raise

        if last_exc is not None:
            raise last_exc
        logger.info(f"resp: {resp}")
        content = getattr(getattr(resp, "message", None), "content", "")
        if not content:
            raise Exception("Empty response from Ollama chat API")
        return normalize_ws(content)

    def generate(self, question, context, system_prompt=SYSTEM_PROMPT2, think=True):
        """Text-only generation — enable thinking for better accuracy."""
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
        system_prompt: str = VISION_PROMPT,
        think: bool = False,
    ) -> str:
        """
        Generate an answer given a question, one or more image paths, and
        optional supporting text context.

        Parameters
        ----------
        question : str
        image_paths : list of absolute file paths to images
        text_context : str
            Any text chunks retrieved alongside the images (may be empty).
        system_prompt : str
        """
        # Build user message content
        user_parts = []
        if text_context:
            user_parts.append(f"Context:\n{text_context}\n")
        user_parts.append(f"Question: {question}")
        user_content = "\n".join(user_parts)

        # Encode images
        encoded_images = []
        for path in image_paths:
            try:
                encoded_images.append(_encode_image(path))
            except Exception as e:
                logger.warning(f"Could not load image {path}: {e}")

        if not encoded_images:
            # Fall back to text-only generation
            logger.warning("No images could be loaded; falling back to text generation")
            return self.generate(question, text_context, system_prompt, think=think)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
                "images": encoded_images,
            },
        ]

        try:
            return self.chat(messages, think)
        except Exception as e:
            logger.error(f"Vision generation error: {e}")
            return f"Error generating response: {str(e)}"
