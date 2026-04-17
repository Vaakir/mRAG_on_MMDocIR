"""Simple LLM wrapper for the agent decision-making using Ollama."""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

try:
    from ollama import Client, ResponseError
except ImportError:
    raise ImportError("ollama package not found. Install it with: pip install ollama")

logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class SimpleLLM:
    """Lightweight LLM wrapper for agent decision-making."""
    
    def __init__(
        self,
        base_url: str = "https://ollama.ux.uis.no",
        model: str = "qwen3:32b",
        api_key: str = None
    ):
        """
        Initialize the LLM wrapper.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.model = model # Model name to use 
        self.api_key = api_key or os.getenv('OLLAMA_API_KEY')
        
        if not self.api_key:
            logger.warning("No OLLAMA_API_KEY found. API calls may fail.")
        
        # Initialize Ollama client
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        self._client = Client(
            host=self.base_url,
            headers=headers if self.api_key else None
        )
        
        logger.info(f"SimpleLLM initialized: {self.model} at {self.base_url}")
    
    def invoke(self, prompt: str, max_retries: int = 1, timeout: int = 300) -> 'SimpleLLMResponse':
        """
        Call the LLM with a prompt. Validates JSON output and retries if needed.
        
        Args:
            prompt: Input prompt string
            max_retries: Max retry attempts if response is not valid JSON (default: 1)
            timeout: Timeout in seconds for LLM call (default: 300 seconds = 5 minutes)
            
        Returns:
            SimpleLLMResponse object with content attribute
        """
        def _make_call(prompt_text: str) -> str:
            """Helper to make actual API call with timeout."""
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._client.generate,
                    model=self.model,
                    prompt=prompt_text,
                    stream=False
                )
                try:
                    response = future.result(timeout=timeout)
                    content = response.get('response', '')
                    if not content:
                        raise ResponseError("Empty response from Ollama")
                    return content
                except FuturesTimeoutError:
                    logger.error(f"LLM call timed out after {timeout} seconds")
                    raise TimeoutError(f"LLM call exceeded {timeout} second timeout")
        
        try:
            # First attempt
            content = _make_call(prompt)
            
            # Validate that response is valid JSON
            if not content.strip().startswith('{'):
                logger.warning(f"LLM response is not JSON. Retrying with correction prompt...")
                
                # Retry with explicit JSON correction instruction
                if max_retries > 0:
                    correction_prompt = f"""{prompt}

IMPORTANT: Your previous response was not valid JSON. 
RESPOND WITH ONLY A JSON OBJECT, NOTHING ELSE.
NO EXPLANATION, NO MARKDOWN, JUST JSON.
Start with {{ and end with }}."""
                    try:
                        content = _make_call(correction_prompt)
                        if not content.strip().startswith('{'):
                            logger.error("Retry failed. Response still not JSON. Using fallback.")
                            return SimpleLLMResponse(content='{"error": "Failed to parse JSON", "fallback": true}')
                    except TimeoutError as timeout_error:
                        logger.error(f"Retry timed out: {timeout_error}")
                        return SimpleLLMResponse(content='{"error": "LLM timeout on retry", "fallback": true}')
                    except Exception as retry_error:
                        logger.error(f"Retry failed with error: {retry_error}. Using fallback.")
                        return SimpleLLMResponse(content='{"error": "Failed to parse JSON", "fallback": true}')
                else:
                    logger.warning("Max retries exhausted. Using fallback.")
                    return SimpleLLMResponse(content='{"error": "Failed to parse JSON", "fallback": true}')
            
            return SimpleLLMResponse(content=content)
            
        except TimeoutError as e:
            logger.error(f"LLM call timed out: {e}")
            return SimpleLLMResponse(content='{"error": "LLM timeout", "fallback": true}')
        except ResponseError as e:
            if getattr(e, 'status_code', None) == 404:
                logger.warning(f"Model {self.model} not found, attempting to pull...")
                try:
                    self._client.pull(self.model)
                    content = _make_call(prompt)
                    
                    # Validate JSON after pull
                    if not content.strip().startswith('{'):
                        logger.warning("Response after pull is still not JSON")
                        return SimpleLLMResponse(content='{"error": "Failed to parse JSON", "fallback": true}')
                    
                    return SimpleLLMResponse(content=content)
                except TimeoutError as timeout_error:
                    logger.error(f"Pull or call timed out: {timeout_error}")
                    raise
                except Exception as pull_error:
                    logger.error(f"Failed to pull model: {pull_error}")
                    raise
            else:
                logger.error(f"Ollama API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise


class SimpleLLMResponse:
    """Wrapper for LLM response to match LangChain interface."""
    
    def __init__(self, content: str):
        """Initialize response with content."""
        self.content = content
    
    def __str__(self) -> str:
        """Return content as string."""
        return self.content
