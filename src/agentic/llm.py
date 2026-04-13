"""Simple LLM wrapper for agent decision-making using Ollama."""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

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
    
    def invoke(self, prompt: str) -> 'SimpleLLMResponse':
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            SimpleLLMResponse object with content attribute
        """
        try:
            response = self._client.generate( # Call the Ollama API to generate a response based on the prompt
                model=self.model,
                prompt=prompt,
                stream=False
            )
            
            content = response.get('response', '') # Extract the generated content from the response
            
            if not content:
                raise ResponseError("Empty response from Ollama")
            
            return SimpleLLMResponse(content=content)
            
        except ResponseError as e:
            if getattr(e, 'status_code', None) == 404:
                logger.warning(f"Model {self.model} not found, attempting to pull...")
                try:
                    self._client.pull(self.model)
                    response = self._client.generate(
                        model=self.model,
                        prompt=prompt,
                        stream=False
                    )
                    content = response.get('response', '')
                    return SimpleLLMResponse(content=content)
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
