# src/generation/generator.py

import os
import requests
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# Simple baseline prompt template
# BASELINE_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

# Context:
# {context}

# Question: {question}

# Instructions:
# - Answer the question based ONLY on the information provided in the context above.
# - If the answer cannot be found in the context, say "I cannot find the answer in the provided context."
# - Be concise and direct in your answer.

# Answer:"""

BASELINE_PROMPT = """"You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the provided context.
- If the question requires counting, listing, or comparison:
  - Identify all relevant items in the context.
  - Apply the required condition step by step.
  - Ensure the final count is correct.
- If the answer is partially available, explain what is missing.
- If the answer cannot be found, say: "I cannot find the answer in the provided context."
- Be concise but ensure accuracy.

Answer:"""

class BaselineGenerator:
    """LLM-based answer generator using Ollama."""
    
    def __init__(
        self,
        base_url: str = "https://ollama.ux.uis.no",
        model: str = "qwen2.5:7b",
        api_key: str = None
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('OLLAMA_API_KEY')
        
        if not self.api_key:
            logger.warning("No OLLAMA_API_KEY found. API calls may fail.")
        else:
            logger.info(f"Loaded API key: {self.api_key[:10]}...")
    
    def generate(
        self,
        question: str,
        context: str,
        prompt_template: str = BASELINE_PROMPT
    ) -> str:
        """Generate an answer given a question and context."""
        
        # Format the prompt
        prompt = prompt_template.format(
            context=context,
            question=question
        )
        print(f"THE PROMPT IS: {prompt}")

        # Call Ollama API with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Prepare headers with API key if available
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                # Use longer timeout (3 minutes) for potentially slow model loading
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 256
                        }
                    },
                    headers=headers,
                    timeout=180  # 3 minutes
                )
                response.raise_for_status()
                result = response.json()
                print(f"RESULT IS: {result}")
                return result.get("response", "").strip()
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Request timed out, retrying ({attempt + 1}/{max_retries})...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Error calling Ollama API: Request timed out after {max_retries} attempts")
                    return "Error generating response"
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling Ollama API: {e}")
                return "Error generating response"
    
    def generate_batch(
        self,
        questions: list,
        contexts: list
    ) -> list:
        """Generate answers for multiple questions."""
        answers = []
        for q, c in zip(questions, contexts):
            answer = self.generate(q, c)
            answers.append(answer)
        return answers