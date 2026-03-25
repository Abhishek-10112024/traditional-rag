"""LLM integration module supporting multiple backends."""

from typing import Optional, List
from loguru import logger
from config import settings
import requests
import json
import os

# Suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to import transformers (for TransformersLLM)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseLLM:
    """Base LLM interface."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    """LLM using Ollama (local or remote)."""
    
    def __init__(self, model: str = None, base_url: str = None):
        """Initialize Ollama LLM.
        
        Args:
            model: Model name (e.g., 'mistral', 'neural-chat', 'orca-mini')
            base_url: Ollama API base URL
        """
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.endpoint = f"{self.base_url}/api/generate"
        
        logger.info(f"Initializing Ollama LLM with model: {self.model} at {self.base_url}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama server")
                return True
            else:
                logger.warning(f"Ollama server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama server at {self.base_url}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return False
    
    def generate(self,
                 prompt: str,
                 temperature: float = None,
                 top_p: float = None,
                 max_tokens: int = None,
                 stream: bool = False,
                 **kwargs) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        temperature = temperature or settings.TEMPERATURE
        top_p = top_p or settings.TOP_P
        max_tokens = max_tokens or settings.MAX_NEW_TOKENS
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "stream": stream,
            }
            
            logger.debug(f"Calling Ollama generate with payload (truncated): temperature={temperature}, top_p={top_p}")
            
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=300,  # 5 minute timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API returned {response.status_code}")
            
            # Parse response
            if stream:
                # For streaming, collect all responses
                full_response = ""
                for line in response.text.strip().split('\n'):
                    if line:
                        try:
                            chunk = json.loads(line)
                            full_response += chunk.get("response", "")
                        except json.JSONDecodeError:
                            continue
                return full_response.strip()
            else:
                # Non-streaming response
                result = response.json()
                return result.get("response", "").strip()
        
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise


class HuggingFaceLLM(BaseLLM):
    """LLM using HuggingFace Inference API."""
    
    def __init__(self, model: str = None, api_token: str = None):
        """Initialize HuggingFace LLM.
        
        Args:
            model: Model ID from HuggingFace Hub
            api_token: HuggingFace API token (from environment if not provided)
        """
        self.model = model or settings.HUGGINGFACE_MODEL
        self.api_token = api_token or settings.HUGGINGFACE_API_TOKEN
        # Updated endpoint: api-inference.huggingface.co is deprecated, use router.huggingface.co
        self.endpoint = f"https://router.huggingface.co/models/{self.model}"
        
        if not self.api_token:
            logger.warning("HUGGINGFACE_API_TOKEN not set. HuggingFace LLM won't work.")
            logger.info("Set HF_TOKEN environment variable in HuggingFace Spaces settings")
        
        logger.info(f"Initializing HuggingFace LLM with model: {self.model}")
    
    def generate(self, prompt: str, max_tokens: int = None, **kwargs) -> str:
        """Generate text using HuggingFace.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.api_token:
            raise Exception("HUGGINGFACE_API_TOKEN not set. Please set HF_TOKEN in your HuggingFace Spaces settings.")
        
        max_tokens = max_tokens or settings.MAX_NEW_TOKENS
        
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            
            # Use text-generation endpoint which works with most models
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": settings.TEMPERATURE,
                    "top_p": settings.TOP_P,
                    "do_sample": True,
                }
            }
            
            logger.debug(f"Calling HuggingFace API for model: {self.model}")
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
            
            if response.status_code != 200:
                error_msg = f"HuggingFace API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                # Text generation response format
                if "generated_text" in result[0]:
                    text = result[0]["generated_text"]
                    # Remove the prompt from the generated text
                    if text.startswith(prompt):
                        text = text[len(prompt):].strip()
                    return text
                else:
                    return result[0].get("text", "").strip()
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].strip()
            else:
                logger.warning(f"Unexpected response format: {result}")
                return ""
        
        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {e}")
            raise


class TransformersLLM(BaseLLM):
    """LLM using transformers library with local models (works offline)."""
    
    def __init__(self, model: str = "google/flan-t5-large"):
        """Initialize Transformers LLM.
        
        Args:
            model: Model name from HuggingFace Hub (google/flan-t5-large recommended for Spaces)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not installed. Install with: pip install transformers torch")
        
        self.model = model
        logger.info(f"Initializing Transformers LLM with model: {model}")
        
        try:
            # Create text generation pipeline with T5 model
            self.pipeline = pipeline(
                "text2text-generation",  # T5 uses text2text-generation task
                model=model,
                device=-1,  # CPU mode
            )
            logger.info(f"Successfully loaded model: {model}")
        except Exception as e:
            logger.error(f"Error loading model {model}: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = None, **kwargs) -> str:
        """Generate text using transformers pipeline.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        max_tokens = max_tokens or settings.MAX_NEW_TOKENS
        
        try:
            logger.debug(f"Generating with {self.model}...")
            
            # T5 uses text2text-generation which doesn't support some of these parameters
            result = self.pipeline(
                prompt,
                max_length=max_tokens,
                num_beams=4,  # Beam search for better quality
                do_sample=False,
            )
            
            # T5 returns list of dicts with 'generated_text' key
            generated_text = result[0]["generated_text"] if result else ""
            
            return generated_text.strip()
        
        except Exception as e:
            logger.error(f"Error generating with Transformers: {e}")
            raise


def get_llm(llm_type: str = None) -> BaseLLM:
    """Get LLM instance based on configuration.
    
    Args:
        llm_type: Type of LLM ('ollama', 'huggingface', or 'transformers')
        
    Returns:
        LLM instance
    """
    llm_type = llm_type or settings.LLM_TYPE
    
    if llm_type == "ollama":
        return OllamaLLM()
    elif llm_type == "huggingface":
        return HuggingFaceLLM(
            model=settings.HUGGINGFACE_MODEL,
            api_token=settings.HUGGINGFACE_API_TOKEN
        )
    elif llm_type == "transformers":
        return TransformersLLM(model="distilgpt2")  # Lightweight model
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


# Global LLM instance
_llm: BaseLLM = None


def get_llm_instance() -> BaseLLM:
    """Get or create LLM instance (singleton)."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm
