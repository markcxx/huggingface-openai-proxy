"""Hugging Face OpenAI API Proxy

A FastAPI-based proxy service that converts OpenAI API requests to Hugging Face API calls,
with support for reasoning models and thinking content processing.
"""

__version__ = "1.0.0"
__author__ = "Hugging Face API Proxy Team"

from .models import *
from .converter import HuggingFaceConverter
from .config import config

__all__ = [
    "HuggingFaceConverter",
    "config",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionStreamResponse",
    "Message",
    "Choice",
    "StreamChoice",
    "Delta",
    "Usage",
    "Model",
    "ModelListResponse",
    "ErrorResponse",
]