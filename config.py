import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from typing import Optional, Dict, Any

load_dotenv()

class Config:
    # Google Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-2.0-flash"
    GEMINI_TEMPERATURE = 0
    
    # Ollama for orchestrator
    OLLAMA_MODEL = "llama3"
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    @classmethod
    def get_gemini_llm(cls) -> ChatGoogleGenerativeAI:
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return ChatGoogleGenerativeAI(
            model=cls.GEMINI_MODEL,
            temperature=cls.GEMINI_TEMPERATURE,
            google_api_key=cls.GOOGLE_API_KEY
        )
    
    @classmethod
    def get_ollama_llm(cls) -> ChatOllama:
        return ChatOllama(
            model=cls.OLLAMA_MODEL,
            base_url=cls.OLLAMA_BASE_URL,
            temperature=0.1
        )