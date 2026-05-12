import google.generativeai as genai
from typing import List, Dict, Any
from .base import BaseAIProvider
from app.core.config import settings

class GeminiProvider(BaseAIProvider):
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            model_to_use = kwargs.get("model", self.model_name)
            model = genai.GenerativeModel(model_to_use)
            
            # Convert OpenAI-style messages to Gemini style
            contents = []
            system_instruction = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [msg["content"]]})
            
            # Re-init model if we have system instruction
            if system_instruction:
                model = genai.GenerativeModel(
                    model_to_use,
                    system_instruction=system_instruction
                )
            
            response = await model.generate_content_async(
                contents,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.7),
                    max_output_tokens=kwargs.get("max_tokens", 1024),
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini Error: {str(e)}")

    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs):
        try:
            model_to_use = kwargs.get("model", self.model_name)
            model = genai.GenerativeModel(model_to_use)
            contents = []
            system_instruction = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [msg["content"]]})
            
            if system_instruction:
                model = genai.GenerativeModel(model_to_use, system_instruction=system_instruction)
            
            response = await model.generate_content_async(
                contents,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.7),
                    max_output_tokens=kwargs.get("max_tokens", 1024),
                ),
                stream=True
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise Exception(f"Gemini Stream Error: {str(e)}")

    def get_name(self) -> str:
        return "gemini"
