class AIProviderError(Exception):
    pass

class ProviderNotFoundError(AIProviderError):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        super().__init__(f"Provider '{provider_name}' not found")

class APIKeyMissingError(AIProviderError):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        super().__init__(f"API key for '{provider_name}' is missing or empty")

class GenerationError(AIProviderError):
    def __init__(self, provider_name: str, original: Exception | None = None):
        self.provider_name = provider_name
        self.original = original
        msg = f"{provider_name} generation failed: {original}" if original else f"{provider_name} generation failed"
        super().__init__(msg)

class StreamingError(AIProviderError):
    def __init__(self, provider_name: str, original: Exception | None = None):
        self.provider_name = provider_name
        self.original = original
        msg = f"{provider_name} streaming failed: {original}" if original else f"{provider_name} streaming failed"
        super().__init__(msg)

class AllProvidersFailedError(AIProviderError):
    def __init__(self, primary: str, fallback: str, last_error: str):
        self.primary = primary
        self.fallback = fallback
        self.last_error = last_error
        super().__init__(f"All AI providers failed. Primary={primary}, Fallback={fallback}. Last error: {last_error}")
