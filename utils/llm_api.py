import httpx
import openai


class BaseLLM:
    REQUEST_TIMEOUT_SEC = 60.0
    MAX_RETRIES_ON_TIMEOUT = 12

    def __init__(self, api_key, base_url, model, seed=None, temperature=None, top_p=None, top_k=None):
        self.api_key = api_key
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.REQUEST_TIMEOUT_SEC,
        )
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def _build_extra_body(self):
        return None

    def _build_kwargs(self, prompt):
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt}],
            "timeout": self.REQUEST_TIMEOUT_SEC,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def generate_response_with_usage(self, prompt):
        last_exc = None
        kwargs = self._build_kwargs(prompt)
        for _ in range(self.MAX_RETRIES_ON_TIMEOUT):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                return content, input_tokens, output_tokens
            except (openai.APITimeoutError, httpx.TimeoutException) as e:
                last_exc = e
                continue
        raise last_exc

    def generate_response(self, prompt):
        content, _, _ = self.generate_response_with_usage(prompt)
        return content


class OpenaiLLM(BaseLLM):
    pass


class OpenrouterLLM(BaseLLM):
    def _build_extra_body(self):
        if self.top_k is None:
            return None
        return {"top_k": self.top_k}


class OllamaLLM(BaseLLM):
    def _build_extra_body(self):
        extra = {}
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        return extra or None


def _detect_provider_from_base_url(base_url):
    url = (base_url or "").lower()
    if "openrouter.ai" in url:
        return "openrouter"
    if "localhost:11434" in url or "127.0.0.1:11434" in url or "ollama" in url:
        return "ollama"
    return "openai"


def create_llm(api_key, base_url, model, provider=None, **kwargs):
    resolved_provider = provider or _detect_provider_from_base_url(base_url)
    if resolved_provider == "openrouter":
        return OpenrouterLLM(api_key, base_url, model, **kwargs)
    if resolved_provider == "ollama":
        return OllamaLLM(api_key, base_url, model, **kwargs)
    return OpenaiLLM(api_key, base_url, model, **kwargs)


class LLMAPI:
    def __init__(self, api_key, base_url, model, provider=None, **kwargs):
        self.impl = create_llm(api_key, base_url, model, provider=provider, **kwargs)

    def generate_response_with_usage(self, prompt):
        return self.impl.generate_response_with_usage(prompt)

    def generate_response(self, prompt):
        return self.impl.generate_response(prompt)
