import numpy as np
import openai
from sentence_transformers import SentenceTransformer


class BaseEmbedding:
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _normalize_input(self, text):
        if isinstance(text, list):
            return "\n".join(str(t) for t in text)
        return str(text)

    def embed_text(self, text):
        raise NotImplementedError

    def embed_texts(self, texts):
        raise NotImplementedError


class SentenceTransformersEmbedding(BaseEmbedding):
    def __init__(self, model, api_key=None, base_url=None):
        super().__init__(api_key or "", base_url or "", model)
        self._st = SentenceTransformer(model)

    def embed_text(self, text):
        normalized_text = self._normalize_input(text)
        v = self._st.encode(
            normalized_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(v, dtype=np.float32)

    def embed_texts(self, texts):
        normalized_texts = [self._normalize_input(t) for t in texts]
        arr = self._st.encode(
            normalized_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [np.array(row, dtype=np.float32) for row in arr]


class OllamaEmbedding(BaseEmbedding):
    def __init__(self, api_key, base_url, model):
        super().__init__(api_key, base_url, model)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def embed_text(self, text):
        normalized_text = self._normalize_input(text)
        response = self.client.embeddings.create(
            model=self.model,
            input=normalized_text,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_texts(self, texts):
        normalized_texts = [self._normalize_input(text) for text in texts]
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=normalized_texts,
            )
            return [np.array(item.embedding, dtype=np.float32) for item in response.data]
        except Exception as e:
            print(f"[Embedding] Batch request failed: {e}")
            print(f"[Embedding] batch_size={len(normalized_texts)} model={self.model!r}")
            for i, t in enumerate(normalized_texts):
                preview = t[:800] + ("..." if len(t) > 800 else "")
                print(f"[Embedding] batch item index={i} char_len={len(t)} preview={preview!r}")
            out = []
            for i, t in enumerate(normalized_texts):
                try:
                    r = self.client.embeddings.create(
                        model=self.model,
                        input=t,
                    )
                    out.append(np.array(r.data[0].embedding, dtype=np.float32))
                except Exception as e2:
                    print(f"[Embedding] Single-item failed at index={i}: {e2}")
                    tail = t[-400:] if len(t) > 400 else t
                    print(
                        f"[Embedding] failed item index={i} char_len={len(t)} "
                        f"head={t[:1200]!r} tail={tail!r}"
                    )
                    raise e2 from e
            print("[Embedding] Single-item fallback OK; batch path failed only.")
            return out
