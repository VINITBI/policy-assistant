"""DeepEval judge backed by a local Ollama model (no external API keys, fully open-source)."""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # keep evaluation fully local

from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.llm.ollama_client import OllamaClient  # noqa: E402


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or get_settings().eval_judge_model
        self.client = OllamaClient()
        super().__init__(self.model_name)

    def load_model(self):
        return self

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_) -> str | BaseModel:
        fmt = schema.model_json_schema() if schema else None
        res = self.client.chat([{"role": "user", "content": prompt}], purpose="eval_judge", temperature=0.0,
                               json_format=fmt, max_tokens=2048, model=self.model_name, allow_fallback=False)
        if schema is not None:
            try:
                return schema.model_validate_json(res.text)
            except Exception:  # noqa: BLE001 - DeepEval will try to parse the raw JSON itself
                return res.text
        return res.text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kw) -> str | BaseModel:
        return await asyncio.to_thread(self.generate, prompt, schema, **kw)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True
