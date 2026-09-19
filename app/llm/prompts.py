"""Loads prompts from prompts/prompts.yaml and renders them safely."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import yaml

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
REQUIRED = ("answer", "condense", "multi_query", "self_eval", "messages", "prompt_instructions")


class PromptRegistry:
    def __init__(self, data: dict[str, Any]) -> None:
        missing = [k for k in REQUIRED if k not in data]
        if missing:
            raise ValueError(f"prompts.yaml is missing sections: {missing}")
        self.data = data
        self.version = str(data.get("version", "0"))

    def system(self, name: str) -> str:
        return self.data[name]["system"].strip()

    def user(self, name: str, **values: Any) -> str:
        """Replace only known {placeholders}; literal braces (e.g. JSON examples) are left untouched."""
        template = self.data[name]["user"]
        return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
                                template).strip()

    def messages(self, name: str, **values: Any) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system(name)}, {"role": "user", "content": self.user(name, **values)}]

    def message(self, key: str) -> str:
        return self.data["messages"][key]

    @property
    def instructions(self) -> list[str]:
        return list(self.data["prompt_instructions"])


@lru_cache(maxsize=1)
def get_prompts() -> PromptRegistry:
    path = get_settings().prompts_file
    with open(path, encoding="utf-8") as fh:
        reg = PromptRegistry(yaml.safe_load(fh))
    log.info("prompts loaded", extra={"file": str(path), "prompt_version": reg.version})
    return reg
