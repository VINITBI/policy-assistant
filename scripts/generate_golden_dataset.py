"""Bootstrap a golden dataset from indexed chunks with the local LLM (review the output before trusting it!).

    python scripts/generate_golden_dataset.py --per-policy 5 --out eval/generated_goldens.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402
from qdrant_client import models  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

settings = get_settings()
setup_logging(settings.log_dir, "INFO")

from app.db.vector_store import get_vector_store  # noqa: E402
from app.llm.ollama_client import get_llm  # noqa: E402
from app.llm.prompts import get_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-policy", type=int, default=5)
    ap.add_argument("--out", default="eval/generated_goldens.yaml")
    args = ap.parse_args()
    store = get_vector_store()
    points, _ = store.client.scroll(store.collection, limit=10000, with_payload=True, with_vectors=False,
                                    scroll_filter=models.Filter(must=[models.FieldCondition(
                                        key="is_active", match=models.MatchValue(value=True))]))
    by_policy: dict[str, list] = {}
    for p in points:
        if p.payload.get("token_count", 0) >= 60:
            by_policy.setdefault(p.payload["policy_name"], []).append(p.payload)
    cases = []
    for name, chunks in by_policy.items():
        for i, ch in enumerate(random.sample(chunks, min(args.per_policy, len(chunks)))):
            res = get_llm().chat(get_prompts().messages("golden_generation", policy_name=name,
                                                        section=ch["section_title"], text=ch["text"]),
                                 purpose="golden_gen", temperature=0.4, json_format="json", max_tokens=300)
            try:
                d = json.loads(re.search(r"\{.*\}", res.text, re.S).group(0))
                cases.append({"id": f"gen-{len(cases) + 1:03d}", "policy": name, "question": d["question"],
                              "expected_answer": d["expected_answer"], "expected_keywords": [],
                              "source_section": ch["section_title"], "source_pages": f"{ch['page_start']}-{ch['page_end']}"})
                print("+", d["question"])
            except Exception as exc:  # noqa: BLE001
                print("skip:", exc)
    Path(args.out).write_text(yaml.safe_dump({"cases": cases}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.out} - review them, then merge into eval/golden_dataset.yaml")


if __name__ == "__main__":
    main()
