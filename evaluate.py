"""Tiny repeatable evaluation for the research agent.

When LANGSMITH_TRACING is enabled, these runs are also recorded in LangSmith.
"""

import json
from pathlib import Path

from assistant_graph import research_agent


def main() -> None:
    examples = json.loads(Path("evaluation_examples.json").read_text())
    passed = 0
    for example in examples:
        result = research_agent.invoke(
            {"question": example["question"], "source_notes": example["source_notes"]}
        )
        text = f"{result.get('summary', '')}\n{result.get('tasks', '')}".lower()
        has_plan = bool(result.get("summary")) and bool(result.get("tasks"))
        uses_expected_concept = all(term.lower() in text for term in example["must_contain"])
        ok = has_plan and uses_expected_concept
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} — {example['question']}")
    print(f"\n{passed}/{len(examples)} examples passed")
    if passed != len(examples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
