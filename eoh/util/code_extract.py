"""Extract (algorithm description, code) from an LLM response.

Logic mirrors upstream `evolution.py:Evolution._extract` line-by-line. The two
output lists preserve upstream's API even though we only ever use the first
element of each.
"""
from __future__ import annotations

import ast
import re


def extract(response: str | None) -> tuple[list[str], list[str]]:
    """Return ([algorithm_descriptions], [code_strings])."""
    if not response:
        return [], []

    # ── code ──────────────────────────────────────────────────────────────
    # 1. Fenced code blocks (most reliable).
    code = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)

    if not code:
        # 2. Locate first top-level Python statement and trim trailing prose
        #    until what's left parses.
        start = re.search(r"^(?:import |from |def |class |@)", response, re.MULTILINE)
        if start:
            candidate = response[start.start():].strip()
            lines = candidate.splitlines()
            for trim in range(len(lines)):
                snippet = "\n".join(lines[:len(lines) - trim]).strip()
                if not snippet:
                    break
                try:
                    ast.parse(snippet)
                    code = [snippet]
                    break
                except SyntaxError:
                    continue

    # Strip any leading {description} line the LLM sometimes puts inside the code block.
    code = [re.sub(r"^\s*\{[^}]*\}\s*\n+", "", c, flags=re.DOTALL).strip() for c in code]
    code = [c for c in code if c]

    # ── algorithm description ────────────────────────────────────────────
    # Search only in text BEFORE the code to avoid matching Python dict literals.
    if "```" in response:
        pre_code = response[:response.find("```")].strip()
    elif code:
        idx = response.find(code[0][:60]) if code[0] else -1
        pre_code = response[:idx].strip() if idx > 0 else response.strip()
    else:
        pre_code = response.strip()

    # ≥ 8 chars to skip empty {}, single-letter vars, dict snippets.
    algorithm = re.findall(r"\{([^{}]{8,})\}", pre_code)

    if not algorithm and pre_code:
        algorithm = [pre_code]

    return algorithm, code


def prepend_imports(code: str, template_program: str) -> str:
    """Prepend template's import lines to `code` if they're missing.

    Mirrors upstream `Evolution._prepend_imports`.
    """
    from eoh.tasks.base import extract_import_lines
    prefix = extract_import_lines(template_program)
    if not prefix:
        return code
    missing = [line for line in prefix.splitlines() if line and line not in code]
    if not missing:
        return code
    return "\n".join(missing) + "\n" + code
