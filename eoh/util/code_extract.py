"""Extract algorithm description and code block from an LLM response."""
from __future__ import annotations

import ast
import re


def extract(response: str | None) -> tuple[list[str], list[str]]:
    """Return ([algorithm_descriptions], [code_strings])."""
    if not response:
        return [], []

    # 1. Fenced code blocks.
    code = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)

    if not code:
        # 2. Find first top-level statement and trim until it parses.
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

    # Strip any leading {description} line inside the code block.
    code = [re.sub(r"^\s*\{[^}]*\}\s*\n+", "", c, flags=re.DOTALL).strip() for c in code]
    code = [c for c in code if c]

    # Search for description only before the code block.
    if "```" in response:
        pre_code = response[:response.find("```")].strip()
    elif code:
        idx = response.find(code[0][:60]) if code[0] else -1
        pre_code = response[:idx].strip() if idx > 0 else response.strip()
    else:
        pre_code = response.strip()

    algorithm = re.findall(r"\{([^{}]{8,})\}", pre_code)

    if not algorithm and pre_code:
        algorithm = [pre_code]

    return algorithm, code


def prepend_imports(code: str, template_program: str) -> str:
    """Prepend missing import lines from template to code."""
    from eoh.tasks.base import extract_import_lines
    prefix = extract_import_lines(template_program)
    if not prefix:
        return code
    missing = [line for line in prefix.splitlines() if line and line not in code]
    if not missing:
        return code
    return "\n".join(missing) + "\n" + code
