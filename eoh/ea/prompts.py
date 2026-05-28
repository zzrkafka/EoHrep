"""Prompt templates — verbatim from upstream `evolution.py:_build_prompt`.

DO NOT EDIT WITHOUT DIFFING UPSTREAM. Any drift here is a reproduction-killing
silent change. See notes/06_eoh_repro_design.md §5.

All builders return a single string (no separate system message — upstream
sends only role=user content).
"""
from __future__ import annotations

from eoh.tasks.base import detect_template_kind


def func_spec(template_program: str) -> str:
    """Render the per-template implementation directive used in every prompt."""
    kind = detect_template_kind(template_program)
    if kind == "class":
        verb = "implement the following Python class"
    elif kind == "multi_function":
        verb = "implement the following Python functions"
    else:
        verb = "implement the following Python function"
    return (
        f"{verb}:\n"
        f"```python\n{template_program.strip()}\n```\n"
        "Do not give additional explanations."
    )


def parent_block(parents: list[dict]) -> str:
    return "\n".join(
        f"No.{i + 1} algorithm and the corresponding code are:\n"
        f"{p['algorithm']}\n{p['code']}"
        for i, p in enumerate(parents)
    )


def build_prompt(
    operator: str,
    task_description: str,
    template_program: str,
    parents: list[dict] | dict | None,
) -> str:
    """Dispatch to the verbatim per-operator template.

    Parent shape:
        i1          → None
        e1, e2      → list[dict] with keys 'algorithm', 'code'
        m1, m2, m3  → single dict with the same keys
    """
    spec = func_spec(template_program)

    if operator == "i1":
        return (
            f"{task_description}\n"
            "First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, {spec}"
        )

    if operator == "e1":
        assert isinstance(parents, list)
        block = parent_block(parents)
        return (
            f"{task_description}\n"
            f"I have {len(parents)} existing algorithms with their codes as follows:\n{block}\n"
            "Please help me create a new algorithm that has a totally different form from the given ones.\n"
            "First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, {spec}"
        )

    if operator == "e2":
        assert isinstance(parents, list)
        block = parent_block(parents)
        return (
            f"{task_description}\n"
            f"I have {len(parents)} existing algorithms with their codes as follows:\n{block}\n"
            "Please help me create a new algorithm that has a totally different form from the given ones "
            "but can be motivated from them.\n"
            "Firstly, identify the common backbone idea in the provided algorithms. "
            "Secondly, based on the backbone idea describe your new algorithm in one sentence. "
            f"The description must be inside a brace. Thirdly, {spec}"
        )

    if operator == "m1":
        assert isinstance(parents, dict)
        return (
            f"{task_description}\n"
            "I have one algorithm with its code as follows.\n"
            f"Algorithm description: {parents['algorithm']}\nCode:\n{parents['code']}\n"
            "Please assist me in creating a new algorithm that has a different form but can be a "
            "modified version of the algorithm provided.\n"
            "First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, {spec}"
        )

    if operator == "m2":
        assert isinstance(parents, dict)
        return (
            f"{task_description}\n"
            "I have one algorithm with its code as follows.\n"
            f"Algorithm description: {parents['algorithm']}\nCode:\n{parents['code']}\n"
            "Please identify the main algorithm parameters and assist me in creating a new algorithm "
            "that has different parameter settings.\n"
            "First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, {spec}"
        )

    if operator == "m3":
        assert isinstance(parents, dict)
        kind = detect_template_kind(template_program)
        if kind == "class":
            keep_str = "keeping the class interface (name, method signatures, inputs, and outputs) unchanged"
        elif kind == "multi_function":
            keep_str = "keeping all function names, inputs, and outputs unchanged"
        else:
            keep_str = "keeping the function name, inputs, and outputs unchanged"
        return (
            "First, identify the main components in the code below. "
            "Next, analyze whether any can be overfit to in-distribution instances. "
            "Then, simplify the components to enhance generalization to out-of-distribution instances. "
            f"Finally, provide the revised code {keep_str}.\n"
            f"{parents['code']}\nDo not give additional explanations."
        )

    raise ValueError(f"Unknown operator: {operator}")
