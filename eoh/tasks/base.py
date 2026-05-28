"""BaseProblem ABC and AST helpers for template inspection."""
from __future__ import annotations

import ast
import itertools
import logging
import sys
import types
import warnings
from abc import ABC, abstractmethod
from typing import Callable

_module_counter = itertools.count()
logger = logging.getLogger("eoh")


def extract_import_lines(template_program: str) -> str:
    tree = ast.parse(template_program)
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))
    return "\n".join(lines)


def detect_template_kind(template_program: str) -> str:
    """Return 'class', 'multi_function', or 'function'."""
    tree = ast.parse(template_program)
    if any(isinstance(n, ast.ClassDef) for n in tree.body):
        return "class"
    top_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    return "multi_function" if len(top_funcs) > 1 else "function"


def get_entry_name(template_program: str) -> str:
    """Return the primary callable name to look up after exec()."""
    tree = ast.parse(template_program)
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    if classes:
        return classes[0]
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if funcs:
        return funcs[-1]
    raise ValueError("No function or class definition found in template_program.")


class BaseProblem(ABC):
    """Base class for EoH tasks. Set `template_program` and `task_description`, implement `evaluate_program`."""

    template_program: str = ""
    task_description: str = ""

    def __init__(self, timeout: int = 40, n_processes: int = 1):
        import multiprocessing
        self.timeout = timeout
        self.n_processes = multiprocessing.cpu_count() if n_processes == -1 else n_processes

    def evaluate(self, code_string: str) -> float:
        """Exec generated code, find the entry callable, and call evaluate_program."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import numpy as np
            module = types.ModuleType(f"heuristic_module_{next(_module_counter)}")
            import_prefix = extract_import_lines(self.template_program)
            if import_prefix:
                exec(import_prefix, module.__dict__)
            module.__dict__.setdefault("np", np)
            exec(code_string, module.__dict__)
            sys.modules[module.__name__] = module
            entry_name = get_entry_name(self.template_program)
            callable_obj: Callable | None = getattr(module, entry_name, None)
            if callable_obj is None:
                raise NameError(f"Entry point '{entry_name}' not defined by generated code")
            return self.evaluate_program(code_string, callable_obj)

    @abstractmethod
    def evaluate_program(self, program_str: str, callable_func: Callable) -> float | None:
        """Return a fitness value (lower is better) or None on failure."""
        ...
