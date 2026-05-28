"""Code similarity primitives: AST signature and token Jaccard."""
from __future__ import annotations

import ast
import hashlib
import re
import tokenize
from io import BytesIO


class _Normalizer(ast.NodeTransformer):
    """Normalize AST: alpha-rename identifiers, mask literals, drop docstrings."""

    def __init__(self) -> None:
        self._name_map: dict[str, str] = {}
        self._next_id = 0

    def _rename(self, name: str) -> str:
        if name not in self._name_map:
            self._name_map[name] = f"v{self._next_id}"
            self._next_id += 1
        return self._name_map[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._rename(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._rename(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        node.value = type(node.value).__name__
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        return node


def ast_signature(code: str) -> str | None:
    """Return a stable hex hash of the normalised AST, or None if not parseable."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    _Normalizer().visit(tree)
    ast.fix_missing_locations(tree)
    dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha1(dump.encode("utf-8")).hexdigest()[:16]


_NAME_RE = re.compile(r"[A-Za-z_]\w*")


def _tokens(code: str) -> set[str]:
    """Return identifier, operator, and number tokens from code."""
    try:
        toks = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
        result: set[str] = set()
        for t in toks:
            if t.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
                if t.string.strip():
                    result.add(t.string)
        return result
    except (tokenize.TokenizeError, IndentationError):
        return set(_NAME_RE.findall(code))


def token_jaccard(a: str, b: str) -> float:
    """|A∩B| / |A∪B|. 0 if both empty."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
