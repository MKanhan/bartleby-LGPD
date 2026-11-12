"""Import bindings for one Python module: which local name points at which module or symbol.

`from langchain_groq import ChatGroq as G` → G is ("langchain_groq", "ChatGroq").
`import google.generativeai as genai` → genai is the module "google.generativeai".
`from google import genai` → genai is the module "google.genai" (a submodule imported by name).

`qualify(expr)` turns a callee expression into its fully qualified dotted path, so
`genai.GenerativeModel` becomes "google.generativeai.GenerativeModel" and the catalog can be
asked about module + name instead of guessing from the method name (spec 65).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ImportTable:
    # local name -> fully qualified dotted path of what it refers to
    names: dict[str, str] = field(default_factory=dict)
    # modules imported with `from X import *` (names resolved through the repo index)
    stars: list[str] = field(default_factory=list)

    @classmethod
    def from_tree(cls, tree: ast.AST) -> ImportTable:
        table = cls()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        table.names[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".")[0]
                        table.names.setdefault(root, root)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    if alias.name == "*":
                        table.stars.append(node.module)
                        continue
                    table.names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.level > 0:
                # Relative import: keep the module tail so the repo index can still find it.
                base = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    target = f"{base}.{alias.name}" if base else alias.name
                    table.names[alias.asname or alias.name] = target
        return table

    def qualify(self, expr: ast.AST) -> str | None:
        """Fully qualified dotted path of a Name/Attribute chain rooted at an imported name."""
        parts: list[str] = []
        cur = expr
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return None
        root = self.names.get(cur.id)
        if root is None:
            return None
        return ".".join([root, *reversed(parts)])


def dotted(expr: ast.AST) -> str | None:
    """Plain dotted text of a Name/Attribute chain (`self.__chat.invoke` → 'self.__chat.invoke')."""
    parts: list[str] = []
    cur = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def split_module_name(qualified: str) -> tuple[str, str]:
    """'langchain_groq.ChatGroq' → ('langchain_groq', 'ChatGroq')."""
    if "." not in qualified:
        return "", qualified
    module, name = qualified.rsplit(".", 1)
    return module, name
