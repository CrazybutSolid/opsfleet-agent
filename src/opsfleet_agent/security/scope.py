"""Per-user data entitlements ("each user may only analyse products related to them").

Scopes come from ``config/users.toml`` in the prototype. In production the same
shape is read from the identity provider / an entitlements table, keyed by the
authenticated principal, never from anything the user types.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class UserScope:
    user_id: str
    display_name: str
    role: str = ""
    all_products: bool = False
    departments: tuple[str, ...] = field(default_factory=tuple)
    categories: tuple[str, ...] = field(default_factory=tuple)
    brands: tuple[str, ...] = field(default_factory=tuple)

    def dimensions(self) -> dict[str, tuple[str, ...]]:
        """Restricted product dimensions -> allowed values (empty dict = unrestricted)."""
        if self.all_products:
            return {}
        dims = {"department": self.departments, "category": self.categories, "brand": self.brands}
        return {k: v for k, v in dims.items() if v}

    def allows(self, dimension: str, value: str) -> bool:
        allowed = self.dimensions().get(dimension)
        if not allowed:
            return True
        return value.casefold() in {a.casefold() for a in allowed}

    def describe(self) -> str:
        if self.all_products:
            return "all products"
        parts = [f"{dim} in ({', '.join(vals)})" for dim, vals in self.dimensions().items()]
        return " AND ".join(parts) if parts else "no products"


class UnknownUser(KeyError):
    pass


def load_users(path: Path) -> dict[str, UserScope]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    users = {}
    for user_id, spec in raw.get("users", {}).items():
        users[user_id] = UserScope(
            user_id=user_id,
            display_name=spec.get("display_name", user_id),
            role=spec.get("role", ""),
            all_products=bool(spec.get("all_products", False)),
            departments=tuple(spec.get("departments", ())),
            categories=tuple(spec.get("categories", ())),
            brands=tuple(spec.get("brands", ())),
        )
    return users


def get_user(path: Path, user_id: str) -> UserScope:
    users = load_users(path)
    if user_id not in users:
        raise UnknownUser(f"Unknown user '{user_id}'. Known users: {', '.join(sorted(users))}")
    return users[user_id]
