"""The data catalogue: which tables and columns the agent may see, and which are PII.

This is the single source of truth for both the SQL guard (what may be queried)
and the schema tool (what the agent tells executives is available). Columns
classified as PII are physically absent from the governed views the agent
queries, so they cannot be selected, joined on, or leaked through ``SELECT *``.
"""

from __future__ import annotations

from dataclasses import dataclass

DATASET = "bigquery-public-data.thelook_ecommerce"

# Columns that identify a person. Matched by name across *all* tables.
PII_COLUMNS: frozenset[str] = frozenset(
    {
        "first_name",
        "last_name",
        "email",
        "street_address",
        "postal_code",
        "latitude",
        "longitude",
        "user_geom",
        "ip_address",
        "phone",
    }
)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    columns: tuple[Column, ...]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def _cols(*specs: tuple[str, str, str]) -> tuple[Column, ...]:
    return tuple(Column(*s) for s in specs)


TABLES: dict[str, Table] = {
    "orders": Table(
        "orders",
        "One row per customer order (header level).",
        _cols(
            ("order_id", "INT64", "Order identifier"),
            ("user_id", "INT64", "Customer id (pseudonymous key; join to users.id)"),
            ("status", "STRING", "Processing | Shipped | Complete | Returned | Cancelled"),
            ("gender", "STRING", "Customer gender at order time (M/F)"),
            ("created_at", "TIMESTAMP", "Order placed"),
            ("returned_at", "TIMESTAMP", "Return registered (null if not returned)"),
            ("shipped_at", "TIMESTAMP", "Shipped"),
            ("delivered_at", "TIMESTAMP", "Delivered"),
            ("num_of_item", "INT64", "Number of items in the order"),
        ),
    ),
    "order_items": Table(
        "order_items",
        "One row per item sold; the revenue fact table (sale_price).",
        _cols(
            ("id", "INT64", "Order item id"),
            ("order_id", "INT64", "Join to orders.order_id"),
            ("user_id", "INT64", "Join to users.id"),
            ("product_id", "INT64", "Join to products.id"),
            ("inventory_item_id", "INT64", "Inventory unit"),
            ("status", "STRING", "Item status (same values as orders.status)"),
            ("created_at", "TIMESTAMP", "Item ordered"),
            ("shipped_at", "TIMESTAMP", "Shipped"),
            ("delivered_at", "TIMESTAMP", "Delivered"),
            ("returned_at", "TIMESTAMP", "Returned"),
            ("sale_price", "FLOAT64", "Revenue for this item (USD)"),
        ),
    ),
    "products": Table(
        "products",
        "Product catalogue.",
        _cols(
            ("id", "INT64", "Product id"),
            ("cost", "FLOAT64", "Unit cost (USD)"),
            ("category", "STRING", "e.g. Jeans, Swim, Active, Outerwear & Coats"),
            ("name", "STRING", "Product name"),
            ("brand", "STRING", "Brand"),
            ("retail_price", "FLOAT64", "List price (USD)"),
            ("department", "STRING", "Men | Women"),
            ("sku", "STRING", "Stock keeping unit"),
            ("distribution_center_id", "INT64", "Fulfilment centre"),
        ),
    ),
    "users": Table(
        "users",
        "Customers. Identity and contact fields are withheld (PII); demographics and geography at city level remain.",
        _cols(
            ("id", "INT64", "Customer id (pseudonymous)"),
            ("age", "INT64", "Age in years"),
            ("gender", "STRING", "M/F"),
            ("state", "STRING", "State / province"),
            ("city", "STRING", "City"),
            ("country", "STRING", "Country"),
            ("traffic_source", "STRING", "Acquisition channel: Search, Organic, Facebook, Email, Display"),
            ("created_at", "TIMESTAMP", "Account created"),
        ),
    ),
}

ALLOWED_TABLES: frozenset[str] = frozenset(TABLES)


def describe_catalog() -> str:
    """Compact, model-friendly description of the governed schema."""
    lines = []
    for t in TABLES.values():
        lines.append(f"- {t.name}: {t.description}")
        for c in t.columns:
            lines.append(f"    - {c.name} ({c.type}): {c.description}")
    lines.append(
        "- Withheld PII columns (never queryable): " + ", ".join(sorted(PII_COLUMNS))
    )
    return "\n".join(lines)
