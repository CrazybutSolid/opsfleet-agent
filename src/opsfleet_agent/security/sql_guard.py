"""SQL guard: validate model-written SQL and rewrite it onto governed views.

The model writes ordinary BigQuery SQL against bare table names (``orders``,
``order_items``, ``products``, ``users``). Before anything reaches BigQuery:

1. **Parse** with sqlglot (BigQuery dialect). Unparseable SQL is a *correctable*
   error that goes back to the model for self-correction.
2. **Validate**: exactly one statement; read-only (SELECT / set operations
   only, no DML/DDL/scripting anywhere in the tree); only the four allowed tables
   of the thelook dataset; no PII columns; no dangerous table functions; literal
   filters on product dimensions must be inside the user's scope.
3. **Rewrite**: every allowed table is bound to a *governed CTE* that (a) projects
   only non-PII columns and (b) applies the user's row-level product scope. CTEs
   are prepended, so ``SELECT *`` and joins can only ever see governed data.
4. **Cap** the result size with a LIMIT.

In production the governed CTEs become BigQuery authorized views + row access
policies + policy tags (column-level security), so the guarantee also holds for
any other client. The guard stays as the first, fast, explainable line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from ..catalog import ALLOWED_TABLES, DATASET, PII_COLUMNS, TABLES
from .scope import UserScope

_PROJECT, _DATASET = DATASET.split(".")

# Statement types that may never appear anywhere in the tree.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    t
    for t in (
        getattr(exp, name, None)
        for name in (
            "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable",
            "TruncateTable", "Command", "Grant", "Revoke", "Copy", "Set", "Use",
            "Transaction", "Commit", "Rollback", "Export", "LoadData", "Into",
        )
    )
    if isinstance(t, type)
)

# Functions that reach outside the dataset or execute remote code / models.
_FORBIDDEN_FUNCTION_PREFIXES = (
    "EXTERNAL_QUERY", "ML.", "AI.", "VECTOR_SEARCH", "APPENDS", "CHANGES",
    "SESSION_USER", "KEYS.", "AEAD.", "READ_", "EXPORT", "NET.",
)

# sqlglot >= 26 stores the WITH clause under "with_" (older versions: "with").
_WITH = "with_" if "with_" in exp.Select.arg_types else "with"

# Product dimensions whose literal values are checked against the user's scope.
_SCOPED_COLUMNS = {"category", "brand", "department"}


class SqlRejected(Exception):
    """The SQL was refused. ``code`` is machine-readable, ``message`` is shown to the model."""

    def __init__(self, code: str, message: str, correctable: bool = True):
        super().__init__(message)
        self.code = code
        self.message = message
        self.correctable = correctable


@dataclass
class GovernedQuery:
    original_sql: str
    sql: str  # what is actually sent to BigQuery
    tables: set[str] = field(default_factory=set)


def _lit(value: str) -> str:
    return exp.Literal.string(value).sql(dialect="bigquery")


def _raw(table: str) -> str:
    return f"`{_PROJECT}.{_DATASET}.{table}`"


def governed_ctes(scope: UserScope) -> dict[str, tuple[str, set[str]]]:
    """Governed definition of every allowed table -> (SQL, governed tables it reads)."""
    cols = {t: ", ".join(TABLES[t].column_names) for t in TABLES}
    # Synthetic data contains rows timestamped in the future; hide them so that
    # "this month" / "to date" metrics are honest.
    not_future = "created_at <= CURRENT_TIMESTAMP()"
    if scope.all_products:
        return {
            "products": (f"SELECT {cols['products']} FROM {_raw('products')}", set()),
            "order_items": (f"SELECT {cols['order_items']} FROM {_raw('order_items')} WHERE {not_future}", set()),
            "orders": (f"SELECT {cols['orders']} FROM {_raw('orders')} WHERE {not_future}", set()),
            "users": (f"SELECT {cols['users']} FROM {_raw('users')}", set()),
        }
    preds = [
        f"{dim} IN ({', '.join(_lit(v) for v in values)})"
        for dim, values in scope.dimensions().items()
    ] or ["FALSE"]  # a user with an empty scope sees nothing
    # Scoped users see only the slice of every table that touches their products:
    # items of their products, orders containing them, customers who bought them.
    return {
        "products": (f"SELECT {cols['products']} FROM {_raw('products')} WHERE {' AND '.join(preds)}", set()),
        "order_items": (
            f"SELECT {cols['order_items']} FROM {_raw('order_items')} "
            f"WHERE {not_future} AND product_id IN (SELECT id FROM products)",
            {"products"},
        ),
        "orders": (
            f"SELECT {cols['orders']} FROM {_raw('orders')} "
            f"WHERE {not_future} AND order_id IN (SELECT order_id FROM order_items)",
            {"order_items", "products"},
        ),
        "users": (
            f"SELECT {cols['users']} FROM {_raw('users')} WHERE id IN (SELECT user_id FROM order_items)",
            {"order_items", "products"},
        ),
    }


def _check_function(node: exp.Expression) -> None:
    name = ""
    if isinstance(node, exp.Anonymous):
        name = str(node.this)
    elif isinstance(node, exp.Func):
        name = node.sql_name()
    upper = name.upper()
    if any(upper.startswith(p) for p in _FORBIDDEN_FUNCTION_PREFIXES):
        raise SqlRejected("FORBIDDEN_FUNCTION", f"Function {name} is not allowed.", correctable=False)


def _check_scope_literals(tree: exp.Expression, scope: UserScope) -> None:
    """Fail fast (with a clear message) when the SQL filters on out-of-scope products.

    Row-level filtering already guarantees no out-of-scope rows are returned; this
    check turns a silent empty result into an explicit, explainable refusal.
    """
    dims = scope.dimensions()
    if not dims:
        return

    for cmp in tree.find_all(exp.EQ, exp.In):
        col = cmp.this if isinstance(cmp, exp.In) else None
        values: list[str] = []
        if isinstance(cmp, exp.EQ):
            left, right = cmp.this, cmp.expression
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                col, values = left, [right.this]
            elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                col, values = right, [left.this]
        elif isinstance(col, exp.Column):
            values = [e.this for e in cmp.expressions if isinstance(e, exp.Literal) and e.is_string]
        if not isinstance(col, exp.Column):
            continue
        dim = col.name.lower()
        if dim not in _SCOPED_COLUMNS or dim not in dims:
            continue
        outside = [v for v in values if not scope.allows(dim, v)]
        if outside:
            raise SqlRejected(
                "OUT_OF_SCOPE",
                f"{dim} {', '.join(repr(v) for v in outside)} is outside this user's product scope "
                f"({scope.describe()}). Do not retry; tell the user which products they can analyse.",
                correctable=False,
            )


def govern(sql: str, scope: UserScope, max_rows: int = 200) -> GovernedQuery:
    sql = (sql or "").strip()
    if not sql:
        raise SqlRejected("EMPTY", "Empty SQL.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except ParseError as e:
        raise SqlRejected("PARSE_ERROR", f"SQL does not parse: {str(e).splitlines()[0]}") from e
    if len(statements) != 1:
        raise SqlRejected(
            "MULTI_STATEMENT", "Exactly one SELECT statement is allowed.", correctable=False
        )
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        if isinstance(tree, _FORBIDDEN_NODES):
            raise SqlRejected(
                "NOT_SELECT", f"Only read-only SELECT queries are allowed (got {tree.key.upper()}).",
                correctable=False,
            )
        raise SqlRejected("PARSE_ERROR", "SQL is not a valid SELECT statement.")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlRejected(
                "NOT_SELECT", f"{node.key.upper()} is not allowed; queries are read-only.",
                correctable=False,
            )
        if isinstance(node, (exp.Func, exp.Anonymous)):
            _check_function(node)

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    clash = cte_names & ALLOWED_TABLES
    if clash:
        raise SqlRejected(
            "RESERVED_NAME", f"CTE name(s) {sorted(clash)} are reserved; rename your CTEs."
        )

    used: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        catalog, db = table.catalog.lower(), table.db.lower()
        if not name:  # e.g. UNNEST or table functions surfaced as tables
            raise SqlRejected("TABLE_NOT_ALLOWED", f"Unsupported FROM item: {table.sql('bigquery')}")
        if name in cte_names and not catalog and not db:
            continue
        qualified_ok = (not catalog and not db) or (
            db == _DATASET and catalog in ("", _PROJECT)
        )
        if not qualified_ok or name not in ALLOWED_TABLES:
            raise SqlRejected(
                "TABLE_NOT_ALLOWED",
                f"Table {table.sql('bigquery')} is not allowed. Use only: {', '.join(sorted(ALLOWED_TABLES))}.",
                correctable=False,
            )
        # Bind to the governed CTE of the same name.
        table.set("catalog", None)
        table.set("db", None)
        used.add(name)

    for col in tree.find_all(exp.Column):
        if col.name.lower() in PII_COLUMNS:
            raise SqlRejected(
                "PII_COLUMN",
                f"Column '{col.name}' is personal data and is withheld. Analyse customers by "
                "pseudonymous id and demographics (age, gender, city, state, country) instead.",
                correctable=False,
            )

    if not used:
        raise SqlRejected("NO_TABLE", "Query must read from at least one allowed table.")

    _check_scope_literals(tree, scope)

    # Prepend governed CTEs (plus their dependencies) ahead of the model's own CTEs.
    defs = governed_ctes(scope)
    needed = set(used)
    for t in used:
        needed |= defs[t][1]
    governed = [
        exp.CTE(this=sqlglot.parse_one(defs[t][0], read="bigquery"), alias=exp.TableAlias(this=exp.to_identifier(t)))
        for t in ("products", "order_items", "orders", "users")
        if t in needed
    ]
    existing = tree.args.get(_WITH)
    own = list(existing.expressions) if existing else []
    tree.set(_WITH, exp.With(expressions=governed + own, recursive=bool(existing and existing.args.get("recursive"))))

    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.this)
        except (AttributeError, ValueError, TypeError):
            current = None
    if current is None or current > max_rows:
        tree = tree.limit(max_rows, copy=False)

    return GovernedQuery(original_sql=sql, sql=tree.sql(dialect="bigquery"), tables=used)
