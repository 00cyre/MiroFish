"""
PostgreSQL-backed graph storage layer.
Replaces Zep Cloud with local Postgres + NetworkX.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS graphs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL REFERENCES graphs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    labels JSONB DEFAULT '[]'::jsonb,
    summary TEXT DEFAULT '',
    attributes JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_nodes_graph_id ON nodes(graph_id);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL REFERENCES graphs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    fact TEXT DEFAULT '',
    source_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    valid_at TIMESTAMPTZ,
    invalid_at TIMESTAMPTZ,
    expired_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_edges_graph_id ON edges(graph_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
"""

# Unique constraint needed for upsert-by-(graph_id, name)
_EXTRA_DDL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_graph_name ON nodes(graph_id, name);
"""

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PgGraphStore:
    """Postgres-backed graph store with NetworkX export."""

    def __init__(self, db_url: str):
        self._db_url = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None

    # -- connection helpers --------------------------------------------------

    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._db_url)
            self._conn.autocommit = True
        return self._conn

    def _cursor(self):
        return self._get_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # -- schema --------------------------------------------------------------

    def initialize(self):
        """Run DDL to create tables and indexes."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(_DDL)
            cur.execute(_EXTRA_DDL)

    # -- graphs --------------------------------------------------------------

    def create_graph(self, name: str, metadata: Optional[Dict] = None) -> str:
        graph_id = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO graphs (id, name, metadata) VALUES (%s, %s, %s)",
                (graph_id, name, json.dumps(metadata or {})),
            )
        return graph_id

    def delete_graph(self, graph_id: str):
        with self._cursor() as cur:
            cur.execute("DELETE FROM graphs WHERE id = %s", (graph_id,))

    def list_graphs(self) -> List[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM graphs ORDER BY created_at DESC")
            return [dict(r) for r in cur.fetchall()]

    # -- nodes ---------------------------------------------------------------

    def upsert_node(
        self,
        graph_id: str,
        name: str,
        labels: Optional[List[str]] = None,
        summary: str = "",
        attributes: Optional[Dict] = None,
    ) -> str:
        node_id = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO nodes (id, graph_id, name, labels, summary, attributes)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (graph_id, name) DO UPDATE
                    SET labels = EXCLUDED.labels,
                        summary = EXCLUDED.summary,
                        attributes = EXCLUDED.attributes
                RETURNING id
                """,
                (
                    node_id,
                    graph_id,
                    name,
                    json.dumps(labels or []),
                    summary,
                    json.dumps(attributes or {}),
                ),
            )
            return cur.fetchone()["id"]

    def get_node(self, node_id: str) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM nodes WHERE id = %s", (node_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_node_by_name(self, graph_id: str, name: str) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM nodes WHERE graph_id = %s AND name = %s",
                (graph_id, name),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_all_nodes(self, graph_id: str) -> List[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, name, labels, summary, attributes, created_at "
                "FROM nodes WHERE graph_id = %s ORDER BY created_at",
                (graph_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    # -- edges ---------------------------------------------------------------

    def upsert_edge(
        self,
        graph_id: str,
        name: str,
        fact: str,
        source_node_id: str,
        target_node_id: str,
    ) -> str:
        edge_id = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO edges (id, graph_id, name, fact, source_node_id, target_node_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (edge_id, graph_id, name, fact, source_node_id, target_node_id),
            )
            return cur.fetchone()["id"]

    def get_all_edges(self, graph_id: str) -> List[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, name, fact, source_node_id, target_node_id, "
                "created_at, valid_at, invalid_at, expired_at "
                "FROM edges WHERE graph_id = %s ORDER BY created_at",
                (graph_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def invalidate_edge(self, edge_id: str):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE edges SET invalid_at = NOW() WHERE id = %s", (edge_id,)
            )

    def expire_edge(self, edge_id: str):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE edges SET expired_at = NOW() WHERE id = %s", (edge_id,)
            )

    # -- search --------------------------------------------------------------

    def search_text(self, graph_id: str, query: str, limit: int = 10) -> List[dict]:
        pattern = f"%{query}%"
        results: List[dict] = []
        with self._cursor() as cur:
            # Search nodes
            cur.execute(
                """
                SELECT id, name, summary
                FROM nodes
                WHERE graph_id = %s AND (name ILIKE %s OR summary ILIKE %s)
                LIMIT %s
                """,
                (graph_id, pattern, pattern, limit),
            )
            for r in cur.fetchall():
                results.append({
                    "type": "node",
                    "id": r["id"],
                    "name": r["name"],
                    "content": f"{r['name']} — {r['summary']}" if r["summary"] else r["name"],
                    "score": 1.0,
                })

            # Search edges
            cur.execute(
                """
                SELECT id, name, fact
                FROM edges
                WHERE graph_id = %s AND (name ILIKE %s OR fact ILIKE %s)
                LIMIT %s
                """,
                (graph_id, pattern, pattern, limit),
            )
            for r in cur.fetchall():
                results.append({
                    "type": "edge",
                    "id": r["id"],
                    "name": r["name"],
                    "content": r["fact"] or r["name"],
                    "score": 1.0,
                })

        return results[:limit]

    # -- stats ---------------------------------------------------------------

    def get_graph_stats(self, graph_id: str) -> dict:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM nodes WHERE graph_id = %s", (graph_id,)
            )
            total_nodes = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT COUNT(*) AS cnt FROM edges WHERE graph_id = %s", (graph_id,)
            )
            total_edges = cur.fetchone()["cnt"]

            # Count by label (labels is a JSONB array)
            cur.execute(
                """
                SELECT label, COUNT(*) AS cnt
                FROM nodes, jsonb_array_elements_text(labels) AS label
                WHERE graph_id = %s
                GROUP BY label
                """,
                (graph_id,),
            )
            entity_types = {r["label"]: r["cnt"] for r in cur.fetchall()}

            # Count by edge name
            cur.execute(
                """
                SELECT name, COUNT(*) AS cnt
                FROM edges
                WHERE graph_id = %s
                GROUP BY name
                """,
                (graph_id,),
            )
            relation_types = {r["name"]: r["cnt"] for r in cur.fetchall()}

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    # -- NetworkX export -----------------------------------------------------

    def build_networkx_graph(self, graph_id: str) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in self.get_all_nodes(graph_id):
            G.add_node(
                node["id"],
                name=node["name"],
                labels=node["labels"],
                summary=node["summary"],
                attributes=node["attributes"],
            )
        for edge in self.get_all_edges(graph_id):
            G.add_edge(
                edge["source_node_id"],
                edge["target_node_id"],
                id=edge["id"],
                name=edge["name"],
                fact=edge["fact"],
                created_at=edge["created_at"],
                valid_at=edge["valid_at"],
                invalid_at=edge["invalid_at"],
                expired_at=edge["expired_at"],
            )
        return G


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store_instance: Optional[PgGraphStore] = None


def get_store() -> PgGraphStore:
    """Return a cached PgGraphStore singleton.

    Reads POSTGRES_URL from env, falling back to ~/.mirofish/config.json.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    db_url = os.environ.get("POSTGRES_URL")
    if not db_url:
        from pathlib import Path

        cfg_file = Path.home() / ".mirofish" / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                db_url = cfg.get("postgresUrl")
            except Exception:
                pass

    if not db_url:
        db_url = "postgresql://localhost:5432/mirofish"

    _store_instance = PgGraphStore(db_url)
    _store_instance.initialize()
    return _store_instance
