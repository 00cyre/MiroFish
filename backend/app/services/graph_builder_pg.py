"""
Graph building service — Postgres + LLM extraction backend.
Drop-in replacement for graph_builder.py (same public interface).
"""

import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.llm_client import LLMClient
from .entity_extractor import EntityExtractor
from .pg_graph_store import PgGraphStore, get_store
from .text_processor import TextProcessor


@dataclass
class GraphInfo:
    """Graph information (mirrors graph_builder.GraphInfo)."""

    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """Build knowledge graphs using Postgres + LLM entity extraction.

    Public interface is identical to the Zep-backed ``GraphBuilderService``.
    """

    def __init__(self, api_key: Optional[str] = None):
        # api_key kept for interface compatibility (ignored — we use Postgres now)
        self.store: PgGraphStore = get_store()
        self.task_manager = TaskManager()
        self._llm = LLMClient()
        self._extractor = EntityExtractor(self._llm)

    # -- public: async build -------------------------------------------------

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """Start a background graph-build and return the task ID."""
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size),
            daemon=True,
        )
        thread.start()
        return task_id

    # -- public: graph CRUD --------------------------------------------------

    def create_graph(self, name: str) -> str:
        return self.store.create_graph(name)

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """No-op for Postgres backend (ontology is used at extraction time)."""
        pass

    def delete_graph(self, graph_id: str):
        self.store.delete_graph(graph_id)

    def list_graphs(self) -> List[GraphInfo]:
        infos = []
        for g in self.store.list_graphs():
            stats = self.store.get_graph_stats(g["id"])
            infos.append(
                GraphInfo(
                    graph_id=g["id"],
                    node_count=stats["total_nodes"],
                    edge_count=stats["total_edges"],
                    entity_types=list(stats["entity_types"].keys()),
                )
            )
        return infos

    def get_graph_info(self, graph_id: str) -> GraphInfo:
        stats = self.store.get_graph_stats(graph_id)
        return GraphInfo(
            graph_id=graph_id,
            node_count=stats["total_nodes"],
            edge_count=stats["total_edges"],
            entity_types=list(stats["entity_types"].keys()),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Return full graph data dict (compatible with Zep version)."""
        nodes = self.store.get_all_nodes(graph_id)
        edges = self.store.get_all_edges(graph_id)

        node_map = {n["id"]: n["name"] for n in nodes}

        nodes_data = []
        for n in nodes:
            created_at = n.get("created_at")
            nodes_data.append({
                "uuid": n["id"],
                "name": n["name"],
                "labels": n.get("labels", []),
                "summary": n.get("summary", ""),
                "attributes": n.get("attributes", {}),
                "created_at": str(created_at) if created_at else None,
            })

        edges_data = []
        for e in edges:
            edges_data.append({
                "uuid": e["id"],
                "name": e["name"],
                "fact": e.get("fact", ""),
                "fact_type": e["name"],
                "source_node_uuid": e["source_node_id"],
                "target_node_uuid": e["target_node_id"],
                "source_node_name": node_map.get(e["source_node_id"], ""),
                "target_node_name": node_map.get(e["target_node_id"], ""),
                "attributes": {},
                "created_at": str(e["created_at"]) if e.get("created_at") else None,
                "valid_at": str(e["valid_at"]) if e.get("valid_at") else None,
                "invalid_at": str(e["invalid_at"]) if e.get("invalid_at") else None,
                "expired_at": str(e["expired_at"]) if e.get("expired_at") else None,
                "episodes": [],
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    # -- worker --------------------------------------------------------------

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="Starting graph construction...",
            )

            # 1. Create graph
            graph_id = self.store.create_graph(graph_name)
            self.task_manager.update_task(
                task_id, progress=10, message=f"Graph created: {graph_id}"
            )

            # 2. Split text into chunks
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id, progress=15, message=f"Text split into {total_chunks} chunks"
            )

            # 3. Extract entities and upsert for each chunk
            for idx, chunk in enumerate(chunks):
                progress = 15 + int((idx + 1) / total_chunks * 75)  # 15% – 90%
                self.task_manager.update_task(
                    task_id,
                    progress=progress,
                    message=f"Processing chunk {idx + 1}/{total_chunks}...",
                )

                extraction = self._extractor.extract(chunk, ontology)

                # Upsert entities as nodes
                name_to_id: Dict[str, str] = {}
                for ent in extraction.get("entities", []):
                    ent_name = ent.get("name", "").strip()
                    if not ent_name:
                        continue
                    node_id = self.store.upsert_node(
                        graph_id=graph_id,
                        name=ent_name,
                        labels=[ent.get("type", "Entity")],
                        summary=ent.get("summary", ""),
                        attributes=ent.get("attributes", {}),
                    )
                    name_to_id[ent_name] = node_id

                # Upsert relationships as edges
                for rel in extraction.get("relationships", []):
                    src_name = rel.get("source", "").strip()
                    tgt_name = rel.get("target", "").strip()
                    if not src_name or not tgt_name:
                        continue

                    # Ensure source/target nodes exist
                    if src_name not in name_to_id:
                        node = self.store.get_node_by_name(graph_id, src_name)
                        if node:
                            name_to_id[src_name] = node["id"]
                        else:
                            name_to_id[src_name] = self.store.upsert_node(
                                graph_id=graph_id, name=src_name, labels=["Entity"]
                            )
                    if tgt_name not in name_to_id:
                        node = self.store.get_node_by_name(graph_id, tgt_name)
                        if node:
                            name_to_id[tgt_name] = node["id"]
                        else:
                            name_to_id[tgt_name] = self.store.upsert_node(
                                graph_id=graph_id, name=tgt_name, labels=["Entity"]
                            )

                    self.store.upsert_edge(
                        graph_id=graph_id,
                        name=rel.get("relation", "RELATED_TO"),
                        fact=rel.get("fact", ""),
                        source_node_id=name_to_id[src_name],
                        target_node_id=name_to_id[tgt_name],
                    )

            # 4. Gather final stats
            self.task_manager.update_task(
                task_id, progress=95, message="Retrieving graph information..."
            )
            graph_info = self.get_graph_info(graph_id)

            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                },
            )

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    # -- compatibility shims used by graph.py build_task ---------------------

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """Process chunks through LLM extraction and store in Postgres.

        Returns an empty list (no episode UUIDs in the Postgres backend).
        """
        total = len(chunks)
        ontology: Dict[str, Any] = {}  # caller should set via set_ontology context

        for idx, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(
                    f"Processing chunk {idx + 1}/{total}...",
                    (idx + 1) / total,
                )
            extraction = self._extractor.extract(chunk, ontology)

            name_to_id: Dict[str, str] = {}
            for ent in extraction.get("entities", []):
                ent_name = ent.get("name", "").strip()
                if not ent_name:
                    continue
                node_id = self.store.upsert_node(
                    graph_id=graph_id,
                    name=ent_name,
                    labels=[ent.get("type", "Entity")],
                    summary=ent.get("summary", ""),
                    attributes=ent.get("attributes", {}),
                )
                name_to_id[ent_name] = node_id

            for rel in extraction.get("relationships", []):
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                if not src or not tgt:
                    continue
                if src not in name_to_id:
                    node = self.store.get_node_by_name(graph_id, src)
                    if node:
                        name_to_id[src] = node["id"]
                    else:
                        name_to_id[src] = self.store.upsert_node(
                            graph_id=graph_id, name=src, labels=["Entity"]
                        )
                if tgt not in name_to_id:
                    node = self.store.get_node_by_name(graph_id, tgt)
                    if node:
                        name_to_id[tgt] = node["id"]
                    else:
                        name_to_id[tgt] = self.store.upsert_node(
                            graph_id=graph_id, name=tgt, labels=["Entity"]
                        )
                self.store.upsert_edge(
                    graph_id=graph_id,
                    name=rel.get("relation", "RELATED_TO"),
                    fact=rel.get("fact", ""),
                    source_node_id=name_to_id[src],
                    target_node_id=name_to_id[tgt],
                )

        return []  # no episode UUIDs

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ):
        """No-op — Postgres backend processes synchronously."""
        if progress_callback:
            progress_callback("Processing complete", 1.0)
