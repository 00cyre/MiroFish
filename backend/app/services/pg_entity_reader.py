"""
PostgreSQL Entity Reading and Filtering Service
Drop-in replacement for zep_entity_reader.py using PgGraphStore
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from .pg_graph_store import get_store
from ..utils.logger import get_logger

logger = get_logger('mirofish.pg_entity_reader')


@dataclass
class EntityNode:
    """Entity node data structure"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # Related edge info
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Related node info
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Get entity type (excluding default Entity label)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Filtered entity collection"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """
    PostgreSQL Entity Reading and Filtering Service

    Drop-in replacement for Zep-based ZepEntityReader.
    All data is read from PgGraphStore (Postgres).

    Main features:
    1. Read all nodes from graph
    2. Filter nodes matching predefined entity types
    3. Get related edges and associated node info for each entity
    """

    def __init__(self, api_key: Optional[str] = None):
        # api_key kept for interface compatibility but ignored
        self.store = get_store()

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Get all nodes from the graph

        Args:
            graph_id: Graph ID

        Returns:
            Node list
        """
        logger.info(f"Getting all nodes from graph {graph_id}...")

        rows = self.store.get_all_nodes(graph_id)

        nodes_data = []
        for row in rows:
            nodes_data.append({
                "uuid": row["id"],
                "name": row["name"],
                "labels": row.get("labels") or [],
                "summary": row.get("summary") or "",
                "attributes": row.get("attributes") or {},
            })

        logger.info(f"Retrieved {len(nodes_data)} nodes")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Get all edges from the graph

        Args:
            graph_id: Graph ID

        Returns:
            Edge list
        """
        logger.info(f"Getting all edges from graph {graph_id}...")

        rows = self.store.get_all_edges(graph_id)

        edges_data = []
        for row in rows:
            edges_data.append({
                "uuid": row["id"],
                "name": row["name"],
                "fact": row.get("fact") or "",
                "source_node_uuid": row["source_node_id"],
                "target_node_uuid": row["target_node_id"],
                "attributes": {},
            })

        logger.info(f"Retrieved {len(edges_data)} edges")
        return edges_data

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        Get all related edges for a specified node

        Args:
            node_uuid: Node UUID

        Returns:
            Edge list
        """
        try:
            # We need to find which graph this node belongs to, then get all edges
            node = self.store.get_node(node_uuid)
            if not node:
                logger.warning(f"Node not found: {node_uuid}")
                return []

            graph_id = node["graph_id"]
            all_edges = self.store.get_all_edges(graph_id)

            edges_data = []
            for edge in all_edges:
                if edge["source_node_id"] == node_uuid or edge["target_node_id"] == node_uuid:
                    edges_data.append({
                        "uuid": edge["id"],
                        "name": edge["name"],
                        "fact": edge.get("fact") or "",
                        "source_node_uuid": edge["source_node_id"],
                        "target_node_uuid": edge["target_node_id"],
                        "attributes": {},
                    })

            return edges_data
        except Exception as e:
            logger.warning(f"Failed to get edges for node {node_uuid}: {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        Filter nodes matching predefined entity types

        Filtering logic:
        - If a node's Labels only contain "Entity", it doesn't match our predefined types, skip it
        - If a node's Labels contain labels other than "Entity" and "Node", it matches predefined types, keep it

        Args:
            graph_id: Graph ID
            defined_entity_types: Predefined entity type list (optional, if provided only keep these types)
            enrich_with_edges: Whether to get related edge info for each entity

        Returns:
            FilteredEntities: Filtered entity collection
        """
        logger.info(f"Starting entity filtering for graph {graph_id}...")

        # Get all nodes
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        # Get all edges (for subsequent association lookup)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []

        # Build node UUID to node data mapping
        node_map = {n["uuid"]: n for n in all_nodes}

        # Filter matching entities
        filtered_entities = []
        entity_types_found = set()

        for node in all_nodes:
            labels = node.get("labels", [])

            # Filtering logic: Labels must contain labels other than "Entity" and "Node"
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

            if not custom_labels:
                continue

            # If predefined types specified, check for match
            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            # Create entity node object
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            # Get related edges and nodes
            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                # Get basic info of associated nodes
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        })

                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(f"Filtering complete: total nodes {total_count}, matching {len(filtered_entities)}, "
                   f"entity types: {entity_types_found}")

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        Get a single entity with full context (edges and associated nodes)

        Args:
            graph_id: Graph ID
            entity_uuid: Entity UUID

        Returns:
            EntityNode or None
        """
        try:
            node = self.store.get_node(entity_uuid)

            if not node:
                return None

            # Get all edges for this graph to find related ones
            all_edges = self.get_all_edges(graph_id)
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}

            # Process related edges and nodes
            related_edges = []
            related_node_uuids = set()

            for edge in all_edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                elif edge["target_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            # Get related node information
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })

            return EntityNode(
                uuid=node["id"],
                name=node["name"],
                labels=node.get("labels") or [],
                summary=node.get("summary") or "",
                attributes=node.get("attributes") or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"Failed to get entity {entity_uuid}: {str(e)}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        Get all entities of a specified type

        Args:
            graph_id: Graph ID
            entity_type: Entity type (e.g., "Student", "PublicFigure", etc.)
            enrich_with_edges: Whether to get related edge information

        Returns:
            List of entities
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities
