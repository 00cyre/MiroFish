"""
PostgreSQL Retrieval Tools Service
Drop-in replacement for zep_tools.py using PgGraphStore + NetworkX

Core retrieval tools (optimized):
1. InsightForge (deep insight retrieval) - Most powerful hybrid retrieval, auto-generates sub-queries and searches across multiple dimensions
2. PanoramaSearch (broad search) - Get the full picture, including expired content
3. QuickSearch (simple search) - Quick retrieval
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .pg_graph_store import get_store
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient

logger = get_logger('mirofish.pg_tools')


@dataclass
class SearchResult:
    """Search result"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }

    def to_text(self) -> str:
        """Convert to text format for LLM comprehension"""
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} relevant items"]

        if self.facts:
            text_parts.append("\n### Related facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node information"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }

    def to_text(self) -> str:
        """Convert to text format"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unknown type")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge information"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Temporal information
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relationship: {source} --[{self.name}]--> {target}\nFact: {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "Unknown"
            invalid_at = self.invalid_at or "Present"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expired: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """Whether it has expired"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """Whether it has been invalidated"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Deep insight retrieval result (InsightForge)
    Contains retrieval results from multiple sub-queries, along with comprehensive analysis
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Retrieval results by dimension
    semantic_facts: List[str] = field(default_factory=list)  # Semantic search results
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Entity insights
    relationship_chains: List[str] = field(default_factory=list)  # Relationship chains

    # Statistics
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }

    def to_text(self) -> str:
        """Convert to detailed text format for LLM comprehension"""
        text_parts = [
            f"## Future Prediction Deep Analysis",
            f"Analysis query: {self.query}",
            f"Prediction scenario: {self.simulation_requirement}",
            f"\n### Prediction Data Statistics",
            f"- Related prediction facts: {self.total_facts}",
            f"- Entities involved: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}"
        ]

        # Sub-queries
        if self.sub_queries:
            text_parts.append(f"\n### Analyzed Sub-queries")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        # Semantic search results
        if self.semantic_facts:
            text_parts.append(f"\n### [Key Facts] (Please cite these verbatim in the report)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Entity insights
        if self.entity_insights:
            text_parts.append(f"\n### [Core Entities]")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f'  Summary: "{entity.get("summary")}"')
                if entity.get('related_facts'):
                    text_parts.append(f"  Related facts: {len(entity.get('related_facts', []))}")

        # Relationship chains
        if self.relationship_chains:
            text_parts.append(f"\n### [Relationship Chains]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Broad search result (Panorama)
    Contains all related information, including expired content
    """
    query: str

    # All nodes
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # All edges (including expired ones)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Currently active facts
    active_facts: List[str] = field(default_factory=list)
    # Expired/invalidated facts (historical records)
    historical_facts: List[str] = field(default_factory=list)

    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }

    def to_text(self) -> str:
        """Convert to text format (full version, no truncation)"""
        text_parts = [
            f"## Broad Search Results (Future Panoramic View)",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Currently active facts: {self.active_count}",
            f"- Historical/expired facts: {self.historical_count}"
        ]

        # Currently active facts (full output, no truncation)
        if self.active_facts:
            text_parts.append(f"\n### [Currently Active Facts] (Simulation results verbatim)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Historical/expired facts (full output, no truncation)
        if self.historical_facts:
            text_parts.append(f"\n### [Historical/Expired Facts] (Evolution process records)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Key entities (full output, no truncation)
        if self.all_nodes:
            text_parts.append(f"\n### [Entities Involved]")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Single agent interview result"""
    agent_name: str
    agent_role: str  # Role type (e.g., student, teacher, media, etc.)
    agent_bio: str  # Biography
    question: str  # Interview question
    response: str  # Interview response
    key_quotes: List[str] = field(default_factory=list)  # Key quotes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Display full agent_bio without truncation
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key Quotes:**\n"
            for quote in self.key_quotes:
                # Clean various quotation marks
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # Remove leading punctuation
                while clean_quote and clean_quote[0] in '\uff0c,\uff1b;\uff1a:\u3001\u3002\uff01\uff1f\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Filter out junk content containing question numbers (Question 1-9)
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Truncate overly long content (truncate at period, not hard cutoff)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interview result (Interview)
    Contains interview responses from multiple simulated agents
    """
    interview_topic: str  # Interview topic
    interview_questions: List[str]  # Interview question list

    # Agents selected for interview
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Interview responses from each agent
    interviews: List[AgentInterview] = field(default_factory=list)

    # Reasoning for agent selection
    selection_reasoning: str = ""
    # Consolidated interview summary
    summary: str = ""

    # Statistics
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }

    def to_text(self) -> str:
        """Convert to detailed text format for LLM comprehension and report citation"""
        text_parts = [
            "## In-Depth Interview Report",
            f"**Interview Topic:** {self.interview_topic}",
            f"**Interviews Conducted:** {self.interviewed_count} / {self.total_agents} simulated agents",
            "\n### Interviewee Selection Reasoning",
            self.selection_reasoning or "(Auto-selected)",
            "\n---",
            "\n### Interview Transcripts",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        text_parts.append("\n### Interview Summary and Key Insights")
        text_parts.append(self.summary or "(No summary)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    PostgreSQL Retrieval Tools Service

    Drop-in replacement for Zep-based ZepToolsService.
    All data is read from PgGraphStore (Postgres) + NetworkX.

    [Core Retrieval Tools - Optimized]
    1. insight_forge - Deep insight retrieval (most powerful, auto-generates sub-queries, multi-dimensional retrieval)
    2. panorama_search - Broad search (get the full picture, including expired content)
    3. quick_search - Simple search (quick retrieval)
    4. interview_agents - In-depth interview (interview simulated agents, get multi-perspective viewpoints)

    [Basic Tools]
    - search_graph - Graph semantic search
    - get_all_nodes - Get all nodes in the graph
    - get_all_edges - Get all edges in the graph (with temporal information)
    - get_node_detail - Get detailed node information
    - get_node_edges - Get edges related to a node
    - get_entities_by_type - Get entities by type
    - get_entity_summary - Get relationship summary for an entity
    """

    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        # api_key kept for interface compatibility but ignored
        self.store = get_store()
        # LLM client used for InsightForge sub-query generation
        self._llm_client = llm_client
        logger.info("ZepToolsService (PgGraphStore) initialized successfully")

    @property
    def llm(self) -> LLMClient:
        """Lazy initialization of LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Graph text search

        Uses PgGraphStore ILIKE search for relevant information in the graph.

        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return
            scope: Search scope, "edges" or "nodes"

        Returns:
            SearchResult: Search results
        """
        logger.info(f"Graph search: graph_id={graph_id}, query={query[:50]}...")

        results = self.store.search_text(graph_id, query, limit=limit)

        facts = []
        edges = []
        nodes = []

        for r in results:
            if r["type"] == "edge" and (scope in ["edges", "both"]):
                facts.append(r["content"])
                edges.append({
                    "uuid": r["id"],
                    "name": r["name"],
                    "fact": r["content"],
                    "source_node_uuid": "",
                    "target_node_uuid": "",
                })
            elif r["type"] == "node" and (scope in ["nodes", "both"]):
                facts.append(f"[{r['name']}]: {r['content']}")
                nodes.append({
                    "uuid": r["id"],
                    "name": r["name"],
                    "labels": [],
                    "summary": r["content"],
                })

        # If scope is "edges" but we also want to capture node matches for facts
        if scope == "edges":
            for r in results:
                if r["type"] == "node":
                    facts.append(f"[{r['name']}]: {r['content']}")

        logger.info(f"Search completed: found {len(facts)} relevant facts")

        return SearchResult(
            facts=facts,
            edges=edges,
            nodes=nodes,
            query=query,
            total_count=len(facts)
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Get all nodes in the graph

        Args:
            graph_id: Graph ID

        Returns:
            List of nodes
        """
        logger.info(f"Fetching all nodes for graph {graph_id}...")

        rows = self.store.get_all_nodes(graph_id)

        result = []
        for row in rows:
            result.append(NodeInfo(
                uuid=row["id"],
                name=row["name"],
                labels=row.get("labels") or [],
                summary=row.get("summary") or "",
                attributes=row.get("attributes") or {}
            ))

        logger.info(f"Fetched {len(result)} nodes")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        Get all edges in the graph (with temporal information)

        Args:
            graph_id: Graph ID
            include_temporal: Whether to include temporal information (default True)

        Returns:
            List of edges (containing created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(f"Fetching all edges for graph {graph_id}...")

        rows = self.store.get_all_edges(graph_id)

        # Build node name map for enrichment
        node_rows = self.store.get_all_nodes(graph_id)
        node_name_map = {r["id"]: r["name"] for r in node_rows}

        result = []
        for row in rows:
            edge_info = EdgeInfo(
                uuid=row["id"],
                name=row["name"],
                fact=row.get("fact") or "",
                source_node_uuid=row["source_node_id"],
                target_node_uuid=row["target_node_id"],
                source_node_name=node_name_map.get(row["source_node_id"]),
                target_node_name=node_name_map.get(row["target_node_id"]),
            )

            if include_temporal:
                edge_info.created_at = str(row["created_at"]) if row.get("created_at") else None
                edge_info.valid_at = str(row["valid_at"]) if row.get("valid_at") else None
                edge_info.invalid_at = str(row["invalid_at"]) if row.get("invalid_at") else None
                edge_info.expired_at = str(row["expired_at"]) if row.get("expired_at") else None

            result.append(edge_info)

        logger.info(f"Fetched {len(result)} edges")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Get detailed information for a single node

        Args:
            node_uuid: Node UUID

        Returns:
            Node information or None
        """
        logger.info(f"Fetching node details: {node_uuid[:8]}...")

        try:
            row = self.store.get_node(node_uuid)

            if not row:
                return None

            return NodeInfo(
                uuid=row["id"],
                name=row["name"],
                labels=row.get("labels") or [],
                summary=row.get("summary") or "",
                attributes=row.get("attributes") or {}
            )
        except Exception as e:
            logger.error(f"Failed to fetch node details: {str(e)}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Get all edges related to a node

        Args:
            graph_id: Graph ID
            node_uuid: Node UUID

        Returns:
            List of edges
        """
        logger.info(f"Fetching edges for node {node_uuid[:8]}...")

        try:
            all_edges = self.get_all_edges(graph_id)

            result = []
            for edge in all_edges:
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)

            logger.info(f"Found {len(result)} edges related to the node")
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch node edges: {str(e)}")
            return []

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str
    ) -> List[NodeInfo]:
        """
        Get entities by type

        Args:
            graph_id: Graph ID
            entity_type: Entity type (e.g., Student, PublicFigure, etc.)

        Returns:
            List of entities matching the type
        """
        logger.info(f"Fetching entities of type {entity_type}...")

        all_nodes = self.get_all_nodes(graph_id)

        filtered = []
        for node in all_nodes:
            if entity_type in node.labels:
                filtered.append(node)

        logger.info(f"Found {len(filtered)} entities of type {entity_type}")
        return filtered

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get relationship summary for a specified entity

        Args:
            graph_id: Graph ID
            entity_name: Entity name

        Returns:
            Entity summary information
        """
        logger.info(f"Fetching relationship summary for entity {entity_name}...")

        # First search for information related to the entity
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )

        # Try to find the entity among all nodes
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get graph statistics

        Args:
            graph_id: Graph ID

        Returns:
            Statistics information
        """
        logger.info(f"Fetching statistics for graph {graph_id}...")

        stats = self.store.get_graph_stats(graph_id)

        return {
            "graph_id": graph_id,
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"],
            "entity_types": stats["entity_types"],
            "relation_types": stats["relation_types"]
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Get simulation-related context information

        Args:
            graph_id: Graph ID
            simulation_requirement: Simulation requirement description
            limit: Quantity limit per information category

        Returns:
            Simulation context information
        """
        logger.info(f"Fetching simulation context: {simulation_requirement[:50]}...")

        # Search for information related to the simulation requirement
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )

        # Get graph statistics
        stats = self.get_graph_statistics(graph_id)

        # Get all entity nodes
        all_nodes = self.get_all_nodes(graph_id)

        # Filter entities with actual types (not pure Entity nodes)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }

    # ========== Core Retrieval Tools (Optimized) ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        [InsightForge - Deep Insight Retrieval]

        The most powerful hybrid retrieval function, automatically decomposes queries and searches across multiple dimensions:
        1. Uses LLM to decompose the query into multiple sub-queries
        2. Performs text search for each sub-query via PgGraphStore
        3. Builds NetworkX graph for relationship chain extraction
        4. Integrates all results to generate deep insights

        Args:
            graph_id: Graph ID
            query: User query
            simulation_requirement: Simulation requirement description
            report_context: Report context (optional, for more precise sub-query generation)
            max_sub_queries: Maximum number of sub-queries

        Returns:
            InsightForgeResult: Deep insight retrieval result
        """
        logger.info(f"InsightForge deep insight retrieval: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )

        # Step 1: Use LLM to generate sub-queries
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"Generated {len(sub_queries)} sub-queries")

        # Step 2: Search for each sub-query
        all_facts = []
        all_edge_ids = set()
        seen_facts = set()

        for sub_query in sub_queries:
            search_results = self.store.search_text(graph_id, sub_query, limit=15)

            for r in search_results:
                if r["content"] not in seen_facts:
                    all_facts.append(r["content"])
                    seen_facts.add(r["content"])
                if r["type"] == "edge":
                    all_edge_ids.add(r["id"])

        # Also search main query
        main_results = self.store.search_text(graph_id, query, limit=20)
        for r in main_results:
            if r["content"] not in seen_facts:
                all_facts.append(r["content"])
                seen_facts.add(r["content"])
            if r["type"] == "edge":
                all_edge_ids.add(r["id"])

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Step 3: Build NetworkX graph and extract entity insights + relationship chains
        G = self.store.build_networkx_graph(graph_id)

        # Find nodes connected to matched edges
        entity_uuids = set()
        for edge_id in all_edge_ids:
            for u, v, data in G.edges(data=True):
                if data.get("id") == edge_id:
                    entity_uuids.add(u)
                    entity_uuids.add(v)
                    break

        # Build entity insights from matched nodes
        entity_insights = []
        node_name_map = {}

        for node_id in entity_uuids:
            if node_id not in G.nodes:
                continue
            node_data = G.nodes[node_id]
            node_name = node_data.get("name", "")
            node_name_map[node_id] = node_name
            labels = node_data.get("labels") or []
            entity_type = next((l for l in labels if l not in ["Entity", "Node"]), "Entity")

            related_facts = [
                f for f in all_facts
                if node_name.lower() in f.lower()
            ]

            entity_insights.append({
                "uuid": node_id,
                "name": node_name,
                "type": entity_type,
                "summary": node_data.get("summary", ""),
                "related_facts": related_facts
            })

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Step 4: Build relationship chains from matched edges
        relationship_chains = []
        for edge_id in all_edge_ids:
            for u, v, data in G.edges(data=True):
                if data.get("id") == edge_id:
                    source_name = node_name_map.get(u) or G.nodes[u].get("name", u[:8]) if u in G.nodes else u[:8]
                    target_name = node_name_map.get(v) or G.nodes[v].get("name", v[:8]) if v in G.nodes else v[:8]
                    relation_name = data.get("name", "")
                    chain = f"{source_name} --[{relation_name}]--> {target_name}"
                    if chain not in relationship_chains:
                        relationship_chains.append(chain)
                    break

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(f"InsightForge complete: {result.total_facts} facts, {result.total_entities} entities, {result.total_relationships} relationships")
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        Use LLM to generate sub-queries

        Decompose complex queries into multiple independently searchable sub-queries
        """
        system_prompt = """You are a professional query analysis expert. Your task is to decompose a complex query into multiple sub-queries that can be independently observed in the simulated world.

Requirements:
1. Each sub-query should be specific enough to find related Agent behaviors or events in the simulated world
2. Sub-queries should cover different dimensions of the original query (e.g., who, what, why, how, when, where)
3. Sub-queries should be relevant to the simulation scenario
4. Return in JSON format: {"sub_queries": ["sub-query 1", "sub-query 2", ...]}"""

        user_prompt = f"""Simulation requirement background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please decompose the following query into {max_queries} sub-queries:
{query}

Return the sub-query list in JSON format."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(f"Failed to generate sub-queries: {str(e)}, using defaults")
            return [
                query,
                f"key participants in {query}",
                f"causes and impacts of {query}",
                f"development process of {query}"
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        [PanoramaSearch - Broad Search]

        Get full panoramic view, including all related content and historical/expired information.

        Args:
            graph_id: Graph ID
            query: Search query (for relevance sorting)
            include_expired: Whether to include expired content (default True)
            limit: Number of results to return

        Returns:
            PanoramaResult: Broad search result
        """
        logger.info(f"PanoramaSearch broad search: {query[:50]}...")

        result = PanoramaResult(query=query)

        # Get all nodes
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # Get all edges (with temporal information)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Classify facts
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            # Determine if expired/invalidated
            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                valid_at = edge.valid_at or "Unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "Unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(edge.fact)

        # Relevance sorting based on query
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('\uff0c', ' ').split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch complete: {result.active_count} active, {result.historical_count} historical")
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        [QuickSearch - Simple Search]

        Fast, lightweight retrieval tool.

        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return

        Returns:
            SearchResult: Search results
        """
        logger.info(f"QuickSearch simple search: {query[:50]}...")

        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )

        logger.info(f"QuickSearch complete: {result.total_count} results")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        [InterviewAgents - In-depth Interview]

        Calls the real OASIS interview API to interview running simulated Agents.
        This method has NO graph store dependency - copied as-is from zep_tools.py.

        Args:
            simulation_id: Simulation ID
            interview_requirement: Interview requirement description
            simulation_requirement: Simulation requirement background (optional)
            max_agents: Maximum number of agents to interview
            custom_questions: Custom interview questions (optional)

        Returns:
            InterviewResult: Interview results
        """
        from .simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents in-depth interview (real API): {interview_requirement[:50]}...")

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )

        # Step 1: Load agent profiles
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(f"No profile files found for simulation {simulation_id}")
            result.summary = "No Agent persona files found for interview"
            return result

        result.total_agents = len(profiles)
        logger.info(f"Loaded {len(profiles)} agent profiles")

        # Step 2: Use LLM to select agents for interview
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"Selected {len(selected_agents)} agents for interview: {selected_indices}")

        # Step 3: Generate interview questions (if not provided)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"Generated {len(result.interview_questions)} interview questions")

        # Combine questions into one interview prompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])

        # Add optimization prefix to constrain Agent response format
        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Based on your persona, all past memories and actions, "
            "answer the following questions directly in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language, do not call any tools\n"
            "2. Do not return JSON format or tool call format\n"
            "3. Do not use Markdown headings (e.g., #, ##, ###)\n"
            "4. Answer each question by number, starting each answer with \"Question X:\" (X is the question number)\n"
            "5. Separate answers to each question with a blank line\n"
            "6. Provide substantive answers, at least 2-3 sentences per question\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: Call real interview API (dual platform)
        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt
                })

            logger.info(f"Calling batch interview API (dual platform): {len(interviews_request)} agents")

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0
            )

            logger.info(f"Interview API returned: {api_result.get('interviews_count', 0)} results, success={api_result.get('success')}")

            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unknown error")
                logger.warning(f"Interview API returned failure: {error_msg}")
                result.summary = f"Interview API call failed: {error_msg}. Please check OASIS simulation environment status."
                return result

            # Step 5: Parse API results
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                twitter_text = twitter_response if twitter_response else "(No response from this platform)"
                reddit_text = reddit_response if reddit_response else "(No response from this platform)"
                response_text = f"[Twitter Platform Response]\n{twitter_text}\n\n[Reddit Platform Response]\n{reddit_text}"

                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'(?:问题|Question)\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题', 'Question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            logger.warning(f"Interview API call failed (environment not running?): {e}")
            result.summary = f"Interview failed: {str(e)}. The simulation environment may have been shut down, please ensure OASIS environment is running."
            return result
        except Exception as e:
            logger.error(f"Interview API call exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Error occurred during interview: {str(e)}"
            return result

        # Step 6: Generate interview summary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )

        logger.info(f"InterviewAgents complete: interviewed {result.interviewed_count} agents (dual platform)")
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Clean JSON tool call wrappers from Agent responses, extract actual content"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load simulation agent profile files"""
        import os
        import csv

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        profiles = []

        # Try Reddit JSON format first
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"Loaded {len(profiles)} profiles from reddit_profiles.json")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read reddit_profiles.json: {e}")

        # Try Twitter CSV format
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown"
                        })
                logger.info(f"Loaded {len(profiles)} profiles from twitter_profiles.csv")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read twitter_profiles.csv: {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """Use LLM to select agents for interview"""

        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)

        system_prompt = """You are a professional interview planning expert. Your task is to select the most suitable interview subjects from the simulated Agent list based on interview requirements.

Selection criteria:
1. Agent's identity/profession is relevant to the interview topic
2. Agent may hold unique or valuable viewpoints
3. Select diverse perspectives (e.g., supporters, opponents, neutral parties, professionals, etc.)
4. Prioritize roles directly related to the event

Return in JSON format:
{
    "selected_indices": [list of selected Agent indices],
    "reasoning": "explanation of selection reasoning"
}"""

        user_prompt = f"""Interview requirement:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else "Not provided"}

Available Agent list ({len(agent_summaries)} total):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select up to {max_agents} Agents most suitable for interview, and explain your selection reasoning."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Auto-selected based on relevance")

            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(f"LLM agent selection failed, using default: {e}")
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Using default selection strategy"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Use LLM to generate interview questions"""

        agent_roles = [a.get("profession", "Unknown") for a in selected_agents]

        system_prompt = """You are a professional journalist/interviewer. Based on the interview requirements, generate 3-5 in-depth interview questions.

Question requirements:
1. Open-ended questions that encourage detailed answers
2. Questions that may yield different answers from different roles
3. Cover multiple dimensions including facts, opinions, and feelings
4. Natural language, like a real interview
5. Keep each question under 50 words, concise and clear
6. Ask directly, do not include background explanations or prefixes

Return in JSON format: {"questions": ["question 1", "question 2", ...]}"""

        user_prompt = f"""Interview requirement: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else "Not provided"}

Interviewee roles: {', '.join(agent_roles)}

Please generate 3-5 interview questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )

            return response.get("questions", [f"What are your views on {interview_requirement}?"])

        except Exception as e:
            logger.warning(f"Failed to generate interview questions: {e}")
            return [
                f"What is your opinion on {interview_requirement}?",
                "What impact does this have on you or the group you represent?",
                "How do you think this issue should be resolved or improved?"
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Generate interview summary"""

        if not interviews:
            return "No interviews completed"

        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}")

        system_prompt = """You are a professional news editor. Based on the responses from multiple interviewees, generate an interview summary.

Summary requirements:
1. Extract the main viewpoints of all parties
2. Identify consensus and disagreements among viewpoints
3. Highlight valuable quotes
4. Be objective and neutral, not favoring any party
5. Keep within 1000 words

Format constraints (must follow):
- Use plain text paragraphs, separate sections with blank lines
- Do not use Markdown headings (e.g., #, ##, ###)
- Do not use dividers (e.g., ---, ***)
- When quoting interviewees, use quotation marks
- You may use **bold** to mark keywords, but do not use other Markdown syntax"""

        user_prompt = f"""Interview topic: {interview_requirement}

Interview content:
{"".join(interview_texts)}

Please generate an interview summary."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate interview summary: {e}")
            return f"Interviewed {len(interviews)} respondents, including: " + ", ".join([i.agent_name for i in interviews])
