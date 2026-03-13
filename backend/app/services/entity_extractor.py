"""
LLM-based entity and relationship extraction from text.
"""

import re
from typing import Any, Dict, List

_SYSTEM_PROMPT = """\
You are an entity and relationship extraction engine.

You will be given a text passage and an ontology definition. Extract ALL entities \
and relationships that match the ontology.

## Ontology

Entity types: {entity_types}
Relation types: {relation_types}

## Rules
1. Only extract entities whose type is one of the listed entity types.
2. Only extract relationships whose relation name is one of the listed relation types.
3. Entity names should be concise canonical forms (proper nouns preferred).
4. Each relationship must reference entities by the exact name you extracted.
5. The "fact" field must be a full sentence describing the relationship.
6. Return ONLY valid JSON — no markdown fences, no commentary.

## Output format
{{
  "entities": [
    {{"name": "...", "type": "...", "summary": "...", "attributes": {{}}}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "relation": "RELATION_NAME", "fact": "..."}}
  ]
}}
"""


def _split_sentences(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into chunks of at most *max_chars*, breaking on sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Rough sentence split
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip() if current else sent
    if current:
        chunks.append(current)
    return chunks


class EntityExtractor:
    """Extract entities and relationships from text using an LLM."""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: An LLMClient (or compatible) instance with a chat_json method.
        """
        self._llm = llm_client

    def extract(self, text: str, ontology: dict) -> dict:
        """Extract entities and relationships constrained by *ontology*.

        Args:
            text: Source text.
            ontology: Dict with keys ``entity_types`` (list of type name strings
                      or dicts with a ``name`` key) and ``relation_types`` (same).

        Returns:
            ``{"entities": [...], "relationships": [...]}``
        """
        # Normalise ontology entries to plain name strings
        entity_types = _type_names(ontology.get("entity_types", []))
        relation_types = _type_names(ontology.get("relation_types", [])
                                     or ontology.get("edge_types", []))

        system = _SYSTEM_PROMPT.format(
            entity_types=", ".join(entity_types),
            relation_types=", ".join(relation_types),
        )

        chunks = _split_sentences(text)
        all_entities: List[dict] = []
        all_rels: List[dict] = []

        for chunk in chunks:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ]
            try:
                result = self._llm.chat_json(messages=messages, temperature=0.1, max_tokens=4096)
            except (ValueError, Exception):
                # If LLM returns invalid JSON, skip this chunk
                continue

            all_entities.extend(result.get("entities", []))
            all_rels.extend(result.get("relationships", []))

        # Deduplicate entities by name (last wins for summary/attributes)
        seen: Dict[str, dict] = {}
        for ent in all_entities:
            seen[ent["name"]] = ent
        deduped = list(seen.values())

        return {"entities": deduped, "relationships": all_rels}


def _type_names(items: List[Any]) -> List[str]:
    """Accept either a list of strings or a list of dicts with a 'name' key."""
    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            names.append(item.get("name", str(item)))
        else:
            names.append(str(item))
    return names
