import re
from typing import Any

from neo4j import AsyncDriver

BOM_LABELS = {"Part", "Assembly", "BOM", "BOMLine", "Supplier", "Revision", "Plant", "Document"}
BOM_RELATIONSHIPS = {"HAS_COMPONENT", "SUPPLIED_BY", "HAS_REVISION", "HAS_DOCUMENT", "MANUFACTURED_AT"}
BLOCKED_CYPHER_TOKENS = {
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "CALL",
    "LOAD CSV",
    "APOC",
}


def safe_limit(value: int, default: int = 10, max_value: int = 100) -> int:
    if value <= 0:
        return default
    return min(value, max_value)


def safe_depth(value: int, default: int = 2, max_value: int = 5) -> int:
    if value <= 0:
        return default
    return min(value, max_value)


def is_safe_cypher(cypher: str) -> bool:
    if not cypher:
        return False
    upper = cypher.upper()
    if not upper.strip().startswith("MATCH"):
        return False
    if any(token in upper for token in BLOCKED_CYPHER_TOKENS):
        return False
    return any(label.upper() in upper for label in BOM_LABELS) and any(
        rel.upper() in upper for rel in BOM_RELATIONSHIPS
    )


def extract_part_candidates(question: str) -> list[str]:
    matches = re.findall(r"\b[A-Za-z]{1,6}[-_]?\d{2,}\b", question)
    deduped: list[str] = []
    seen = set()
    for match in matches:
        normalized = match.upper()
        if normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


async def query_graph_statistics(driver: AsyncDriver, database: str) -> dict[str, int]:
    records, _, _ = await driver.execute_query(
        """
        RETURN
            COUNT {(:Part)} AS parts,
            COUNT {(:Assembly)} AS assemblies,
            COUNT {(:Supplier)} AS suppliers,
            COUNT {()-[:HAS_COMPONENT]->()} AS bom_links,
            COUNT {()-[:SUPPLIED_BY]->()} AS supply_links,
            COUNT {()} AS total_nodes,
            COUNT {()-[]->()} AS total_relationships
        """,
        database_=database,
    )
    return dict(records[0]) if records else {
        "parts": 0,
        "assemblies": 0,
        "suppliers": 0,
        "bom_links": 0,
        "supply_links": 0,
        "total_nodes": 0,
        "total_relationships": 0,
    }


async def query_get_bom(
    driver: AsyncDriver,
    database: str,
    root_part: str,
    page_size: int,
    cursor: int,
    max_depth: int,
) -> list[dict[str, Any]]:
    skip = max(cursor, 0)
    limit = safe_limit(page_size) + 1
    depth = safe_depth(max_depth)
    records, _, _ = await driver.execute_query(
        """
        MATCH path=(root:Assembly {part_number: $root_part})-[:HAS_COMPONENT*1..$depth]->(child)
        RETURN
            root.part_number AS root_part,
            child.part_number AS child_part,
            labels(child) AS child_labels,
            length(path) AS level
        ORDER BY level ASC, child_part ASC
        SKIP $skip
        LIMIT $limit
        """,
        root_part=root_part,
        depth=depth,
        skip=skip,
        limit=limit,
        database_=database,
    )
    return [record.data() for record in records]


async def query_get_children(
    driver: AsyncDriver,
    database: str,
    parent_part: str,
    page_size: int,
    cursor: int,
) -> list[dict[str, Any]]:
    skip = max(cursor, 0)
    limit = safe_limit(page_size) + 1
    records, _, _ = await driver.execute_query(
        """
        MATCH (parent {part_number: $parent_part})-[r:HAS_COMPONENT]->(child)
        RETURN
            parent.part_number AS parent_part,
            child.part_number AS child_part,
            labels(child) AS child_labels,
            coalesce(r.quantity, 1) AS quantity,
            coalesce(r.unit, "ea") AS unit
        ORDER BY child_part ASC
        SKIP $skip
        LIMIT $limit
        """,
        parent_part=parent_part,
        skip=skip,
        limit=limit,
        database_=database,
    )
    return [record.data() for record in records]


async def query_where_used(
    driver: AsyncDriver,
    database: str,
    component_part: str,
    page_size: int,
    cursor: int,
) -> list[dict[str, Any]]:
    skip = max(cursor, 0)
    limit = safe_limit(page_size) + 1
    records, _, _ = await driver.execute_query(
        """
        MATCH (parent)-[r:HAS_COMPONENT]->(child {part_number: $component_part})
        RETURN
            parent.part_number AS parent_part,
            labels(parent) AS parent_labels,
            child.part_number AS component_part,
            coalesce(r.quantity, 1) AS quantity,
            coalesce(r.unit, "ea") AS unit
        ORDER BY parent_part ASC
        SKIP $skip
        LIMIT $limit
        """,
        component_part=component_part,
        skip=skip,
        limit=limit,
        database_=database,
    )
    return [record.data() for record in records]


async def query_get_suppliers(
    driver: AsyncDriver,
    database: str,
    part_number: str,
    page_size: int,
    cursor: int,
) -> list[dict[str, Any]]:
    skip = max(cursor, 0)
    limit = safe_limit(page_size) + 1
    records, _, _ = await driver.execute_query(
        """
        MATCH (p {part_number: $part_number})-[r:SUPPLIED_BY]->(s:Supplier)
        RETURN
            p.part_number AS part_number,
            s.supplier_id AS supplier_id,
            s.name AS supplier_name,
            coalesce(r.lead_time_days, null) AS lead_time_days,
            coalesce(r.preferred, false) AS preferred
        ORDER BY preferred DESC, supplier_name ASC
        SKIP $skip
        LIMIT $limit
        """,
        part_number=part_number,
        skip=skip,
        limit=limit,
        database_=database,
    )
    return [record.data() for record in records]


async def query_get_revisions(
    driver: AsyncDriver,
    database: str,
    part_number: str,
    page_size: int,
    cursor: int,
) -> list[dict[str, Any]]:
    skip = max(cursor, 0)
    limit = safe_limit(page_size) + 1
    records, _, _ = await driver.execute_query(
        """
        MATCH (p {part_number: $part_number})-[:HAS_REVISION]->(r:Revision)
        RETURN
            p.part_number AS part_number,
            r.revision_id AS revision_id,
            r.effective_from AS effective_from,
            r.effective_to AS effective_to,
            coalesce(r.status, "unknown") AS status
        ORDER BY effective_from DESC
        SKIP $skip
        LIMIT $limit
        """,
        part_number=part_number,
        skip=skip,
        limit=limit,
        database_=database,
    )
    return [record.data() for record in records]


def query_graph_context_sync(graph: Any, question: str, max_depth: int = 2, limit: int = 10) -> list[dict[str, Any]]:
    candidates = extract_part_candidates(question)
    if not candidates:
        return []

    depth = safe_depth(max_depth)
    bounded_limit = safe_limit(limit)
    rows: list[dict[str, Any]] = []

    for candidate in candidates[:3]:
        result = graph.query(
            """
            MATCH path=(root {part_number: $part_number})-[:HAS_COMPONENT*1..$depth]->(child)
            RETURN
                root.part_number AS root_part,
                child.part_number AS child_part,
                labels(child) AS child_labels,
                length(path) AS level
            ORDER BY level ASC, child_part ASC
            LIMIT $limit
            """,
            params={"part_number": candidate, "depth": depth, "limit": bounded_limit},
        )
        if result:
            rows.extend(result)

    return rows[:bounded_limit]
