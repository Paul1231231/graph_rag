from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from neo4j import AsyncDriver, AsyncGraphDatabase

from agent_contracts import get_neo4j_settings
from graph_access import (
    query_get_bom,
    query_get_children,
    query_get_revisions,
    query_get_suppliers,
    query_graph_statistics,
    query_where_used,
    safe_depth,
    safe_limit,
)

load_dotenv()


@dataclass
class AppContext:
    driver: AsyncDriver
    database: str


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    settings = get_neo4j_settings()
    driver = AsyncGraphDatabase.driver(
        settings["uri"],
        auth=(settings["username"], settings["password"]),
    )
    try:
        yield AppContext(driver=driver, database=settings["database"])
    finally:
        await driver.close()


mcp = FastMCP("BOM GraphRAG Server", lifespan=app_lifespan)


def _paged_response(query_name: str, items: list[dict[str, Any]], cursor: int, page_size: int) -> dict[str, Any]:
    bounded_page_size = safe_limit(page_size)
    has_more = len(items) > bounded_page_size
    page_items = items[:bounded_page_size]
    next_cursor = cursor + bounded_page_size if has_more else None
    return {
        "query": query_name,
        "ok": True,
        "data": page_items,
        "pagination": {
            "cursor": cursor,
            "page_size": bounded_page_size,
            "next_cursor": next_cursor,
            "has_more": has_more,
        },
    }


@mcp.tool()
async def graph_statistics(ctx: Context) -> dict[str, Any]:
    context = ctx.request_context.lifespan_context
    stats = await query_graph_statistics(context.driver, context.database)
    return {"query": "graph_statistics", "ok": True, "data": stats, "pagination": None}


@mcp.tool()
async def get_bom(
    root_part: str,
    page_size: int = 20,
    cursor: int = 0,
    max_depth: int = 2,
    ctx: Context = None,
) -> dict[str, Any]:
    context = ctx.request_context.lifespan_context
    rows = await query_get_bom(
        context.driver,
        context.database,
        root_part=root_part,
        page_size=safe_limit(page_size),
        cursor=max(cursor, 0),
        max_depth=safe_depth(max_depth),
    )
    return _paged_response("get_bom", rows, max(cursor, 0), page_size)


@mcp.tool()
async def get_children(
    parent_part: str,
    page_size: int = 20,
    cursor: int = 0,
    ctx: Context = None,
) -> dict[str, Any]:
    context = ctx.request_context.lifespan_context
    rows = await query_get_children(
        context.driver,
        context.database,
        parent_part=parent_part,
        page_size=safe_limit(page_size),
        cursor=max(cursor, 0),
    )
    return _paged_response("get_children", rows, max(cursor, 0), page_size)


@mcp.tool()
async def where_used(
    component_part: str,
    page_size: int = 20,
    cursor: int = 0,
    ctx: Context = None,
) -> dict[str, Any]:
    context = ctx.request_context.lifespan_context
    rows = await query_where_used(
        context.driver,
        context.database,
        component_part=component_part,
        page_size=safe_limit(page_size),
        cursor=max(cursor, 0),
    )
    return _paged_response("where_used", rows, max(cursor, 0), page_size)


@mcp.tool()
async def get_suppliers(
    part_number: str,
    page_size: int = 20,
    cursor: int = 0,
    ctx: Context = None,
) -> dict[str, Any]:
    context = ctx.request_context.lifespan_context
    rows = await query_get_suppliers(
        context.driver,
        context.database,
        part_number=part_number,
        page_size=safe_limit(page_size),
        cursor=max(cursor, 0),
    )
    return _paged_response("get_suppliers", rows, max(cursor, 0), page_size)


@mcp.tool()
async def get_revisions(
    part_number: str,
    page_size: int = 20,
    cursor: int = 0,
    ctx: Context = None,
) -> dict[str, Any]:
    context = ctx.request_context.lifespan_context
    rows = await query_get_revisions(
        context.driver,
        context.database,
        part_number=part_number,
        page_size=safe_limit(page_size),
        cursor=max(cursor, 0),
    )
    return _paged_response("get_revisions", rows, max(cursor, 0), page_size)


@mcp.tool()
async def explain_bom(
    query_type: str,
    item_id: str,
    page_size: int = 5,
    ctx: Context = None,
) -> dict[str, Any]:
    context = ctx.request_context.lifespan_context
    bounded_page_size = safe_limit(page_size, default=5, max_value=20)

    query_map = {
        "get_bom": lambda: query_get_bom(context.driver, context.database, item_id, bounded_page_size, 0, 2),
        "get_children": lambda: query_get_children(context.driver, context.database, item_id, bounded_page_size, 0),
        "where_used": lambda: query_where_used(context.driver, context.database, item_id, bounded_page_size, 0),
        "get_suppliers": lambda: query_get_suppliers(context.driver, context.database, item_id, bounded_page_size, 0),
        "get_revisions": lambda: query_get_revisions(context.driver, context.database, item_id, bounded_page_size, 0),
    }

    if query_type not in query_map:
        return {
            "query": "explain_bom",
            "ok": False,
            "error": f"Unsupported query_type '{query_type}'.",
            "supported_query_types": list(query_map.keys()),
        }

    rows = await query_map[query_type]()
    page_rows = rows[:bounded_page_size]
    if not page_rows:
        summary = f"No BOM data found for {query_type} on '{item_id}'."
    else:
        summary = f"{query_type} returned {len(page_rows)} record(s) for '{item_id}'."

    return {
        "query": "explain_bom",
        "ok": True,
        "data": {"summary": summary, "rows": page_rows},
        "pagination": None,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
