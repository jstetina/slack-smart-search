#!/usr/bin/env python3
"""
Smart Search MCP Server

Exposes semantic search over Slack messages as an MCP tool.
Databases should be mounted at /data/db/
"""

import os
import sys
import io
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# Suppress verbose logging and warnings before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pymilvus import MilvusClient

# Suppress HF warning during sentence_transformers import
_old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    from sentence_transformers import SentenceTransformer
finally:
    sys.stderr = _old_stderr

# Configuration
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "http")
DB_PATH = Path(os.environ.get("DB_PATH", "/data/db"))
PUBLIC_DB = DB_PATH / "slack_public.db"
PRIVATE_DB = DB_PATH / "slack_private.db"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "slack_messages")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
WORKSPACE_URL = os.environ.get("WORKSPACE_URL", "https://redhat-internal.slack.com")
TOP_K = int(os.environ.get("TOP_K", "10"))

# Disable DNS rebinding protection for container networking
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

# Initialize MCP server with security settings
mcp = FastMCP("slack-smart-search", transport_security=transport_security)

# Global state for lazy initialization
_model = None
_public_client = None
_private_client = None


def get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        print(f"[*] Loading embedding model: {EMBEDDING_MODEL}", file=sys.stderr)
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        finally:
            sys.stderr = old_stderr
    return _model


def get_public_client():
    """Lazy connect to public database."""
    global _public_client
    if _public_client is None and PUBLIC_DB.exists():
        print(f"[*] Connecting to public database: {PUBLIC_DB}", file=sys.stderr)
        _public_client = MilvusClient(uri=str(PUBLIC_DB))
    return _public_client


def get_private_client():
    """Lazy connect to private database."""
    global _private_client
    if _private_client is None and PRIVATE_DB.exists():
        print(f"[*] Connecting to private database: {PRIVATE_DB}", file=sys.stderr)
        _private_client = MilvusClient(uri=str(PRIVATE_DB))
    return _private_client


def format_timestamp(ts: str) -> str:
    """Convert Slack timestamp to readable format."""
    try:
        dt = datetime.fromtimestamp(float(ts))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return ts


def make_slack_url(channel_id: str, ts: str) -> str:
    """Construct a Slack message URL."""
    if not channel_id or not ts:
        return ""
    ts_for_url = ts.replace(".", "")
    return f"{WORKSPACE_URL}/archives/{channel_id}/p{ts_for_url}"


def format_result(result: dict) -> dict:
    """Format a search result for response."""
    text = result.get("text", "")
    ts_raw = result.get("ts", "")
    user = result.get("user_name") or result.get("user", "unknown")
    channel = result.get("channel_id", "")
    
    # Create a clean text preview (first 300 chars, no newlines)
    text_preview = text[:300].replace("\n", " ").strip()
    if len(text) > 300:
        text_preview += "..."
    
    return {
        "text": text_preview,
        "full_text": text if len(text) > 300 else None,  # Only include if truncated
        "user": user,
        "channel_id": channel,
        "timestamp": format_timestamp(ts_raw),
        "url": make_slack_url(channel, ts_raw),
    }


@mcp.tool(
    description="""
Deep semantic nearest-neighbor search over historical Slack messages.
Use this when exact keyword matching fails or is impractical. Finds messages
by meaning rather than literal text - ideal for describing problems in natural
language, discovering prior RHOAI discussions, finding answers when you don't
know the exact terminology, or locating past troubleshooting threads.

Parameters:
- query: Natural language description of what you're looking for
- top_k: (optional) Number of results to return, default 10
- search_scope: (optional) "public", "private", or "all" databases, default "public"
- user: (optional) Filter by user ID or username (partial match)
- start_date: (optional) Filter messages after this date (YYYY-MM-DD)
- end_date: (optional) Filter messages before this date (YYYY-MM-DD)
"""
)
async def smart_search(
    query: str,
    top_k: int = TOP_K,
    search_scope: Literal["public", "private", "all"] = "public"
) -> dict:
    """
    Semantic search across Slack messages.
    Returns a dict with "message" (always set) and "results" (list) so the client always has visible content.
    """
    try:
        search_public = search_scope in ("public", "all")
        search_private = search_scope in ("private", "all")
        public_client = get_public_client() if search_public else None
        private_client = get_private_client() if search_private else None

        if not public_client and not private_client:
            return {
                "message": "No search databases available. Mount Slack DBs at DB_PATH (e.g. /app/db with slack_public.db and/or slack_private.db) or run 'make dump' to index messages.",
                "results": [],
            }

        model = get_model()
        if model is None:
            return {"message": "Failed to load embedding model.", "results": []}

        query_vector = model.encode(query, show_progress_bar=False).tolist()
        results = []

        if public_client:
            try:
                public_results = public_client.search(
                    collection_name=COLLECTION_NAME,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "user", "user_name", "ts", "channel_id"],
                )
                for hit in public_results[0]:
                    entity = hit.get("entity", hit)
                    entity["distance"] = hit.get("distance", 0)
                    entity["source"] = "public"
                    results.append(entity)
            except Exception as e:
                print(f"[!] Public search error: {e}", file=sys.stderr)

        if private_client:
            try:
                private_results = private_client.search(
                    collection_name=COLLECTION_NAME,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "user", "user_name", "ts", "channel_id"],
                )
                for hit in private_results[0]:
                    entity = hit.get("entity", hit)
                    entity["distance"] = hit.get("distance", 0)
                    entity["source"] = "private"
                    results.append(entity)
            except Exception as e:
                print(f"[!] Private search error: {e}", file=sys.stderr)

        results.sort(key=lambda x: x.get("distance", 999))
        results = results[:top_k]
        formatted = [format_result(r) for r in results]
        return {
            "message": f"Found {len(formatted)} result(s) for \"{query[:50]}{'...' if len(query) > 50 else ''}\".",
            "results": formatted,
        }
    except Exception as e:
        return {"message": f"Search failed: {e}", "results": []}


@mcp.tool()
async def search_stats() -> dict:
    """
    Get statistics about the search databases.
    
    Returns:
        Dictionary with database stats (message counts, etc.)
    """
    stats = {
        "public_db": {"available": False, "path": str(PUBLIC_DB)},
        "private_db": {"available": False, "path": str(PRIVATE_DB)},
        "embedding_model": EMBEDDING_MODEL,
        "workspace_url": WORKSPACE_URL,
    }
    
    public_client = get_public_client()
    if public_client:
        stats["public_db"]["available"] = True
        try:
            info = public_client.get_collection_stats(COLLECTION_NAME)
            stats["public_db"]["message_count"] = info.get("row_count", 0)
        except:
            pass
    
    private_client = get_private_client()
    if private_client:
        stats["private_db"]["available"] = True
        try:
            info = private_client.get_collection_stats(COLLECTION_NAME)
            stats["private_db"]["message_count"] = info.get("row_count", 0)
        except:
            pass
    
    return stats


if __name__ == "__main__":
    sys.stderr.write(f"Starting Slack Smart Search MCP Server\n")
    sys.stderr.write(f"Transport: {MCP_TRANSPORT}\n")
    sys.stderr.write(f"DB Path: {DB_PATH}\n")
    sys.stderr.flush()

    if MCP_TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    else:
        # For HTTP transport, get the ASGI app and run with uvicorn
        import uvicorn
        app = mcp.streamable_http_app()
        sys.stderr.write(f"Starting HTTP server on 0.0.0.0:8000\n")
        sys.stderr.flush()
        uvicorn.run(app, host="0.0.0.0", port=8000, server_header=False, forwarded_allow_ips="*")
