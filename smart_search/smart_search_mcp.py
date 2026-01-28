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
from typing import Literal

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
from pymilvus import MilvusClient

# Suppress HF warning during sentence_transformers import
_old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    from sentence_transformers import SentenceTransformer
finally:
    sys.stderr = _old_stderr

# Configuration
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
DB_PATH = Path(os.environ.get("DB_PATH", "/data/db"))
PUBLIC_DB = DB_PATH / "slack_public.db"
PRIVATE_DB = DB_PATH / "slack_private.db"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "slack_messages")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
WORKSPACE_URL = os.environ.get("WORKSPACE_URL", "https://redhat-internal.slack.com")
TOP_K = int(os.environ.get("TOP_K", "10"))

# Initialize MCP server
mcp = FastMCP(
    "smart_search",
    host="127.0.0.1" if MCP_TRANSPORT == "stdio" else "0.0.0.0"
)

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
    
    return {
        "text": text,
        "user": user,
        "channel_id": channel,
        "timestamp": format_timestamp(ts_raw),
        "url": make_slack_url(channel, ts_raw),
        "source": result.get("source", ""),
        "distance": result.get("distance", 0),
    }


@mcp.tool()
async def smart_search(
    query: str,
    top_k: int = TOP_K,
    search_scope: Literal["public", "private", "all"] = "public"
) -> list[dict]:
    """
    Semantic search across Slack messages.
    
    Args:
        query: The search query (natural language)
        top_k: Number of results to return (default 10)
        search_scope: Which databases to search - "public", "private", or "all"
    
    Returns:
        List of matching messages with text, user, channel, timestamp, and URL
    """
    model = get_model()
    if model is None:
        return [{"error": "Failed to load embedding model"}]
    
    # Generate embedding for query
    query_vector = model.encode(query, show_progress_bar=False).tolist()
    
    results = []
    search_public = search_scope in ("public", "all")
    search_private = search_scope in ("private", "all")
    
    # Search public database
    if search_public:
        client = get_public_client()
        if client:
            try:
                public_results = client.search(
                    collection_name=COLLECTION_NAME,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "user", "user_name", "ts", "channel_id"],
                )
                for hit in public_results[0]:
                    hit["entity"]["distance"] = hit["distance"]
                    hit["entity"]["source"] = "public"
                    results.append(hit["entity"])
            except Exception as e:
                print(f"[!] Public search error: {e}", file=sys.stderr)
    
    # Search private database
    if search_private:
        client = get_private_client()
        if client:
            try:
                private_results = client.search(
                    collection_name=COLLECTION_NAME,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "user", "user_name", "ts", "channel_id"],
                )
                for hit in private_results[0]:
                    hit["entity"]["distance"] = hit["distance"]
                    hit["entity"]["source"] = "private"
                    results.append(hit["entity"])
            except Exception as e:
                print(f"[!] Private search error: {e}", file=sys.stderr)
    
    # Sort by distance (lower is better) and limit
    results.sort(key=lambda x: x.get("distance", 999))
    results = results[:top_k]
    
    # Format results
    return [format_result(r) for r in results]


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
    mcp.run(transport=MCP_TRANSPORT)
