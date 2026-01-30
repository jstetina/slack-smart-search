#!/usr/bin/env python3
"""
Interactive Slack Search

Semantic search across dumped Slack messages using embeddings.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

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

# Resolve project root and chdir so scripts work from any cwd; config paths are under config/
_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
os.chdir(_project_root)
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config import DEFAULT_CONFIG_PATH
from pymilvus import MilvusClient

# Suppress HF warning during sentence_transformers import
import io
_old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    from sentence_transformers import SentenceTransformer
finally:
    sys.stderr = _old_stderr

# Configuration
CONFIG_PATH = DEFAULT_CONFIG_PATH
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10


def load_config():
    """Load configuration from file."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def format_timestamp(ts: str) -> str:
    """Convert Slack timestamp to readable format."""
    try:
        dt = datetime.fromtimestamp(float(ts))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return ts


def make_slack_url(channel_id: str, ts: str, workspace_url: str = "https://redhat-internal.slack.com") -> str:
    """Construct a Slack message URL.
    
    Format: https://redhat-internal.slack.com/archives/C03UGJY6Z1A/p1761801162728869
    The timestamp 1761801162.728869 becomes p1761801162728869 (remove dot, add p prefix)
    """
    if not channel_id or not ts:
        return ""
    # Remove the dot from timestamp
    ts_for_url = ts.replace(".", "")
    return f"{workspace_url}/archives/{channel_id}/p{ts_for_url}"


def format_result(result: dict, index: int, workspace_url: str = "https://redhat-internal.slack.com") -> str:
    """Format a search result for display."""
    text = result.get("text", "")
    if len(text) > 200:
        text = text[:200] + "..."
    
    ts_raw = result.get("ts", "")
    ts = format_timestamp(ts_raw)
    # Prefer user_name, fall back to user ID
    user = result.get("user_name") or result.get("user", "unknown")
    channel = result.get("channel_id", "")
    url = make_slack_url(channel, ts_raw, workspace_url)
    
    return f"""
[{index}] {ts} | {user} | {channel}
    {text}
    {url}
"""


class SlackSearch:
    """Interactive Slack search."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.public_client = None
        self.private_client = None
        self.collection_name = config.get("collection_name", "slack_messages")
        self.embedding_model = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self.workspace_url = config.get("workspace_url", "https://redhat-internal.slack.com")
    
    def connect(self):
        """Connect to databases and load model."""
        print(f"[*] Loading embedding model: {self.embedding_model}")
        # Suppress HF hub warnings during model load
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            self.model = SentenceTransformer(self.embedding_model)
        finally:
            sys.stderr = old_stderr
        
        public_db = self.config.get("public_db", "./db/slack_public.db")
        private_db = self.config.get("private_db", "./db/slack_private.db")
        
        if Path(public_db).exists():
            print(f"[*] Connecting to public database: {public_db}")
            self.public_client = MilvusClient(uri=public_db)
        
        if Path(private_db).exists():
            print(f"[*] Connecting to private database: {private_db}")
            self.private_client = MilvusClient(uri=private_db)
        
        if not self.public_client and not self.private_client:
            print("[!] No databases found. Run 'make dump' first.")
            return False
        
        return True
    
    def search(self, query: str, top_k: int = TOP_K, search_public: bool = True, search_private: bool = True):
        """Search for messages matching the query."""
        # Generate embedding for query
        query_vector = self.model.encode(query, show_progress_bar=False).tolist()
        
        results = []
        
        # Search public database
        if search_public and self.public_client:
            try:
                public_results = self.public_client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "user", "user_name", "ts", "channel_id", "raw_json"],
                )
                for hit in public_results[0]:
                    hit["entity"]["distance"] = hit["distance"]
                    hit["entity"]["source"] = "public"
                    results.append(hit["entity"])
            except Exception as e:
                print(f"[!] Public search error: {e}")
        
        # Search private database
        if search_private and self.private_client:
            try:
                private_results = self.private_client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "user", "user_name", "ts", "channel_id", "raw_json"],
                )
                for hit in private_results[0]:
                    hit["entity"]["distance"] = hit["distance"]
                    hit["entity"]["source"] = "private"
                    results.append(hit["entity"])
            except Exception as e:
                print(f"[!] Private search error: {e}")
        
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x.get("distance", 999))
        
        return results[:top_k]
    
    def interactive(self):
        """Run interactive search loop."""
        print("\n" + "=" * 60)
        print("Slack Smart Search")
        print("=" * 60)
        print("Commands:")
        print("  <query>     - Search for messages")
        print("  /public     - Search only public (default)")
        print("  /private    - Search only private")
        print("  /all        - Search both")
        print("  /quit       - Exit")
        print("=" * 60 + "\n")
        
        search_public = True
        search_private = False
        
        while True:
            try:
                query = input("search> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            
            if not query:
                continue
            
            if query == "/quit" or query == "/q":
                print("Bye!")
                break
            elif query == "/public":
                search_public = True
                search_private = False
                print("[*] Searching public only")
                continue
            elif query == "/private":
                search_public = False
                search_private = True
                print("[*] Searching private only")
                continue
            elif query == "/all":
                search_public = True
                search_private = True
                print("[*] Searching all databases")
                continue
            
            results = self.search(query, search_public=search_public, search_private=search_private)
            
            if not results:
                print("No results found.")
                continue
            
            print(f"\n--- Found {len(results)} results ---")
            for i, result in enumerate(results, 1):
                source = result.get("source", "")
                print(f"[{source}]", end="")
                print(format_result(result, i, self.workspace_url))


def main():
    config = load_config()
    search = SlackSearch(config)
    
    if not search.connect():
        return
    
    search.interactive()


if __name__ == "__main__":
    main()
