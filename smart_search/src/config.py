"""Configuration and path constants for the dump/search tools."""

import os
import json
from pathlib import Path

# Project root: parent of src/; config files live under config/
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
CONFIG_DIR = _PROJECT_ROOT / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "dump_config.json"
SLACK_API_BASE = "https://slack.com/api"

# Model dimensions (must match the model used)
EMBEDDING_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _parse_timestamp(value: str) -> str:
    """Parse timestamp value to Unix timestamp string. Used by Config.load()."""
    from helpers import parse_timestamp
    return parse_timestamp(value)


class Config:
    """Configuration holder."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.load()

    def load(self):
        """Load configuration from file or environment."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = json.load(f)
        else:
            data = {}

        # Slack tokens from env or config
        self.xoxc_token = os.environ.get("SLACK_XOXC_TOKEN", data.get("xoxc_token", ""))
        self.xoxd_token = os.environ.get("SLACK_XOXD_TOKEN", data.get("xoxd_token", ""))

        # Milvus database files (separate files for public/private so public can be shared)
        self.public_db = data.get("public_db", "./slack_public.db")
        self.private_db = data.get("private_db", "./slack_private.db")
        self.milvus_token = os.environ.get("MILVUS_TOKEN", data.get("milvus_token", ""))

        # Collection name (same for both databases since they're separate files)
        self.collection_name = data.get("collection_name", "slack_messages")

        # Embedding model
        self.embedding_model = data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self.embedding_dim = EMBEDDING_DIMS.get(self.embedding_model, 384)

        # Channels to dump - now split by visibility
        self.public_channels = data.get("public_channels", [])
        self.private_channels = data.get("private_channels", [])

        # Start timestamp - only index messages after this time
        self.start_timestamp = _parse_timestamp(data.get("start_timestamp", "0"))

        # Rate limiting
        self.request_delay = data.get("request_delay", 1.0)
