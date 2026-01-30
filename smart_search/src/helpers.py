"""Helper utilities and support classes for dump/search."""

import json
from datetime import datetime
from pathlib import Path

# Project root and config dir (same as config.py)
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
CONFIG_DIR = _PROJECT_ROOT / "config"
DB_DIR = _PROJECT_ROOT / "db"

DEFAULT_PROGRESS_PATH = DB_DIR / "dump_progress.json"
DEFAULT_RAW_RESPONSES_DIR = _PROJECT_ROOT / "raw_responses"


def parse_timestamp(value: str) -> str:
    """
    Parse a timestamp value to Unix timestamp string.

    Supports:
      - Unix timestamp (e.g., "1704067200")
      - YYYY-MM-DD format (e.g., "2024-01-01")
      - YYYY-MM-DD HH:MM:SS format (e.g., "2024-01-01 12:00:00")

    Returns Unix timestamp as string.
    """
    if not value or value == "0":
        return "0"

    if value.replace(".", "").isdigit():
        return value

    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(value, fmt)
            return str(dt.timestamp())
        except ValueError:
            continue

    print(f"[!] Warning: Could not parse timestamp '{value}', using 0")
    return "0"


class RawResponseLogger:
    """Logs raw API responses to JSONL files for future reference."""

    def __init__(self, output_dir: Path = DEFAULT_RAW_RESPONSES_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_messages(self, channel_id: str, messages: list[dict], source: str = "history"):
        """Append raw messages to channel's JSONL file."""
        if not messages:
            return

        filepath = self.output_dir / f"{channel_id}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            entry = {
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "count": len(messages),
                "messages": messages,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class ProgressTracker:
    """Tracks progress for interrupt/resume support."""

    def __init__(self, progress_path: Path = DEFAULT_PROGRESS_PATH):
        self.progress_path = progress_path
        self.progress: dict[str, dict] = {}
        self.load()

    def load(self):
        """Load progress from file."""
        if self.progress_path.exists():
            with open(self.progress_path) as f:
                self.progress = json.load(f)
        else:
            self.progress = {}

    def save(self):
        """Save progress to file."""
        with open(self.progress_path, "w") as f:
            json.dump(self.progress, f, indent=2)

    def get_channel_progress(self, channel_id: str) -> dict:
        """Get progress for a specific channel."""
        return self.progress.get(channel_id, {})

    def update_channel_progress(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        newest_ts: str | None = None,
        messages_indexed: int | None = None,
        completed: bool = False,
    ):
        """Update progress for a channel."""
        if channel_id not in self.progress:
            self.progress[channel_id] = {
                "oldest_ts": None,
                "newest_ts": None,
                "messages_indexed": 0,
                "completed": False,
                "last_updated": None,
            }

        if oldest_ts is not None:
            self.progress[channel_id]["oldest_ts"] = oldest_ts
        if newest_ts is not None:
            self.progress[channel_id]["newest_ts"] = newest_ts
        if messages_indexed is not None:
            self.progress[channel_id]["messages_indexed"] = messages_indexed
        if completed:
            self.progress[channel_id]["completed"] = True

        self.progress[channel_id]["last_updated"] = datetime.now().isoformat()
        self.save()

    def is_channel_completed(self, channel_id: str, start_timestamp: str) -> bool:
        """Check if a channel has been fully indexed since start_timestamp."""
        progress = self.get_channel_progress(channel_id)
        if not progress.get("completed"):
            return False
        oldest = progress.get("oldest_ts")
        if oldest and float(oldest) <= float(start_timestamp):
            return True
        return False
