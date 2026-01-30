#!/usr/bin/env python3
"""
Slack Channel Dump Script

This script dumps Slack channel messages to Milvus vector databases.
It uses two separate collections:
  - Public channels: Can be shared with others
  - Private channels: User-specific, not shared

It is idempotent (won't add duplicates) and supports interrupt/resume.

Usage:
    python slack_dump.py

Configuration via environment variables or config file (config/dump_config.json).
"""

import os
import sys
import signal
import asyncio
import logging
import time
import warnings
from datetime import datetime

# Suppress verbose HuggingFace/transformers logging and warnings before imports
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
from pathlib import Path
_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
os.chdir(_project_root)
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config import Config
from helpers import ProgressTracker, RawResponseLogger
from slack_client import SlackClient
from milvus_store import MilvusStore

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n[!] Shutdown requested. Finishing current operation...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def resolve_mentions_in_messages(
    slack: SlackClient,
    messages: list[dict],
) -> list[dict]:
    """Resolve user mentions and author IDs in messages.

    - Stores original message as '_raw_json_original' before any modifications
    - Transforms <@USER_ID> to @display_name in the text field
    - Adds 'user_name' field with resolved author name

    Modifies messages in place and returns them.
    """
    for msg in messages:
        text = msg.get("text", "")
        if text and "<@" in text:
            resolved_text = await slack.resolve_user_mentions(text)
            msg["text"] = resolved_text

        user_id = msg.get("user") or msg.get("bot_id", "")
        if user_id:
            user_name = await slack.get_user_display_name(user_id)
            msg["user_name"] = user_name

    return messages


async def dump_channel(
    slack: SlackClient,
    store: MilvusStore,
    progress: ProgressTracker,
    raw_logger: RawResponseLogger,
    channel_id: str,
    start_timestamp: str,
    request_delay: float,
    visibility: str,
) -> int:
    """
    Dump a single channel's messages.

    Returns the number of messages indexed.
    """
    global shutdown_requested

    print(f"\n[*] Processing {visibility} channel: {channel_id}")

    channel_info = await slack.get_channel_info(channel_id)
    if channel_info:
        channel_name = channel_info.get("name", channel_id)
        print(f"[*] Channel name: #{channel_name}")

    channel_progress = progress.get_channel_progress(channel_id)
    messages_indexed = channel_progress.get("messages_indexed", 0)
    cursor = None
    total_new = 0
    newest_ts_seen: str | None = None

    is_completed = progress.is_channel_completed(channel_id, start_timestamp)
    fetch_start_ts = str(time.time())

    if is_completed:
        saved_newest = channel_progress.get("newest_ts")
        if not saved_newest:
            saved_newest = store.get_newest_message_ts(channel_id)
            if saved_newest:
                print(f"[*] Recovered newest_ts from database: {saved_newest}")
                progress.update_channel_progress(channel_id, newest_ts=saved_newest)

        if saved_newest:
            print(f"[*] Fetching new messages since {saved_newest}")
            oldest_for_fetch = None
            latest_for_fetch = fetch_start_ts
            stop_at_ts = float(saved_newest)
        else:
            print("[*] Channel completed but no messages found, skipping")
            return 0
    else:
        stop_at_ts = None
        latest_for_fetch = channel_progress.get("oldest_ts")
        oldest_for_fetch = start_timestamp

        if latest_for_fetch:
            print(f"[*] Resuming historical fetch from timestamp {latest_for_fetch}")
        else:
            existing_oldest = store.get_oldest_message_ts(channel_id)
            if existing_oldest:
                print(f"[*] Found existing messages, oldest: {existing_oldest}")
                latest_for_fetch = existing_oldest

    while not shutdown_requested:
        messages, next_cursor, has_more = await slack.get_channel_history(
            channel_id,
            cursor=cursor,
            oldest=oldest_for_fetch,
            latest=latest_for_fetch,
            limit=100,
        )

        if not messages:
            print("[*] No more messages to fetch")
            break

        raw_logger.log_messages(channel_id, messages, source="history")

        new_messages = []
        for msg in messages:
            ts = msg.get("ts", "")
            if not store.message_exists(channel_id, ts):
                new_messages.append(msg)

        if new_messages:
            await resolve_mentions_in_messages(slack, new_messages)

        if new_messages:
            inserted = store.insert_messages_batch(channel_id, new_messages)
            total_new += inserted
            messages_indexed += inserted
            print(f"[*] Indexed {inserted} messages (total: {messages_indexed})")

        if messages:
            oldest_in_batch = min(messages, key=lambda m: float(m.get("ts", "0")))
            newest_in_batch = max(messages, key=lambda m: float(m.get("ts", "0")))
            oldest_ts = oldest_in_batch.get("ts")
            batch_newest_ts = newest_in_batch.get("ts")

            if newest_ts_seen is None or float(batch_newest_ts) > float(newest_ts_seen):
                newest_ts_seen = batch_newest_ts

            progress.update_channel_progress(
                channel_id,
                oldest_ts=oldest_ts if not is_completed else None,
                newest_ts=newest_ts_seen,
                messages_indexed=messages_indexed,
            )

            if stop_at_ts and float(oldest_ts) <= stop_at_ts:
                print(f"[*] Reached previously indexed messages (oldest in batch: {oldest_ts})")
                new_message_ts_set = {m.get("ts") for m in new_messages}
                for msg in messages:
                    if shutdown_requested:
                        break
                    msg_ts = msg.get("ts")
                    if msg_ts in new_message_ts_set and msg.get("reply_count", 0) > 0:
                        print(f"[*] Fetching thread replies for {msg_ts}")
                        replies = await slack.get_thread_replies(channel_id, msg_ts)
                        if replies:
                            raw_logger.log_messages(channel_id, replies, source=f"thread:{msg_ts}")
                            new_replies = [r for r in replies if not store.message_exists(channel_id, r.get("ts", ""))]
                            if new_replies:
                                await resolve_mentions_in_messages(slack, new_replies)
                                inserted = store.insert_messages_batch(channel_id, new_replies)
                                total_new += inserted
                                messages_indexed += inserted
                                print(f"[*] Indexed {inserted} thread replies")
                        await asyncio.sleep(request_delay)
                progress.update_channel_progress(
                    channel_id,
                    newest_ts=fetch_start_ts,
                    messages_indexed=messages_indexed,
                )
                print(f"[*] Channel {channel_id} incremental update complete")
                break

        for msg in messages:
            if shutdown_requested:
                break
            if msg.get("reply_count", 0) > 0:
                thread_ts = msg.get("ts")
                print(f"[*] Fetching thread replies for {thread_ts}")
                replies = await slack.get_thread_replies(channel_id, thread_ts)
                if replies:
                    raw_logger.log_messages(channel_id, replies, source=f"thread:{thread_ts}")
                    new_replies = [r for r in replies if not store.message_exists(channel_id, r.get("ts", ""))]
                    if new_replies:
                        await resolve_mentions_in_messages(slack, new_replies)
                        inserted = store.insert_messages_batch(channel_id, new_replies)
                        total_new += inserted
                        messages_indexed += inserted
                        print(f"[*] Indexed {inserted} thread replies")
                await asyncio.sleep(request_delay)

        if not has_more:
            progress.update_channel_progress(
                channel_id,
                completed=True if not is_completed else None,
                newest_ts=fetch_start_ts,
                messages_indexed=messages_indexed,
            )
            if is_completed:
                print(f"[*] Channel {channel_id} incremental update complete")
            else:
                print(f"[*] Channel {channel_id} fully indexed")
            break

        cursor = next_cursor
        await asyncio.sleep(request_delay)

    return total_new


async def main():
    """Main entry point."""
    global shutdown_requested

    print("=" * 60)
    print("Slack Channel Dump Script")
    print("=" * 60)

    config = Config()

    if not config.xoxc_token or not config.xoxd_token:
        print("[!] Slack tokens not configured.")
        print("    Set SLACK_XOXC_TOKEN and SLACK_XOXD_TOKEN environment variables")
        print("    or create config/dump_config.json with xoxc_token and xoxd_token fields")
        sys.exit(1)

    if not config.public_channels and not config.private_channels:
        print("[!] No channels configured.")
        print("    Add 'public_channels' and/or 'private_channels' lists to config/dump_config.json")
        sys.exit(1)

    print(f"[*] Public channels configured: {len(config.public_channels)}")
    print(f"[*] Private channels configured: {len(config.private_channels)}")
    print(f"[*] Start timestamp: {config.start_timestamp}")
    if config.start_timestamp != "0":
        dt = datetime.fromtimestamp(float(config.start_timestamp))
        print(f"    ({dt.isoformat()})")

    progress = ProgressTracker()
    total_indexed = 0

    async with SlackClient(config) as slack:
        print("\n[*] Resolving channel names...")
        public_channel_ids = await slack.resolve_channels(config.public_channels)
        private_channel_ids = await slack.resolve_channels(config.private_channels)

        print(f"[*] Resolved {len(public_channel_ids)} public channels")
        print(f"[*] Resolved {len(private_channel_ids)} private channels")

        if not public_channel_ids and not private_channel_ids:
            print("[!] No valid channels found after resolution.")
            sys.exit(1)

        public_store = None
        private_store = None

        if public_channel_ids:
            print(f"\n[*] Setting up public database: {config.public_db}")
            public_store = MilvusStore(
                config.public_db,
                config.milvus_token,
                config.collection_name,
                config.embedding_dim,
                config.embedding_model,
            )
            public_store.connect()

        if private_channel_ids:
            print(f"\n[*] Setting up private database: {config.private_db}")
            private_store = MilvusStore(
                config.private_db,
                config.milvus_token,
                config.collection_name,
                config.embedding_dim,
                config.embedding_model,
            )
            private_store.connect()

        raw_logger = RawResponseLogger()
        print(f"[*] Raw API responses will be saved to: {raw_logger.output_dir}")

        if public_store and public_channel_ids:
            print("\n" + "-" * 40)
            print("[*] Processing PUBLIC channels")
            print("-" * 40)
            for channel_id in public_channel_ids:
                if shutdown_requested:
                    print("\n[!] Shutdown requested, saving progress...")
                    break
                try:
                    indexed = await dump_channel(
                        slack,
                        public_store,
                        progress,
                        raw_logger,
                        channel_id,
                        config.start_timestamp,
                        config.request_delay,
                        "public",
                    )
                    total_indexed += indexed
                except Exception as e:
                    print(f"[!] Error processing channel {channel_id}: {e}")
                    continue

        if not shutdown_requested and private_store and private_channel_ids:
            print("\n" + "-" * 40)
            print("[*] Processing PRIVATE channels")
            print("-" * 40)
            for channel_id in private_channel_ids:
                if shutdown_requested:
                    print("\n[!] Shutdown requested, saving progress...")
                    break
                try:
                    indexed = await dump_channel(
                        slack,
                        private_store,
                        progress,
                        raw_logger,
                        channel_id,
                        config.start_timestamp,
                        config.request_delay,
                        "private",
                    )
                    total_indexed += indexed
                except Exception as e:
                    print(f"[!] Error processing channel {channel_id}: {e}")
                    continue

    print("\n" + "=" * 60)
    print(f"[*] Done! Total messages indexed: {total_indexed}")
    if public_channel_ids:
        print(f"    Public database: {config.public_db}")
    if private_channel_ids:
        print(f"    Private database: {config.private_db}")
    if shutdown_requested:
        print("[*] Script interrupted. Run again to continue from where you left off.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
