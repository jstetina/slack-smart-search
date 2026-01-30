"""Slack API client."""

import re
from typing import Any

import httpx

from config import Config, SLACK_API_BASE


class SlackClient:
    """Slack API client."""

    def __init__(self, config: Config):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._user_cache: dict[str, str] = {}

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Make a request to the Slack API."""
        url = f"{SLACK_API_BASE}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.config.xoxc_token}",
            "Content-Type": "application/json",
            "User-Agent": "SlackDump/1.0",
        }
        cookies = {"d": self.config.xoxd_token}

        try:
            if method.upper() == "GET":
                response = await self._client.request(
                    method, url, headers=headers, cookies=cookies, params=payload
                )
            else:
                response = await self._client.request(
                    method, url, headers=headers, cookies=cookies, json=payload
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[!] API request failed: {e}")
            return None

    async def get_channel_history(
        self,
        channel_id: str,
        cursor: str | None = None,
        oldest: str | None = None,
        latest: str | None = None,
        limit: int = 100,
    ) -> tuple[list[dict], str | None, bool]:
        """
        Get channel history with pagination.

        Returns: (messages, next_cursor, has_more)
        """
        payload = {"channel": channel_id, "limit": limit}
        if cursor:
            payload["cursor"] = cursor
        if oldest:
            payload["oldest"] = oldest
        if latest:
            payload["latest"] = latest

        data = await self._request("conversations.history", payload=payload)
        if data and data.get("ok"):
            messages = data.get("messages", [])
            response_metadata = data.get("response_metadata", {})
            next_cursor = response_metadata.get("next_cursor")
            has_more = data.get("has_more", False)
            return messages, next_cursor, has_more

        error = data.get("error", "unknown") if data else "request failed"
        print(f"[!] Failed to get channel history: {error}")
        return [], None, False

    async def get_thread_replies(
        self, channel_id: str, thread_ts: str
    ) -> list[dict]:
        """Get all replies in a thread."""
        payload = {"channel": channel_id, "ts": thread_ts}
        data = await self._request("conversations.replies", payload=payload)
        if data and data.get("ok"):
            messages = data.get("messages", [])
            return messages[1:] if len(messages) > 1 else []

        return []

    async def get_channel_info(self, channel_id: str) -> dict | None:
        """Get channel information."""
        payload = {"channel": channel_id}
        data = await self._request("conversations.info", payload=payload)
        if data and data.get("ok"):
            return data.get("channel")
        return None

    async def get_user_info(self, user_id: str) -> dict | None:
        """Get user information."""
        payload = {"user": user_id}
        data = await self._request("users.info", payload=payload)
        if data and data.get("ok"):
            return data.get("user")
        return None

    async def get_user_display_name(self, user_id: str) -> str:
        """Get user display name, with caching."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        user_info = await self.get_user_info(user_id)
        if user_info:
            profile = user_info.get("profile", {})
            display_name = (
                (profile.get("display_name") or "").strip()
                or (profile.get("real_name") or "").strip()
                or (user_info.get("real_name") or "").strip()
                or (user_info.get("name") or "").strip()
                or user_id
            )
            self._user_cache[user_id] = display_name
            return display_name

        self._user_cache[user_id] = user_id
        return user_id

    async def resolve_user_mentions(self, text: str) -> str:
        """Resolve <@USER_ID> or <@USER_ID|name> mentions in text to @display_name."""
        pattern = r"<@([A-Z0-9]+)(?:\|([^>]+))?>"
        matches = list(re.finditer(pattern, text))
        if not matches:
            return text

        result = text
        for match in reversed(matches):
            user_id = match.group(1)
            slack_provided_name = match.group(2)
            if slack_provided_name:
                display_name = slack_provided_name
            else:
                display_name = await self.get_user_display_name(user_id)
            result = result[:match.start()] + f"@{display_name}" + result[match.end():]

        return result

    async def resolve_channel(self, channel: str) -> str | None:
        """
        Resolve a channel name or ID to a channel ID.

        Accepts:
          - Channel ID (C12345678, D12345678, G12345678) - returned as-is
          - Channel name with or without # (e.g., "general" or "#general")

        Returns the channel ID or None if not found.
        """
        channel = channel.lstrip("#")

        if len(channel) >= 9 and channel[0] in "CDG" and channel[1:].isalnum():
            return channel

        print(f"[*] Resolving channel name: {channel}")

        cursor = None
        while True:
            payload = {
                "types": "public_channel,private_channel,mpim,im",
                "limit": 200,
            }
            if cursor:
                payload["cursor"] = cursor

            data = await self._request("conversations.list", payload=payload)
            if not data or not data.get("ok"):
                error = data.get("error", "unknown") if data else "request failed"
                print(f"[!] Failed to list conversations: {error}")
                return None

            channels = data.get("channels", [])
            for ch in channels:
                if ch.get("name") == channel or ch.get("name_normalized") == channel:
                    channel_id = ch.get("id")
                    print(f"[*] Resolved '{channel}' to {channel_id}")
                    return channel_id

            response_metadata = data.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        print(f"[!] Could not find channel: {channel}")
        return None

    async def resolve_channels(self, channels: list[str]) -> list[str]:
        """Resolve a list of channel names/IDs to channel IDs."""
        resolved = []
        for channel in channels:
            channel_id = await self.resolve_channel(channel)
            if channel_id:
                resolved.append(channel_id)
            else:
                print(f"[!] Skipping unresolved channel: {channel}")
        return resolved
