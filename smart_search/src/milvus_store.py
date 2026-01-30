"""Milvus vector store and embedding model."""

import json
import sys
import io
from pathlib import Path

from pymilvus import MilvusClient, DataType

from config import DEFAULT_EMBEDDING_MODEL

_old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    from sentence_transformers import SentenceTransformer
finally:
    sys.stderr = _old_stderr


class EmbeddingModel:
    """Singleton embedding model to avoid loading multiple times."""

    _model = None
    _model_name = None

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
        if cls._model is None or cls._model_name != model_name:
            print(f"[*] Loading embedding model: {model_name}")
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
        return cls._model

    @classmethod
    def encode(cls, text: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
        """Generate embedding for a single text."""
        model = cls.get_model(model_name)
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    @classmethod
    def encode_batch(cls, texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        model = cls.get_model(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()


class MilvusStore:
    """Milvus vector database storage for a specific collection."""

    def __init__(self, milvus_uri: str, milvus_token: str, collection_name: str, embedding_dim: int = 384, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.client: MilvusClient | None = None

    def connect(self):
        """Connect to Milvus and ensure collection exists."""
        print(f"[*] Connecting to Milvus at {self.milvus_uri}")
        if self.milvus_uri.startswith("./") or self.milvus_uri.startswith("/"):
            db_path = Path(self.milvus_uri)
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(
            uri=self.milvus_uri,
            token=self.milvus_token if self.milvus_token else None,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if self.client.has_collection(self.collection_name):
            print(f"[*] Collection '{self.collection_name}' already exists")
            return

        print(f"[*] Creating collection '{self.collection_name}'")

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=100,
            is_primary=True,
        )
        schema.add_field(
            field_name="channel_id",
            datatype=DataType.VARCHAR,
            max_length=50,
        )
        schema.add_field(
            field_name="ts",
            datatype=DataType.VARCHAR,
            max_length=30,
        )
        schema.add_field(
            field_name="thread_ts",
            datatype=DataType.VARCHAR,
            max_length=30,
        )
        schema.add_field(
            field_name="user",
            datatype=DataType.VARCHAR,
            max_length=50,
        )
        schema.add_field(
            field_name="user_name",
            datatype=DataType.VARCHAR,
            max_length=200,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="msg_type",
            datatype=DataType.VARCHAR,
            max_length=50,
        )
        schema.add_field(
            field_name="raw_json",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dim,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",
            metric_type="L2",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"[*] Collection '{self.collection_name}' created")

    def message_exists(self, channel_id: str, ts: str) -> bool:
        """Check if a message already exists in the database."""
        msg_id = f"{channel_id}:{ts}"
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'id == "{msg_id}"',
            output_fields=["id"],
            limit=1,
        )
        return len(results) > 0

    def get_newest_message_ts(self, channel_id: str) -> str | None:
        """Get the newest message timestamp for a channel."""
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'channel_id == "{channel_id}"',
            output_fields=["ts"],
            limit=1000,
        )
        if not results:
            return None
        timestamps = [r["ts"] for r in results if r.get("ts")]
        if timestamps:
            return max(timestamps, key=float)
        return None

    def get_oldest_message_ts(self, channel_id: str) -> str | None:
        """Get the oldest message timestamp for a channel."""
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'channel_id == "{channel_id}"',
            output_fields=["ts"],
            limit=1000,
        )
        if not results:
            return None
        timestamps = [r["ts"] for r in results if r.get("ts")]
        if timestamps:
            return min(timestamps, key=float)
        return None

    def insert_message(self, channel_id: str, message: dict) -> bool:
        """Insert a message into the database."""
        ts = message.get("ts", "")
        msg_id = f"{channel_id}:{ts}"

        text = message.get("text", "")
        if len(text) > 65000:
            text = text[:65000] + "..."

        raw_json = json.dumps(message, ensure_ascii=False)
        if len(raw_json) > 65000:
            essential = {
                k: message[k]
                for k in ["ts", "user", "text", "type", "thread_ts", "reply_count", "reactions", "files", "attachments"]
                if k in message
            }
            raw_json = json.dumps(essential, ensure_ascii=False)
            if len(raw_json) > 65000:
                raw_json = raw_json[:65000]

        embedding_text = text if text else "(empty message)"
        vector = EmbeddingModel.encode(embedding_text, self.embedding_model)

        data = {
            "id": msg_id,
            "channel_id": channel_id,
            "ts": ts,
            "thread_ts": message.get("thread_ts", ""),
            "user": message.get("user", message.get("bot_id", "")),
            "text": text,
            "msg_type": message.get("type", "message"),
            "raw_json": raw_json,
            "vector": vector,
        }

        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=[data],
            )
            return True
        except Exception as e:
            print(f"[!] Failed to insert message {msg_id}: {e}")
            return False

    def insert_messages_batch(self, channel_id: str, messages: list[dict]) -> int:
        """Insert multiple messages in a batch. Returns count of inserted messages."""
        if not messages:
            return 0

        texts = []
        prepared_data = []

        for message in messages:
            ts = message.get("ts", "")
            msg_id = f"{channel_id}:{ts}"

            text = message.get("text", "")
            if len(text) > 65000:
                text = text[:65000] + "..."

            raw_json = message.get("_raw_json_original") or json.dumps(message, ensure_ascii=False)
            if len(raw_json) > 65000:
                essential = {
                    k: message[k]
                    for k in ["ts", "user", "text", "type", "thread_ts", "reply_count", "reactions", "files", "attachments"]
                    if k in message
                }
                raw_json = json.dumps(essential, ensure_ascii=False)
                if len(raw_json) > 65000:
                    raw_json = raw_json[:65000]

            texts.append(text if text else "(empty message)")
            user_id = message.get("user", message.get("bot_id", ""))
            user_name = message.get("user_name", user_id)
            prepared_data.append({
                "id": msg_id,
                "channel_id": channel_id,
                "ts": ts,
                "thread_ts": message.get("thread_ts", ""),
                "user": user_id,
                "user_name": user_name,
                "text": text,
                "msg_type": message.get("type", "message"),
                "raw_json": raw_json,
            })

        vectors = EmbeddingModel.encode_batch(texts, self.embedding_model)

        data_list = []
        for i, data in enumerate(prepared_data):
            data["vector"] = vectors[i]
            data_list.append(data)

        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=data_list,
            )
            return len(data_list)
        except Exception as e:
            print(f"[!] Batch insert failed: {e}")
            count = 0
            for msg in messages:
                if self.insert_message(channel_id, msg):
                    count += 1
            return count
