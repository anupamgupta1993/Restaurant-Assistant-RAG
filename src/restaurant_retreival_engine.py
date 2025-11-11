from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import pandas as pd
import os
import math
import tqdm.auto as tqdm_auto
import threading
import time


class EmbeddingService:
    """Handles local embedding generation using FastEmbed."""

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-small-en"):
        self.model_name = model_name

    def embed_text(self, text: str) -> models.Document:
        text = " ".join(text.split())  # normalize whitespace
        return models.Document(text=text, model=self.model_name)


class DataLoader:
    """Responsible for loading and preparing restaurant data."""

    def __init__(self, restaurants_path: str, menu_path: str):
        self.restaurants_path = restaurants_path
        self.menu_path = menu_path

    @staticmethod
    def _safe_value(val: Any) -> str:
        if val is None:
            return "Not available"
        if isinstance(val, float) and math.isnan(val):
            return "Not available"
        return str(val)

    def load_and_merge_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.restaurants_path):
            raise FileNotFoundError(f"File not found: {self.restaurants_path}")
        if not os.path.exists(self.menu_path):
            raise FileNotFoundError(f"File not found: {self.menu_path}")

        df_rest = pd.read_csv(self.restaurants_path)
        df_rest.drop_duplicates(subset=["name"], inplace=True)

        df_menu = pd.read_csv(self.menu_path, nrows=100000)
        df_menu.drop_duplicates(inplace=True)

        df = pd.merge(
            df_rest, df_menu,
            left_on="id", right_on="restaurant_id",
            how="inner"
        )
        df.drop(columns=["id", "position"], inplace=True)
        df[["city", "state"]] = df["full_address"].str.extract(
            r",\s*([^,]+?)\s*,\s*([A-Z]{2})\b"
        )

        return df.to_dict(orient="records")

    def format_embedding_text(self, record: Dict[str, Any]) -> str:
        sv = self._safe_value
        return (
            f"{sv(record.get('name_x'))} - {sv(record.get('category_x'))} - {sv(record.get('full_address'))}. "
            f"Menu item: {sv(record.get('name_y'))} - {sv(record.get('category_y'))}. "
            f"Description: {sv(record.get('description'))}. "
            f"Price Range: {sv(record.get('price_range'))}. "
            f"Ratings: {sv(record.get('ratings'))}."
        )


class RestaurantVectorStore:
    """Encapsulates Qdrant client operations: collection management and indexing."""

    def __init__(self, host: str = "http://localhost:6333", batch_size: int = 500):
        self.client = QdrantClient(host)
        self.batch_size = batch_size


    def create_collection(self, name: str, vector_size: int = 512) -> None:
        if self.client.collection_exists(collection_name=name):
            print(f"Collection '{name}' already exists. Deleting...")
            self.client.delete_collection(collection_name=name)

        print(f"Creating collection '{name}'...")
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
    def _batch_upsert(self, name: str, points):
        total = len(points)
        with tqdm_auto.tqdm(total=total, desc=f"Indexing → {name}", unit="pts") as pbar:
            for i in range(0, total, self.batch_size):
                batch = points[i : i + self.batch_size]
                self.client.upsert(collection_name=name, points=batch)
                pbar.update(len(batch))
                time.sleep(0.05)  # prevents UI from freezing with large batches

        print(f"✅ Finished upserting {total} points into '{name}'")
    
    def upsert_points_async(self, name: str, points):
        worker = threading.Thread(
            target=self._batch_upsert,
            args=(name, points),
            daemon=True
        )
        worker.start()
        return worker  # so caller can optionally `.join()`


class RestaurantSearchEngine:
    """High-level interface for indexing and searching restaurant data."""

    def __init__(
        self,
        vector_store: RestaurantVectorStore,
        embedding_service: EmbeddingService,
        data_loader: DataLoader,
        default_collection: str = "restaurants"
    ):
        self.vector_store = vector_store
        self.embedding = embedding_service
        self.data_loader = data_loader
        self.default_collection = default_collection

    def initialize_collection(self, collection_name: Optional[str] = None) -> None:
        coll = collection_name or self.default_collection
        self.vector_store.create_collection(name=coll)

    def index_data(self, collection_name: Optional[str] = None , data: Optional[List[Dict[str, Any]]] = None) -> None:
        coll = collection_name or self.default_collection
        data = data or self.data_loader.load_and_merge_data()

        points = []
        for idx, record in enumerate(data):
            text = self.data_loader.format_embedding_text(record)
            doc = self.embedding.embed_text(text)

            point = models.PointStruct(
                id=idx,
                vector=doc,
                payload=record
            )
            points.append(point)

        # self.vector_store.upsert_points(coll, points)
        return self.vector_store.upsert_points_async(coll, points)

        print(f"Indexed {len(points)} records into '{coll}'.")

    def search(self, query: str, collection_name: Optional[str] = None, num_results: int = 5):
        coll = collection_name or self.default_collection
        query_doc = self.embedding.embed_text(query)
        return self.vector_store.client.query_points(
            collection_name=coll,
            query=query_doc,
            limit=num_results,
            with_payload=True,
        )
