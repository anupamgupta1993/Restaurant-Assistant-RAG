from typing import Any, Optional
from restaurant_retreival_engine import RestaurantVectorStore, EmbeddingService, DataLoader, RestaurantSearchEngine

# Cached instances - created once and reused
_vector_store: Optional[RestaurantVectorStore] = None
_embedding: Optional[EmbeddingService] = None
_data_loader: Optional[DataLoader] = None

def _get_or_create_instances():
    """Get or create cached instances of vector store, embedding, and data loader."""
    global _vector_store, _embedding, _data_loader
    
    if _vector_store is None:
        _vector_store = RestaurantVectorStore()
    
    if _embedding is None:
        _embedding = EmbeddingService()
    
    if _data_loader is None:
        _data_loader = DataLoader(
            "../data/restaurants.csv",
            "../data/restaurant-menus.csv"
        )
    
    return _vector_store, _embedding, _data_loader

def load_index():
    vector_store, embedding, data_loader = _get_or_create_instances()

    engine = RestaurantSearchEngine(vector_store, embedding, data_loader)

    # Check if collection already exists
    collection_name = engine.default_collection
    if vector_store.client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' already exists. Skipping indexing.")
    else:
        print(f"Collection '{collection_name}' does not exist. Creating and indexing...")
        engine.initialize_collection()
        thread = engine.index_data()
        # Optionally block until indexing finishes
        # thread.join()
    
    return engine
    