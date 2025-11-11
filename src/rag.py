from typing import Optional
from llm_utility import RAGQueryEngine

# Cached instance - created once and reused
_rag_engine: Optional[RAGQueryEngine] = None

def _get_or_create_rag_engine() -> RAGQueryEngine:
    """Get or create cached instance of RAGQueryEngine."""
    global _rag_engine
    
    if _rag_engine is None:
        _rag_engine = RAGQueryEngine()
    
    return _rag_engine

def rag_llm(question: str) -> str:
    """Process a question using the RAG pipeline."""
    rag_engine = _get_or_create_rag_engine()
    
    results = rag_engine.retrieval_index.search(question, num_results=5)

    res = []
    for point in results.points:
        res.append(point.payload)

    ans = rag_engine.query_llm(question, res)
    return ans

    
