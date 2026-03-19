"""Dense retrieval pipeline using FAISS and PubMedBERT embeddings."""


def build_faiss_index(embeddings):
    """Build a FAISS index from document embeddings."""
    # TODO: Implement FAISS index construction
    raise NotImplementedError


def retrieve_passages(query, index, top_k=10):
    """Encode query and retrieve top-K passages from FAISS index."""
    # TODO: Implement query encoding and retrieval
    raise NotImplementedError
