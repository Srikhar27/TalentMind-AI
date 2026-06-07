import faiss
import numpy as np

class CandidateRetrievalEngine:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None

    def build_index(self, embeddings):
        """
        Builds a FAISS index for Cosine Similarity search.
        Normalizes vectors beforehand so that Inner Product index equals Cosine Similarity.
        """
        print(f"Building FAISS IndexFlatIP of dimension {self.dimension}...")
        
        # Copy to avoid side-effects
        vectors = embeddings.astype('float32')
        
        # L2 normalization for Cosine Similarity
        faiss.normalize_L2(vectors)
        
        # IndexFlatIP uses Inner Product
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        print(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def retrieve(self, query_embedding, top_n=2000):
        """
        Retrieves top_n candidate indices and their cosine similarities.
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built. Call build_index first.")
            
        # Normalize query embedding
        query = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search the index
        similarities, indices = self.index.search(query, top_n)
        
        # Return list of (candidate_idx, similarity_score)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1: # FAISS returns -1 if there are fewer results than top_n
                results.append((int(idx), float(sim)))
                
        return results
