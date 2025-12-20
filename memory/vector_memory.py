"""
Vector-based episodic memory using FAISS for fast similarity search
"""
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import faiss
import pickle
import os
import config


class VectorMemory:
    """Stores and retrieves reflections using FAISS vector similarity"""

    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize vector memory with FAISS index

        Args:
            embedding_dim: Dimension of embeddings (1536 for OpenAI text-embedding-ada-002)
        """
        self.memories = []
        self.embedding_dim = embedding_dim

        # Initialize FAISS index (using IndexFlatL2 for exact search)
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Initialize embeddings model
        self.embeddings_model = None
        try:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {e}")
            print("Memory will use fallback keyword matching")

    def add_memory(self, task_description: str, prediction: float, actual: float,
                   reflection: str, success: bool, metadata: Optional[Dict] = None):
        """
        Add a new memory with embedding to FAISS index

        Args:
            task_description: Description of the task
            prediction: Predicted value
            actual: Actual value
            reflection: Reflection text
            success: Whether prediction was successful
            metadata: Additional metadata
        """
        memory_id = len(self.memories)

        memory = {
            'id': memory_id,
            'timestamp': datetime.now(),
            'task_description': task_description,
            'prediction': prediction,
            'actual': actual,
            'reflection': reflection,
            'success': success,
            'metadata': metadata or {}
        }

        # Generate embedding
        if self.embeddings_model:
            try:
                # Combine task and reflection for richer embedding
                combined_text = f"{task_description}\n{reflection}"
                embedding = self.embeddings_model.embed_query(combined_text)

                # Add to FAISS index
                embedding_array = np.array([embedding], dtype='float32')
                self.index.add(embedding_array)

                memory['has_embedding'] = True
            except Exception as e:
                print(f"Warning: Failed to generate embedding: {e}")
                memory['has_embedding'] = False
        else:
            memory['has_embedding'] = False

        self.memories.append(memory)

        # Maintain max memory size
        if len(self.memories) > config.MAX_MEMORY_ITEMS:
            # Remove oldest memory
            removed = self.memories.pop(0)
            # Rebuild FAISS index (since we can't easily remove from IndexFlatL2)
            self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild FAISS index after removing memories"""
        if not self.embeddings_model:
            return

        # Create new index
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Re-add all embeddings
        for memory in self.memories:
            if memory.get('has_embedding'):
                try:
                    combined_text = f"{memory['task_description']}\n{memory['reflection']}"
                    embedding = self.embeddings_model.embed_query(combined_text)
                    embedding_array = np.array([embedding], dtype='float32')
                    self.index.add(embedding_array)
                except:
                    pass

    def retrieve_relevant_memories(self, task_description: str,
                                   top_k: int = None,
                                   metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant memories using FAISS similarity search

        Args:
            task_description: Current task
            top_k: Number of memories to retrieve
            metadata_filter: Filter by metadata

        Returns:
            List of relevant memories with similarity scores
        """
        if not self.memories:
            return []

        top_k = top_k or config.MEMORY_RETRIEVAL_TOP_K

        # Use FAISS if available
        if self.embeddings_model and self.index.ntotal > 0:
            try:
                return self._retrieve_with_faiss(task_description, top_k, metadata_filter)
            except Exception as e:
                print(f"FAISS retrieval failed: {e}, falling back to keyword matching")
                return self._retrieve_with_keywords(task_description, top_k, metadata_filter)
        else:
            return self._retrieve_with_keywords(task_description, top_k, metadata_filter)

    def _retrieve_with_faiss(self, query: str, top_k: int,
                            metadata_filter: Optional[Dict]) -> List[Dict[str, Any]]:
        """Retrieve memories using FAISS vector search"""
        # Generate query embedding
        query_embedding = self.embeddings_model.embed_query(query)
        query_array = np.array([query_embedding], dtype='float32')

        # Search FAISS index (k * 2 to allow for filtering)
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_array, search_k)

        # Get memories with scores
        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.memories):
                continue

            memory = self.memories[idx].copy()

            # Apply metadata filter
            if metadata_filter:
                if not all(memory['metadata'].get(k) == v for k, v in metadata_filter.items()):
                    continue

            # Convert L2 distance to similarity score (lower is better, so invert)
            # Using negative exponential to convert distance to similarity
            similarity_score = np.exp(-dist)

            # Boost failed predictions (more valuable to learn from)
            if not memory['success']:
                similarity_score *= 1.3

            memory['score'] = float(similarity_score)
            retrieved.append(memory)

        # Sort by score and return top k
        retrieved.sort(key=lambda x: x['score'], reverse=True)
        return retrieved[:top_k]

    def _retrieve_with_keywords(self, query: str, top_k: int,
                                metadata_filter: Optional[Dict]) -> List[Dict[str, Any]]:
        """Fallback: retrieve using keyword matching"""
        query_words = set(query.lower().split())

        scored_memories = []
        for memory in self.memories:
            # Apply metadata filter
            if metadata_filter:
                if not all(memory['metadata'].get(k) == v for k, v in metadata_filter.items()):
                    continue

            # Calculate keyword overlap
            memory_text = f"{memory['task_description']} {memory['reflection']}".lower()
            memory_words = set(memory_text.split())

            overlap = len(query_words.intersection(memory_words))
            score = overlap / len(query_words) if query_words else 0

            # Boost failed predictions
            if not memory['success']:
                score *= 1.5

            memory_copy = memory.copy()
            memory_copy['score'] = score
            scored_memories.append(memory_copy)

        scored_memories.sort(key=lambda x: x['score'], reverse=True)
        return scored_memories[:top_k]

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all stored memories"""
        return self.memories

    def clear_memories(self):
        """Clear all memories and reset index"""
        self.memories = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.memories:
            return {
                'total_memories': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'success_rate': 0.0,
                'using_faiss': False
            }

        successful = sum(1 for m in self.memories if m['success'])

        return {
            'total_memories': len(self.memories),
            'successful_predictions': successful,
            'failed_predictions': len(self.memories) - successful,
            'success_rate': successful / len(self.memories) * 100,
            'using_faiss': self.embeddings_model is not None,
            'index_size': self.index.ntotal
        }

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for LLM prompt"""
        if not memories:
            return "No relevant past experiences found."

        formatted = []
        for i, memory in enumerate(memories, 1):
            score_pct = memory.get('score', 0) * 100
            formatted.append(f"\nRelevant Experience {i} (Similarity: {score_pct:.1f}%):")
            formatted.append(f"Task: {memory['task_description']}")
            formatted.append(f"Prediction: ${memory['prediction']:,.2f} | Actual: ${memory['actual']:,.2f}")
            formatted.append(f"Outcome: {'✓ SUCCESS' if memory['success'] else '✗ FAILED'}")
            formatted.append(f"Key Lesson: {memory['reflection']}")

        return "\n".join(formatted)

    def save_to_disk(self, filepath: str = "memory_cache.pkl"):
        """Save memories to disk"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'memories': self.memories,
                    'embedding_dim': self.embedding_dim
                }, f)
            print(f"Saved {len(self.memories)} memories to {filepath}")
        except Exception as e:
            print(f"Warning: Could not save memories: {e}")

    def load_from_disk(self, filepath: str = "memory_cache.pkl"):
        """Load memories from disk"""
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.memories = data['memories']
            self.embedding_dim = data.get('embedding_dim', 1536)

            # Rebuild FAISS index
            self._rebuild_index()

            print(f"Loaded {len(self.memories)} memories from {filepath}")
        except Exception as e:
            print(f"Warning: Could not load memories: {e}")
