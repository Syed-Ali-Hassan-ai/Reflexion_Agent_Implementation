"""
Episodic memory for storing and retrieving reflections
"""
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import config


class EpisodicMemory:
    """Stores and retrieves past reflections using simple similarity matching"""

    def __init__(self, use_embeddings: bool = False):
        """
        Initialize episodic memory

        Args:
            use_embeddings: Whether to use embeddings for similarity (optional)
        """
        self.memories = []
        self.use_embeddings = use_embeddings
        self.embeddings_model = None

        if use_embeddings:
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings_model = OpenAIEmbeddings()
            except:
                print("Warning: Could not initialize embeddings. Using keyword matching instead.")
                self.use_embeddings = False

    def add_memory(self, task_description: str, prediction: float, actual: float,
                   reflection: str, success: bool, metadata: Optional[Dict] = None):
        """
        Add a new memory (reflection) to the store

        Args:
            task_description: Description of the task
            prediction: Predicted value
            actual: Actual value
            reflection: Reflection text about what went wrong/right
            success: Whether the prediction was successful
            metadata: Additional metadata (store, dept, etc.)
        """
        memory = {
            'id': len(self.memories) + 1,
            'timestamp': datetime.now(),
            'task_description': task_description,
            'prediction': prediction,
            'actual': actual,
            'reflection': reflection,
            'success': success,
            'metadata': metadata or {},
            'embedding': None
        }

        # Generate embedding if enabled
        if self.use_embeddings and self.embeddings_model:
            try:
                memory['embedding'] = self.embeddings_model.embed_query(
                    f"{task_description} {reflection}"
                )
            except:
                pass

        self.memories.append(memory)

        # Keep only the most recent memories
        if len(self.memories) > config.MAX_MEMORY_ITEMS:
            self.memories = self.memories[-config.MAX_MEMORY_ITEMS:]

    def retrieve_relevant_memories(self, task_description: str,
                                   top_k: int = None,
                                   metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past memories for a given task

        Args:
            task_description: Current task description
            top_k: Number of memories to retrieve (default from config)
            metadata_filter: Filter by metadata (e.g., same store/dept)

        Returns:
            List of relevant memories
        """
        if not self.memories:
            return []

        top_k = top_k or config.MEMORY_RETRIEVAL_TOP_K

        # Filter by metadata if provided
        filtered_memories = self.memories
        if metadata_filter:
            filtered_memories = [
                m for m in self.memories
                if all(m['metadata'].get(k) == v for k, v in metadata_filter.items())
            ]

        if not filtered_memories:
            filtered_memories = self.memories

        # Calculate similarity scores
        if self.use_embeddings and self.embeddings_model:
            scored_memories = self._retrieve_with_embeddings(task_description, filtered_memories)
        else:
            scored_memories = self._retrieve_with_keywords(task_description, filtered_memories)

        # Sort by relevance and return top k
        scored_memories.sort(key=lambda x: x['score'], reverse=True)
        return scored_memories[:top_k]

    def _retrieve_with_keywords(self, query: str, memories: List[Dict]) -> List[Dict]:
        """
        Retrieve memories using keyword matching

        Args:
            query: Query string
            memories: List of memories to search

        Returns:
            List of memories with relevance scores
        """
        query_words = set(query.lower().split())

        scored_memories = []
        for memory in memories:
            # Combine task description and reflection for matching
            memory_text = f"{memory['task_description']} {memory['reflection']}".lower()
            memory_words = set(memory_text.split())

            # Calculate overlap score
            overlap = len(query_words.intersection(memory_words))
            score = overlap / len(query_words) if query_words else 0

            # Boost score for failures (more valuable to learn from)
            if not memory['success']:
                score *= 1.5

            # Boost recent memories slightly
            recency_boost = 1.0 + (0.1 * (self.memories.index(memory) / len(self.memories)))
            score *= recency_boost

            scored_memories.append({
                **memory,
                'score': score
            })

        return scored_memories

    def _retrieve_with_embeddings(self, query: str, memories: List[Dict]) -> List[Dict]:
        """
        Retrieve memories using embedding similarity

        Args:
            query: Query string
            memories: List of memories to search

        Returns:
            List of memories with relevance scores
        """
        try:
            query_embedding = self.embeddings_model.embed_query(query)

            scored_memories = []
            for memory in memories:
                if memory['embedding'] is None:
                    continue

                # Calculate cosine similarity
                similarity = np.dot(query_embedding, memory['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory['embedding'])
                )

                # Boost score for failures
                if not memory['success']:
                    similarity *= 1.2

                scored_memories.append({
                    **memory,
                    'score': float(similarity)
                })

            return scored_memories

        except Exception as e:
            print(f"Error with embeddings, falling back to keywords: {e}")
            return self._retrieve_with_keywords(query, memories)

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all stored memories"""
        return self.memories

    def clear_memories(self):
        """Clear all memories"""
        self.memories = []

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics about stored memories"""
        if not self.memories:
            return {
                'total_memories': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'success_rate': 0.0
            }

        successful = sum(1 for m in self.memories if m['success'])
        failed = len(self.memories) - successful

        return {
            'total_memories': len(self.memories),
            'successful_predictions': successful,
            'failed_predictions': failed,
            'success_rate': successful / len(self.memories) * 100 if self.memories else 0.0
        }

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format memories into a string for inclusion in prompts

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted string
        """
        if not memories:
            return "No relevant past experiences found."

        formatted = []
        for i, memory in enumerate(memories, 1):
            formatted.append(f"\nExperience {i}:")
            formatted.append(f"Task: {memory['task_description']}")
            formatted.append(f"Prediction: ${memory['prediction']:,.2f}")
            formatted.append(f"Actual: ${memory['actual']:,.2f}")
            formatted.append(f"Outcome: {'SUCCESS' if memory['success'] else 'FAILED'}")
            formatted.append(f"Lesson Learned: {memory['reflection']}")

        return "\n".join(formatted)
