"""
Caching utilities for faster agent responses
"""
import hashlib
import json
import pickle
import os
from functools import wraps
from typing import Any, Callable
from datetime import datetime, timedelta


class ResponseCache:
    """Cache for LLM responses and tool results"""

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        """
        Initialize cache

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a stable string representation
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get(self, cache_key: str) -> Any:
        """
        Get value from cache

        Args:
            cache_key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(cache_key)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # Check if expired
            if datetime.now() - data['timestamp'] > self.ttl:
                os.remove(cache_path)
                return None

            return data['value']

        except Exception as e:
            print(f"Cache read error: {e}")
            return None

    def set(self, cache_key: str, value: Any):
        """
        Store value in cache

        Args:
            cache_key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            data = {
                'timestamp': datetime.now(),
                'value': value
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

        except Exception as e:
            print(f"Cache write error: {e}")

    def clear(self):
        """Clear all cache entries"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            print(f"Cache clear error: {e}")

    def clear_expired(self):
        """Remove expired cache entries"""
        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.pkl'):
                    continue

                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)

                    if datetime.now() - data['timestamp'] > self.ttl:
                        os.remove(filepath)
                except:
                    pass
        except Exception as e:
            print(f"Cache cleanup error: {e}")


# Global cache instance
_global_cache = ResponseCache()


def cached(cache_instance: ResponseCache = None):
    """
    Decorator to cache function results

    Args:
        cache_instance: Cache instance to use (default: global cache)

    Example:
        @cached()
        def expensive_function(x, y):
            return x + y
    """
    cache = cache_instance or _global_cache

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._get_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result)

            return result

        return wrapper

    return decorator


class LRUCache:
    """In-memory LRU cache for frequently accessed data"""

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache

        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache = {}
        self.access_order = []

    def get(self, key: str) -> Any:
        """Get value and update access order"""
        if key not in self.cache:
            return None

        # Update access order (move to end)
        self.access_order.remove(key)
        self.access_order.append(key)

        return self.cache[key]

    def set(self, key: str, value: Any):
        """Set value and manage cache size"""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


# Global LRU cache for in-memory caching
_lru_cache = LRUCache(max_size=100)


def get_global_cache() -> ResponseCache:
    """Get global cache instance"""
    return _global_cache


def get_lru_cache() -> LRUCache:
    """Get global LRU cache instance"""
    return _lru_cache
