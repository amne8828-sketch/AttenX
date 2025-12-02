"""
Redis utility module for caching and session management
Supports both local Redis and Render Redis add-on
"""

import os
import redis
from typing import Optional, Any
import json
import pickle
from datetime import timedelta

class RedisClient:
    """Redis client wrapper with connection pooling"""
    
    def __init__(self):
        """Initialize Redis client with connection from environment"""
        self.redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        
        try:
            # Create connection pool
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=10,
                decode_responses=False  # We'll handle encoding ourselves
            )
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            print(f"✅ Redis connected: {self.redis_url}")
            self._connected = True
            
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            print("Running without Redis caching")
            self.client = None
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self._connected
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with automatic deserialization"""
        if not self.is_connected:
            return None
        
        try:
            value = self.client.get(key)
            if value is None:
                return None
            
            # Try to unpickle, fallback to decode
            try:
                return pickle.loads(value)
            except:
                return value.decode('utf-8')
                
        except Exception as e:
            print(f"Redis GET error for key '{key}': {e}")
            return None
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        Set value in Redis with automatic serialization
        
        Args:
            key: Redis key
            value: Value to store (will be pickled)
            expire: Expiration time in seconds (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (str, int, float)):
                serialized = str(value).encode('utf-8')
            else:
                serialized = pickle.dumps(value)
            
            # Set with optional expiration
            if expire:
                return self.client.setex(key, expire, serialized)
            else:
                return self.client.set(key, serialized)
                
        except Exception as e:
            print(f"Redis SET error for key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.is_connected:
            return False
        
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Redis DELETE error for key '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.is_connected:
            return False
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            print(f"Redis EXISTS error for key '{key}': {e}")
            return False
    
    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter"""
        if not self.is_connected:
            return None
        
        try:
            return self.client.incr(key, amount)
        except Exception as e:
            print(f"Redis INCR error for key '{key}': {e}")
            return None
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on existing key"""
        if not self.is_connected:
            return False
        
        try:
            return self.client.expire(key, seconds)
        except Exception as e:
            print(f"Redis EXPIRE error for key '{key}': {e}")
            return False
    
    def get_many(self, keys: list) -> dict:
        """Get multiple keys at once"""
        if not self.is_connected:
            return {}
        
        try:
            values = self.client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = pickle.loads(value)
                    except:
                        result[key] = value.decode('utf-8')
            return result
        except Exception as e:
            print(f"Redis MGET error: {e}")
            return {}
    
    def set_many(self, mapping: dict, expire: Optional[int] = None) -> bool:
        """Set multiple key-value pairs at once"""
        if not self.is_connected:
            return False
        
        try:
            # Serialize all values
            serialized = {}
            for key, value in mapping.items():
                if isinstance(value, (str, int, float)):
                    serialized[key] = str(value).encode('utf-8')
                else:
                    serialized[key] = pickle.dumps(value)
            
            # Set all at once
            result = self.client.mset(serialized)
            
            # Set expiration if specified
            if expire and result:
                pipeline = self.client.pipeline()
                for key in mapping.keys():
                    pipeline.expire(key, expire)
                pipeline.execute()
            
            return result
        except Exception as e:
            print(f"Redis MSET error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.is_connected:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Redis CLEAR_PATTERN error for pattern '{pattern}': {e}")
            return 0
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get time to live for a key"""
        if not self.is_connected:
            return None
        
        try:
            return self.client.ttl(key)
        except Exception as e:
            print(f"Redis TTL error for key '{key}': {e}")
            return None

# Global Redis client instance
_redis_client = None

def get_redis_client() -> RedisClient:
    """Get or create global Redis client instance"""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client

# Convenience functions
def cache_get(key: str) -> Optional[Any]:
    """Get value from cache"""
    return get_redis_client().get(key)

def cache_set(key: str, value: Any, expire: int = 3600) -> bool:
    """Set value in cache with 1 hour default expiration"""
    return get_redis_client().set(key, value, expire)

def cache_delete(key: str) -> bool:
    """Delete value from cache"""
    return get_redis_client().delete(key)

def cache_clear(pattern: str = "*") -> int:
    """Clear cache matching pattern"""
    return get_redis_client().clear_pattern(pattern)
