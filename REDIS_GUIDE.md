# Redis Integration Guide for AttendX

## Overview

Redis has been integrated into AttendX for:
- **Session Management**: Server-side sessions with Flask-Session
- **Caching**: Application-level caching with Flask-Caching
- **Rate Limiting**: Enhanced rate limiting with Redis backend
- **Temporary Data**: Key-value storage for temporary data

## Local Development Setup

### Using Docker Compose (Recommended)

Redis is automatically configured in `docker-compose.yml`:

```bash
cd attendX
docker-compose up -d
```

This starts:
- MongoDB on port 27017
- **Redis on port 6379**
- Flask app on port 10000
- Go2RTC on port 1984

### Environment Variable

```bash
REDIS_URL=redis://redis:6379/0  # Docker
# or
REDIS_URL=redis://localhost:6379/0  # Local Redis
```

## Render Deployment

### 1. Add Redis Add-on

In your Render dashboard:
1. Go to your web service
2. Click "Add-ons" → "Redis"
3. Select a plan (free tier available)
4. Copy the Redis URL

### 2. Configure Environment Variable

Add to your Render environment variables:
```
REDIS_URL=redis://red-xxxxx:6379
```

Render automatically injects the Redis URL when you add the add-on.

## Usage in Code

### Using Redis Utility

```python
from redis_utils import get_redis_client, cache_get, cache_set

# Get Redis client
redis_client = get_redis_client()

# Simple caching
cache_set('user:123', {'name': 'John', 'email': 'john@example.com'}, expire=3600)
user_data = cache_get('user:123')

# Direct Redis operations
redis_client.set('counter', 0, expire=300)
redis_client.incr('counter')
count = redis_client.get('counter')

# Batch operations
redis_client.set_many({
    'key1': 'value1',
    'key2': 'value2'
}, expire=600)

# Clear cache by pattern
redis_client.clear_pattern('user:*')
```

### Example: Caching Face Embeddings

```python
from redis_utils import cache_get, cache_set
import hashlib

def get_face_embedding_cached(image):
    # Generate cache key from image hash
    img_hash = hashlib.md5(image.tobytes()).hexdigest()
    cache_key = f'embedding:{img_hash}'
    
    # Try cache first
    cached = cache_get(cache_key)
    if cached:
        return cached
    
    # Generate embedding if not cached
    embedding = get_face_embedding(image)
    
    # Cache for 1 hour
    cache_set(cache_key, embedding, expire=3600)
    
    return embedding
```

### Example: Rate Limiting with Redis

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis_utils import get_redis_client

# Configure Flask-Limiter with Redis
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=os.environ.get('REDIS_URL', 'memory://'),
    strategy="fixed-window"
)

@app.route('/api/enroll', methods=['POST'])
@limiter.limit("5 per hour")
def enroll():
    # Your enrollment logic
    pass
```

### Example: Session Management

```python
from flask import Flask, session
from flask_session import Session

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = get_redis_client().client
Session(app)

@app.route('/login')
def login():
    session['user_id'] = 123
    session['email'] = 'user@example.com'
    return 'Logged in'
```

## Redis Utility Functions

### Connection
- `get_redis_client()` - Get global Redis client instance

### Basic Operations
- `cache_get(key)` - Get value
- `cache_set(key, value, expire)` - Set value with expiration
- `cache_delete(key)` - Delete key
- `cache_clear(pattern)` - Clear keys by pattern

### Advanced Operations
- `redis_client.exists(key)` - Check if key exists
- `redis_client.incr(key, amount)` - Increment counter
- `redis_client.expire(key, seconds)` - Set expiration
- `redis_client.get_ttl(key)` - Get time to live
- `redis_client.get_many(keys)` - Batch get
- `redis_client.set_many(mapping, expire)` - Batch set

## Fallback Behavior

If Redis is unavailable, the application gracefully continues:
- Cache operations return `None` or `False`
- No errors thrown
- Application functionality preserved
- Logs warning: "Running without Redis caching"

## Performance Benefits

1. **Faster Authentication**: Session data in memory
2. **Reduced Database Load**: Cached queries
3. **Rate Limiting**: Distributed rate limiting across instances
4. **Temporary Storage**: Fast key-value storage for enrollment sessions

## Monitoring

Check Redis connection:
```python
from redis_utils import get_redis_client

redis_client = get_redis_client()
if redis_client.is_connected:
    print("✅ Redis connected")
else:
    print("❌ Redis not available")
```

## Redis on Render

Render provides managed Redis with:
- ✅ Automatic backups
- ✅ High availability
- ✅ SSL/TLS encryption
- ✅ Free tier available
- ✅ Automatic scaling

### Free Tier Limits
- 25 MB storage
- Shared CPU
- Perfect for development and small projects

### Paid Tiers
- More storage and memory
- Dedicated resources
- Better performance

## Troubleshooting

### Connection Failed
```python
# Check Redis URL
import os
print(os.environ.get('REDIS_URL'))

# Test connection
from redis_utils import get_redis_client
client = get_redis_client()
print(client.is_connected)
```

### Performance Issues
- Check Redis memory usage
- Review cache expiration times
- Monitor key count with `redis_client.client.dbsize()`

### Clear All Cache
```python
from redis_utils import cache_clear
cache_clear('*')  # Clear all keys
```

## Best Practices

1. **Set Expiration**: Always set expiration on cached data
2. **Use Patterns**: Organize keys with prefixes (e.g., `user:123`, `session:abc`)
3. **Handle Failures**: Redis failures shouldn't crash the app
4. **Monitor Memory**: Keep track of Redis memory usage
5. **Cache Invalidation**: Clear stale data when updating database

## Next Steps

- [ ] Enable Redis sessions in Flask app
- [ ] Add caching to database queries
- [ ] Implement rate limiting with Redis
- [ ] Cache face embeddings
- [ ] Monitor Redis performance
