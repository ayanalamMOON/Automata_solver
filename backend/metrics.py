from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Request metrics
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Business metrics
AUTOMATA_OPERATIONS_TOTAL = Counter(
    'automata_operations_total',
    'Total automata operations',
    ['operation_type', 'automata_type']
)

AUTOMATA_OPERATION_DURATION_SECONDS = Histogram(
    'automata_operation_duration_seconds',
    'Automata operation latency',
    ['operation_type', 'automata_type']
)

# Cache metrics
CACHE_HITS_TOTAL = Counter(
    'cache_hits_total',
    'Total cache hits'
)

CACHE_MISSES_TOTAL = Counter(
    'cache_misses_total',
    'Total cache misses'
)

# Resource metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

MEMORY_USAGE_BYTES = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes'
)

# OpenAI API metrics
OPENAI_API_REQUESTS_TOTAL = Counter(
    'openai_api_requests_total',
    'Total OpenAI API requests',
    ['endpoint']
)

OPENAI_API_DURATION_SECONDS = Histogram(
    'openai_api_duration_seconds',
    'OpenAI API request latency',
    ['endpoint']
)

# Redis metrics
REDIS_OPERATIONS_TOTAL = Counter(
    'redis_operations_total',
    'Total Redis operations',
    ['operation_type']
)

REDIS_OPERATION_DURATION_SECONDS = Histogram(
    'redis_operation_duration_seconds',
    'Redis operation latency',
    ['operation_type']
)

# Error metrics
ERROR_COUNTS_TOTAL = Counter(
    'error_counts_total',
    'Total error counts',
    ['error_type']
)

class MetricsMiddleware:
    """Middleware to collect HTTP request metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        start_time = time.time()
        
        # Track active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await self.app(scope, receive, send)
            
            # Record request metrics
            method = scope.get("method", "UNKNOWN")
            path = scope["path"]
            status = response.status_code
            
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=path,
                status=status
            ).inc()
            
            # Record request duration
            duration = time.time() - start_time
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            return response
            
        finally:
            # Decrease active connections count
            ACTIVE_CONNECTIONS.dec()

def track_operation(operation_type: str, automata_type: str):
    """Decorator to track automata operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record operation metrics
                AUTOMATA_OPERATIONS_TOTAL.labels(
                    operation_type=operation_type,
                    automata_type=automata_type
                ).inc()
                
                # Record operation duration
                duration = time.time() - start_time
                AUTOMATA_OPERATION_DURATION_SECONDS.labels(
                    operation_type=operation_type,
                    automata_type=automata_type
                ).observe(duration)
                
                return result
                
            except Exception as e:
                # Record error metrics
                ERROR_COUNTS_TOTAL.labels(
                    error_type=type(e).__name__
                ).inc()
                raise
                
        return wrapper
    return decorator

def track_cache(func):
    """Decorator to track cache operations"""
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            
            if result is not None:
                CACHE_HITS_TOTAL.inc()
            else:
                CACHE_MISSES_TOTAL.inc()
            
            return result
            
        except Exception as e:
            ERROR_COUNTS_TOTAL.labels(
                error_type=type(e).__name__
            ).inc()
            raise
            
    return wrapper

def track_openai_api(endpoint: str):
    """Decorator to track OpenAI API operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record API metrics
                OPENAI_API_REQUESTS_TOTAL.labels(
                    endpoint=endpoint
                ).inc()
                
                # Record API duration
                duration = time.time() - start_time
                OPENAI_API_DURATION_SECONDS.labels(
                    endpoint=endpoint
                ).observe(duration)
                
                return result
                
            except Exception as e:
                ERROR_COUNTS_TOTAL.labels(
                    error_type=type(e).__name__
                ).inc()
                raise
                
        return wrapper
    return decorator

def track_redis_operation(operation_type: str):
    """Decorator to track Redis operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record Redis metrics
                REDIS_OPERATIONS_TOTAL.labels(
                    operation_type=operation_type
                ).inc()
                
                # Record operation duration
                duration = time.time() - start_time
                REDIS_OPERATION_DURATION_SECONDS.labels(
                    operation_type=operation_type
                ).observe(duration)
                
                return result
                
            except Exception as e:
                ERROR_COUNTS_TOTAL.labels(
                    error_type=type(e).__name__
                ).inc()
                raise
                
        return wrapper
    return decorator