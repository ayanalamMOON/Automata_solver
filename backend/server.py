import uvicorn
import os
import argparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
from logging.config import dictConfig
import uuid
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
from automata_solver import AutomataSolver, BatchProcessor, AutomataExporter, StepByStepBuilder
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import sys

# Logging configuration
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.processors.JSONRenderer(),
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        }
    },
    "loggers": {
        "": {"handlers": ["console"], "level": "INFO"},
    }
}

# Configure logging
dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Exception handler for rate limiting
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Metrics
REQUESTS = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])

@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add correlation ID to log context
    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
        }
    )
    
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        logger.error(
            "request_failed",
            extra={
                "request_id": request_id,
                "error": str(e),
                "path": request.url.path,
            }
        )
        raise

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    health_status: Dict[str, Any] = {
        "status": "healthy",
        "components": {}
    }
    
    # Check Redis connection
    try:
        redis_client = get_redis_client()
        redis_client.ping()
        health_status["components"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check OpenAI API
    try:
        openai.api_key
        health_status["components"]["openai"] = {"status": "configured"}
    except Exception as e:
        health_status["components"]["openai"] = {
            "status": "unconfigured",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Add system info
    health_status["components"]["system"] = {
        "status": "healthy",
        "cpu_count": os.cpu_count(),
        "python_version": sys.version
    }
    
    return health_status

# Initialize components
solver = AutomataSolver()
batch_processor = BatchProcessor()
builder = StepByStepBuilder()

class BatchSubmission(BaseModel):
    submissions: List[Dict]

class AutomataDefinition(BaseModel):
    automata_type: str
    name: str
    definition: Dict

class BuilderAction(BaseModel):
    action: str
    params: Dict

@app.post("/api/batch-process")
async def process_batch(submission: BatchSubmission):
    """Process multiple automata submissions in parallel"""
    try:
        results = await batch_processor.process_batch_submissions(submission.submissions)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/export")
async def export_automata(automata: AutomataDefinition):
    """Export automata to different formats"""
    try:
        instance = solver.create_automata(automata.automata_type, automata.definition)
        return {
            "jflap": AutomataExporter.to_jflap(instance),
            "dot": AutomataExporter.to_dot(instance)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/import")
async def import_automata(file_content: str, format: str):
    """Import automata from different formats"""
    try:
        if format.lower() == "jflap":
            automata = AutomataExporter.from_jflap(file_content)
            return {
                "status": "success",
                "automata": {
                    "type": automata.__class__.__name__,
                    "definition": automata.__dict__
                }
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/builder/start")
async def start_building(params: AutomataDefinition):
    """Start building a new automata"""
    try:
        state = builder.start_new(params.automata_type, params.name)
        return {"status": "success", "state": state}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/builder/action")
async def builder_action(action: BuilderAction):
    """Perform a builder action (add state, add transition, etc.)"""
    try:
        if action.action == "add_state":
            state = builder.add_state(**action.params)
        elif action.action == "add_transition":
            state = builder.add_transition(**action.params)
        elif action.action == "undo":
            state = builder.undo()
        elif action.action == "redo":
            state = builder.redo()
        elif action.action == "simulate":
            state = builder.simulate_input(**action.params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action.action}")
        return {"status": "success", "state": state}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def run_server(host="0.0.0.0", port=8000, reload=False):
    """
    Run the FastAPI server with the specified configuration.
    
    Args:
        host: The host to bind the server to
        port: The port to bind the server to
        reload: Whether to enable auto-reload for development
    """
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        workers=os.cpu_count()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Automata Solver API server")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.environ.get("PORT", 8000)), 
        help="Port to bind the server to (default: 8000 or from PORT env variable)"
    )
    parser.add_argument(
        "--dev", 
        action="store_true", 
        help="Run in development mode with auto-reload"
    )
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.dev)