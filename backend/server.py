import uvicorn
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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