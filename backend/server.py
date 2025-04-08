import uvicorn
import os
import argparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from automata_solver import AutomataSolver, BatchProcessor, AutomataExporter, StepByStepBuilder

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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