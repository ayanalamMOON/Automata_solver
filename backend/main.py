from fastapi import FastAPI, Request, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from automata_solver import (
    convert_regex_to_dfa,
    minimize_automaton,
    export_automaton,
    validate_regex
)
from ai_explainer import explain_automata
from PIL import Image
import pytesseract
import io
import os
import logging
from automata.fa.dfa import DFA
from typing import List, Dict, Optional
import uvicorn
from security import get_current_active_user, User, require_admin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create FastAPI application with custom OpenAPI schema
app = FastAPI(
    title="Automata Solver API",
    description="""
    API for solving automata problems, converting regex to DFA, and explaining automata concepts.
    
    Features:
    - Convert regular expressions to DFAs
    - Minimize automata
    - Validate and analyze automata solutions
    - Batch processing capabilities
    - AI-powered explanations
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="Automata Solver API",
        version="1.0.0",
        description="Complete API documentation for the Automata Solver",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": "token",
                    "scopes": {}
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://localhost:8080",
    os.environ.get("PRODUCTION_FRONTEND_URL", "")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def error_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Request to {request.url} failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "message": str(e)},
        )

# Request/Response Models
class RegexInput(BaseModel):
    regex: str

class AutomatonData(BaseModel):
    states: List[str]
    alphabet: List[str]
    transitions: Dict[str, Dict[str, str]]
    start_state: str
    accept_states: List[str]

class BulkAutomataRequest(BaseModel):
    items: List[Dict] = Field(..., description="List of automata to process")
    parallel: bool = Field(True, description="Whether to process in parallel")

class BulkAutomataResponse(BaseModel):
    results: List[Dict]
    failed_count: int
    success_count: int

class AutomataAnalysisRequest(BaseModel):
    automaton: Dict = Field(..., description="Automaton to analyze")
    properties: List[str] = Field(..., description="Properties to analyze")

@app.get("/")
def home():
    return {"message": "Welcome to Automata Solver API", "status": "active"}

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring services"""
    return {"status": "healthy"}

@app.post("/convert")
async def convert(input: RegexInput):
    logger.info(f"Converting regex: {input.regex}")
    dfa_svg = convert_regex_to_dfa(input.regex)
    return {"dfa_svg": dfa_svg}

@app.get("/explain/{query}")
async def explain(query: str):
    logger.info(f"Explaining query: {query}")
    explanation = explain_automata(query)
    return {"explanation": explanation}

@app.post("/minimize")
async def minimize(automaton: AutomatonData):
    logger.info("Minimizing automaton")
    minimized = minimize_automaton(automaton.dict())
    return minimized

@app.post("/export/{format}")
async def export(automaton: AutomatonData, format: str):
    if format not in ['svg', 'pdf', 'png']:
        logger.warning(f"Unsupported export format requested: {format}")
        return {"error": "Unsupported format"}

    logger.info(f"Exporting automaton as {format}")
    result = export_automaton(automaton.dict(), format)
    return {"data": result}

@app.post("/validate_regex")
async def validate_regex_endpoint(input: RegexInput):
    logger.info(f"Validating regex: {input.regex}")
    is_valid = validate_regex(input.regex)
    return {"is_valid": is_valid}

@app.post("/ai_suggestions")
async def ai_suggestions(input: RegexInput):
    logger.info(f"Getting AI suggestions for: {input.regex}")
    suggestions = explain_automata(f"Improve regex: {input.regex}")
    return {"suggestions": suggestions}

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        # Perform OCR using pytesseract
        extracted_text = pytesseract.image_to_string(img)

        return JSONResponse(content={"text": extracted_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/test_string")
async def test_string(data: AutomatonData):
    try:
        dfa = DFA(
            states=set(data.states),
            input_symbols=set(data.alphabet),
            transitions=data.transitions,
            initial_state=data.start_state,
            final_states=set(data.accept_states),
        )
        is_accepted = dfa.accepts_input(data.string)
        execution_path = dfa.read_input_stepwise(data.string)
        return {"is_accepted": is_accepted, "execution_path": execution_path}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test_openai")
async def test_openai():
    """Test endpoint to check OpenAI API access."""
    try:
        from ai_explainer import explain_automata
        test_query = "What is a DFA?"
        response = explain_automata(test_query)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post(
    "/api/bulk/convert",
    response_model=BulkAutomataResponse,
    summary="Convert multiple regular expressions to DFAs",
    description="Process multiple regular expressions in parallel, converting each to a DFA"
)
async def bulk_convert(
    request: BulkAutomataRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Convert multiple regular expressions to DFAs in parallel.
    
    - **items**: List of regular expressions to convert
    - **parallel**: Whether to process in parallel
    
    Returns processed results with success/failure counts.
    """
    results = []
    success_count = 0
    failed_count = 0
    
    for item in request.items:
        try:
            dfa = convert_regex_to_dfa(item["regex"])
            results.append({"success": True, "dfa": dfa})
            success_count += 1
        except Exception as e:
            results.append({"success": False, "error": str(e)})
            failed_count += 1
    
    return {
        "results": results,
        "success_count": success_count,
        "failed_count": failed_count
    }

@app.post(
    "/api/bulk/minimize",
    response_model=BulkAutomataResponse,
    summary="Minimize multiple automata",
    description="Process and minimize multiple automata in parallel"
)
async def bulk_minimize(
    request: BulkAutomataRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Minimize multiple automata in parallel.
    
    - **items**: List of automata to minimize
    - **parallel**: Whether to process in parallel
    
    Returns minimized automata with success/failure counts.
    """
    results = []
    success_count = 0
    failed_count = 0
    
    for item in request.items:
        try:
            minimized = minimize_automaton(item["automaton"])
            results.append({"success": True, "minimized": minimized})
            success_count += 1
        except Exception as e:
            results.append({"success": False, "error": str(e)})
            failed_count += 1
    
    return {
        "results": results,
        "success_count": success_count,
        "failed_count": failed_count
    }

@app.post(
    "/api/analyze",
    summary="Analyze automaton properties",
    description="Analyze various properties of an automaton"
)
async def analyze_automaton(
    request: AutomataAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze properties of an automaton.
    
    - **automaton**: The automaton to analyze
    - **properties**: List of properties to analyze (e.g., "deterministic", "minimal")
    
    Returns analysis results for each requested property.
    """
    results = {}
    
    try:
        for property in request.properties:
            if property == "deterministic":
                results["deterministic"] = is_deterministic(request.automaton)
            elif property == "minimal":
                results["minimal"] = is_minimal(request.automaton)
            elif property == "complete":
                results["complete"] = is_complete(request.automaton)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return results

@app.get(
    "/api/admin/stats",
    summary="Get system statistics",
    dependencies=[Depends(require_admin)]
)
async def get_system_stats():
    """
    Get system statistics (admin only).
    
    Returns various system metrics and statistics.
    """
    # Implementation of system statistics
    pass

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error processing request: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
        reload=os.environ.get("ENV", "development") == "development"
    )
