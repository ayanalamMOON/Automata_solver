from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
from typing import List, Dict
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create FastAPI application with metadata
app = FastAPI(
    title="Automata Solver API",
    description="API for solving automata problems, converting regex to DFA, and explaining automata concepts",
    version="1.0.0",
)

# Configure CORS for production
origins = [
    "http://localhost:5173",  # Local development frontend
    "http://localhost:8080",  # Alternative local frontend
]

# Add production URLs if environment variables are set
production_url = os.environ.get("PRODUCTION_FRONTEND_URL")
if production_url:
    origins.append(production_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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


class RegexInput(BaseModel):
    regex: str


class AutomatonData(BaseModel):
    states: List[str]
    alphabet: List[str]
    transitions: Dict[str, Dict[str, str]]
    start_state: str
    accept_states: List[str]


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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
        reload=os.environ.get("ENV", "development") == "development"
    )
