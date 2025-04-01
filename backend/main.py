from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from automata_solver import convert_regex_to_dfa, minimize_automaton, export_automaton
from ai_explainer import explain_automata

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegexInput(BaseModel):
    type: str
    value: str

class AutomatonData(BaseModel):
    states: list
    transitions: dict
    initial_state: str
    final_states: list
    input_symbols: list

@app.get("/")
def home():
    return {"message": "Welcome to Automata Solver API"}

@app.post("/convert")
async def convert(input: RegexInput):
    dfa_svg = convert_regex_to_dfa(input.value)
    return {"dfa_svg": dfa_svg}

@app.get("/explain/{query}")
async def explain(query: str):
    explanation = explain_automata(query)
    return {"explanation": explanation}

@app.post("/minimize")
async def minimize(automaton: AutomatonData):
    minimized = minimize_automaton(automaton.dict())
    return minimized

@app.post("/export/{format}")
async def export(automaton: AutomatonData, format: str):
    if format not in ['svg', 'pdf', 'png']:
        return {"error": "Unsupported format"}
    result = export_automaton(automaton.dict(), format)
    return {"data": result}
