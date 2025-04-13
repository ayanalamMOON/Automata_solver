from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from automata_solver import AutomataSolver, DFA, AutomataError

app = FastAPI()

class AutomatonData(BaseModel):
    states: List[str]
    alphabet: List[str]
    transitions: Dict[str, Dict[str, str]]
    start_state: str
    accept_states: List[str]
    input_string: Optional[str] = None

@app.post("/api/simulate/step_by_step")
async def simulate_step_by_step(data: AutomatonData):
    """
    Simulate an automaton step by step on an input string.
    Returns detailed step-by-step execution information for visualization.
    """
    try:
        # Create DFA object
        dfa = AutomataSolver._create_dfa({
            "name": "Simulation DFA",
            "states": [{"name": s, "initial": s == data.start_state, "final": s in data.accept_states} 
                      for s in data.states],
            "transitions": [{"from": from_state, "symbol": symbol, "to": to_state} 
                          for from_state, trans in data.transitions.items() 
                          for symbol, to_state in trans.items()]
        })
        
        # Run step by step simulation
        if not data.input_string:
            raise HTTPException(status_code=400, message="Input string is required")
            
        result = dfa.simulate_step_by_step(data.input_string)
        return result
        
    except AutomataError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))