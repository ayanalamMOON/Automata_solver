from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from automata_solver import (
    AutomataSolver, 
    DFA, 
    AutomataError, 
    validate_regex, 
    convert_regex_to_dfa,
    create_nfa_from_regex,
    convert_nfa_to_dfa
)
from grammar_solver import Grammar, GrammarError, ValidationError

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegexInput(BaseModel):
    regex: str

class AutomatonData(BaseModel):
    states: List[str]
    alphabet: List[str]
    transitions: Dict[str, Dict[str, str]]
    start_state: str
    accept_states: List[str]
    input_string: Optional[str] = None

class ProductionRule(BaseModel):
    left: str
    right: List[str]

class GrammarInput(BaseModel):
    name: str
    productions: List[ProductionRule]
    start_symbol: str

@app.post("/api/validate_regex")
async def validate_regex_endpoint(input: RegexInput):
    """
    Validate a regular expression
    
    Args:
        input: RegexInput object containing the regex to validate
        
    Returns:
        Dict with is_valid flag
    """
    try:
        is_valid = validate_regex(input.regex)
        return {"is_valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/convert")
async def convert_regex_endpoint(input: RegexInput):
    """
    Convert a regular expression to a DFA
    
    Args:
        input: RegexInput object containing the regex to convert
        
    Returns:
        Complete DFA definition with visualization state
    """
    try:
        if not validate_regex(input.regex):
            raise HTTPException(status_code=400, detail="Invalid regular expression")
            
        # Create NFA from regex
        nfa = create_nfa_from_regex(input.regex)
        
        # Convert NFA to DFA
        dfa = convert_nfa_to_dfa(nfa)
        
        # Get visualization state
        vis_state = dfa.get_visualization_state()
        
        # Return complete DFA definition
        return {
            "states": list(dfa.states.keys()),
            "alphabet": list(dfa.alphabet),
            "transitions": {
                from_state: {
                    symbol: dfa.transitions[(from_state, symbol)]
                    for symbol in dfa.alphabet
                    if (from_state, symbol) in dfa.transitions
                }
                for from_state in dfa.states
            },
            "start_state": next(s.name for s in dfa.states.values() if s.is_initial),
            "accept_states": [s.name for s in dfa.states.values() if s.is_final],
            "visualization_state": vis_state
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
            raise HTTPException(status_code=400, detail="Input string is required")
            
        result = dfa.simulate_step_by_step(data.input_string)
        return result
        
    except AutomataError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/grammar/validate")
async def validate_grammar(input: GrammarInput):
    """Validate a context-free grammar"""
    try:
        grammar = Grammar(input.name)
        for prod in input.productions:
            grammar.add_production(prod.left, prod.right)
        grammar.set_start_symbol(input.start_symbol)
        return {"valid": True}
    except (GrammarError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/grammar/to_cnf")
async def convert_to_cnf(input: GrammarInput):
    """Convert a grammar to Chomsky Normal Form"""
    try:
        grammar = Grammar(input.name)
        for prod in input.productions:
            grammar.add_production(prod.left, prod.right)
        grammar.set_start_symbol(input.start_symbol)
        
        cnf = grammar.to_cnf()
        return {
            "name": cnf.name,
            "productions": [
                {"left": p.left, "right": p.right}
                for p in cnf.productions
            ],
            "start_symbol": cnf.start_symbol
        }
    except (GrammarError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/grammar/check_ll1")
async def check_ll1(input: GrammarInput):
    """Check if a grammar is LL(1)"""
    try:
        grammar = Grammar(input.name)
        for prod in input.productions:
            grammar.add_production(prod.left, prod.right)
        grammar.set_start_symbol(input.start_symbol)
        
        is_ll1 = grammar.is_ll1()
        return {"is_ll1": is_ll1}
    except (GrammarError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/grammar/parse")
async def parse_string(input: GrammarInput, string: str):
    """Generate a parse tree for an input string"""
    try:
        grammar = Grammar(input.name)
        for prod in input.productions:
            grammar.add_production(prod.left, prod.right)
        grammar.set_start_symbol(input.start_symbol)
        
        tree = grammar.generate_parse_tree(string)
        return {
            "visualization": tree.visualize(),
            "valid": True
        }
    except (GrammarError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/grammar/batch")
async def batch_process_grammar(input: GrammarInput, operations: List[str]):
    """Process multiple grammar operations in a single request"""
    try:
        grammar = Grammar(input.name)
        for prod in input.productions:
            grammar.add_production(prod.left, prod.right)
        grammar.set_start_symbol(input.start_symbol)
        
        results = {}
        
        for operation in operations:
            if operation == "validate":
                results["validate"] = {"valid": True}
            
            elif operation == "optimize":
                optimized = Grammar(f"Optimized {input.name}")
                optimized.terminals = grammar.terminals.copy()
                optimized.non_terminals = grammar.non_terminals.copy()
                optimized.productions = [Production(p.left, p.right) for p in grammar.productions]
                optimized.start_symbol = grammar.start_symbol
                optimized.optimize()
                
                results["optimize"] = {
                    "name": optimized.name,
                    "productions": [
                        {"left": p.left, "right": p.right}
                        for p in optimized.productions
                    ],
                    "start_symbol": optimized.start_symbol
                }
            
            elif operation == "cnf":
                cnf = grammar.to_cnf()
                results["cnf"] = {
                    "name": cnf.name,
                    "productions": [
                        {"left": p.left, "right": p.right}
                        for p in cnf.productions
                    ],
                    "start_symbol": cnf.start_symbol
                }
            
            elif operation == "ll1":
                results["ll1"] = {"is_ll1": grammar.is_ll1()}
                
            elif operation == "lr0":
                results["lr0"] = {"is_lr0": grammar.is_lr0()}
                
        return results
        
    except (GrammarError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/grammar/batch_parse")
async def batch_parse_strings(input: GrammarInput, strings: List[str]):
    """Parse multiple strings using the grammar"""
    try:
        grammar = Grammar(input.name)
        for prod in input.productions:
            grammar.add_production(prod.left, prod.right)
        grammar.set_start_symbol(input.start_symbol)
        
        results = {}
        for i, string in enumerate(strings):
            try:
                tree = grammar.generate_parse_tree(string)
                results[string] = {
                    "valid": True,
                    "visualization": tree.visualize()
                }
            except GrammarError as e:
                results[string] = {
                    "valid": False,
                    "error": str(e)
                }
                
        return results
        
    except (GrammarError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/grammar/optimize")
async def optimize_grammar(input: GrammarInput):
    """Apply all optimization algorithms to the grammar"""
    try:
        grammar = Grammar(input.name)
        for prod in input.productions:
            grammar.add_production(prod.left, prod.right)
        grammar.set_start_symbol(input.start_symbol)
        
        grammar.optimize()
        
        return {
            "name": grammar.name,
            "productions": [
                {"left": p.left, "right": p.right}
                for p in grammar.productions
            ],
            "start_symbol": grammar.start_symbol
        }
    except (GrammarError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))