from typing import Dict, List, Set, Tuple, Union
import graphviz
from dataclasses import dataclass
import json
import asyncio
import xml.etree.ElementTree as ET

@dataclass
class State:
    name: str
    is_initial: bool = False
    is_final: bool = False

class AutomataBase:
    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, State] = {}
        self.alphabet: Set[str] = set()
        
    def add_state(self, name: str, is_initial: bool = False, is_final: bool = False) -> None:
        self.states[name] = State(name, is_initial, is_final)
    
    def visualize(self) -> str:
        """Generate a graphical representation of the automata using graphviz"""
        dot = graphviz.Digraph(comment=f'{self.name} Visualization')
        dot.attr(rankdir='LR')
        
        # Add states
        for state in self.states.values():
            shape = 'doublecircle' if state.is_final else 'circle'
            dot.node(state.name, shape=shape)
            
            # Add initial state marker
            if state.is_initial:
                dot.node('', shape='none')
                dot.edge('', state.name)
        
        return dot.source

class DFA(AutomataBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.transitions: Dict[Tuple[str, str], str] = {}
    
    def add_transition(self, from_state: str, symbol: str, to_state: str) -> None:
        self.alphabet.add(symbol)
        self.transitions[(from_state, symbol)] = to_state
    
    def simulate(self, input_string: str) -> Tuple[bool, List[str]]:
        """
        Simulate the DFA on an input string
        Returns: (accepted, path)
        """
        if not self.states:
            return False, []
        
        current = next((s for s in self.states.values() if s.is_initial), None)
        if not current:
            return False, []
            
        path = [current.name]
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                return False, path
                
            if (current.name, symbol) not in self.transitions:
                return False, path
                
            current = self.states[self.transitions[(current.name, symbol)]]
            path.append(current.name)
        
        return current.is_final, path

class NFA(AutomataBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.transitions: Dict[Tuple[str, str], Set[str]] = {}
        self.epsilon_transitions: Dict[str, Set[str]] = {}
    
    def add_transition(self, from_state: str, symbol: str, to_states: Set[str]) -> None:
        self.alphabet.add(symbol)
        self.transitions[(from_state, symbol)] = to_states
    
    def add_epsilon_transition(self, from_state: str, to_states: Set[str]) -> None:
        if from_state not in self.epsilon_transitions:
            self.epsilon_transitions[from_state] = set()
        self.epsilon_transitions[from_state].update(to_states)

class PDA(AutomataBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.stack_alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[str, str, str], Set[Tuple[str, List[str]]]] = {}
    
    def add_transition(self, from_state: str, input_symbol: str, 
                      stack_symbol: str, to_state: str, push_symbols: List[str]) -> None:
        self.alphabet.add(input_symbol)
        self.stack_alphabet.add(stack_symbol)
        for symbol in push_symbols:
            self.stack_alphabet.add(symbol)
            
        key = (from_state, input_symbol, stack_symbol)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].add((to_state, push_symbols))

class AutomataSolver:
    @staticmethod
    def create_automata(type_name: str, definition: Dict) -> Union[DFA, NFA, PDA]:
        """Factory method to create different types of automata"""
        if type_name.upper() == "DFA":
            return AutomataSolver._create_dfa(definition)
        elif type_name.upper() == "NFA":
            return AutomataSolver._create_nfa(definition)
        elif type_name.upper() == "PDA":
            return AutomataSolver._create_pda(definition)
        else:
            raise ValueError(f"Unsupported automata type: {type_name}")
    
    @staticmethod
    def _create_dfa(definition: Dict) -> DFA:
        dfa = DFA(definition.get('name', 'Unnamed DFA'))
        
        # Add states
        for state_def in definition.get('states', []):
            dfa.add_state(
                state_def['name'],
                state_def.get('initial', False),
                state_def.get('final', False)
            )
        
        # Add transitions
        for transition in definition.get('transitions', []):
            dfa.add_transition(
                transition['from'],
                transition['symbol'],
                transition['to']
            )
            
        return dfa

    @staticmethod
    def verify_solution(automata: Union[DFA, NFA, PDA], test_cases: List[Dict]) -> Dict:
        """Verify a solution against test cases"""
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for test in test_cases:
            input_string = test['input']
            expected = test['expected']
            
            if isinstance(automata, DFA):
                accepted, path = automata.simulate(input_string)
                passed = accepted == expected
                
                results['details'].append({
                    'input': input_string,
                    'expected': expected,
                    'actual': accepted,
                    'path': path,
                    'passed': passed
                })
                
                if passed:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
        
        return results

class BatchProcessor:
    def __init__(self):
        self.solver = AutomataSolver()

    async def process_batch_submissions(self, submissions: List[Dict]) -> List[Dict]:
        """
        Process multiple automata submissions in parallel
        
        Args:
            submissions: List of dictionaries containing:
                - automata_type: str (DFA, NFA, PDA)
                - definition: Dict (automata definition)
                - test_cases: List[Dict] (test cases to verify)
                
        Returns:
            List of results for each submission
        """
        results = []
        
        # Process submissions in parallel using asyncio
        async def process_single(submission):
            try:
                automata = self.solver.create_automata(
                    submission['automata_type'],
                    submission['definition']
                )
                verification = self.solver.verify_solution(
                    automata,
                    submission['test_cases']
                )
                return {
                    'success': True,
                    'verification': verification,
                    'visualization': automata.visualize()
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }

        # Create tasks for all submissions
        tasks = [process_single(sub) for sub in submissions]
        results = await asyncio.gather(*tasks)
        
        return results
