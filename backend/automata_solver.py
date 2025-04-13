from typing import Dict, List, Set, Tuple, Union, Optional
import graphviz
from dataclasses import dataclass
import json
import asyncio
import xml.etree.ElementTree as ET
from pydantic import BaseModel, validator, Field, field_validator
import logging
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

# Functions that need to be importable at module level
def convert_regex_to_dfa(regex: str) -> str:
    """
    Convert a regular expression to a DFA and return its visualization
    
    Args:
        regex: The regular expression to convert
        
    Returns:
        A string containing the DOT representation of the DFA
        
    Raises:
        ValidationError: If the regex is invalid
    """
    try:
        # First validate the regex
        if not validate_regex(regex):
            raise ValidationError("Invalid regular expression")
            
        # Create NFA from regex
        nfa = create_nfa_from_regex(regex)
        
        # Convert NFA to DFA
        dfa = convert_nfa_to_dfa(nfa)
        
        # Minimize the DFA
        minimized_dfa = minimize_automaton(dfa)
        
        # Generate visualization
        return minimized_dfa.visualize()
        
    except Exception as e:
        logger.error(f"Failed to convert regex to DFA: {str(e)}")
        raise

# Export the public interface
__all__ = [
    'AutomataSolver',
    'DFA',
    'NFA', 
    'PDA',
    'BatchProcessor',
    'convert_regex_to_dfa'
]

class AutomataError(Exception):
    """Base exception for automata-related errors"""
    pass

class ValidationError(AutomataError):
    """Raised when automata validation fails"""
    pass

class StateValidationSchema(BaseModel):
    name: str
    is_initial: bool = False
    is_final: bool = False
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("State name cannot be empty")
        return v

class TransitionValidationSchema(BaseModel):
    from_state: str
    symbol: str
    to_state: str = Field(..., description="Destination state")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Transition symbol cannot be empty")
        return v

@dataclass
class State:
    name: str
    is_initial: bool = False
    is_final: bool = False

class AutomataBase(ABC):
    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, State] = {}
        self.alphabet: Set[str] = set()
        
    def validate_state(self, state_data: Dict) -> None:
        """Validate state data using pydantic schema"""
        try:
            StateValidationSchema(**state_data)
        except Exception as e:
            raise ValidationError(f"Invalid state data: {str(e)}")
    
    def add_state(self, name: str, is_initial: bool = False, is_final: bool = False) -> None:
        """
        Add a state to the automaton
        
        Args:
            name: State identifier
            is_initial: Whether this is the initial state
            is_final: Whether this is a final/accepting state
            
        Raises:
            ValidationError: If state data is invalid
        """
        try:
            self.validate_state({
                "name": name,
                "is_initial": is_initial,
                "is_final": is_final
            })
            self.states[name] = State(name, is_initial, is_final)
            logger.info(f"Added state {name} to {self.name}")
        except Exception as e:
            logger.error(f"Failed to add state {name}: {str(e)}")
            raise
    
    @abstractmethod
    def validate_transition(self, transition_data: Dict) -> None:
        """Validate transition data"""
        pass
    
    def visualize(self) -> str:
        """Generate a graphical representation of the automata using graphviz"""
        try:
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
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            raise AutomataError("Failed to generate visualization")

class DFA(AutomataBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.transitions: Dict[Tuple[str, str], str] = {}
    
    def validate_transition(self, transition_data: Dict) -> None:
        """Validate DFA transition data"""
        try:
            TransitionValidationSchema(**transition_data)
        except Exception as e:
            raise ValidationError(f"Invalid transition data: {str(e)}")
    
    def add_transition(self, from_state: str, symbol: str, to_state: str) -> None:
        """
        Add a transition to the DFA
        
        Args:
            from_state: Source state
            symbol: Transition symbol
            to_state: Destination state
            
        Raises:
            ValidationError: If transition data is invalid
            AutomataError: If states don't exist
        """
        try:
            self.validate_transition({
                "from_state": from_state,
                "symbol": symbol,
                "to_state": to_state
            })
            
            if from_state not in self.states or to_state not in self.states:
                raise AutomataError("Source or destination state does not exist")
                
            self.alphabet.add(symbol)
            self.transitions[(from_state, symbol)] = to_state
            logger.info(f"Added transition {from_state} --{symbol}--> {to_state}")
        except Exception as e:
            logger.error(f"Failed to add transition: {str(e)}")
            raise
    
    def simulate(self, input_string: str) -> Tuple[bool, List[str]]:
        """
        Simulate the DFA on an input string
        
        Args:
            input_string: The input string to simulate
            
        Returns:
            Tuple of (accepted, path)
            
        Raises:
            AutomataError: If simulation fails
        """
        try:
            if not self.states:
                raise AutomataError("No states in automaton")
            
            current = next((s for s in self.states.values() if s.is_initial), None)
            if not current:
                raise AutomataError("No initial state defined")
                
            path = [current.name]
            
            for symbol in input_string:
                if symbol not in self.alphabet:
                    raise AutomataError(f"Invalid symbol in input: {symbol}")
                    
                if (current.name, symbol) not in self.transitions:
                    return False, path
                    
                current = self.states[self.transitions[(current.name, symbol)]]
                path.append(current.name)
            
            return current.is_final, path
        except AutomataError:
            raise
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise AutomataError("Simulation failed")

class NFA(AutomataBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.transitions: Dict[Tuple[str, str], Set[str]] = {}
        self.epsilon_transitions: Dict[str, Set[str]] = {}
    
    def validate_transition(self, transition_data: Dict) -> None:
        """Validate NFA transition data"""
        try:
            # Implement validation logic for NFA transitions
            pass
        except Exception as e:
            logger.error(f"Failed to validate NFA transition: {str(e)}")
            raise ValidationError("Invalid NFA transition data")
    
    def add_transition(self, from_state: str, symbol: str, to_states: Set[str]) -> None:
        try:
            self.alphabet.add(symbol)
            self.transitions[(from_state, symbol)] = to_states
        except Exception as e:
            logger.error(f"Failed to add NFA transition: {str(e)}")
            raise AutomataError("Failed to add NFA transition")
    
    def add_epsilon_transition(self, from_state: str, to_states: Set[str]) -> None:
        try:
            if from_state not in self.epsilon_transitions:
                self.epsilon_transitions[from_state] = set()
            self.epsilon_transitions[from_state].update(to_states)
        except Exception as e:
            logger.error(f"Failed to add epsilon transition: {str(e)}")
            raise AutomataError("Failed to add epsilon transition")

class PDA(AutomataBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.stack_alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[str, str, str], Set[Tuple[str, Tuple[str, ...]]]] = {}
    
    def validate_transition(self, transition_data: Dict) -> None:
        """Validate PDA transition data"""
        try:
            if not all(key in transition_data for key in ['from', 'input', 'stack', 'to', 'push']):
                raise ValidationError("Missing required transition fields")
            if not isinstance(transition_data['push'], (list, tuple)):
                raise ValidationError("Push symbols must be a list or tuple")
        except Exception as e:
            logger.error(f"Failed to validate PDA transition: {str(e)}")
            raise ValidationError("Invalid PDA transition data")
    
    def add_transition(self, from_state: str, input_symbol: str, 
                      stack_symbol: str, to_state: str, push_symbols: List[str]) -> None:
        """
        Add a transition to the PDA
        
        Args:
            from_state: Source state
            input_symbol: Input symbol to consume
            stack_symbol: Stack symbol to pop
            to_state: Destination state
            push_symbols: List of symbols to push onto stack
            
        Raises:
            ValidationError: If transition data is invalid
            AutomataError: If states don't exist
        """
        try:
            self.validate_transition({
                'from': from_state,
                'input': input_symbol,
                'stack': stack_symbol,
                'to': to_state,
                'push': push_symbols
            })
            
            if from_state not in self.states or to_state not in self.states:
                raise AutomataError("Source or destination state does not exist")
                
            self.alphabet.add(input_symbol)
            self.stack_alphabet.add(stack_symbol)
            for symbol in push_symbols:
                self.stack_alphabet.add(symbol)
                
            key = (from_state, input_symbol, stack_symbol)
            if key not in self.transitions:
                self.transitions[key] = set()
            self.transitions[key].add((to_state, tuple(push_symbols)))
        except Exception as e:
            logger.error(f"Failed to add PDA transition: {str(e)}")
            raise AutomataError("Failed to add PDA transition")

class AutomataSolver:
    @staticmethod
    def create_automata(type_name: str, definition: Dict) -> Union[DFA, NFA, PDA]:
        """
        Factory method to create different types of automata
        """
        try:
            if type_name.upper() == "DFA":
                return AutomataSolver._create_dfa(definition)
            elif type_name.upper() == "NFA":
                return AutomataSolver._create_nfa(definition)
            elif type_name.upper() == "PDA":
                return AutomataSolver._create_pda(definition)
            else:
                raise ValueError(f"Unsupported automata type: {type_name}")
        except Exception as e:
            logger.error(f"Failed to create automata: {str(e)}")
            raise

    @staticmethod
    def verify_solution(automata: Union[DFA, NFA, PDA], test_cases: List[Dict]) -> Dict:
        """
        Verify a solution against test cases
        """
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        try:
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
        except Exception as e:
            logger.error(f"Solution verification failed: {str(e)}")
            raise AutomataError("Failed to verify solution")

    @staticmethod
    def _create_dfa(definition: Dict) -> DFA:
        """
        Create a DFA from a definition
        
        Args:
            definition: DFA definition
            
        Returns:
            Created DFA instance
            
        Raises:
            ValidationError: If definition is invalid
        """
        try:
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
        except Exception as e:
            logger.error(f"Failed to create DFA: {str(e)}")
            raise ValidationError(f"Invalid DFA definition: {str(e)}")

    @staticmethod
    def is_deterministic(automaton: Dict) -> bool:
        """Check if an automaton is deterministic"""
        transitions = {}
        for state in automaton['states']:
            for symbol in automaton['alphabet']:
                count = sum(1 for t in automaton['transitions'].items() 
                          if t[0] == state and t[1].get(symbol))
                if count != 1:
                    return False
        return True

    @staticmethod
    def is_minimal(automaton: Dict) -> bool:
        """Check if an automaton is minimal"""
        # Convert to DFA and minimize
        dfa = AutomataSolver._create_dfa(automaton)
        minimized = minimize_automaton(dfa)
        
        # Compare state count
        return len(minimized.states) >= len(automaton['states'])

    @staticmethod
    def is_complete(automaton: Dict) -> bool:
        """Check if an automaton is complete"""
        for state in automaton['states']:
            for symbol in automaton['alphabet']:
                if not any(t[0] == state and symbol in t[1] 
                         for t in automaton['transitions'].items()):
                    return False
        return True

def validate_regex(regex: str) -> bool:
    """
    Validate a regular expression
    
    Args:
        regex: The regular expression to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check for balanced parentheses
        parens = 0
        for char in regex:
            if char == '(':
                parens += 1
            elif char == ')':
                parens -= 1
            if parens < 0:
                return False
                
        # All parentheses should be matched
        if parens != 0:
            return False
            
        # Check for invalid operators
        valid_operators = {'*', '+', '|', '?'}
        prev_char = None
        for char in regex:
            if char in valid_operators and (prev_char is None or prev_char in valid_operators):
                return False
            prev_char = char
            
        return True
        
    except Exception as e:
        logger.error(f"Regex validation failed: {str(e)}")
        return False

class RegexNode:
    """Node in a regex syntax tree"""
    def __init__(self, type: str, value: str = None, left=None, right=None):
        self.type = type  # 'concat', 'union', 'star', 'char'
        self.value = value
        self.left = left
        self.right = right

# Core regex parsing functionality
def parse_regex(regex: str) -> 'RegexNode':
    """Parse regex into syntax tree using recursive descent"""
    class Parser:
        def __init__(self):
            self.pos = 0
            self.regex = regex
        
        def parse_char(self) -> 'RegexNode':
            if self.pos < len(self.regex) and self.regex[self.pos] not in "()|+?*":
                char = self.regex[self.pos]
                self.pos += 1
                return RegexNode("char", char)
            return None
        
        def parse_group(self) -> 'RegexNode':
            if self.pos < len(self.regex) and self.regex[self.pos] == "(":
                self.pos += 1
                node = self.parse_union()
                if self.pos < len(self.regex) and self.regex[self.pos] == ")":
                    self.pos += 1
                    return node
            return self.parse_char()
        
        def parse_star(self) -> 'RegexNode':
            node = self.parse_group()
            while node and self.pos < len(self.regex) and self.regex[self.pos] in "*+?":
                op = self.regex[self.pos]
                if op == "*":
                    node = RegexNode("star", left=node)
                elif op == "+":
                    node = RegexNode(
                        "concat",
                        left=node,
                        right=RegexNode("star", left=node)
                    )
                elif op == "?":
                    node = RegexNode(
                        "union",
                        left=node,
                        right=RegexNode("char", "")
                    )
                self.pos += 1
            return node
        
        def parse_concat(self) -> 'RegexNode':
            left = self.parse_star()
            if not left:
                return None
            right = self.parse_concat()
            if not right:
                return left
            return RegexNode("concat", left=left, right=right)
        
        def parse_union(self) -> 'RegexNode':
            left = self.parse_concat()
            if self.pos < len(self.regex) and self.regex[self.pos] == "|":
                self.pos += 1
                right = self.parse_union()
                return RegexNode("union", left=left, right=right)
            return left
            
        def parse(self) -> 'RegexNode':
            result = self.parse_union()
            if self.pos < len(self.regex):
                raise ValidationError(f"Unexpected character at position {self.pos}")
            return result
    
    parser = Parser()
    return parser.parse()

def create_nfa_from_regex(regex: str) -> NFA:
    """
    Create an NFA from a regular expression using Thompson's construction
    
    Args:
        regex: The regular expression to convert
        
    Returns:
        An NFA instance
        
    Raises:
        ValidationError: If the regex is invalid
    """
    try:
        if not validate_regex(regex):
            raise ValidationError("Invalid regular expression")
            
        # Parse regex into syntax tree
        ast = parse_regex(regex)
        
        # Create NFA using Thompson's construction
        nfa = NFA(f"NFA for {regex}")
        start_state = "q0"
        nfa.add_state(start_state, True, False)
        
        state_counter = 1
        
        def new_state() -> str:
            nonlocal state_counter
            state = f"q{state_counter}"
            state_counter += 1
            return state
        
        def thompson_construct(node: RegexNode, start: str, end: str):
            if node.type == "char":
                nfa.add_transition(start, node.value, {end})
                
            elif node.type == "concat":
                middle = new_state()
                nfa.add_state(middle)
                thompson_construct(node.left, start, middle)
                thompson_construct(node.right, middle, end)
                
            elif node.type == "union":
                nfa.add_epsilon_transition(start, {new_left := new_state()})
                nfa.add_epsilon_transition(start, {new_right := new_state()})
                nfa.add_state(new_left)
                nfa.add_state(new_right)
                
                thompson_construct(node.left, new_left, end)
                thompson_construct(node.right, new_right, end)
                
            elif node.type == "star":
                middle = new_state()
                nfa.add_state(middle)
                nfa.add_epsilon_transition(start, {middle})
                nfa.add_epsilon_transition(start, {end})
                thompson_construct(node.left, middle, middle)
                nfa.add_epsilon_transition(middle, {end})
        
        # Create final state
        final_state = new_state()
        nfa.add_state(final_state, False, True)
        
        # Build NFA recursively
        thompson_construct(ast, start_state, final_state)
        
        return nfa
        
    except Exception as e:
        logger.error(f"Failed to create NFA from regex: {str(e)}")
        raise ValidationError(f"Failed to create NFA: {str(e)}")

def convert_nfa_to_dfa(nfa: NFA) -> DFA:
    """
    Convert an NFA to a DFA using the subset construction algorithm
    
    Args:
        nfa: The NFA to convert
        
    Returns:
        A DFA instance
    """
    try:
        dfa = DFA(f"DFA for {nfa.name}")
        
        # Get epsilon closure of a set of states
        def epsilon_closure(states: Set[str]) -> Set[str]:
            closure = states.copy()
            stack = list(states)
            
            while stack:
                state = stack.pop()
                if state in nfa.epsilon_transitions:
                    for next_state in nfa.epsilon_transitions[state]:
                        if next_state not in closure:
                            closure.add(next_state)
                            stack.append(next_state)
            
            return closure
        
        # Get next states for a set of states and a symbol
        def move(states: Set[str], symbol: str) -> Set[str]:
            next_states = set()
            for state in states:
                if (state, symbol) in nfa.transitions:
                    next_states.update(nfa.transitions[(state, symbol)])
            return next_states
        
        # Initialize DFA with start state
        initial_states = epsilon_closure({next(s.name for s in nfa.states.values() if s.is_initial)})
        dfa_states = {frozenset(initial_states): f"q0"}
        dfa.add_state("q0", True, any(nfa.states[s].is_final for s in initial_states))
        
        # Process unmarked DFA states
        unprocessed = [initial_states]
        state_counter = 1
        
        while unprocessed:
            current_states = unprocessed.pop()
            current_name = dfa_states[frozenset(current_states)]
            
            # Process each input symbol
            for symbol in nfa.alphabet:
                # Get next state set
                next_states = epsilon_closure(move(current_states, symbol))
                if not next_states:
                    continue
                    
                # Add new DFA state if needed
                next_frozen = frozenset(next_states)
                if next_frozen not in dfa_states:
                    new_name = f"q{state_counter}"
                    state_counter += 1
                    dfa_states[next_frozen] = new_name
                    dfa.add_state(
                        new_name,
                        False,
                        any(nfa.states[s].is_final for s in next_states)
                    )
                    unprocessed.append(next_states)
                
                # Add transition
                dfa.add_transition(
                    current_name,
                    symbol,
                    dfa_states[next_frozen]
                )
        
        return dfa
    except Exception as e:
        logger.error(f"Failed to convert NFA to DFA: {str(e)}")
        raise AutomataError("Failed to convert NFA to DFA")

def minimize_automaton(automaton: Union[DFA, Dict]) -> DFA:
    """
    Minimize a DFA using Hopcroft's algorithm
    
    Args:
        automaton: The DFA to minimize or its definition as a dictionary
        
    Returns:
        A minimized DFA instance
    """
    try:
        if isinstance(automaton, dict):
            dfa = DFA(automaton.get('name', 'Unnamed DFA'))
            # Convert dictionary to DFA instance
            for state in automaton.get('states', []):
                dfa.add_state(
                    state['name'],
                    state.get('initial', False),
                    state.get('final', False)
                )
            for transition in automaton.get('transitions', []):
                dfa.add_transition(
                    transition['from'],
                    transition['symbol'],
                    transition['to']
                )
            automaton = dfa
        
        # Implementation of Hopcroft's algorithm
        def get_reachable_states(dfa: DFA) -> Set[str]:
            reachable = set()
            queue = [next(s.name for s in dfa.states.values() if s.is_initial)]
            
            while queue:
                state = queue.pop()
                if state not in reachable:
                    reachable.add(state)
                    for symbol in dfa.alphabet:
                        if (state, symbol) in dfa.transitions:
                            queue.append(dfa.transitions[(state, symbol)])
            
            return reachable
        
        # Remove unreachable states
        reachable = get_reachable_states(automaton)
        minimized = DFA(f"Minimized {automaton.name}")
        
        # Initialize partitions (accepting and non-accepting states)
        accepting = {s.name for s in automaton.states.values() if s.is_final and s.name in reachable}
        non_accepting = reachable - accepting
        partitions = {frozenset(accepting), frozenset(non_accepting)} - {frozenset()}
        
        # Refine partitions
        while True:
            new_partitions = set()
            for partition in partitions:
                # Try to split partition
                splits = {}
                for state in partition:
                    key = []
                    for symbol in sorted(automaton.alphabet):
                        dest = None
                        if (state, symbol) in automaton.transitions:
                            dest_state = automaton.transitions[(state, symbol)]
                            # Find which partition contains the destination state
                            for p in partitions:
                                if dest_state in p:
                                    dest = hash(p)
                                    break
                        key.append((symbol, dest))
                    key = tuple(key)
                    
                    if key not in splits:
                        splits[key] = set()
                    splits[key].add(state)
                
                # Add resulting splits to new partitions
                new_partitions.update(frozenset(states) for states in splits.values())
            
            if new_partitions == partitions:
                break
            partitions = new_partitions
        
        # Create minimized DFA
        state_map = {}
        counter = 0
        
        # Add states
        for partition in partitions:
            if not partition:
                continue
            new_name = f"q{counter}"
            counter += 1
            
            # Check if this partition contains initial/final states
            is_initial = any(automaton.states[s].is_initial for s in partition)
            is_final = any(automaton.states[s].is_final for s in partition)
            
            minimized.add_state(new_name, is_initial, is_final)
            for state in partition:
                state_map[state] = new_name
        
        # Add transitions
        for (state, symbol), dest in automaton.transitions.items():
            if state in state_map and dest in state_map:
                minimized.add_transition(
                    state_map[state],
                    symbol,
                    state_map[dest]
                )
        
        return minimized
    except Exception as e:
        logger.error(f"Failed to minimize automaton: {str(e)}")
        raise AutomataError("Failed to minimize automaton")

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
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to process batch submissions: {str(e)}")
            raise AutomataError("Failed to process batch submissions")
        
        return results

def is_deterministic(automaton):
    """Check if an automaton is deterministic"""
    # Check if each state has exactly one transition for each symbol
    for state in automaton['states']:
        if state not in automaton['transitions']:
            return False
        for symbol in automaton['alphabet']:
            transitions = automaton['transitions'][state].get(symbol, [])
            if not isinstance(transitions, str) or len(transitions) != 1:
                return False
    return True

def is_minimal(automaton):
    """Check if a DFA is minimal"""
    # First ensure it's deterministic
    if not is_deterministic(automaton):
        return False
        
    # Get all reachable states from start state
    reachable = get_reachable_states(automaton)
    
    # Check if all states are reachable
    if len(reachable) != len(automaton['states']):
        return False
        
    # Check for equivalent states
    for s1 in automaton['states']:
        for s2 in automaton['states']:
            if s1 < s2:  # Check each pair only once
                if are_states_equivalent(automaton, s1, s2):
                    return False
    return True

def get_reachable_states(automaton):
    """Get all states reachable from the start state"""
    reachable = {automaton['start_state']}
    queue = [automaton['start_state']]
    
    while queue:
        state = queue.pop(0)
        for symbol in automaton['alphabet']:
            next_state = automaton['transitions'][state].get(symbol)
            if next_state and next_state not in reachable:
                reachable.add(next_state)
                queue.append(next_state)
    return reachable

def are_states_equivalent(automaton, s1, s2):
    """Check if two states are equivalent"""
    # States are equivalent if they can't be distinguished by any input string
    # Start with basic distinguishability (accepting vs non-accepting)
    if (s1 in automaton['accept_states']) != (s2 in automaton['accept_states']):
        return False
        
    # Check transitions for all input symbols
    for symbol in automaton['alphabet']:
        next1 = automaton['transitions'][s1].get(symbol)
        next2 = automaton['transitions'][s2].get(symbol)
        if next1 != next2:
            return False
    return True
