from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import graphviz
import base64
import re

def validate_regex(regex: str) -> bool:
    try:
        re.compile(regex)
        return True
    except re.error:
        return False

def convert_regex_to_dfa(regex: str) -> str:
    if not validate_regex(regex):
        return "Invalid regex"

    # Dummy logic, replace with actual regex-to-DFA conversion
    dfa = DFA(
        states={'q0', 'q1'},
        input_symbols={'0', '1'},
        transitions={'q0': {'0': 'q0', '1': 'q1'}, 'q1': {'0': 'q0', '1': 'q1'}},
        initial_state='q0',
        final_states={'q1'}
    )
    
    dot = graphviz.Digraph(format="svg")
    for state in dfa.states:
        dot.node(state, shape="doublecircle" if state in dfa.final_states else "circle")
    for src, transitions in dfa.transitions.items():
        for symbol, dest in transitions.items():
            dot.edge(src, dest, label=symbol)
    
    # Add zoom and pan functionality
    dot.attr('graph', {'rankdir': 'LR', 'splines': 'polyline', 'nodesep': '0.5', 'ranksep': '0.5'})
    dot.attr('node', {'shape': 'circle', 'style': 'filled', 'fillcolor': 'lightgrey', 'fontname': 'Helvetica'})
    dot.attr('edge', {'fontname': 'Helvetica'})
    
    svg_data = dot.pipe().decode('utf-8')
    return svg_data

def minimize_automaton(automaton: dict) -> dict:
    try:
        # Convert the input data to DFA object
        dfa = DFA(
            states=set(automaton['states']),
            input_symbols=set(automaton['input_symbols']),
            transitions=automaton['transitions'],
            initial_state=automaton['initial_state'],
            final_states=set(automaton['final_states'])
        )
        
        # Minimize the DFA
        minimized_dfa = dfa.minify()
        
        # Convert back to dictionary format
        return {
            'states': list(minimized_dfa.states),
            'input_symbols': list(minimized_dfa.input_symbols),
            'transitions': minimized_dfa.transitions,
            'initial_state': minimized_dfa.initial_state,
            'final_states': list(minimized_dfa.final_states)
        }
    except Exception as e:
        # Return the original automaton with an error message
        automaton['error'] = str(e)
        return automaton

def export_automaton(automaton: dict, format: str = 'svg') -> str:
    try:
        # Create a Graphviz graph
        dot = graphviz.Digraph(format=format)
        
        # Add graph styling
        dot.attr('graph', {'rankdir': 'LR', 'splines': 'polyline'})
        dot.attr('node', {'fontname': 'Helvetica'})
        dot.attr('edge', {'fontname': 'Helvetica'})
        
        # Add states/nodes
        for state in automaton['states']:
            is_final = state in automaton['final_states']
            is_initial = state == automaton['initial_state']
            
            # Style based on state type
            shape = "doublecircle" if is_final else "circle"
            color = "blue" if is_initial else "black"
            
            dot.node(state, shape=shape, color=color)
        
        # Add explicit initial state indicator
        dot.node('', shape='none')
        dot.edge('', automaton['initial_state'], label='')
        
        # Add transitions/edges
        for src, trans in automaton['transitions'].items():
            for symbol, dest in trans.items():
                # Handle NFA case where destination could be a list
                if isinstance(dest, list):
                    for d in dest:
                        dot.edge(src, d, label=symbol)
                else:
                    dot.edge(src, dest, label=symbol)
        
        # Return the result in requested format
        if format == 'svg':
            return dot.pipe().decode('utf-8')
        else:
            # For PDF/PNG, return base64 encoded data
            return base64.b64encode(dot.pipe()).decode('utf-8')
            
    except Exception as e:
        # If there's an error, return a simple error message
        error_dot = graphviz.Digraph(format=format)
        error_dot.node('error', label=f"Error: {str(e)}", shape='box', color='red')
        
        if format == 'svg':
            return error_dot.pipe().decode('utf-8')
        else:
            return base64.b64encode(error_dot.pipe()).decode('utf-8')
