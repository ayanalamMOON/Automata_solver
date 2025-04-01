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
    # Dummy logic, replace with actual minimization logic
    return automaton

def export_automaton(automaton: dict, format: str) -> str:
    # Dummy logic, replace with actual export logic
    return "exported_data"
