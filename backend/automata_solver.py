from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import graphviz
import base64

def convert_regex_to_dfa(regex: str) -> str:
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
    for
