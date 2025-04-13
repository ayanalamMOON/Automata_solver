import pytest
from automata_solver import AutomataSolver, DFA, NFA, PDA, AutomataError, ValidationError
from typing import Dict, Set

def test_dfa_creation():
    """Test DFA creation with valid definition"""
    definition = {
        'name': 'Test DFA',
        'states': [
            {'name': 'q0', 'initial': True, 'final': False},
            {'name': 'q1', 'initial': False, 'final': True}
        ],
        'transitions': [
            {'from': 'q0', 'symbol': 'a', 'to': 'q1'}
        ]
    }
    
    dfa = AutomataSolver._create_dfa(definition)
    assert isinstance(dfa, DFA)
    assert dfa.name == 'Test DFA'
    assert len(dfa.states) == 2
    assert ('q0', 'a') in dfa.transitions
    assert dfa.transitions[('q0', 'a')] == 'q1'

def test_dfa_validation():
    """Test DFA validation with invalid data"""
    with pytest.raises(ValidationError):
        dfa = DFA('Test')
        dfa.add_state('')  # Empty state name

def test_dfa_simulation():
    """Test DFA simulation with various inputs"""
    dfa = DFA('Binary Divisible by 2')
    
    # Create DFA that accepts binary numbers divisible by 2
    dfa.add_state('q0', True, False)
    dfa.add_state('q1', False, True)
    
    dfa.add_transition('q0', '0', 'q1')
    dfa.add_transition('q0', '1', 'q0')
    dfa.add_transition('q1', '0', 'q1')
    dfa.add_transition('q1', '1', 'q0')
    
    # Test various inputs
    accepted, path = dfa.simulate('10')
    assert accepted
    assert path == ['q0', 'q0', 'q1']
    
    accepted, path = dfa.simulate('101')
    assert not accepted
    assert path == ['q0', 'q0', 'q1', 'q0']

def test_nfa_creation():
    """Test NFA creation and epsilon transitions"""
    nfa = NFA('Test NFA')
    nfa.add_state('q0', True, False)
    nfa.add_state('q1', False, True)
    
    nfa.add_transition('q0', 'a', {'q0', 'q1'})
    nfa.add_epsilon_transition('q0', {'q1'})
    
    assert ('q0', 'a') in nfa.transitions
    assert nfa.transitions[('q0', 'a')] == {'q0', 'q1'}
    assert 'q0' in nfa.epsilon_transitions
    assert nfa.epsilon_transitions['q0'] == {'q1'}

def test_pda_creation():
    """Test PDA creation and stack operations"""
    pda = PDA('Test PDA')
    pda.add_state('q0', True, False)
    pda.add_state('q1', False, True)
    
    # Add transition that pushes 'X' onto stack
    pda.add_transition('q0', 'a', 'Z', 'q1', ['X', 'Z'])
    
    key = ('q0', 'a', 'Z')
    assert key in pda.transitions
    assert ('q1', ['X', 'Z']) in pda.transitions[key]

def test_batch_processing():
    """Test batch processing of automata submissions"""
    submissions = [
        {
            'automata_type': 'DFA',
            'definition': {
                'name': 'Test 1',
                'states': [
                    {'name': 'q0', 'initial': True, 'final': True}
                ],
                'transitions': []
            },
            'test_cases': [
                {'input': '', 'expected': True}
            ]
        }
    ]
    
    solver = AutomataSolver()
    results = solver.verify_solution(
        solver.create_automata('DFA', submissions[0]['definition']),
        submissions[0]['test_cases']
    )
    
    assert results['passed'] == 1
    assert results['failed'] == 0

def test_error_handling():
    """Test error handling for invalid operations"""
    dfa = DFA('Error Test')
    
    # Test adding invalid transition
    with pytest.raises(AutomataError):
        dfa.add_transition('q0', 'a', 'q1')  # States don't exist
    
    # Test simulating with invalid input
    dfa.add_state('q0', True, True)
    with pytest.raises(AutomataError):
        dfa.simulate('abc')  # Symbol not in alphabet

def test_visualization():
    """Test automata visualization"""
    dfa = DFA('Visualization Test')
    dfa.add_state('q0', True, False)
    dfa.add_state('q1', False, True)
    dfa.add_transition('q0', 'a', 'q1')
    
    dot_source = dfa.visualize()
    assert 'digraph' in dot_source
    assert 'q0' in dot_source
    assert 'q1' in dot_source