import pytest
from automata_solver import AutomataSolver, DFA, NFA, PDA, AutomataError, ValidationError, convert_regex_to_dfa, validate_regex, create_nfa_from_regex, convert_nfa_to_dfa
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
    push_symbols = ['X', 'Z']
    pda.add_transition('q0', 'a', 'Z', 'q1', push_symbols)
    
    key = ('q0', 'a', 'Z')
    assert key in pda.transitions
    assert ('q1', tuple(push_symbols)) in pda.transitions[key]  # Check for tuple

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

def test_convert_regex_to_dfa():
    """Test the module-level convert_regex_to_dfa function"""
    # Test basic regex conversion
    result = convert_regex_to_dfa('a(b|c)*')
    assert isinstance(result, str)
    assert 'digraph' in result
    
    # Test invalid regex
    with pytest.raises(ValidationError):
        convert_regex_to_dfa('a**b')  # Invalid: consecutive stars
        
    # Test empty regex
    with pytest.raises(ValidationError):
        convert_regex_to_dfa('')

def test_regex_validation_simple():
    """Test regex validation with simple patterns"""
    assert validate_regex('a') is True
    assert validate_regex('ab') is True
    assert validate_regex('a|b') is True
    assert validate_regex('a*') is True
    assert validate_regex('(a|b)*') is True

def test_regex_validation_advanced():
    """Test regex validation with more complex patterns"""
    assert validate_regex('(a|b)*abb') is True
    assert validate_regex('((a|b)*|c)d') is True
    assert validate_regex('a+b?c*') is True

def test_regex_validation_invalid():
    """Test regex validation with invalid patterns"""
    assert validate_regex('') is False
    assert validate_regex('*') is False
    assert validate_regex('a**b') is False
    assert validate_regex('(ab') is False
    assert validate_regex('ab)') is False
    assert validate_regex('a||b') is False

def test_create_nfa_from_regex():
    """Test NFA creation from regex"""
    # Test simple regex
    nfa = create_nfa_from_regex('ab')
    assert len(nfa.states) > 0
    assert 'a' in nfa.alphabet
    assert 'b' in nfa.alphabet

    # Test alternation
    nfa = create_nfa_from_regex('a|b')
    assert len(nfa.states) > 0
    assert 'a' in nfa.alphabet
    assert 'b' in nfa.alphabet

    # Test Kleene star
    nfa = create_nfa_from_regex('a*')
    assert len(nfa.states) > 0
    assert 'a' in nfa.alphabet

    # Test combination
    nfa = create_nfa_from_regex('(a|b)*abb')
    assert len(nfa.states) > 0
    assert 'a' in nfa.alphabet
    assert 'b' in nfa.alphabet

def test_nfa_to_dfa_conversion():
    """Test NFA to DFA conversion"""
    # Create NFA first
    nfa = create_nfa_from_regex('(a|b)*abb')
    
    # Convert to DFA
    dfa = convert_nfa_to_dfa(nfa)
    
    # Test DFA properties
    assert isinstance(dfa, DFA)
    assert len(dfa.states) > 0
    assert len(dfa.transitions) > 0
    assert len([s for s in dfa.states.values() if s.is_initial]) == 1
    assert len([s for s in dfa.states.values() if s.is_final]) > 0

    # Test DFA acceptance
    assert dfa.simulate('abb')[0] is True
    assert dfa.simulate('aabb')[0] is True
    assert dfa.simulate('ab')[0] is False

def test_regex_to_dfa_full():
    """Test complete regex to DFA conversion pipeline"""
    # Test conversion and simulation
    dfa = convert_regex_to_dfa('(a|b)*abb')
    assert isinstance(dfa, DFA)

    # Test accepted strings
    assert dfa.simulate('abb')[0] is True
    assert dfa.simulate('aabb')[0] is True
    assert dfa.simulate('babb')[0] is True
    assert dfa.simulate('abababb')[0] is True

    # Test rejected strings
    assert dfa.simulate('ab')[0] is False
    assert dfa.simulate('ba')[0] is False
    assert dfa.simulate('abba')[0] is False