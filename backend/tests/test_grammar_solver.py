import pytest
from grammar_solver import Grammar, Production, ParseTree, GrammarError, ValidationError, convert_to_gnf

def test_grammar_creation():
    """Test basic grammar creation and validation"""
    grammar = Grammar("Test Grammar")
    grammar.add_production("S", ["A", "B"])
    grammar.add_production("A", ["a"])
    grammar.add_production("B", ["b"])
    grammar.set_start_symbol("S")
    
    assert "S" in grammar.non_terminals
    assert "a" in grammar.terminals
    assert len(grammar.productions) == 3
    assert grammar.start_symbol == "S"

def test_invalid_grammar():
    """Test grammar validation errors"""
    grammar = Grammar("Invalid Grammar")
    
    # Test invalid start symbol
    with pytest.raises(ValidationError):
        grammar.set_start_symbol("x")  # Non-existent symbol
        
    # Test production with undefined symbols
    grammar.add_production("S", ["A"])  # Valid
    with pytest.raises(GrammarError):
        grammar.add_production("", ["a"])  # Empty left-hand side

def test_cnf_checking():
    """Test Chomsky Normal Form validation"""
    grammar = Grammar("CNF Test")
    
    # Valid CNF productions
    grammar.add_production("S", ["A", "B"])
    grammar.add_production("A", ["a"])
    grammar.add_production("B", ["b"])
    grammar.set_start_symbol("S")
    
    assert grammar.is_cnf() == True
    
    # Add non-CNF production
    grammar.add_production("C", ["A", "B", "C"])  # Too many symbols
    assert grammar.is_cnf() == False

def test_cnf_conversion():
    """Test conversion to Chomsky Normal Form"""
    grammar = Grammar("Pre-CNF")
    grammar.add_production("S", ["A", "B", "C"])  # Will need conversion
    grammar.add_production("A", ["a"])
    grammar.add_production("B", ["b"])
    grammar.add_production("C", ["c"])
    grammar.set_start_symbol("S")
    
    cnf = grammar.to_cnf()
    assert cnf.is_cnf() == True
    # Verify that the language is preserved (basic check)
    assert cnf.start_symbol is not None
    assert len(cnf.productions) >= len(grammar.productions)

def test_parse_tree():
    """Test parse tree creation and visualization"""
    tree = ParseTree("S")
    tree.add_child(ParseTree("A"))
    tree.add_child(ParseTree("B"))
    
    viz = tree.visualize()
    assert "digraph" in viz
    assert "S" in viz
    assert "A" in viz
    assert "B" in viz

def test_ll1_checking():
    """Test LL(1) grammar checking"""
    grammar = Grammar("LL(1) Test")
    grammar.add_production("E", ["T", "E'"])
    grammar.add_production("E'", ["+", "T", "E'"])
    grammar.add_production("E'", [])  # epsilon
    grammar.add_production("T", ["F", "T'"])
    grammar.add_production("T'", ["*", "F", "T'"])
    grammar.add_production("T'", [])  # epsilon
    grammar.add_production("F", ["(", "E", ")"])
    grammar.add_production("F", ["id"])
    grammar.set_start_symbol("E")
    
    # TODO: Once implemented, uncomment and update test
    # assert grammar.is_ll1() == True

def test_derivation_visualization():
    """Test derivation visualization"""
    grammar = Grammar("Derivation Test")
    grammar.add_production("S", ["A", "B"])
    grammar.add_production("A", ["a"])
    grammar.add_production("B", ["b"])
    grammar.set_start_symbol("S")
    
    # Create a sample derivation
    derivation = [
        (grammar.productions[0], 0),  # S → AB
        (grammar.productions[1], 0),  # A → a
        (grammar.productions[2], 1)   # B → b
    ]
    
    viz = grammar.visualize_derivation(derivation)
    assert "digraph" in viz
    assert "Derivation" in viz

def test_gnf_conversion():
    """Test conversion to Greibach Normal Form"""
    grammar = Grammar("Pre-GNF")
    grammar.add_production("S", ["A", "B"])
    grammar.add_production("A", ["a"])
    grammar.add_production("B", ["b"])
    grammar.set_start_symbol("S")
    
    # TODO: Once implemented, uncomment and update test
    # gnf = convert_to_gnf(grammar)
    # Verify GNF properties
    # 1. All productions should start with a terminal
    # 2. Followed by zero or more non-terminals