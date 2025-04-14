from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import graphviz
import logging
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

class GrammarError(Exception):
    """Base exception for grammar-related errors"""
    pass

class ValidationError(GrammarError):
    """Raised when grammar validation fails"""
    pass

@dataclass
class Production:
    """Represents a production rule in a grammar"""
    left: str  # Left-hand side (non-terminal)
    right: List[str]  # Right-hand side (sequence of terminals/non-terminals)
    
    def __str__(self) -> str:
        return f"{self.left} → {' '.join(self.right)}"

@dataclass
class LR0Item:
    """Represents an LR(0) item in the parsing process"""
    prod: Production
    dot_position: int  # Position of the dot in the production
    
    def __str__(self) -> str:
        right = list(self.prod.right)
        right.insert(self.dot_position, "•")
        return f"{self.prod.left} → {' '.join(right)}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LR0Item):
            return NotImplemented
        return (self.prod == other.prod and 
                self.dot_position == other.dot_position)
    
    def __hash__(self) -> int:
        return hash((self.prod.left, tuple(self.prod.right), self.dot_position))

class LR0State:
    """Represents a state in the LR(0) automaton"""
    def __init__(self, items: Set[LR0Item]):
        self.items = items
        self.transitions: Dict[str, 'LR0State'] = {}
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LR0State):
            return NotImplemented
        return self.items == other.items
    
    def __hash__(self) -> int:
        return hash(frozenset(self.items))

class Grammar:
    """Base class for Context-Free Grammars"""
    def __init__(self, name: str):
        self.name = name
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = set()
        self.productions: List[Production] = []
        self.start_symbol: Optional[str] = None
        
    def add_production(self, left: str, right: List[str]) -> None:
        """Add a production rule to the grammar"""
        try:
            prod = Production(left, right)
            self.productions.append(prod)
            self.non_terminals.add(left)
            for symbol in right:
                if symbol.isupper():  # Convention: uppercase for non-terminals
                    self.non_terminals.add(symbol)
                else:
                    self.terminals.add(symbol)
        except Exception as e:
            logger.error(f"Failed to add production: {str(e)}")
            raise GrammarError(f"Failed to add production: {str(e)}")
            
    def set_start_symbol(self, symbol: str) -> None:
        """Set the start symbol of the grammar"""
        if symbol not in self.non_terminals:
            raise ValidationError(f"Start symbol {symbol} must be a non-terminal")
        self.start_symbol = symbol
        
    def is_cnf(self) -> bool:
        """Check if grammar is in Chomsky Normal Form"""
        if not self.start_symbol:
            return False
            
        for prod in self.productions:
            # CNF rules must be of form A → BC or A → a
            if len(prod.right) == 1:
                if prod.right[0] in self.non_terminals:  # A → B not allowed
                    return False
            elif len(prod.right) == 2:
                if not all(sym in self.non_terminals for sym in prod.right):
                    return False
            else:
                return False
        return True
    
    def _eliminate_epsilon_productions(self) -> None:
        """Eliminate ε-productions from the grammar"""
        # Find all nullable non-terminals
        nullable = set()
        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                if not prod.right and prod.left not in nullable:
                    nullable.add(prod.left)
                    changed = True
                elif all(sym in nullable for sym in prod.right) and prod.left not in nullable:
                    nullable.add(prod.left)
                    changed = True
        
        # Generate new productions
        new_productions = []
        for prod in self.productions:
            if not prod.right:  # Skip original ε-productions
                continue
                
            # Generate all possible combinations of including/excluding nullable symbols
            def generate_combinations(symbols: List[str], current: List[str], pos: int) -> None:
                if pos == len(symbols):
                    if current:  # Don't add empty productions
                        new_productions.append(Production(prod.left, current))
                    return
                    
                if symbols[pos] in nullable:
                    # Skip this symbol
                    generate_combinations(symbols, current, pos + 1)
                # Include this symbol
                generate_combinations(symbols, current + [symbols[pos]], pos + 1)
                
            generate_combinations(prod.right, [], 0)
            
        self.productions = new_productions
    
    def _eliminate_unit_productions(self) -> None:
        """Eliminate unit productions (A → B) from the grammar"""
        # Find all unit pairs
        unit_pairs = {(nt, nt) for nt in self.non_terminals}
        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                if len(prod.right) == 1 and prod.right[0] in self.non_terminals:
                    for pair in list(unit_pairs):
                        if pair[1] == prod.left and (pair[0], prod.right[0]) not in unit_pairs:
                            unit_pairs.add((pair[0], prod.right[0]))
                            changed = True
        
        # Generate new productions
        new_productions = []
        for prod in self.productions:
            if len(prod.right) == 1 and prod.right[0] in self.non_terminals:
                continue  # Skip unit productions
            for A, B in unit_pairs:
                if B == prod.left:
                    new_productions.append(Production(A, prod.right))
                    
        self.productions = new_productions
    
    def _convert_long_productions(self) -> None:
        """Convert productions with more than 2 symbols on RHS"""
        new_productions = []
        new_non_terminals = set()
        
        for prod in self.productions:
            if len(prod.right) <= 2:
                new_productions.append(prod)
                continue
                
            # Create new non-terminals and break down the production
            current = prod.left
            symbols = prod.right
            while len(symbols) > 2:
                new_nt = f"{current}_{len(new_non_terminals)}"
                while new_nt in self.non_terminals or new_nt in new_non_terminals:
                    new_nt += "'"
                new_non_terminals.add(new_nt)
                new_productions.append(Production(current, [symbols[0], new_nt]))
                current = new_nt
                symbols = symbols[1:]
            new_productions.append(Production(current, symbols))
            
        self.productions = new_productions
        self.non_terminals.update(new_non_terminals)
    
    def _convert_terminal_productions(self) -> None:
        """Convert productions with mixed terminals/non-terminals in RHS"""
        new_productions = []
        terminal_map = {}  # Maps terminals to their non-terminal replacements
        
        for prod in self.productions:
            if len(prod.right) == 1 and prod.right[0] in self.terminals:
                new_productions.append(prod)  # Keep A → a productions
                continue
                
            new_right = []
            for symbol in prod.right:
                if symbol in self.terminals:
                    if symbol not in terminal_map:
                        new_nt = f"T_{len(terminal_map)}"
                        while new_nt in self.non_terminals:
                            new_nt += "'"
                        terminal_map[symbol] = new_nt
                        self.non_terminals.add(new_nt)
                        new_productions.append(Production(new_nt, [symbol]))
                    new_right.append(terminal_map[symbol])
                else:
                    new_right.append(symbol)
            new_productions.append(Production(prod.left, new_right))
            
        self.productions = new_productions
    
    def to_cnf(self) -> 'Grammar':
        """Convert grammar to Chomsky Normal Form"""
        cnf = Grammar(f"CNF of {self.name}")
        
        # Copy the original grammar
        cnf.terminals = self.terminals.copy()
        cnf.non_terminals = self.non_terminals.copy()
        cnf.productions = [Production(p.left, p.right) for p in self.productions]
        
        # Step 1: Create new start symbol
        new_start = f"{self.start_symbol}'"
        while new_start in cnf.non_terminals:
            new_start += "'"
        cnf.non_terminals.add(new_start)
        cnf.productions.insert(0, Production(new_start, [self.start_symbol]))
        cnf.start_symbol = new_start
        
        # Step 2: Eliminate ε-productions
        cnf._eliminate_epsilon_productions()
        
        # Step 3: Eliminate unit productions
        cnf._eliminate_unit_productions()
        
        # Step 4: Convert long productions (more than 2 symbols on RHS)
        cnf._convert_long_productions()
        
        # Step 5: Convert productions with mixed terminals/non-terminals
        cnf._convert_terminal_productions()
        
        return cnf
        
    def _calculate_first_sets(self) -> Dict[str, Set[str]]:
        """Calculate FIRST sets for all symbols"""
        first: Dict[str, Set[str]] = {
            terminal: {terminal} for terminal in self.terminals
        }
        first.update({
            non_terminal: set() for non_terminal in self.non_terminals
        })
        
        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                current_first = first[prod.left].copy()
                
                # Calculate first set for right-hand side
                first_of_rhs = set()
                all_nullable = True
                
                for symbol in prod.right:
                    if symbol not in first:
                        first[symbol] = {symbol}  # Handle terminal symbols
                    symbol_first = first[symbol]
                    first_of_rhs.update(symbol_first - {""})  # Add non-epsilon symbols
                    
                    if "" not in symbol_first:
                        all_nullable = False
                        break
                
                if all_nullable and prod.right:  # If all symbols are nullable
                    first_of_rhs.add("")
                elif not prod.right:  # Empty production
                    first_of_rhs.add("")
                
                # Update first set of left-hand side
                if not first_of_rhs.issubset(first[prod.left]):
                    first[prod.left].update(first_of_rhs)
                    changed = True
                    
        return first
    
    def _calculate_follow_sets(self, first_sets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Calculate FOLLOW sets for all non-terminals"""
        follow = {non_terminal: set() for non_terminal in self.non_terminals}
        follow[self.start_symbol].add("$")  # End marker in follow set of start symbol
        
        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                for i, symbol in enumerate(prod.right):
                    if symbol in self.non_terminals:
                        # Calculate first of remaining string
                        first_of_rest = set()
                        all_nullable = True
                        
                        for next_sym in prod.right[i+1:]:
                            if next_sym not in first_sets:
                                first_sets[next_sym] = {next_sym}
                            sym_first = first_sets[next_sym]
                            first_of_rest.update(sym_first - {""})
                            
                            if "" not in sym_first:
                                all_nullable = False
                                break
                                
                        # Add first of rest to follow set
                        if not first_of_rest.issubset(follow[symbol]):
                            follow[symbol].update(first_of_rest)
                            changed = True
                            
                        # If nullable, add follow of left-hand side
                        if all_nullable or not prod.right[i+1:]:
                            if not follow[prod.left].issubset(follow[symbol]):
                                follow[symbol].update(follow[prod.left])
                                changed = True
                                
        return follow
    
    def is_ll1(self) -> bool:
        """Check if grammar is LL(1)"""
        # Calculate FIRST and FOLLOW sets
        first_sets = self._calculate_first_sets()
        follow_sets = self._calculate_follow_sets(first_sets)
        
        # Group productions by left-hand side
        productions_by_lhs = {}
        for prod in self.productions:
            if prod.left not in productions_by_lhs:
                productions_by_lhs[prod.left] = []
            productions_by_lhs[prod.left].append(prod)
            
        # Check LL(1) conditions for each non-terminal
        for non_terminal, prods in productions_by_lhs.items():
            # Calculate first set for each production
            for i, prod1 in enumerate(prods):
                first1 = set()
                all_nullable1 = True
                
                for symbol in prod1.right:
                    sym_first = first_sets[symbol]
                    first1.update(sym_first - {""})
                    if "" not in sym_first:
                        all_nullable1 = False
                        break
                        
                if all_nullable1:
                    first1.add("")
                    
                # Compare with other productions
                for prod2 in prods[i+1:]:
                    first2 = set()
                    all_nullable2 = True
                    
                    for symbol in prod2.right:
                        sym_first = first_sets[symbol]
                        first2.update(sym_first - {""})
                        if "" not in sym_first:
                            all_nullable2 = False
                            break
                            
                    if all_nullable2:
                        first2.add("")
                        
                    # Check for intersection of FIRST sets
                    if first1 & first2:
                        return False
                        
                    # If both can derive ε, check FOLLOW sets
                    if "" in first1 and "" in first2:
                        return False
                        
                    # If one can derive ε, check for intersection with FOLLOW
                    if "" in first1 and first2 & follow_sets[non_terminal]:
                        return False
                    if "" in first2 and first1 & follow_sets[non_terminal]:
                        return False
                        
        return True
    
    def _compute_closure(self, items: Set[LR0Item]) -> Set[LR0Item]:
        """Compute closure of a set of LR(0) items"""
        closure = items.copy()
        changed = True
        
        while changed:
            changed = False
            new_items = set()
            
            for item in closure:
                # If dot is before a non-terminal
                if (item.dot_position < len(item.prod.right) and 
                    item.prod.right[item.dot_position] in self.non_terminals):
                    
                    nt = item.prod.right[item.dot_position]
                    # Add all productions for this non-terminal
                    for prod in self.productions:
                        if prod.left == nt:
                            new_item = LR0Item(prod, 0)
                            if new_item not in closure:
                                new_items.add(new_item)
                                changed = True
                                
            closure.update(new_items)
        
        return closure
    
    def _goto(self, state: Set[LR0Item], symbol: str) -> Set[LR0Item]:
        """Compute GOTO(I,X) for an LR(0) state"""
        next_items = set()
        
        for item in state:
            if (item.dot_position < len(item.prod.right) and 
                item.prod.right[item.dot_position] == symbol):
                
                next_items.add(LR0Item(
                    item.prod,
                    item.dot_position + 1
                ))
                
        return self._compute_closure(next_items)
    
    def _build_lr0_automaton(self) -> Tuple[List[LR0State], LR0State]:
        """Build the LR(0) parsing automaton"""
        # Add augmented production S' → S
        augmented_start = f"{self.start_symbol}'"
        augmented_prod = Production(augmented_start, [self.start_symbol])
        
        # Create initial state
        initial_item = LR0Item(augmented_prod, 0)
        initial_state = LR0State(self._compute_closure({initial_item}))
        
        states = [initial_state]
        state_map = {initial_state: 0}  # For indexing states
        
        # Process states until no new ones are found
        processed = set()
        to_process = {initial_state}
        
        while to_process:
            current = to_process.pop()
            if current in processed:
                continue
                
            processed.add(current)
            
            # Find all symbols after dots in current state
            symbols = set()
            for item in current.items:
                if item.dot_position < len(item.prod.right):
                    symbols.add(item.prod.right[item.dot_position])
                    
            # Compute GOTO for each symbol
            for symbol in symbols:
                next_items = self._goto(current.items, symbol)
                if not next_items:
                    continue
                    
                next_state = LR0State(next_items)
                
                if next_state not in state_map:
                    states.append(next_state)
                    state_map[next_state] = len(states) - 1
                    to_process.add(next_state)
                    
                current.transitions[symbol] = next_state
                
        return states, initial_state
    
    def _parse_lr0(self, input_string: str) -> ParseTree:
        """Parse input using LR(0) parsing algorithm"""
        # Build LR(0) automaton
        states, initial_state = self._build_lr0_automaton()
        
        # Initialize parse configuration
        input_tokens = list(input_string) + ["$"]
        stack = [(initial_state, None)]  # (state, symbol) pairs
        current_token = 0
        
        # Build parse tree as we go
        root = ParseTree("ROOT")
        node_stack = []
        
        while True:
            current_state = stack[-1][0]
            current_symbol = input_tokens[current_token]
            
            # Find applicable reduction or shift
            can_shift = False
            reduction = None
            
            for item in current_state.items:
                if item.dot_position == len(item.prod.right):
                    # Found a reduction
                    if reduction:
                        raise GrammarError("Grammar is not LR(0): Reduce/reduce conflict")
                    reduction = item.prod
                elif (item.dot_position < len(item.prod.right) and 
                      item.prod.right[item.dot_position] == current_symbol):
                    can_shift = True
                    
            if can_shift and reduction:
                raise GrammarError("Grammar is not LR(0): Shift/reduce conflict")
                
            if can_shift:
                # Shift
                next_state = current_state.transitions[current_symbol]
                stack.append((next_state, current_symbol))
                if current_symbol != "$":
                    node = ParseTree(current_symbol)
                    node_stack.append(node)
                current_token += 1
            elif reduction:
                # Reduce
                if reduction.left == f"{self.start_symbol}'":
                    # Accept
                    if len(node_stack) == 1:
                        root.add_child(node_stack[0])
                        return root
                    raise GrammarError("Parsing failed: Stack not properly reduced")
                    
                # Pop symbols from stack
                popped_nodes = []
                for _ in range(len(reduction.right)):
                    stack.pop()
                    if node_stack:
                        popped_nodes.insert(0, node_stack.pop())
                        
                # Create new node for the reduction
                new_node = ParseTree(reduction.left)
                for node in popped_nodes:
                    new_node.add_child(node)
                node_stack.append(new_node)
                
                # GOTO
                next_state = stack[-1][0].transitions[reduction.left]
                stack.append((next_state, reduction.left))
            else:
                raise GrammarError(f"Parsing failed: No valid action for {current_symbol}")
    
    def generate_parse_tree(self, input_string: str) -> ParseTree:
        """Generate parse tree for an input string"""
        # First try LL(1) parsing
        try:
            if self.is_ll1():
                return self._parse_ll1(input_string)
        except GrammarError as e:
            logger.info(f"LL(1) parsing failed: {str(e)}")
            
        # Try LR(0) parsing as fallback
        try:
            return self._parse_lr0(input_string)
        except GrammarError as e:
            logger.info(f"LR(0) parsing failed: {str(e)}")
            raise GrammarError("Could not parse input - grammar is neither LL(1) nor LR(0)")

    def is_lr0(self) -> bool:
        """Check if grammar is LR(0)"""
        try:
            states, _ = self._build_lr0_automaton()
            
            # Check for conflicts in each state
            for state in states:
                reductions = []
                can_shift = False
                
                for item in state.items:
                    if item.dot_position == len(item.prod.right):
                        reductions.append(item.prod)
                    elif item.dot_position < len(item.prod.right):
                        can_shift = True
                        
                # Check for conflicts
                if len(reductions) > 1:  # Reduce/reduce conflict
                    return False
                if can_shift and reductions:  # Shift/reduce conflict
                    return False
                    
            return True
        except Exception:
            return False
        
    def visualize_derivation(self, steps: List[Tuple[Production, int]]) -> str:
        """Generate visualization of a derivation sequence"""
        dot = graphviz.Digraph(comment=f"Derivation in {self.name}")
        dot.attr(rankdir="LR")  # Left to right layout
        
        # Track the sentential form at each step
        sentential_form = [self.start_symbol]
        nodes = []
        
        # Add initial sentential form
        current_id = "step_0"
        dot.node(current_id, " ".join(sentential_form))
        nodes.append(current_id)
        
        # Process each derivation step
        for i, (prod, pos) in enumerate(steps):
            # Apply production at position
            new_form = sentential_form[:pos] + prod.right + sentential_form[pos + 1:]
            sentential_form = new_form
            
            # Add new node
            next_id = f"step_{i + 1}"
            dot.node(next_id, " ".join(sentential_form))
            nodes.append(next_id)
            
            # Add edge with production label
            dot.edge(nodes[-2], nodes[-1], label=str(prod))
        
        return dot.source

    def remove_unreachable_symbols(self) -> None:
        """Remove unreachable symbols from the grammar"""
        # Find all reachable symbols starting from start symbol
        reachable = {self.start_symbol}
        changed = True
        
        while changed:
            changed = False
            for prod in self.productions:
                if prod.left in reachable:
                    for symbol in prod.right:
                        if symbol not in reachable:
                            reachable.add(symbol)
                            changed = True
        
        # Keep only productions with reachable symbols
        self.productions = [p for p in self.productions if p.left in reachable and 
                          all(sym in reachable for sym in p.right)]
        self.non_terminals = {nt for nt in self.non_terminals if nt in reachable}
        self.terminals = {t for t in self.terminals if t in reachable}

    def remove_useless_symbols(self) -> None:
        """Remove symbols that don't derive any terminal string"""
        # Find all productive symbols (can derive terminal strings)
        productive = set(self.terminals)
        changed = True
        
        while changed:
            changed = False
            for prod in self.productions:
                if (prod.left not in productive and 
                    all(sym in productive for sym in prod.right)):
                    productive.add(prod.left)
                    changed = True
        
        # Keep only productions with productive symbols
        self.productions = [p for p in self.productions if p.left in productive and 
                          all(sym in productive for sym in p.right)]
        self.non_terminals = {nt for nt in self.non_terminals if nt in productive}
        
        # Then remove unreachable symbols
        self.remove_unreachable_symbols()

    def left_factor(self) -> None:
        """Apply left factoring to the grammar"""
        changed = True
        while changed:
            changed = False
            # Group productions by left-hand side
            prods_by_lhs = {}
            for prod in self.productions:
                if prod.left not in prods_by_lhs:
                    prods_by_lhs[prod.left] = []
                prods_by_lhs[prod.left].append(prod.right)
            
            new_productions = []
            for left, rights in prods_by_lhs.items():
                # Find common prefixes
                while len(rights) > 1:
                    prefix_groups = {}
                    for right in rights:
                        if not right:  # Skip empty productions
                            new_productions.append(Production(left, []))
                            continue
                        prefix = right[0]
                        if prefix not in prefix_groups:
                            prefix_groups[prefix] = []
                        prefix_groups[prefix].append(right)
                    
                    # Find longest common prefix for each group
                    for prefix, group in prefix_groups.items():
                        if len(group) <= 1:  # No factoring needed
                            new_productions.append(Production(left, group[0]))
                            rights.remove(group[0])
                            continue
                            
                        # Find length of common prefix
                        common_len = 0
                        while all(len(r) > common_len and 
                                all(r[i] == group[0][i] for r in group)
                                for i in range(common_len + 1)):
                            common_len += 1
                            
                        if common_len > 0:
                            changed = True
                            # Create new non-terminal
                            new_nt = f"{left}_{len(self.non_terminals)}"
                            while new_nt in self.non_terminals:
                                new_nt += "'"
                            self.non_terminals.add(new_nt)
                            
                            # Add production with common prefix
                            new_productions.append(
                                Production(left, group[0][:common_len] + [new_nt])
                            )
                            
                            # Add productions for the rest
                            for right in group:
                                if right[common_len:]:  # Non-empty suffix
                                    new_productions.append(
                                        Production(new_nt, right[common_len:])
                                    )
                                else:  # Empty suffix
                                    new_productions.append(Production(new_nt, []))
                                rights.remove(right)
                
                # Add any remaining productions
                for right in rights:
                    new_productions.append(Production(left, right))
            
            if changed:
                self.productions = new_productions

    def optimize(self) -> None:
        """Apply all optimization algorithms to the grammar"""
        # Remove useless and unreachable symbols
        self.remove_useless_symbols()
        
        # Left factor the grammar
        self.left_factor()
        
        # Eliminate immediate left recursion
        self._eliminate_immediate_left_recursion()
        
        # Final cleanup of unreachable symbols
        self.remove_unreachable_symbols()

    def _eliminate_immediate_left_recursion(self) -> None:
        """Eliminate immediate left recursion from the grammar"""
        new_productions = []
        
        # Group productions by left-hand side
        prods_by_lhs = {}
        for prod in self.productions:
            if prod.left not in prods_by_lhs:
                prods_by_lhs[prod.left] = []
            prods_by_lhs[prod.left].append(prod)
        
        # Process each non-terminal
        for nt, prods in prods_by_lhs.items():
            # Separate recursive and non-recursive productions
            recursive = []
            non_recursive = []
            
            for prod in prods:
                if prod.right and prod.right[0] == nt:
                    recursive.append(prod)
                else:
                    non_recursive.append(prod)
            
            if recursive:  # Has left recursion
                # Create new non-terminal
                new_nt = f"{nt}'"
                while new_nt in self.non_terminals:
                    new_nt += "'"
                self.non_terminals.add(new_nt)
                
                # Add new productions
                for prod in non_recursive:
                    new_productions.append(
                        Production(nt, prod.right + [new_nt])
                    )
                
                for prod in recursive:
                    new_productions.append(
                        Production(new_nt, prod.right[1:] + [new_nt])
                    )
                    
                # Add epsilon production for new non-terminal
                new_productions.append(Production(new_nt, []))
                
            else:  # No left recursion
                new_productions.extend(non_recursive)
        
        self.productions = new_productions

class ParseTree:
    """Represents a parse tree for a string in the grammar"""
    def __init__(self, label: str):
        self.label = label
        self.children: List[ParseTree] = []
        
    def add_child(self, child: 'ParseTree') -> None:
        """Add a child node to this node"""
        self.children.append(child)
        
    def visualize(self) -> str:
        """Generate visualization of the parse tree"""
        dot = graphviz.Digraph(comment="Parse Tree")
        dot.attr(rankdir="TB")  # Top to bottom layout
        
        def add_nodes(node: ParseTree, parent_id: Optional[str] = None, rank: int = 0) -> None:
            node_id = str(id(node))
            
            # Different styles for terminals and non-terminals
            if node.label.isupper() or node.label == "ROOT":
                dot.node(node_id, node.label, shape="circle")
            else:
                dot.node(node_id, node.label, shape="box")
                
            if parent_id:
                dot.edge(parent_id, node_id)
                
            # Add invisible edges between siblings for better layout
            prev_child_id = None
            for child in node.children:
                child_id = str(id(child))
                if prev_child_id:
                    dot.edge(prev_child_id, child_id, style="invis")
                prev_child_id = child_id
                add_nodes(child, node_id, rank + 1)
                
            # Create subgraph for rank constraints
            with dot.subgraph() as s:
                s.attr(rank="same")
                if prev_child_id:  # If node has children
                    for child in node.children:
                        s.node(str(id(child)))
                
        add_nodes(self)
        return dot.source

def convert_to_gnf(grammar: Grammar) -> Grammar:
    """Convert a grammar to Greibach Normal Form"""
    # First convert to CNF to simplify the process
    cnf = grammar.to_cnf()
    gnf = Grammar(f"GNF of {grammar.name}")
    
    # Copy the grammar
    gnf.terminals = cnf.terminals.copy()
    gnf.non_terminals = cnf.non_terminals.copy()
    gnf.productions = [Production(p.left, p.right) for p in cnf.productions]
    gnf.start_symbol = cnf.start_symbol
    
    # Order non-terminals (any order is fine as we'll eliminate left recursion)
    ordered_non_terminals = list(gnf.non_terminals)
    
    # Eliminate left recursion
    for i, Ai in enumerate(ordered_non_terminals):
        # Eliminate immediate left recursion
        ai_productions = []  # Ai → Aiα
        bi_productions = []  # Ai → β where β doesn't start with Ai
        
        new_prods = [p for p in gnf.productions if p.left == Ai]
        for prod in new_prods:
            if prod.right and prod.right[0] == Ai:
                ai_productions.append(prod)
            else:
                bi_productions.append(prod)
                
        if ai_productions:
            # Create new non-terminal Ai'
            new_nt = f"{Ai}'"
            while new_nt in gnf.non_terminals:
                new_nt += "'"
            gnf.non_terminals.add(new_nt)
            
            # Replace Ai → Aiα with Ai → βAi' and Ai' → αAi'|ε
            new_productions = []
            for prod in bi_productions:
                new_productions.append(Production(Ai, prod.right + [new_nt]))
            for prod in ai_productions:
                new_productions.append(Production(new_nt, prod.right[1:] + [new_nt]))
            new_productions.append(Production(new_nt, []))  # Ai' → ε
            
            # Update productions
            gnf.productions = [p for p in gnf.productions if p.left != Ai] + new_productions
    
    # Convert to GNF form
    changed = True
    while changed:
        changed = False
        new_productions = []
        
        for prod in gnf.productions:
            if not prod.right:  # Skip ε-productions for now
                new_productions.append(prod)
                continue
                
            if prod.right[0] in gnf.terminals:  # Already in GNF
                new_productions.append(prod)
                continue
                
            # Find productions for the leftmost non-terminal
            left_nt = prod.right[0]
            replacements = [p for p in gnf.productions if p.left == left_nt]
            
            # Replace the leftmost non-terminal with its productions
            for repl in replacements:
                if not repl.right:  # Skip ε-productions in replacement
                    continue
                new_right = repl.right + prod.right[1:]
                new_productions.append(Production(prod.left, new_right))
                changed = True
                
        gnf.productions = new_productions
    
    # Clean up by removing unreachable productions
    reachable = {gnf.start_symbol}
    changed = True
    while changed:
        changed = False
        for prod in gnf.productions:
            if prod.left in reachable:
                for symbol in prod.right:
                    if symbol in gnf.non_terminals and symbol not in reachable:
                        reachable.add(symbol)
                        changed = True
    
    # Keep only productions with reachable non-terminals
    gnf.productions = [p for p in gnf.productions if p.left in reachable]
    gnf.non_terminals = {nt for nt in gnf.non_terminals if nt in reachable}
    
    return gnf