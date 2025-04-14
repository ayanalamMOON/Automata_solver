import React, { useState, useCallback } from 'react';
import './GrammarBuilder.css';

const BatchProcessor = ({ grammar, onResult }) => {
    const [strings, setStrings] = useState('');
    const [operations, setOperations] = useState([]);
    const [processing, setProcessing] = useState(false);

    const availableOperations = [
        { id: 'validate', label: 'Validate Grammar' },
        { id: 'optimize', label: 'Optimize Grammar' },
        { id: 'cnf', label: 'Convert to CNF' },
        { id: 'll1', label: 'Check LL(1)' },
        { id: 'lr0', label: 'Check LR(0)' }
    ];

    const handleBatchProcess = async () => {
        if (!operations.length) return;
        setProcessing(true);
        
        try {
            const response = await fetch('/api/grammar/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...grammar,
                    operations
                })
            });
            
            const data = await response.json();
            if (response.ok) {
                onResult({
                    type: 'Batch',
                    results: data
                });
            }
        } catch (err) {
            console.error('Batch processing failed:', err);
        } finally {
            setProcessing(false);
        }
    };

    const handleBatchParse = async () => {
        if (!strings.trim()) return;
        setProcessing(true);
        
        try {
            const stringList = strings.split('\n').filter(s => s.trim());
            const response = await fetch('/api/grammar/batch_parse', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...grammar,
                    strings: stringList
                })
            });
            
            const data = await response.json();
            if (response.ok) {
                onResult({
                    type: 'BatchParse',
                    results: data
                });
            }
        } catch (err) {
            console.error('Batch parsing failed:', err);
        } finally {
            setProcessing(false);
        }
    };

    return (
        <div className="batch-processor">
            <h3>Batch Processing</h3>
            
            <div className="operation-selector">
                <h4>Select Operations</h4>
                {availableOperations.map(op => (
                    <label key={op.id}>
                        <input
                            type="checkbox"
                            checked={operations.includes(op.id)}
                            onChange={e => {
                                if (e.target.checked) {
                                    setOperations([...operations, op.id]);
                                } else {
                                    setOperations(operations.filter(id => id !== op.id));
                                }
                            }}
                        />
                        {op.label}
                    </label>
                ))}
                <button 
                    onClick={handleBatchProcess}
                    disabled={processing || !operations.length}>
                    Process Operations
                </button>
            </div>

            <div className="batch-parser">
                <h4>Batch String Parsing</h4>
                <textarea
                    value={strings}
                    onChange={e => setStrings(e.target.value)}
                    placeholder="Enter strings to parse (one per line)"
                    rows={5}
                />
                <button 
                    onClick={handleBatchParse}
                    disabled={processing || !strings.trim()}>
                    Parse Strings
                </button>
            </div>
        </div>
    );
};

const GrammarBuilder = () => {
    const [grammar, setGrammar] = useState({
        name: '',
        productions: [],
        start_symbol: ''
    });
    
    const [newProduction, setNewProduction] = useState({
        left: '',
        right: ''
    });
    
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    
    const addProduction = useCallback(() => {
        if (!newProduction.left || !newProduction.right) return;
        
        setGrammar(prev => ({
            ...prev,
            productions: [
                ...prev.productions,
                {
                    left: newProduction.left,
                    right: newProduction.right.split(' ').filter(Boolean)
                }
            ]
        }));
        
        setNewProduction({ left: '', right: '' });
    }, [newProduction]);
    
    const removeProduction = useCallback((index) => {
        setGrammar(prev => ({
            ...prev,
            productions: prev.productions.filter((_, i) => i !== index)
        }));
    }, []);
    
    const validateGrammar = async () => {
        try {
            const response = await fetch('/api/grammar/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(grammar)
            });
            
            const data = await response.json();
            if (response.ok) {
                setError(null);
                return true;
            } else {
                setError(data.detail);
                return false;
            }
        } catch (err) {
            setError(err.message);
            return false;
        }
    };
    
    const convertToCNF = async () => {
        if (!await validateGrammar()) return;
        
        try {
            const response = await fetch('/api/grammar/to_cnf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(grammar)
            });
            
            const data = await response.json();
            if (response.ok) {
                setResult({
                    type: 'CNF',
                    grammar: data
                });
                setError(null);
            } else {
                setError(data.detail);
            }
        } catch (err) {
            setError(err.message);
        }
    };
    
    const checkLL1 = async () => {
        if (!await validateGrammar()) return;
        
        try {
            const response = await fetch('/api/grammar/check_ll1', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(grammar)
            });
            
            const data = await response.json();
            if (response.ok) {
                setResult({
                    type: 'LL1',
                    isLL1: data.is_ll1
                });
                setError(null);
            } else {
                setError(data.detail);
            }
        } catch (err) {
            setError(err.message);
        }
    };
    
    const parseString = async (input) => {
        if (!await validateGrammar()) return;
        
        try {
            const response = await fetch('/api/grammar/parse', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...grammar,
                    string: input
                })
            });
            
            const data = await response.json();
            if (response.ok) {
                setResult({
                    type: 'ParseTree',
                    visualization: data.visualization
                });
                setError(null);
            } else {
                setError(data.detail);
            }
        } catch (err) {
            setError(err.message);
        }
    };
    
    return (
        <div className="grammar-builder">
            <h2>Grammar Builder</h2>
            
            <div className="grammar-input">
                <div className="form-group">
                    <label>Grammar Name:</label>
                    <input
                        type="text"
                        value={grammar.name}
                        onChange={e => setGrammar(prev => ({ ...prev, name: e.target.value }))}
                        placeholder="Enter grammar name"
                    />
                </div>
                
                <div className="form-group">
                    <label>Start Symbol:</label>
                    <input
                        type="text"
                        value={grammar.start_symbol}
                        onChange={e => setGrammar(prev => ({ ...prev, start_symbol: e.target.value }))}
                        placeholder="Enter start symbol"
                    />
                </div>
                
                <div className="productions">
                    <h3>Productions</h3>
                    {grammar.productions.map((prod, index) => (
                        <div key={index} className="production">
                            <span>{prod.left} → {prod.right.join(' ')}</span>
                            <button onClick={() => removeProduction(index)}>Remove</button>
                        </div>
                    ))}
                    
                    <div className="add-production">
                        <input
                            type="text"
                            value={newProduction.left}
                            onChange={e => setNewProduction(prev => ({ ...prev, left: e.target.value }))}
                            placeholder="Left-hand side"
                        />
                        <span>→</span>
                        <input
                            type="text"
                            value={newProduction.right}
                            onChange={e => setNewProduction(prev => ({ ...prev, right: e.target.value }))}
                            placeholder="Right-hand side (space-separated)"
                        />
                        <button onClick={addProduction}>Add</button>
                    </div>
                </div>
                
                <div className="actions">
                    <button onClick={validateGrammar}>Validate</button>
                    <button onClick={convertToCNF}>Convert to CNF</button>
                    <button onClick={checkLL1}>Check LL(1)</button>
                    <button onClick={() => parseString(prompt('Enter string to parse:'))}>
                        Parse String
                    </button>
                </div>
                
                {error && <div className="error">{error}</div>}
                
                {result && (
                    <div className="result">
                        <h3>Result</h3>
                        {result.type === 'CNF' && (
                            <div>
                                <h4>Chomsky Normal Form:</h4>
                                <div className="cnf-grammar">
                                    {result.grammar.productions.map((prod, index) => (
                                        <div key={index}>
                                            {prod.left} → {prod.right.join(' ')}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        {result.type === 'LL1' && (
                            <div>
                                <h4>LL(1) Analysis:</h4>
                                <p>This grammar is {result.isLL1 ? '' : 'not'} LL(1)</p>
                            </div>
                        )}
                        {result.type === 'ParseTree' && (
                            <div>
                                <h4>Parse Tree:</h4>
                                <div className="parse-tree" 
                                     dangerouslySetInnerHTML={{ __html: result.visualization }} />
                            </div>
                        )}
                        {result.type === 'Batch' && (
                            <div>
                                <h4>Batch Processing Results:</h4>
                                {Object.entries(result.results).map(([op, res]) => (
                                    <div key={op} className="batch-result">
                                        <h5>{op}</h5>
                                        <pre>{JSON.stringify(res, null, 2)}</pre>
                                    </div>
                                ))}
                            </div>
                        )}
                        {result.type === 'BatchParse' && (
                            <div>
                                <h4>Batch Parsing Results:</h4>
                                {Object.entries(result.results).map(([str, res]) => (
                                    <div key={str} className="parse-result">
                                        <h5>String: {str}</h5>
                                        {res.valid ? (
                                            <div className="parse-tree"
                                                 dangerouslySetInnerHTML={{ __html: res.visualization }} />
                                        ) : (
                                            <div className="error">{res.error}</div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
            
            <BatchProcessor 
                grammar={grammar}
                onResult={result => setResult(result)}
            />
        </div>
    );
};

export default GrammarBuilder;