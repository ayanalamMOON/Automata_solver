import React, { useState, useCallback, useEffect } from 'react';
import './RegexPlayground.css';
import AutomataVisualizer from './AutomataVisualizer';

const RegexPlayground = () => {
    const [regex, setRegex] = useState('');
    const [isValid, setIsValid] = useState(true);
    const [validationMessage, setValidationMessage] = useState('');
    const [automaton, setAutomaton] = useState(null);
    const [testString, setTestString] = useState('');
    const [testResult, setTestResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const validateRegex = useCallback(async (input) => {
        try {
            const response = await fetch('/api/validate_regex', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ regex: input }),
            });

            const data = await response.json();
            setIsValid(data.is_valid);
            setValidationMessage(data.is_valid ? 'Valid regular expression' : 'Invalid regular expression');
            return data.is_valid;
        } catch (error) {
            setError('Error validating regular expression');
            return false;
        }
    }, []);

    const convertToDFA = useCallback(async () => {
        if (!regex) return;

        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/convert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ regex }),
            });

            const data = await response.json();
            if (data.error) {
                setError(data.error);
                return;
            }

            setAutomaton(data);
        } catch (error) {
            setError('Error converting regular expression to DFA');
        } finally {
            setIsLoading(false);
        }
    }, [regex]);

    const testInput = useCallback(async () => {
        if (!automaton || !testString) return;

        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/simulate/step_by_step', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    states: automaton.states,
                    alphabet: automaton.alphabet,
                    transitions: automaton.transitions,
                    start_state: automaton.start_state,
                    accept_states: automaton.accept_states,
                    input_string: testString
                }),
            });

            const data = await response.json();
            setTestResult(data);
        } catch (error) {
            setError('Error testing input string');
        } finally {
            setIsLoading(false);
        }
    }, [automaton, testString]);

    // Handle regex input changes
    const handleRegexChange = useCallback(async (e) => {
        const input = e.target.value;
        setRegex(input);
        if (input) {
            await validateRegex(input);
        } else {
            setIsValid(true);
            setValidationMessage('');
        }
    }, [validateRegex]);

    // Convert to DFA when regex is valid
    useEffect(() => {
        if (isValid && regex) {
            convertToDFA();
        } else {
            setAutomaton(null);
        }
    }, [isValid, regex, convertToDFA]);

    return (
        <div className="regex-playground">
            <h2>Regular Expression Playground</h2>
            
            <div className="input-section">
                <div className="regex-input">
                    <label htmlFor="regex">Regular Expression:</label>
                    <input
                        id="regex"
                        type="text"
                        value={regex}
                        onChange={handleRegexChange}
                        placeholder="Enter a regular expression"
                        className={isValid ? '' : 'invalid'}
                    />
                    {validationMessage && (
                        <div className={`validation-message ${isValid ? 'valid' : 'invalid'}`}>
                            {validationMessage}
                        </div>
                    )}
                </div>

                <div className="test-input">
                    <label htmlFor="test-string">Test String:</label>
                    <div className="test-input-group">
                        <input
                            id="test-string"
                            type="text"
                            value={testString}
                            onChange={(e) => setTestString(e.target.value)}
                            placeholder="Enter a test string"
                        />
                        <button 
                            onClick={testInput}
                            disabled={!automaton || !testString || isLoading}
                        >
                            Test
                        </button>
                    </div>
                </div>
            </div>

            {isLoading && (
                <div className="loading">Converting regular expression to DFA...</div>
            )}

            {error && (
                <div className="error-message">{error}</div>
            )}

            {automaton && (
                <div className="visualization-section">
                    <h3>Equivalent DFA</h3>
                    <AutomataVisualizer visualizationState={automaton.visualization_state} />
                </div>
            )}

            {testResult && (
                <div className="test-results">
                    <h3>Test Results</h3>
                    <div className={`result ${testResult.accepted ? 'accepted' : 'rejected'}`}>
                        String {testResult.accepted ? 'ACCEPTED' : 'REJECTED'}
                    </div>
                    {testResult.steps && (
                        <div className="steps">
                            <h4>Execution Steps:</h4>
                            {testResult.steps.map((step, index) => (
                                <div key={index} className="step">
                                    <div>Step {step.step}</div>
                                    <div>Current State: {step.current_state}</div>
                                    <div>Input Symbol: {step.input_symbol || 'ε'}</div>
                                    <div>Processed: {step.processed_input}</div>
                                    <div>Remaining: {step.remaining_input}</div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default RegexPlayground;