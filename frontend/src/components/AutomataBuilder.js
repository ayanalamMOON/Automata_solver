import React, { useState, useEffect } from 'react';
import './AutomataBuilder.css';

const AutomataBuilder = () => {
    const [builderState, setBuilderState] = useState(null);
    const [currentAction, setCurrentAction] = useState('add_state');
    const [stateParams, setStateParams] = useState({
        name: '',
        is_initial: false,
        is_final: false
    });
    const [transitionParams, setTransitionParams] = useState({
        from_state: '',
        symbol: '',
        to_state: ''
    });
    const [simulationInput, setSimulationInput] = useState('');
    const [simulationResult, setSimulationResult] = useState(null);

    const startNewAutomata = async (type) => {
        try {
            const response = await fetch('/api/builder/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    automata_type: type,
                    name: `New ${type}`
                }),
            });
            const data = await response.json();
            setBuilderState(data.state);
        } catch (error) {
            console.error('Error starting new automata:', error);
        }
    };

    const performAction = async (action, params) => {
        try {
            const response = await fetch('/api/builder/action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action, params }),
            });
            const data = await response.json();
            if (action === 'simulate') {
                setSimulationResult(data.state);
            } else {
                setBuilderState(data.state);
            }
        } catch (error) {
            console.error('Error performing action:', error);
        }
    };

    const handleAddState = (e) => {
        e.preventDefault();
        performAction('add_state', stateParams);
        setStateParams({ name: '', is_initial: false, is_final: false });
    };

    const handleAddTransition = (e) => {
        e.preventDefault();
        performAction('add_transition', transitionParams);
        setTransitionParams({ from_state: '', symbol: '', to_state: '' });
    };

    const handleSimulate = (e) => {
        e.preventDefault();
        performAction('simulate', { input_string: simulationInput });
    };

    const exportAutomata = async (format) => {
        try {
            const response = await fetch('/api/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    automata_type: builderState.type,
                    name: 'Exported Automata',
                    definition: builderState
                }),
            });
            const data = await response.json();
            
            // Create download link
            const blob = new Blob([data[format]], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `automata.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error exporting automata:', error);
        }
    };

    const importAutomata = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const response = await fetch('/api/import', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        file_content: e.target.result,
                        format: file.name.endsWith('.jff') ? 'jflap' : 'dot'
                    }),
                });
                const data = await response.json();
                setBuilderState(data.automata);
            } catch (error) {
                console.error('Error importing automata:', error);
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="automata-builder">
            <div className="builder-controls">
                <div className="new-automata">
                    <button onClick={() => startNewAutomata('DFA')}>New DFA</button>
                    <button onClick={() => startNewAutomata('NFA')}>New NFA</button>
                    <button onClick={() => startNewAutomata('PDA')}>New PDA</button>
                </div>

                <div className="action-controls">
                    <select value={currentAction} onChange={(e) => setCurrentAction(e.target.value)}>
                        <option value="add_state">Add State</option>
                        <option value="add_transition">Add Transition</option>
                        <option value="simulate">Simulate Input</option>
                    </select>

                    {currentAction === 'add_state' && (
                        <form onSubmit={handleAddState}>
                            <input
                                type="text"
                                placeholder="State name"
                                value={stateParams.name}
                                onChange={(e) => setStateParams({...stateParams, name: e.target.value})}
                            />
                            <label>
                                <input
                                    type="checkbox"
                                    checked={stateParams.is_initial}
                                    onChange={(e) => setStateParams({...stateParams, is_initial: e.target.checked})}
                                />
                                Initial State
                            </label>
                            <label>
                                <input
                                    type="checkbox"
                                    checked={stateParams.is_final}
                                    onChange={(e) => setStateParams({...stateParams, is_final: e.target.checked})}
                                />
                                Final State
                            </label>
                            <button type="submit">Add State</button>
                        </form>
                    )}

                    {currentAction === 'add_transition' && (
                        <form onSubmit={handleAddTransition}>
                            <input
                                type="text"
                                placeholder="From state"
                                value={transitionParams.from_state}
                                onChange={(e) => setTransitionParams({...transitionParams, from_state: e.target.value})}
                            />
                            <input
                                type="text"
                                placeholder="Symbol"
                                value={transitionParams.symbol}
                                onChange={(e) => setTransitionParams({...transitionParams, symbol: e.target.value})}
                            />
                            <input
                                type="text"
                                placeholder="To state"
                                value={transitionParams.to_state}
                                onChange={(e) => setTransitionParams({...transitionParams, to_state: e.target.value})}
                            />
                            <button type="submit">Add Transition</button>
                        </form>
                    )}

                    {currentAction === 'simulate' && (
                        <form onSubmit={handleSimulate}>
                            <input
                                type="text"
                                placeholder="Input string"
                                value={simulationInput}
                                onChange={(e) => setSimulationInput(e.target.value)}
                            />
                            <button type="submit">Simulate</button>
                        </form>
                    )}
                </div>

                <div className="history-controls">
                    <button 
                        onClick={() => performAction('undo')}
                        disabled={!builderState?.can_undo}
                    >
                        Undo
                    </button>
                    <button 
                        onClick={() => performAction('redo')}
                        disabled={!builderState?.can_redo}
                    >
                        Redo
                    </button>
                </div>

                <div className="import-export">
                    <button onClick={() => exportAutomata('jflap')}>Export to JFLAP</button>
                    <button onClick={() => exportAutomata('dot')}>Export to DOT</button>
                    <input
                        type="file"
                        accept=".jff,.dot"
                        onChange={importAutomata}
                        style={{ display: 'none' }}
                        id="import-file"
                    />
                    <button onClick={() => document.getElementById('import-file').click()}>
                        Import Automata
                    </button>
                </div>
            </div>

            {builderState && (
                <div className="visualization">
                    <div dangerouslySetInnerHTML={{ __html: builderState.visualization }} />
                </div>
            )}

            {simulationResult && (
                <div className="simulation-results">
                    <h3>Simulation Results</h3>
                    <p>Accepted: {simulationResult.accepted ? 'Yes' : 'No'}</p>
                    <div className="simulation-steps">
                        {simulationResult.steps.map((step, index) => (
                            <div key={index} className="step">
                                <span>Step {step.step}: </span>
                                <span>State: {step.current_state}</span>
                                <span>Remaining Input: {step.remaining_input}</span>
                                <span>Processed: {step.processed_input}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default AutomataBuilder;