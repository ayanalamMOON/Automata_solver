import React, { useState } from "react";
import { convertRegex, explainAutomata, minimizeAutomaton, exportAutomaton } from "./api";

const App = () => {
    const [regex, setRegex] = useState("");
    const [svg, setSvg] = useState("");
    const [explanation, setExplanation] = useState("");
    const [automaton, setAutomaton] = useState({
        states: [],
        transitions: {},
        initial_state: "",
        final_states: [],
        input_symbols: []
    });
    const [editingState, setEditingState] = useState(false);
    const [newState, setNewState] = useState({ name: "", isFinal: false });

    const handleConvert = async () => {
        const dfaSvg = await convertRegex(regex);
        setSvg(dfaSvg);
    };

    const handleExplain = async () => {
        const exp = await explainAutomata(regex);
        setExplanation(exp);
    };

    const handleMinimize = async () => {
        const minimized = await minimizeAutomaton(automaton);
        setAutomaton(minimized);
        const exported = await exportAutomaton(minimized, 'svg');
        setSvg(exported);
    };

    const handleExport = async (format) => {
        const exported = await exportAutomaton(automaton, format);
        if (format === 'svg') {
            setSvg(exported);
        } else {
            const link = document.createElement('a');
            link.href = `data:application/${format};base64,${exported}`;
            link.download = `automaton.${format}`;
            link.click();
        }
    };

    const addState = () => {
        if (newState.name) {
            setAutomaton(prev => ({
                ...prev,
                states: [...prev.states, newState.name],
                final_states: newState.isFinal ? 
                    [...prev.final_states, newState.name] : 
                    prev.final_states,
                transitions: {
                    ...prev.transitions,
                    [newState.name]: {}
                }
            }));
            setNewState({ name: "", isFinal: false });
            setEditingState(false);
        }
    };

    return (
        <div className="container">
            <h1>Automata Solver</h1>
            
            <div className="section">
                <input 
                    type="text" 
                    value={regex} 
                    onChange={(e) => setRegex(e.target.value)} 
                    placeholder="Enter Regex" 
                />
                <button onClick={handleConvert}>Convert to DFA</button>
                <button onClick={handleExplain}>Explain</button>
            </div>

            <div className="section">
                <button onClick={() => setEditingState(!editingState)}>
                    {editingState ? "Cancel" : "Add State"}
                </button>
                
                {editingState && (
                    <div className="state-editor">
                        <input
                            type="text"
                            value={newState.name}
                            onChange={(e) => setNewState({...newState, name: e.target.value})}
                            placeholder="State name"
                        />
                        <label>
                            <input
                                type="checkbox"
                                checked={newState.isFinal}
                                onChange={(e) => setNewState({...newState, isFinal: e.target.checked})}
                            />
                            Final State
                        </label>
                        <button onClick={addState}>Add</button>
                    </div>
                )}
            </div>

            <div className="section">
                <button onClick={handleMinimize}>Minimize Automaton</button>
                <button onClick={() => handleExport('svg')}>Export SVG</button>
                <button onClick={() => handleExport('png')}>Export PNG</button>
                <button onClick={() => handleExport('pdf')}>Export PDF</button>
            </div>

            <div className="display-section">
                {svg && <div dangerouslySetInnerHTML={{ __html: svg }} />}
                {explanation && <p>{explanation}</p>}
            </div>
            
            <style jsx>{`
                .container {
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .section {
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #eee;
                    border-radius: 5px;
                }
                .state-editor {
                    margin-top: 10px;
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                button {
                    margin: 0 5px;
                    padding: 8px 16px;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background: #0056b3;
                }
                input[type="text"] {
                    padding: 8px;
                    margin-right: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .display-section {
                    margin-top: 20px;
                    padding: 20px;
                    border: 1px solid #eee;
                    border-radius: 5px;
                }
            `}</style>
        </div>
    );
};

export default App;
