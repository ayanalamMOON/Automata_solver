import React, { useState, useEffect, useRef } from "react";
import { convertRegex, explainAutomata, minimizeAutomaton, exportAutomaton, validateRegex, aiSuggestions } from "./api";
import * as d3 from "d3";

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
    const [isValidRegex, setIsValidRegex] = useState(true);
    const [loading, setLoading] = useState(false);
    const [suggestions, setSuggestions] = useState("");
    const svgRef = useRef(null);

    useEffect(() => {
        const validate = async () => {
            if (regex.trim() !== "") {
                const result = await validateRegex(regex);
                setIsValidRegex(result.is_valid);
            }
        };
        validate();
    }, [regex]);

    const handleConvert = async () => {
        if (!isValidRegex) return;
        
        setLoading(true);
        try {
            const dfaSvg = await convertRegex(regex);
            setSvg(dfaSvg);
        } catch (error) {
            console.error("Error converting regex:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleExplain = async () => {
        setLoading(true);
        try {
            const exp = await explainAutomata(regex);
            setExplanation(exp);
        } catch (error) {
            console.error("Error getting explanation:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleMinimize = async () => {
        setLoading(true);
        try {
            const minimized = await minimizeAutomaton(automaton);
            setAutomaton(minimized);
            const exported = await exportAutomaton(minimized, 'svg');
            setSvg(exported);
        } catch (error) {
            console.error("Error minimizing automaton:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = async (format) => {
        setLoading(true);
        try {
            const exported = await exportAutomaton(automaton, format);
            if (format === 'svg') {
                setSvg(exported);
            } else {
                const link = document.createElement('a');
                link.href = `data:application/${format};base64,${exported}`;
                link.download = `automaton.${format}`;
                link.click();
            }
        } catch (error) {
            console.error(`Error exporting as ${format}:`, error);
        } finally {
            setLoading(false);
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

    const handleSuggestions = async () => {
        setLoading(true);
        try {
            const result = await aiSuggestions(regex);
            setSuggestions(result.suggestions);
        } catch (error) {
            console.error("Error getting suggestions:", error);
        } finally {
            setLoading(false);
        }
    };

    // Set up D3 zoom functionality after SVG is rendered
    useEffect(() => {
        if (svg && svgRef.current) {
            // Need to wait for the SVG to be fully rendered in the DOM
            setTimeout(() => {
                const svgElement = d3.select(svgRef.current).select("svg");
                if (!svgElement.empty()) {
                    // Add zoom behavior if not already added
                    if (!svgElement.property("__zoom_added")) {
                        const zoomBehavior = d3.zoom()
                            .scaleExtent([0.1, 10])
                            .on("zoom", (event) => {
                                svgElement.select("g")
                                    .attr("transform", event.transform);
                            });
                        
                        svgElement.call(zoomBehavior);
                        svgElement.property("__zoom_added", true);
                    }
                }
            }, 300); // Small delay to ensure SVG is in the DOM
        }
    }, [svg]);

    return (
        <div className="container">
            <h1>Automata Solver</h1>
            
            <div className="section">
                <input 
                    type="text" 
                    value={regex} 
                    onChange={(e) => setRegex(e.target.value)} 
                    placeholder="Enter Regex" 
                    className={isValidRegex ? "" : "invalid"}
                />
                <button onClick={handleConvert} disabled={!isValidRegex}>Convert to DFA</button>
                <button onClick={handleExplain}>Explain</button>
                <button onClick={handleSuggestions}>AI Suggestions</button>
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
                {loading && <div className="loading">Loading...</div>}
                {svg && (
                    <div 
                        ref={svgRef} 
                        className="svg-container" 
                        dangerouslySetInnerHTML={{ __html: svg }}
                    />
                )}
                {explanation && (
                    <div className="explanation">
                        <h3>Explanation</h3>
                        <p>{explanation}</p>
                    </div>
                )}
                {suggestions && (
                    <div className="suggestions">
                        <h3>AI Suggestions</h3>
                        <p>{suggestions}</p>
                    </div>
                )}
            </div>
            
            <style jsx="true">{`
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
                }
                input[type="text"] {
                    padding: 8px;
                    margin-right: 10px;
                }
                .display-section {
                    margin-top: 20px;
                    padding: 20px;
                    border: 1px solid #eee;
                    border-radius: 5px;
                }
                .svg-container {
                    overflow: auto;
                    border: 1px solid #ddd;
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                    background: #f9f9f9;
                }
                .loading {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100px;
                    font-weight: bold;
                }
                .explanation, .suggestions {
                    margin-top: 20px;
                    padding: 15px;
                    background: #f5f5f5;
                    border-radius: 5px;
                    border-left: 4px solid #646cff;
                }
            `}</style>
        </div>
    );
};

export default App;
