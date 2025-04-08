import React, { useState } from 'react';
import './App.css';
import AutomataBuilder from './components/AutomataBuilder';
import AnswerAnalyzer from './components/AnswerAnalyzer';

function App() {
    const [currentTool, setCurrentTool] = useState('builder');

    return (
        <div className="app">
            <header className="app-header">
                <h1>Automata Solver</h1>
                <nav>
                    <button 
                        className={currentTool === 'builder' ? 'active' : ''}
                        onClick={() => setCurrentTool('builder')}
                    >
                        Automata Builder
                    </button>
                    <button 
                        className={currentTool === 'analyzer' ? 'active' : ''}
                        onClick={() => setCurrentTool('analyzer')}
                    >
                        Answer Analyzer
                    </button>
                </nav>
            </header>

            <main className="app-content">
                {currentTool === 'builder' ? (
                    <AutomataBuilder />
                ) : (
                    <AnswerAnalyzer />
                )}
            </main>
        </div>
    );
}

export default App;