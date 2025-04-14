import React, { useState } from 'react';
import './App.css';
import AutomataBuilder from './components/AutomataBuilder';
import AutomataVisualizer from './components/AutomataVisualizer';
import GrammarBuilder from './components/GrammarBuilder';
import RegexPlayground from './components/RegexPlayground';

function App() {
    const [view, setView] = useState('automata');

    return (
        <div className="App">
            <nav className="app-nav">
                <button 
                    className={view === 'automata' ? 'active' : ''} 
                    onClick={() => setView('automata')}>
                    Automata Tools
                </button>
                <button 
                    className={view === 'grammar' ? 'active' : ''} 
                    onClick={() => setView('grammar')}>
                    Grammar Tools
                </button>
                <button 
                    className={view === 'regex' ? 'active' : ''} 
                    onClick={() => setView('regex')}>
                    Regex Tools
                </button>
            </nav>

            <main className="app-content">
                {view === 'automata' && (
                    <div className="automata-section">
                        <AutomataBuilder />
                        <AutomataVisualizer />
                    </div>
                )}
                {view === 'grammar' && (
                    <div className="grammar-section">
                        <GrammarBuilder />
                    </div>
                )}
                {view === 'regex' && (
                    <div className="regex-section">
                        <RegexPlayground />
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;