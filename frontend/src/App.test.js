import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

describe('App', () => {
  test('renders header', () => {
    render(<App />);
    const headerElement = screen.getByText(/Automata Solver/i);
    expect(headerElement).toBeInTheDocument();
  });

  test('switches between builder and analyzer views', () => {
    render(<App />);
    
    // Should start with builder view
    expect(screen.getByText('New DFA')).toBeInTheDocument();
    
    // Switch to analyzer
    fireEvent.click(screen.getByText('Answer Analyzer'));
    expect(screen.getByLabelText(/question/i)).toBeInTheDocument();
    
    // Switch back to builder
    fireEvent.click(screen.getByText('Automata Builder'));
    expect(screen.getByText('New DFA')).toBeInTheDocument();
  });

  test('maintains active state in navigation', () => {
    render(<App />);
    
    // Builder should be active initially
    const builderButton = screen.getByText('Automata Builder');
    expect(builderButton).toHaveClass('active');
    
    // Switch to analyzer
    fireEvent.click(screen.getByText('Answer Analyzer'));
    expect(screen.getByText('Answer Analyzer')).toHaveClass('active');
    expect(builderButton).not.toHaveClass('active');
  });
});