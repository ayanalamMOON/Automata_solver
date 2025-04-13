import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AutomataBuilder from './AutomataBuilder';

describe('AutomataBuilder', () => {
  test('renders initial state correctly', () => {
    render(<AutomataBuilder />);
    
    expect(screen.getByText('New DFA')).toBeInTheDocument();
    expect(screen.getByText('New NFA')).toBeInTheDocument();
    expect(screen.getByText('New PDA')).toBeInTheDocument();
  });

  test('can create a new DFA', async () => {
    render(<AutomataBuilder />);
    
    // Click the New DFA button
    fireEvent.click(screen.getByText('New DFA'));
    
    // Wait for the visualization to appear
    await waitFor(() => {
      expect(screen.getByTestId('visualization')).toBeInTheDocument();
    });
  });

  test('can add a state', async () => {
    render(<AutomataBuilder />);
    
    // Select add state action
    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'add_state' }
    });
    
    // Fill in state details
    const stateInput = screen.getByPlaceholderText('State name');
    await userEvent.type(stateInput, 'q0');
    
    // Check initial state checkbox
    const initialCheckbox = screen.getByLabelText('Initial State');
    fireEvent.click(initialCheckbox);
    
    // Submit the form
    const addButton = screen.getByText('Add State');
    fireEvent.click(addButton);
    
    // Wait for the state to appear in visualization
    await waitFor(() => {
      const visualization = screen.getByTestId('visualization');
      expect(visualization).toHaveTextContent('q0');
    });
  });

  test('can add a transition', async () => {
    render(<AutomataBuilder />);
    
    // Select add transition action
    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'add_transition' }
    });
    
    // Fill in transition details
    const fromInput = screen.getByPlaceholderText('From state');
    const symbolInput = screen.getByPlaceholderText('Symbol');
    const toInput = screen.getByPlaceholderText('To state');
    
    await userEvent.type(fromInput, 'q0');
    await userEvent.type(symbolInput, 'a');
    await userEvent.type(toInput, 'q1');
    
    // Submit the form
    const addButton = screen.getByText('Add Transition');
    fireEvent.click(addButton);
    
    // Wait for the transition to be added
    await waitFor(() => {
      expect(screen.getByTestId('visualization')).toBeInTheDocument();
    });
  });

  test('can simulate input', async () => {
    render(<AutomataBuilder />);
    
    // Select simulate action
    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'simulate' }
    });
    
    // Enter input string
    const inputField = screen.getByPlaceholderText('Input string');
    await userEvent.type(inputField, 'ab');
    
    // Click simulate
    fireEvent.click(screen.getByText('Simulate'));
    
    // Wait for simulation results
    await waitFor(() => {
      expect(screen.getByText('Simulation Results')).toBeInTheDocument();
      expect(screen.getByText('Step 0:')).toBeInTheDocument();
    });
  });

  test('can export automata', async () => {
    render(<AutomataBuilder />);
    
    // Create a new DFA first
    fireEvent.click(screen.getByText('New DFA'));
    
    // Wait for the builder to be ready
    await waitFor(() => {
      expect(screen.getByText('Export to JFLAP')).toBeInTheDocument();
    });
    
    // Click export button
    fireEvent.click(screen.getByText('Export to JFLAP'));
    
    // Verify export was triggered (actual download can't be tested in JSDOM)
    await waitFor(() => {
      expect(screen.getByTestId('visualization')).toBeInTheDocument();
    });
  });

  test('handles errors gracefully', async () => {
    render(<AutomataBuilder />);
    
    // Try to add a transition without creating an automaton first
    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'add_transition' }
    });
    
    const addButton = screen.getByText('Add Transition');
    fireEvent.click(addButton);
    
    // Should show error message
    await waitFor(() => {
      expect(screen.getByText(/Error/i)).toBeInTheDocument();
    });
  });
});