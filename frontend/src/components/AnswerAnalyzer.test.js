import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AnswerAnalyzer from './AnswerAnalyzer';

describe('AnswerAnalyzer', () => {
  const mockQuestion = 'Create a DFA that accepts all strings ending in "ab"';
  const mockAnswer = 'q0 -> q1 [a] q1 -> q2 [b]';

  beforeEach(() => {
    render(<AnswerAnalyzer />);
  });

  test('renders input fields', () => {
    expect(screen.getByLabelText(/question/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/your answer/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /analyze/i })).toBeInTheDocument();
  });

  test('can submit question and answer for analysis', async () => {
    // Fill in the question
    const questionInput = screen.getByLabelText(/question/i);
    await userEvent.type(questionInput, mockQuestion);

    // Fill in the answer
    const answerInput = screen.getByLabelText(/your answer/i);
    await userEvent.type(answerInput, mockAnswer);

    // Submit for analysis
    const analyzeButton = screen.getByRole('button', { name: /analyze/i });
    fireEvent.click(analyzeButton);

    // Wait for analysis results
    await waitFor(() => {
      expect(screen.getByTestId('analysis-results')).toBeInTheDocument();
    });
  });

  test('displays error message when fields are empty', async () => {
    // Click analyze without filling fields
    const analyzeButton = screen.getByRole('button', { name: /analyze/i });
    fireEvent.click(analyzeButton);

    // Should show validation error
    expect(screen.getByText(/please fill in both/i)).toBeInTheDocument();
  });

  test('handles API errors gracefully', async () => {
    // Fill in the fields
    const questionInput = screen.getByLabelText(/question/i);
    const answerInput = screen.getByLabelText(/your answer/i);

    await userEvent.type(questionInput, 'Invalid question');
    await userEvent.type(answerInput, 'Invalid answer');

    // Submit for analysis
    const analyzeButton = screen.getByRole('button', { name: /analyze/i });
    fireEvent.click(analyzeButton);

    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });

  test('can reset form', async () => {
    // Fill in the fields
    const questionInput = screen.getByLabelText(/question/i);
    const answerInput = screen.getByLabelText(/your answer/i);

    await userEvent.type(questionInput, mockQuestion);
    await userEvent.type(answerInput, mockAnswer);

    // Click reset button
    const resetButton = screen.getByRole('button', { name: /reset/i });
    fireEvent.click(resetButton);

    // Check if fields are cleared
    expect(questionInput).toHaveValue('');
    expect(answerInput).toHaveValue('');
  });

  test('shows loading state during analysis', async () => {
    // Fill in the fields
    const questionInput = screen.getByLabelText(/question/i);
    const answerInput = screen.getByLabelText(/your answer/i);

    await userEvent.type(questionInput, mockQuestion);
    await userEvent.type(answerInput, mockAnswer);

    // Submit for analysis
    const analyzeButton = screen.getByRole('button', { name: /analyze/i });
    fireEvent.click(analyzeButton);

    // Check for loading indicator
    expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();

    // Wait for analysis to complete
    await waitFor(() => {
      expect(screen.queryByTestId('loading-indicator')).not.toBeInTheDocument();
    });
  });
});