import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RegexPlayground from './RegexPlayground';

describe('RegexPlayground', () => {
    beforeEach(() => {
        render(<RegexPlayground />);
    });

    test('renders regular expression input', () => {
        expect(screen.getByLabelText(/regular expression/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/test string/i)).toBeInTheDocument();
    });

    test('validates regular expression input', async () => {
        const regexInput = screen.getByLabelText(/regular expression/i);
        await userEvent.type(regexInput, 'a*b');

        await waitFor(() => {
            expect(screen.getByText(/valid regular expression/i)).toBeInTheDocument();
        });

        // Test invalid regex
        await userEvent.clear(regexInput);
        await userEvent.type(regexInput, 'a**b');

        await waitFor(() => {
            expect(screen.getByText(/invalid regular expression/i)).toBeInTheDocument();
        });
    });

    test('converts regex to DFA and shows visualization', async () => {
        const regexInput = screen.getByLabelText(/regular expression/i);
        await userEvent.type(regexInput, 'a*b');

        await waitFor(() => {
            expect(screen.getByText(/equivalent dfa/i)).toBeInTheDocument();
        });
    });

    test('can test input strings against the DFA', async () => {
        // First enter a valid regex
        const regexInput = screen.getByLabelText(/regular expression/i);
        await userEvent.type(regexInput, 'a*b');

        // Wait for DFA to be created
        await waitFor(() => {
            expect(screen.getByText(/equivalent dfa/i)).toBeInTheDocument();
        });

        // Enter and test a test string
        const testInput = screen.getByLabelText(/test string/i);
        await userEvent.type(testInput, 'aab');

        const testButton = screen.getByText(/test/i);
        fireEvent.click(testButton);

        await waitFor(() => {
            expect(screen.getByText(/test results/i)).toBeInTheDocument();
        });
    });

    test('handles API errors gracefully', async () => {
        const regexInput = screen.getByLabelText(/regular expression/i);
        await userEvent.type(regexInput, '[invalid'); // This should trigger an API error

        await waitFor(() => {
            expect(screen.getByText(/error/i)).toBeInTheDocument();
        });
    });

    test('test button is disabled when no automaton or test string', () => {
        const testButton = screen.getByText(/test/i);
        expect(testButton).toBeDisabled();
    });

    test('shows loading state during conversion', async () => {
        const regexInput = screen.getByLabelText(/regular expression/i);
        await userEvent.type(regexInput, 'a*b');

        await waitFor(() => {
            expect(screen.getByText(/converting/i)).toBeInTheDocument();
        });
    });
});