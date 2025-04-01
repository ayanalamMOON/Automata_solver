const API_URL = 'http://localhost:8000';

export const convertRegex = async (regex) => {
    const response = await fetch(`${API_URL}/convert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'regex', value: regex })
    });
    const data = await response.json();
    return data.dfa_svg;
};

export const explainAutomata = async (query) => {
    const response = await fetch(`${API_URL}/explain/${encodeURIComponent(query)}`);
    const data = await response.json();
    return data.explanation;
};

export const minimizeAutomaton = async (automatonData) => {
    const response = await fetch(`${API_URL}/minimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(automatonData)
    });
    return response.json();
};

export const exportAutomaton = async (automatonData, format) => {
    const response = await fetch(`${API_URL}/export/${format}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(automatonData)
    });
    const data = await response.json();
    return data.data;
};
