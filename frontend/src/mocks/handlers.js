import { rest } from 'msw';

export const handlers = [
  // Mock the builder start endpoint
  rest.post('/api/builder/start', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        status: 'success',
        state: {
          type: req.body.automata_type,
          states: [],
          transitions: [],
          visualization: 'digraph { /* Test visualization */ }',
          can_undo: false,
          can_redo: false
        }
      })
    );
  }),

  // Mock the builder action endpoint
  rest.post('/api/builder/action', (req, res, ctx) => {
    const { action, params } = req.body;
    
    if (action === 'add_state') {
      return res(
        ctx.status(200),
        ctx.json({
          status: 'success',
          state: {
            type: 'DFA',
            states: [{ name: params.name, is_initial: params.is_initial, is_final: params.is_final }],
            transitions: [],
            visualization: 'digraph { /* Updated visualization */ }',
            can_undo: true,
            can_redo: false
          }
        })
      );
    }

    if (action === 'simulate') {
      return res(
        ctx.status(200),
        ctx.json({
          status: 'success',
          state: {
            accepted: true,
            steps: [
              {
                step: 0,
                current_state: 'q0',
                remaining_input: 'ab',
                processed_input: ''
              },
              {
                step: 1,
                current_state: 'q1',
                remaining_input: 'b',
                processed_input: 'a'
              }
            ]
          }
        })
      );
    }

    return res(ctx.status(200), ctx.json({ status: 'success', state: {} }));
  }),

  // Mock the export endpoint
  rest.post('/api/export', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        jflap: '<?xml version="1.0" encoding="UTF-8"?><structure>...</structure>',
        dot: 'digraph { /* Test export */ }'
      })
    );
  }),

  // Mock the import endpoint
  rest.post('/api/import', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        status: 'success',
        automata: {
          type: 'DFA',
          definition: {
            states: [],
            transitions: []
          }
        }
      })
    );
  }),

  // Validate regex
  rest.post('/api/validate_regex', (req, res, ctx) => {
    const { regex } = req.body;
    const isValid = !regex.includes('**'); // Simple validation for testing
    return res(
      ctx.status(200),
      ctx.json({
        is_valid: isValid
      })
    );
  }),

  // Convert regex to DFA
  rest.post('/api/convert', (req, res, ctx) => {
    const { regex } = req.body;
    if (regex.includes('invalid')) {
      return res(
        ctx.status(400),
        ctx.json({
          error: 'Invalid regular expression'
        })
      );
    }
    return res(
      ctx.status(200),
      ctx.json({
        states: ['q0', 'q1', 'q2'],
        alphabet: ['a', 'b'],
        transitions: {
          'q0': { 'a': 'q1', 'b': 'q2' },
          'q1': { 'a': 'q1', 'b': 'q2' },
          'q2': { 'a': 'q1', 'b': 'q2' }
        },
        start_state: 'q0',
        accept_states: ['q2'],
        visualization_state: {
          nodes: [
            { id: 'q0', active: true, accepting: false },
            { id: 'q1', active: false, accepting: false },
            { id: 'q2', active: false, accepting: true }
          ],
          edges: [
            { from: 'q0', to: 'q1', symbol: 'a' },
            { from: 'q0', to: 'q2', symbol: 'b' },
            { from: 'q1', to: 'q1', symbol: 'a' },
            { from: 'q1', to: 'q2', symbol: 'b' },
            { from: 'q2', to: 'q1', symbol: 'a' },
            { from: 'q2', to: 'q2', symbol: 'b' }
          ]
        }
      })
    );
  })
];