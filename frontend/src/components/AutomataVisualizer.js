import React, { useCallback } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from '@reactflow/core';
import '@reactflow/core/dist/style.css';
import './AutomataVisualizer.css';

/**
 * Custom node component for automata states
 * @param {Object} data - Node data containing label, active state, and accepting state
 */
const CustomNode = ({ data }) => {
  return (
    <div className={`automata-node ${data.isActive ? 'active' : ''} ${data.isAccepting ? 'accepting' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-label">{data.label}</div>
      {data.isAccepting && <div className="accepting-marker">★</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
};

const nodeTypes = {
  custom: CustomNode,
};

/**
 * AutomataVisualizer component for rendering automaton state diagrams
 * @param {Object} visualizationState - The current visualization state of the automaton
 * @param {Array} visualizationState.nodes - Array of node objects with id, active, and accepting properties
 * @param {Array} visualizationState.edges - Array of edge objects with from, to, symbol, and active properties
 */
const AutomataVisualizer = ({ visualizationState }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  /**
   * Updates the visualization with new state data
   * @param {Object} state - New visualization state
   */
  const updateVisualization = useCallback((state) => {
    if (!state) return;

    // Convert visualization state to nodes and edges
    const newNodes = state.nodes.map((node, index) => ({
      id: node.id,
      type: 'custom',
      position: getNodePosition(index, state.nodes.length),
      data: {
        label: node.id,
        isActive: node.active,
        isAccepting: node.accepting,
      },
    }));

    const newEdges = state.edges.map((edge, index) => ({
      id: `${edge.from}-${edge.symbol}-${edge.to}`,
      source: edge.from,
      target: edge.to,
      label: edge.symbol,
      animated: edge.active,
      type: 'smoothstep',
      className: edge.active ? 'active-edge' : '',
    }));

    setNodes(newNodes);
    setEdges(newEdges);
  }, [setNodes, setEdges]);

  // Update visualization when state changes
  React.useEffect(() => {
    updateVisualization(visualizationState);
  }, [visualizationState, updateVisualization]);

  return (
    <div className="automata-visualizer">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

/**
 * Calculate node positions in a circular layout
 * @param {number} index - Index of the node
 * @param {number} total - Total number of nodes
 * @returns {Object} Position object with x and y coordinates
 */
const getNodePosition = (index, total) => {
  const radius = 200;
  const angle = (2 * Math.PI * index) / total;
  return {
    x: radius * Math.cos(angle) + radius,
    y: radius * Math.sin(angle) + radius,
  };
};

export default AutomataVisualizer;