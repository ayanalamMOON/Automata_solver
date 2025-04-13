import React, { useState } from 'react';
import './StepVisualizer.css';

const StepVisualizer = ({ simulationData }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const steps = simulationData?.steps || [];
  const totalSteps = steps.length;
  
  // No data to visualize
  if (!simulationData || !steps.length) {
    return <div className="step-visualizer empty">No simulation data available</div>;
  }

  const goToStep = (stepIndex) => {
    if (stepIndex >= 0 && stepIndex < totalSteps) {
      setCurrentStep(stepIndex);
    }
  };

  const step = steps[currentStep];
  const isLastStep = currentStep === totalSteps - 1;
  const isFirstStep = currentStep === 0;
  const accepted = simulationData.accepted;

  return (
    <div className="step-visualizer">
      <h3>Step-by-Step Execution</h3>
      
      <div className="step-navigation">
        <button 
          onClick={() => goToStep(0)} 
          disabled={isFirstStep}
          title="Go to start">
          &lt;&lt;
        </button>
        <button 
          onClick={() => goToStep(currentStep - 1)} 
          disabled={isFirstStep}
          title="Previous step">
          &lt;
        </button>
        <span className="step-counter">
          Step {currentStep} of {totalSteps - 1}
        </span>
        <button 
          onClick={() => goToStep(currentStep + 1)} 
          disabled={isLastStep}
          title="Next step">
          &gt;
        </button>
        <button 
          onClick={() => goToStep(totalSteps - 1)} 
          disabled={isLastStep}
          title="Go to end">
          &gt;&gt;
        </button>
      </div>

      <div className="step-details">
        <div className="state-info">
          <div className="state-label">Current State:</div>
          <div className={`state-value ${step.is_accepting ? 'accepting-state' : ''}`}>
            {step.current_state}
            {step.is_accepting && <span className="state-marker">⭐</span>}
          </div>
        </div>

        {step.input_symbol !== null && (
          <div className="transition-info">
            <span className="symbol-info">
              Reading: <strong>{step.input_symbol || 'ε'}</strong>
            </span>
            <span className="transition-arrow">→</span>
            <span className={`next-state ${step.is_accepting ? 'accepting-state' : ''}`}>
              {step.next_state || 'None'}
            </span>
          </div>
        )}

        <div className="tape-visualization">
          <div className="processed-input">
            {step.processed_input}
          </div>
          <div className="head-position">↓</div>
          <div className="remaining-input">
            {step.remaining_input}
          </div>
        </div>

        {isLastStep && (
          <div className={`final-result ${accepted ? 'accepted' : 'rejected'}`}>
            String {accepted ? 'ACCEPTED' : 'REJECTED'}
          </div>
        )}

        {step.is_error && (
          <div className="error-message">
            {step.error_message}
          </div>
        )}
      </div>

      <div className="step-controls">
        <button
          className="auto-play-button"
          onClick={() => {
            const timer = setInterval(() => {
              setCurrentStep(prev => {
                if (prev >= totalSteps - 1) {
                  clearInterval(timer);
                  return prev;
                }
                return prev + 1;
              });
            }, 800);
          }}
          disabled={isLastStep}
        >
          Auto Play
        </button>
        <button
          className="reset-button"
          onClick={() => setCurrentStep(0)}
          disabled={isFirstStep}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default StepVisualizer;