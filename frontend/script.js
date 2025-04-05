// Global variables
const backendBaseUrl = 'http://localhost:8000';
let currentAutomaton = null;
let currentTool = null;
let automatonHistory = [];
let testHistory = [];
let isDarkTheme = false;
let zoomInstance = null;

// DOM ready function
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Initialize UI components
    setupTabs();
    setupPanelToggles();
    setupModals();
    setupThemeToggle();
    setupDropzone();
    setupVisualizationTools();
    setupFormHandlers();
    initializeAutomatonGraph();
    
    // Check for saved theme preference
    loadThemePreference();
    
    // Welcome toast
    showToast('Welcome to Automata Solver!', 'info');
}

// Tab functionality
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabContainer = button.closest('.tabs').parentElement;
            const tabName = button.dataset.tab;
            
            // Remove active class from all tabs and hide all content
            tabContainer.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            tabContainer.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));
            
            // Activate the clicked tab
            button.classList.add('active');
            tabContainer.querySelector(`#${tabName}-tab`).classList.remove('hidden');
            
            // Animate the tab appearance
            animateElement(tabContainer.querySelector(`#${tabName}-tab`), 'fadeIn');
        });
    });
    
    // Same for help tabs
    const helpTabs = document.querySelectorAll('.help-tab');
    helpTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.help-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.help-content').forEach(c => c.classList.add('hidden'));
            
            tab.classList.add('active');
            const contentId = `${tab.dataset.tab}-help`;
            document.getElementById(contentId).classList.remove('hidden');
        });
    });
}

// Panel toggle functionality
function setupPanelToggles() {
    const toggleButtons = document.querySelectorAll('.panel-toggle');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const panel = button.closest('.panel');
            const content = panel.querySelector('.panel-content');
            const icon = button.querySelector('i');
            
            if (content.style.display === 'none') {
                // Expand panel
                slideDown(content);
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-up');
            } else {
                // Collapse panel
                slideUp(content);
                icon.classList.remove('fa-chevron-up');
                icon.classList.add('fa-chevron-down');
            }
        });
    });
}

// Modal functionality
function setupModals() {
    // Export modal
    document.getElementById('export-btn').addEventListener('click', () => {
        showModal('export-modal');
    });
    
    // Examples modal
    document.getElementById('example-automata').addEventListener('click', () => {
        showModal('examples-modal');
    });
    
    // Help modal
    document.getElementById('help-button').addEventListener('click', () => {
        showModal('help-modal');
    });
    
    // Close buttons for all modals
    document.querySelectorAll('.close-modal, .cancel-btn').forEach(button => {
        button.addEventListener('click', () => {
            const modal = button.closest('.modal');
            hideModal(modal.id);
        });
    });
    
    // Example selection
    document.querySelectorAll('.example-card').forEach(card => {
        card.addEventListener('click', () => {
            const regexValue = card.dataset.example;
            document.getElementById('regex-input').value = regexValue;
            hideModal('examples-modal');
            
            // Ensure the regex tab is active
            const regexTab = document.querySelector('[data-tab="regex"]');
            regexTab.click();
            
            // Show a success message
            showToast(`Example "${regexValue}" loaded`, 'success');
        });
    });
    
    // Export functionality
    document.getElementById('download-btn').addEventListener('click', () => {
        const format = document.querySelector('input[name="export-format"]:checked').value;
        const includeDetails = document.querySelector('input[value="details"]').checked;
        const includeExplanation = document.querySelector('input[value="explanation"]').checked;
        
        if (!currentAutomaton) {
            showToast('No automaton to export', 'error');
            return;
        }
        
        exportAutomaton(currentAutomaton, format, includeDetails, includeExplanation);
        hideModal('export-modal');
    });
}

// Theme toggle functionality
function setupThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    
    themeToggle.addEventListener('click', () => {
        toggleTheme();
    });
}

// Toggle between light and dark themes
function toggleTheme() {
    isDarkTheme = !isDarkTheme;
    
    if (isDarkTheme) {
        document.body.classList.add('dark-theme');
        document.getElementById('theme-toggle').innerHTML = '<i class="fas fa-sun"></i>';
    } else {
        document.body.classList.remove('dark-theme');
        document.getElementById('theme-toggle').innerHTML = '<i class="fas fa-moon"></i>';
    }
    
    // Save theme preference
    localStorage.setItem('darkTheme', isDarkTheme);
    
    // Re-render automaton if exists (for theme-specific colors)
    if (currentAutomaton) {
        renderAutomaton(currentAutomaton);
    }
}

// Load theme preference from local storage
function loadThemePreference() {
    const savedTheme = localStorage.getItem('darkTheme');
    
    if (savedTheme === 'true') {
        toggleTheme();
    }
}

// Dropzone setup for image uploads
function setupDropzone() {
    const dropzone = document.getElementById('image-dropzone');
    const fileInput = document.getElementById('image-upload');
    const previewContainer = document.querySelector('.uploaded-image-preview');
    const previewImage = document.getElementById('uploaded-image');
    const removeButton = document.getElementById('remove-image');
    const processButton = document.getElementById('process-image');
    
    // Open file dialog when dropzone is clicked
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', () => {
        handleImageUpload(fileInput.files[0]);
    });
    
    // Handle drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('active');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('active');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Remove uploaded image
    removeButton.addEventListener('click', () => {
        previewContainer.classList.add('hidden');
        dropzone.classList.remove('hidden');
        fileInput.value = '';
        previewImage.src = '';
    });
    
    // Process image with OCR
    processButton.addEventListener('click', () => {
        if (fileInput.files.length === 0) {
            showToast('Please upload an image first', 'error');
            return;
        }
        
        processImageWithOCR(fileInput.files[0]);
    });
}

// Handle image upload and preview
function handleImageUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
        showToast('Please upload a valid image file', 'error');
        return;
    }
    
    const dropzone = document.getElementById('image-dropzone');
    const previewContainer = document.querySelector('.uploaded-image-preview');
    const previewImage = document.getElementById('uploaded-image');
    
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropzone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        animateElement(previewContainer, 'fadeIn');
    };
    
    reader.readAsDataURL(file);
}

// Process image with OCR
function processImageWithOCR(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    showToast('Processing image...', 'info');
    
    // Show loading spinner
    const explanationLoader = document.getElementById('explanation-loader');
    explanationLoader.classList.remove('hidden');
    
    fetch(`${backendBaseUrl}/upload`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        explanationLoader.classList.add('hidden');
        
        if (data.error) {
            showToast(`Error: ${data.error}`, 'error');
            return;
        }
        
        // If we got extracted text
        if (data.text) {
            // Switch to explanation tab
            document.querySelector('[data-tab="explanation"]').click();
            
            // Display the extracted text
            const explanationContent = document.getElementById('explanation-content');
            explanationContent.innerHTML = `
                <div class="extracted-text">
                    <h4>Extracted Text from Image</h4>
                    <pre>${data.text}</pre>
                </div>
            `;
            
            // Try to detect regex patterns
            detectRegexPatterns(data.text);
        }
    })
    .catch(error => {
        explanationLoader.classList.add('hidden');
        showToast(`Error: ${error.message}`, 'error');
    });
}

// Try to detect regex patterns in extracted text
function detectRegexPatterns(text) {
    // Look for potential regex patterns in the text
    // This is a simple example, you'd want more sophisticated detection in production
    const regexPatterns = [
        /regex:?\s*([a-z0-9\(\)\[\]\{\}\.\*\+\?\|\^\$\\]+)/i,
        /regular expression:?\s*([a-z0-9\(\)\[\]\{\}\.\*\+\?\|\^\$\\]+)/i,
        /pattern:?\s*([a-z0-9\(\)\[\]\{\}\.\*\+\?\|\^\$\\]+)/i
    ];
    
    for (const pattern of regexPatterns) {
        const match = text.match(pattern);
        if (match && match[1]) {
            const detectedRegex = match[1].trim();
            
            // Ask user if they want to use this regex
            if (confirm(`Detected regex pattern: "${detectedRegex}". Use this pattern?`)) {
                document.getElementById('regex-input').value = detectedRegex;
                document.querySelector('[data-tab="regex"]').click();
                return;
            }
        }
    }
}

// Visualization tools setup
function setupVisualizationTools() {
    // Zoom controls
    document.getElementById('zoom-in').addEventListener('click', () => {
        if (zoomInstance) {
            zoomInstance.zoomIn();
        }
    });
    
    document.getElementById('zoom-out').addEventListener('click', () => {
        if (zoomInstance) {
            zoomInstance.zoomOut();
        }
    });
    
    document.getElementById('reset-zoom').addEventListener('click', () => {
        if (zoomInstance) {
            zoomInstance.reset();
        }
    });
    
    // Fullscreen
    document.getElementById('fullscreen-btn').addEventListener('click', () => {
        const graphContainer = document.getElementById('automaton-graph');
        
        if (!document.fullscreenElement) {
            if (graphContainer.requestFullscreen) {
                graphContainer.requestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    });
}

// Initialize the automaton graph area
function initializeAutomatonGraph() {
    const graphContainer = document.getElementById('automaton-graph');
    
    // Initialize panzoom for the graph container
    // We'll attach this to an SVG element once we have one
}

// Setup form handlers
function setupFormHandlers() {
    // Validate regex
    document.getElementById('validate-regex').addEventListener('click', () => {
        const regexInput = document.getElementById('regex-input').value.trim();
        
        if (!regexInput) {
            showToast('Please enter a regular expression', 'error');
            return;
        }
        
        validateRegex(regexInput);
    });
    
    // Convert regex to automaton
    document.getElementById('convert-regex').addEventListener('click', () => {
        const regexInput = document.getElementById('regex-input').value.trim();
        
        if (!regexInput) {
            showToast('Please enter a regular expression', 'error');
            return;
        }
        
        convertRegexToAutomaton(regexInput);
    });
    
    // Build automaton from formal definition
    document.getElementById('build-automaton').addEventListener('click', () => {
        buildAutomatonFromDefinition();
    });
    
    // State and alphabet input change handlers
    document.getElementById('states-input').addEventListener('input', updateStateSelections);
    document.getElementById('alphabet-input').addEventListener('input', updateAlphabetSelections);
    
    // Transitions builder - add transition
    document.querySelector('.add-transition').addEventListener('click', addTransitionRow);
    
    // Minimize automaton
    document.getElementById('minimize-automaton').addEventListener('click', () => {
        if (!currentAutomaton) {
            showToast('No automaton to minimize', 'error');
            return;
        }
        
        minimizeAutomaton(currentAutomaton);
    });
    
    // Complement automaton
    document.getElementById('complement-automaton').addEventListener('click', () => {
        if (!currentAutomaton) {
            showToast('No automaton to complement', 'error');
            return;
        }
        
        complementAutomaton(currentAutomaton);
    });
    
    // Test string
    document.getElementById('test-string-btn').addEventListener('click', () => {
        const testString = document.getElementById('test-string').value;
        
        if (!currentAutomaton) {
            showToast('No automaton to test against', 'error');
            return;
        }
        
        testStringAgainstAutomaton(testString, currentAutomaton);
    });
    
    // Visual editor tools
    document.querySelectorAll('.tool-btn[data-tool]').forEach(button => {
        button.addEventListener('click', () => {
            currentTool = button.dataset.tool;
            
            // Update active tool UI
            document.querySelectorAll('.tool-btn[data-tool]').forEach(btn => {
                btn.classList.remove('active');
            });
            button.classList.add('active');
        });
    });
    
    // New project
    document.getElementById('new-project').addEventListener('click', () => {
        if (confirm('Create a new project? Any unsaved changes will be lost.')) {
            clearAllInputs();
            resetAutomatonDisplay();
            showToast('Created new project', 'info');
        }
    });
    
    // Save project
    document.getElementById('save-project').addEventListener('click', () => {
        saveProject();
    });
}

// Validate regex
function validateRegex(regex) {
    fetch(`${backendBaseUrl}/validate_regex`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            type: 'regex',
            value: regex
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.is_valid) {
            showToast('Regex is valid', 'success');
        } else {
            showToast('Invalid regex syntax', 'error');
        }
    })
    .catch(error => {
        showToast(`Error: ${error.message}`, 'error');
    });
}

// Convert regex to automaton
function convertRegexToAutomaton(regex) {
    // Show loading state
    showToast('Converting regex to automaton...', 'info');
    const graphContainer = document.getElementById('automaton-graph');
    graphContainer.innerHTML = `
        <div class="loading-spinner">
            <i class="fas fa-spinner fa-spin"></i>
            <span>Generating automaton...</span>
        </div>
    `;
    
    fetch(`${backendBaseUrl}/convert`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            type: 'regex',
            value: regex
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showToast(`Error: ${data.error}`, 'error');
            return;
        }
        
        if (data.dfa_svg) {
            // Display the SVG
            graphContainer.innerHTML = data.dfa_svg;
            
            // Store the automaton data
            currentAutomaton = {
                type: 'DFA',
                regex: regex,
                svg: data.dfa_svg,
                // We'd ideally get these from the backend
                states: extractStatesFromSVG(data.dfa_svg),
                alphabet: extractAlphabetFromSVG(data.dfa_svg),
                initialState: extractInitialStateFromSVG(data.dfa_svg),
                finalStates: extractFinalStatesFromSVG(data.dfa_svg)
            };
            
            // Update automaton details
            updateAutomatonDetails(currentAutomaton);
            
            // Setup pan and zoom
            setupPanZoom();
            
            // Get explanation
            getAutomatonExplanation(regex);
            
            // Add to history
            addToAutomatonHistory(currentAutomaton);
            
            showToast('Automaton generated successfully', 'success');
        }
    })
    .catch(error => {
        graphContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-circle empty-icon"></i>
                <p>Error: ${error.message}</p>
            </div>
        `;
        showToast(`Error: ${error.message}`, 'error');
    });
}

// Get automaton explanation
function getAutomatonExplanation(regex) {
    const explanationLoader = document.getElementById('explanation-loader');
    const explanationContent = document.getElementById('explanation-content');
    
    explanationLoader.classList.remove('hidden');
    
    fetch(`${backendBaseUrl}/explain/${encodeURIComponent(regex)}`)
    .then(response => response.json())
    .then(data => {
        explanationLoader.classList.add('hidden');
        
        if (data.explanation) {
            explanationContent.innerHTML = `
                <div class="explanation-text">
                    ${marked.parse(data.explanation)}
                </div>
            `;
        } else {
            explanationContent.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-lightbulb empty-icon"></i>
                    <p>No explanation available</p>
                </div>
            `;
        }
    })
    .catch(error => {
        explanationLoader.classList.add('hidden');
        explanationContent.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-circle empty-icon"></i>
                <p>Error loading explanation: ${error.message}</p>
            </div>
        `;
    });
}

// Update state selections when states input changes
function updateStateSelections() {
    const statesInput = document.getElementById('states-input').value;
    const states = statesInput.split(',').map(s => s.trim()).filter(s => s);
    
    // Update initial state dropdown
    const initialStateSelect = document.getElementById('initial-state-select');
    initialStateSelect.innerHTML = '<option value="">Select a state</option>';
    
    states.forEach(state => {
        const option = document.createElement('option');
        option.value = state;
        option.textContent = state;
        initialStateSelect.appendChild(option);
    });
    
    // Update final states checkboxes
    const finalStatesContainer = document.getElementById('final-states-checkboxes');
    finalStatesContainer.innerHTML = '';
    
    states.forEach(state => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = state;
        checkbox.name = 'final-states';
        
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(state));
        
        finalStatesContainer.appendChild(label);
    });
    
    // Update transition state dropdowns
    updateTransitionStateOptions();
}

// Update alphabet selections when alphabet input changes
function updateAlphabetSelections() {
    const alphabetInput = document.getElementById('alphabet-input').value;
    const alphabet = alphabetInput.split(',').map(s => s.trim()).filter(s => s);
    
    // Update transitions input symbol options
    const inputSymbolSelects = document.querySelectorAll('.input-symbol');
    
    inputSymbolSelects.forEach(select => {
        const currentValue = select.value;
        select.innerHTML = '<option value="">Input</option>';
        
        alphabet.forEach(symbol => {
            const option = document.createElement('option');
            option.value = symbol;
            option.textContent = symbol;
            select.appendChild(option);
        });
        
        // Restore selected value if possible
        if (currentValue && alphabet.includes(currentValue)) {
            select.value = currentValue;
        }
    });
}

// Update transition state options
function updateTransitionStateOptions() {
    const statesInput = document.getElementById('states-input').value;
    const states = statesInput.split(',').map(s => s.trim()).filter(s => s);
    
    const fromStateSelects = document.querySelectorAll('.from-state');
    const toStateSelects = document.querySelectorAll('.to-state');
    
    // Update 'from' state dropdowns
    fromStateSelects.forEach(select => {
        const currentValue = select.value;
        select.innerHTML = '<option value="">From</option>';
        
        states.forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            select.appendChild(option);
        });
        
        // Restore selected value if possible
        if (currentValue && states.includes(currentValue)) {
            select.value = currentValue;
        }
    });
    
    // Update 'to' state dropdowns
    toStateSelects.forEach(select => {
        const currentValue = select.value;
        select.innerHTML = '<option value="">To</option>';
        
        states.forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            select.appendChild(option);
        });
        
        // Restore selected value if possible
        if (currentValue && states.includes(currentValue)) {
            select.value = currentValue;
        }
    });
}

// Add transition row
function addTransitionRow() {
    const transitionsBuilder = document.getElementById('transitions-builder');
    const newRow = document.createElement('div');
    newRow.className = 'transition-row';
    
    const fromStateSelect = document.createElement('select');
    fromStateSelect.className = 'from-state';
    fromStateSelect.innerHTML = '<option value="">From</option>';
    
    const inputSymbolSelect = document.createElement('select');
    inputSymbolSelect.className = 'input-symbol';
    inputSymbolSelect.innerHTML = '<option value="">Input</option>';
    
    const toStateSelect = document.createElement('select');
    toStateSelect.className = 'to-state';
    toStateSelect.innerHTML = '<option value="">To</option>';
    
    const removeButton = document.createElement('button');
    removeButton.className = 'remove-transition';
    removeButton.innerHTML = '<i class="fas fa-minus"></i>';
    removeButton.addEventListener('click', function() {
        transitionsBuilder.removeChild(newRow);
    });
    
    newRow.appendChild(fromStateSelect);
    newRow.appendChild(inputSymbolSelect);
    newRow.appendChild(toStateSelect);
    newRow.appendChild(removeButton);
    
    transitionsBuilder.appendChild(newRow);
    
    // Fill in states and input symbols
    updateStateSelections();
    updateAlphabetSelections();
}

// Build automaton from formal definition
function buildAutomatonFromDefinition() {
    const statesInput = document.getElementById('states-input').value;
    const alphabetInput = document.getElementById('alphabet-input').value;
    const initialState = document.getElementById('initial-state-select').value;
    const finalStatesElements = document.querySelectorAll('input[name="final-states"]:checked');
    
    // Validate inputs
    if (!statesInput || !alphabetInput || !initialState || finalStatesElements.length === 0) {
        showToast('Please fill in all required fields', 'error');
        return;
    }
    
    const states = statesInput.split(',').map(s => s.trim()).filter(s => s);
    const alphabet = alphabetInput.split(',').map(s => s.trim()).filter(s => s);
    const finalStates = Array.from(finalStatesElements).map(el => el.value);
    
    // Collect transitions
    const transitions = {};
    const transitionRows = document.querySelectorAll('.transition-row');
    let hasValidTransitions = false;
    
    states.forEach(state => {
        transitions[state] = {};
    });
    
    transitionRows.forEach(row => {
        const fromState = row.querySelector('.from-state').value;
        const inputSymbol = row.querySelector('.input-symbol').value;
        const toState = row.querySelector('.to-state').value;
        
        if (fromState && inputSymbol && toState) {
            transitions[fromState][inputSymbol] = toState;
            hasValidTransitions = true;
        }
    });
    
    if (!hasValidTransitions) {
        showToast('Please add at least one valid transition', 'error');
        return;
    }
    
    // Create the automaton object
    const automaton = {
        states: states,
        input_symbols: alphabet,
        transitions: transitions,
        initial_state: initialState,
        final_states: finalStates
    };
    
    // Render the automaton
    showToast('Building automaton...', 'info');
    renderAutomaton(automaton);
}

// Render an automaton
function renderAutomaton(automaton) {
    // Store the current automaton
    currentAutomaton = automaton;
    
    // Send to backend for visualization
    fetch(`${backendBaseUrl}/export/svg`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(automaton)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showToast(`Error: ${data.error}`, 'error');
            return;
        }
        
        // Display the SVG
        const graphContainer = document.getElementById('automaton-graph');
        graphContainer.innerHTML = data.data;
        
        // Update automaton details
        updateAutomatonDetails(automaton);
        
        // Setup pan and zoom
        setupPanZoom();
        
        // Add to history
        addToAutomatonHistory(automaton);
        
        showToast('Automaton rendered successfully', 'success');
    })
    .catch(error => {
        showToast(`Error: ${error.message}`, 'error');
    });
}

// Minimize automaton
function minimizeAutomaton(automaton) {
    showToast('Minimizing automaton...', 'info');
    
    fetch(`${backendBaseUrl}/minimize`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(automaton)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showToast(`Error: ${data.error}`, 'error');
            return;
        }
        
        // Render the minimized automaton
        renderAutomaton(data);
        showToast('Automaton minimized', 'success');
    })
    .catch(error => {
        showToast(`Error: ${error.message}`, 'error');
    });
}

// Complement automaton
function complementAutomaton(automaton) {
    // Create a copy of the automaton
    const complemented = JSON.parse(JSON.stringify(automaton));
    
    // Swap final and non-final states
    const allStates = new Set(complemented.states);
    const finalStates = new Set(complemented.final_states);
    const newFinalStates = [];
    
    allStates.forEach(state => {
        if (!finalStates.has(state)) {
            newFinalStates.push(state);
        }
    });
    
    complemented.final_states = newFinalStates;
    
    // Render the complemented automaton
    renderAutomaton(complemented);
    showToast('Automaton complemented', 'success');
}

// Test string against automaton
function testStringAgainstAutomaton(testString, automaton) {
    const testResult = document.querySelector('.test-result');
    const successIcon = document.querySelector('.success-icon');
    const errorIcon = document.querySelector('.error-icon');
    const resultText = document.getElementById('test-result-text');
    
    // Show loading state
    testResult.classList.remove('hidden');
    testResult.classList.remove('success', 'error');
    successIcon.classList.add('hidden');
    errorIcon.classList.add('hidden');
    resultText.textContent = 'Testing string...';
    
    // Send to backend for testing
    fetch(`${backendBaseUrl}/test_string`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            automaton: automaton,
            string: testString
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            testResult.classList.add('error');
            testResult.classList.remove('success');
            successIcon.classList.add('hidden');
            errorIcon.classList.remove('hidden');
            resultText.textContent = `Error: ${data.error}`;
            return;
        }
        
        const isAccepted = data.is_accepted;
        
        if (isAccepted) {
            testResult.classList.add('success');
            testResult.classList.remove('error');
            successIcon.classList.remove('hidden');
            errorIcon.classList.add('hidden');
            resultText.textContent = `The string "${testString}" is accepted by the automaton.`;
            
            // Show execution path if available
            if (data.execution_path && data.execution_path.length > 0) {
                const pathInfo = document.createElement('div');
                pathInfo.className = 'execution-path';
                pathInfo.innerHTML = `
                    <p class="path-title">Execution path:</p>
                    <div class="path-steps">
                        ${data.execution_path.map((step, index) => `
                            <div class="path-step">
                                <span class="step-number">${index + 1}</span>
                                <span class="from-state">${step.from_state}</span>
                                <span class="step-symbol">${step.symbol}</span>
                                <i class="fas fa-long-arrow-alt-right"></i>
                                <span class="to-state">${step.to_state}</span>
                            </div>
                        `).join('')}
                    </div>
                `;
                resultText.appendChild(pathInfo);
                
                // Highlight the path in the graph if possible
                highlightExecutionPath(data.execution_path);
            }
        } else {
            testResult.classList.add('error');
            testResult.classList.remove('success');
            successIcon.classList.add('hidden');
            errorIcon.classList.remove('hidden');
            resultText.textContent = `The string "${testString}" is rejected by the automaton.`;
            
            // Show reason if available
            if (data.execution_path && data.execution_path.length > 0) {
                const pathInfo = document.createElement('div');
                pathInfo.className = 'execution-path';
                pathInfo.innerHTML = `
                    <p class="path-title">Execution path (failed at step ${data.execution_path.length}):</p>
                    <div class="path-steps">
                        ${data.execution_path.map((step, index) => `
                            <div class="path-step ${index === data.execution_path.length - 1 ? 'failed-step' : ''}">
                                <span class="step-number">${index + 1}</span>
                                <span class="from-state">${step.from_state}</span>
                                <span class="step-symbol">${step.symbol}</span>
                                <i class="fas fa-long-arrow-alt-right"></i>
                                <span class="to-state">${step.to_state || '?'}</span>
                            </div>
                        `).join('')}
                    </div>
                `;
                resultText.appendChild(pathInfo);
            }
        }
        
        // Add to test history
        addToTestHistory(testString, isAccepted);
        
        // Clear the input
        document.getElementById('test-string').value = '';
    })
    .catch(error => {
        testResult.classList.add('error');
        testResult.classList.remove('success');
        successIcon.classList.add('hidden');
        errorIcon.classList.remove('hidden');
        resultText.textContent = `Error: ${error.message}`;
    });
}

// Highlight execution path in the graph
function highlightExecutionPath(executionPath) {
    // Reset any previous highlights
    const svg = document.querySelector('#automaton-graph svg');
    if (!svg) return;
    
    // Remove previous highlights
    svg.querySelectorAll('.highlighted-node, .highlighted-edge').forEach(el => {
        el.classList.remove('highlighted-node', 'highlighted-edge');
    });
    
    // Apply highlights - this is a simple implementation
    // In a full implementation, we would need to match the state and transition IDs in the SVG
    executionPath.forEach(step => {
        // Try to find and highlight nodes
        const fromNode = svg.querySelector(`[id*="${step.from_state}"]`);
        const toNode = svg.querySelector(`[id*="${step.to_state}"]`);
        
        if (fromNode) fromNode.classList.add('highlighted-node');
        if (toNode) toNode.classList.add('highlighted-node');
        
        // Try to find and highlight edges
        // This is approximate - would need better ID matching in production
        const edges = svg.querySelectorAll('g.edge');
        edges.forEach(edge => {
            const title = edge.querySelector('title');
            if (title && title.textContent.includes(`${step.from_state} -> ${step.to_state}`)) {
                edge.classList.add('highlighted-edge');
            }
        });
    });
    
    // Add CSS for highlights if not already present
    if (!document.getElementById('highlight-styles')) {
        const style = document.createElement('style');
        style.id = 'highlight-styles';
        style.textContent = `
            .highlighted-node ellipse {
                fill: #ff9500 !important;
                stroke-width: 2px !important;
            }
            .highlighted-edge path {
                stroke: #ff9500 !important;
                stroke-width: 2px !important;
            }
            .execution-path {
                margin-top: 1rem;
                padding: 0.8rem;
                background-color: rgba(0,0,0,0.05);
                border-radius: 4px;
            }
            .path-title {
                font-weight: 500;
                margin-bottom: 0.5rem;
            }
            .path-steps {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            .path-step {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem;
                background-color: rgba(255,255,255,0.5);
                border-radius: 4px;
            }
            .step-number {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                background-color: #6200ea;
                color: white;
                font-weight: 500;
                font-size: 0.8rem;
            }
            .from-state, .to-state {
                font-family: 'Roboto Mono', monospace;
                padding: 0.2rem 0.4rem;
                background-color: rgba(0,0,0,0.05);
                border-radius: 4px;
            }
            .step-symbol {
                font-family: 'Roboto Mono', monospace;
                font-weight: 500;
            }
            .failed-step {
                background-color: rgba(176, 0, 32, 0.1);
            }
            .dark-theme .execution-path {
                background-color: rgba(255,255,255,0.05);
            }
            .dark-theme .path-step {
                background-color: rgba(255,255,255,0.05);
            }
            .dark-theme .from-state, .dark-theme .to-state {
                background-color: rgba(255,255,255,0.1);
            }
        `;
        document.head.appendChild(style);
    }
}

// Add to test history
function addToTestHistory(testString, isAccepted) {
    const testHistoryList = document.getElementById('test-history-list');
    const listItem = document.createElement('li');
    listItem.className = isAccepted ? 'accepted' : 'rejected';
    
    const icon = document.createElement('i');
    icon.className = isAccepted ? 'fas fa-check' : 'fas fa-times';
    
    listItem.appendChild(icon);
    listItem.appendChild(document.createTextNode(` "${testString}"`));
    
    testHistoryList.prepend(listItem);
    
    // Store in history array
    testHistory.push({ string: testString, accepted: isAccepted });
    
    // Limit history size
    if (testHistoryList.children.length > 10) {
        testHistoryList.removeChild(testHistoryList.lastChild);
    }
}

// Update automaton details display
function updateAutomatonDetails(automaton) {
    document.getElementById('automaton-type').textContent = automaton.type || 'DFA';
    document.getElementById('automaton-states').textContent = automaton.states.join(', ');
    document.getElementById('automaton-alphabet').textContent = automaton.input_symbols.join(', ');
    document.getElementById('automaton-initial').textContent = automaton.initial_state;
    document.getElementById('automaton-final').textContent = automaton.final_states.join(', ');
}

// Setup pan and zoom functionality
function setupPanZoom() {
    const svg = document.querySelector('#automaton-graph svg');
    
    if (svg) {
        // Reset any existing instance
        if (zoomInstance) {
            zoomInstance.destroy();
        }
        
        // Create new panzoom instance
        zoomInstance = panzoom(svg, {
            maxZoom: 5,
            minZoom: 0.5,
            zoomSpeed: 0.1,
            bounds: true,
            boundsPadding: 0.1
        });
    }
}

// Add automaton to history
function addToAutomatonHistory(automaton) {
    automatonHistory.push(JSON.parse(JSON.stringify(automaton)));
    
    // Limit history size
    if (automatonHistory.length > 10) {
        automatonHistory.shift();
    }
}

// Save project to local storage
function saveProject() {
    const project = {
        automaton: currentAutomaton,
        regexInput: document.getElementById('regex-input').value,
        statesInput: document.getElementById('states-input').value,
        alphabetInput: document.getElementById('alphabet-input').value,
        initialState: document.getElementById('initial-state-select').value,
        finalStates: Array.from(document.querySelectorAll('input[name="final-states"]:checked')).map(el => el.value),
        testHistory: testHistory,
        timestamp: new Date().toISOString()
    };
    
    localStorage.setItem('automataSolverProject', JSON.stringify(project));
    showToast('Project saved', 'success');
}

// Load project from local storage
function loadProject() {
    const savedProject = localStorage.getItem('automataSolverProject');
    
    if (savedProject) {
        const project = JSON.parse(savedProject);
        
        // Load inputs
        document.getElementById('regex-input').value = project.regexInput || '';
        document.getElementById('states-input').value = project.statesInput || '';
        document.getElementById('alphabet-input').value = project.alphabetInput || '';
        
        // Update state and alphabet selections
        if (project.statesInput) {
            updateStateSelections();
        }
        
        if (project.alphabetInput) {
            updateAlphabetSelections();
        }
        
        // Set initial state
        if (project.initialState) {
            document.getElementById('initial-state-select').value = project.initialState;
        }
        
        // Set final states
        if (project.finalStates) {
            project.finalStates.forEach(state => {
                const checkbox = document.querySelector(`input[name="final-states"][value="${state}"]`);
                if (checkbox) {
                    checkbox.checked = true;
                }
            });
        }
        
        // Load test history
        if (project.testHistory) {
            testHistory = project.testHistory;
            
            const testHistoryList = document.getElementById('test-history-list');
            testHistoryList.innerHTML = '';
            
            testHistory.forEach(test => {
                const listItem = document.createElement('li');
                listItem.className = test.accepted ? 'accepted' : 'rejected';
                
                const icon = document.createElement('i');
                icon.className = test.accepted ? 'fas fa-check' : 'fas fa-times';
                
                listItem.appendChild(icon);
                listItem.appendChild(document.createTextNode(` "${test.string}"`));
                
                testHistoryList.appendChild(listItem);
            });
        }
        
        // Load automaton if available
        if (project.automaton) {
            currentAutomaton = project.automaton;
            renderAutomaton(currentAutomaton);
        }
        
        showToast('Project loaded', 'success');
        return true;
    }
    
    return false;
}

// Clear all inputs
function clearAllInputs() {
    document.getElementById('regex-input').value = '';
    document.getElementById('states-input').value = '';
    document.getElementById('alphabet-input').value = '';
    document.getElementById('initial-state-select').innerHTML = '<option value="">Select a state</option>';
    document.getElementById('final-states-checkboxes').innerHTML = '';
    document.getElementById('transitions-builder').innerHTML = `
        <div class="transition-row">
            <select class="from-state">
                <option value="">From</option>
            </select>
            <select class="input-symbol">
                <option value="">Input</option>
            </select>
            <select class="to-state">
                <option value="">To</option>
            </select>
            <button class="add-transition"><i class="fas fa-plus"></i></button>
        </div>
    `;
    document.getElementById('test-string').value = '';
    document.querySelector('.test-result').classList.add('hidden');
    document.getElementById('test-history-list').innerHTML = '';
    
    // Re-setup the add transition button
    document.querySelector('.add-transition').addEventListener('click', addTransitionRow);
    
    // Clear history
    testHistory = [];
}

// Reset automaton display
function resetAutomatonDisplay() {
    currentAutomaton = null;
    
    // Reset graph container
    const graphContainer = document.getElementById('automaton-graph');
    graphContainer.innerHTML = `
        <div class="empty-state">
            <i class="fas fa-project-diagram empty-icon"></i>
            <p>Convert a regex or build an automaton to see the visualization</p>
        </div>
    `;
    
    // Reset details
    document.getElementById('automaton-type').textContent = '-';
    document.getElementById('automaton-states').textContent = '-';
    document.getElementById('automaton-alphabet').textContent = '-';
    document.getElementById('automaton-initial').textContent = '-';
    document.getElementById('automaton-final').textContent = '-';
    
    // Reset explanation
    document.getElementById('explanation-content').innerHTML = `
        <div class="empty-state">
            <i class="fas fa-lightbulb empty-icon"></i>
            <p>Generate an automaton to see an explanation</p>
        </div>
    `;
}

// Utility Functions

// Show a toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = document.createElement('div');
    icon.className = 'toast-icon';
    
    switch (type) {
        case 'success':
            icon.innerHTML = '<i class="fas fa-check-circle"></i>';
            break;
        case 'error':
            icon.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
            break;
        default:
            icon.innerHTML = '<i class="fas fa-info-circle"></i>';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'toast-message';
    messageDiv.textContent = message;
    
    const closeButton = document.createElement('button');
    closeButton.className = 'toast-close';
    closeButton.innerHTML = '<i class="fas fa-times"></i>';
    closeButton.addEventListener('click', () => {
        toast.remove();
    });
    
    toast.appendChild(icon);
    toast.appendChild(messageDiv);
    toast.appendChild(closeButton);
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode === toastContainer) {
            toastContainer.removeChild(toast);
        }
    }, 5000);
}

// Show a modal
function showModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('hidden');
    modal.classList.add('active');
    
    // Prevent scrolling of the background
    document.body.style.overflow = 'hidden';
}

// Hide a modal
function hideModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('active');
    
    // Add a small delay before hiding completely
    setTimeout(() => {
        modal.classList.add('hidden');
    }, 300);
    
    // Allow scrolling again
    document.body.style.overflow = '';
}

// Animate an element
function animateElement(element, animationName) {
    element.style.animation = 'none';
    element.offsetHeight; // Trigger reflow
    element.style.animation = `${animationName} 0.3s ease forwards`;
}

// SlideUp animation
function slideUp(element) {
    const height = element.offsetHeight;
    element.style.height = height + 'px';
    element.style.overflow = 'hidden';
    element.style.transition = 'height 0.3s ease';
    
    // Trigger reflow
    element.offsetHeight;
    
    element.style.height = '0';
    
    setTimeout(() => {
        element.style.display = 'none';
        element.style.height = '';
        element.style.overflow = '';
        element.style.transition = '';
    }, 300);
}

// SlideDown animation
function slideDown(element) {
    element.style.display = 'block';
    element.style.overflow = 'hidden';
    element.style.height = '0';
    element.style.transition = 'height 0.3s ease';
    
    // Trigger reflow
    element.offsetHeight;
    
    const height = element.scrollHeight;
    element.style.height = height + 'px';
    
    setTimeout(() => {
        element.style.height = '';
        element.style.overflow = '';
        element.style.transition = '';
    }, 300);
}

// Helper functions for extracting information from SVG (placeholder implementations)
function extractStatesFromSVG(svg) {
    // In a real implementation, parse the SVG to extract states
    // This is just a placeholder
    return ['q0', 'q1', 'q2'];
}

function extractAlphabetFromSVG(svg) {
    // In a real implementation, parse the SVG to extract alphabet
    // This is just a placeholder
    return ['0', '1'];
}

function extractInitialStateFromSVG(svg) {
    // In a real implementation, parse the SVG to extract initial state
    // This is just a placeholder
    return 'q0';
}

function extractFinalStatesFromSVG(svg) {
    // In a real implementation, parse the SVG to extract final states
    // This is just a placeholder
    return ['q2'];
}

// Export automaton (placeholder implementation)
function exportAutomaton(automaton, format, includeDetails, includeExplanation) {
    fetch(`${backendBaseUrl}/export/${format}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(automaton)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showToast(`Error: ${data.error}`, 'error');
            return;
        }
        
        // Create a download link
        const link = document.createElement('a');
        
        if (format === 'svg') {
            // SVG can be directly embedded
            const blob = new Blob([data.data], { type: 'image/svg+xml' });
            link.href = URL.createObjectURL(blob);
        } else {
            // For PNG and PDF, we get base64 data
            link.href = `data:${format === 'png' ? 'image/png' : 'application/pdf'};base64,${data.data}`;
        }
        
        link.download = `automaton.${format}`;
        link.click();
        
        showToast(`Automaton exported as ${format.toUpperCase()}`, 'success');
    })
    .catch(error => {
        showToast(`Error: ${error.message}`, 'error');
    });
}

// Check if we have a saved project to load
window.addEventListener('load', () => {
    if (localStorage.getItem('automataSolverProject')) {
        if (confirm('Would you like to load your last saved project?')) {
            loadProject();
        }
    }
});