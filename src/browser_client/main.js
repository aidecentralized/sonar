import { data } from '@tensorflow/tfjs';
import { processData, WebRTCCommUtils } from './client.js'

// Get DOM elements
const trainDataInput = document.getElementById('train-data-input');
const testDataInput = document.getElementById('test-data-input');
const loadDataButton = document.getElementById('load-data-button');
const startButton = document.getElementById('start-button');
const consoleOutput = document.getElementById('console-output');
const saveConfigButton = document.getElementById('save-config-button');

// Data storage
let trainDataset = null;
let testDataset = null;
let combinedDataset = null;

// ** Set your session parameters here **
const SESSION_ID = 1111; // Change this to a fixed or generated session ID
const MAX_CLIENTS = 3;
const IS_CREATOR = false; // Set to true if this should create a session

// ** Start WebRTC Comm Utils **
const signalingServer = 'ws://localhost:8765'; // Your WebSocket server

// Initialize configuration
let config = {
  "signaling_server": signalingServer,
  "num_users": MAX_CLIENTS,
  "session_id": SESSION_ID,
  "epochs": 10,
  "num_collaborators": 1,
  "algos": {
    "node_0": {
      "topology": "ring"
    }
  }
};

function disableButtons() {
  startButton.disabled = true;
  saveConfigButton.disabled = true;
  loadDataButton.disabled = false;
}

function enableButtons() {
  startButton.disabled = false;
  saveConfigButton.disabled = false;
}

function displayMessage(message) {
  const p = document.createElement('p');
  p.textContent = message;
  consoleOutput.appendChild(p);
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// Update the load button status whenever datasets change
function updateLoadButtonState() {
  // Enable load button if training data is available (test data is optional)
  loadDataButton.disabled = !trainDataset;
}

// Initialize with disabled load button
updateLoadButtonState();

saveConfigButton.addEventListener('click', function() {
    config.algos.node_0.topology = document.getElementById('topology').value;
    config.signaling_server = document.getElementById('signaling_server').value;
    config.num_users = document.getElementById('num_users').value;
    config.session_id = document.getElementById('session_id').value;
    config.epochs = document.getElementById('epochs').value;
    config.num_collaborators = document.getElementById('num_collaborators').value;
    displayMessage('Config Saved:');
    displayMessage(JSON.stringify(config, null, 2));
});

// Handle training data file input
trainDataInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    displayMessage('Loading training data: ' + file.name);
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const rawData = JSON.parse(e.target.result);
            trainDataset = processData(rawData);
            displayMessage(`Training data loaded: ${trainDataset.images.length} samples`);
            updateLoadButtonState();
        } catch (error) {
            displayMessage(`Error loading training data: ${error.message}`);
            trainDataset = null;
            updateLoadButtonState();
        }
    };
    reader.readAsText(file);
});

// Handle test data file input
testDataInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    displayMessage('Loading test data: ' + file.name);
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const rawData = JSON.parse(e.target.result);
            testDataset = processData(rawData);
            displayMessage(`Test data loaded: ${testDataset.images.length} samples`);
            updateLoadButtonState();
        } catch (error) {
            displayMessage(`Error loading test data: ${error.message}`);
            testDataset = null;
            updateLoadButtonState();
        }
    };
    reader.readAsText(file);
});

// Handle the load data button click
loadDataButton.addEventListener('click', function() {
    if (!trainDataset) {
        displayMessage('Please load at least a training dataset');
        return;
    }
    
    // Combine the datasets
    if (testDataset) {
        // Case 1: Both training and test datasets are provided
        combinedDataset = {
            train: trainDataset,
            test: testDataset,
            isCustomSplit: false
        };
        displayMessage(`Datasets combined: ${trainDataset.images.length} training samples, ${testDataset.images.length} test samples`);
    } else {
        // Case 2: Only training dataset is provided
        // We'll mark this dataset for internal splitting in the model
        // The actual split will happen in model.js
        combinedDataset = {
            train: trainDataset,
            test: null,
            isCustomSplit: true
        };
        displayMessage(`Training dataset loaded: ${trainDataset.images.length} samples (20% will be used for validation)`);
    }
    
    enableButtons();
});

startButton.addEventListener('click', function() {
    if (!combinedDataset) {
        displayMessage('Please load at least a training dataset first');
        return;
    }

    disableButtons();
    displayMessage('Starting training...');
    
    try {
        const node = new WebRTCCommUtils(config, combinedDataset);
        node.startTraining().catch(err => {
            displayMessage(`Error: ${err.message}`);
            console.error(err);
        });
    } catch (error) {
        displayMessage(`Error: ${error.message}`);
        console.error(error);
    }
});