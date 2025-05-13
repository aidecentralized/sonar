import { data } from '@tensorflow/tfjs';
import { processData, WebRTCCommUtils } from './client.js'
import { supportedDatasets } from './model.js';

const trainDataInput = document.getElementById('train-data-input');
const testDataInput = document.getElementById('test-data-input');
const startButton = document.getElementById('start-button');
const consoleOutput = document.getElementById('console-output');
const saveConfigButton = document.getElementById('save-config-button');
const fileDropdown = document.getElementById('fileDropdown');
const datasetDropdown = document.getElementById('datasetDropdown');

// ** Set your session parameters here **
const SESSION_ID = "1111"; // Change this to a fixed or generated session ID
const MAX_CLIENTS = 3;
const IS_CREATOR = false; // Set to true if this should create a session

// ** Start WebRTC Comm Utils **
const signalingServer = 'ws://localhost:8765'; // Your WebSocket server

const samplePartitions = [
    'cifar10_client_0_test.json', 
    'cifar10_client_1_test.json',
    'cifar10_client_2_test.json',
    'cifar10_client_3_test.json',
    'cifar10_client_4_test.json',
    'cifar10_client_5_test.json',
    'cifar10_client_6_test.json',
    'cifar10_client_7_test.json',
    'cifar10_client_8_test.json',
    'cifar10_client_9_test.json',
    'mnist_client_0_test.json', 
    'mnist_client_1_test.json',
    'mnist_client_2_test.json',
    'mnist_client_3_test.json',
    'mnist_client_4_test.json',
    'mnist_client_5_test.json',
    'mnist_client_6_test.json',
    'mnist_client_7_test.json',
    'mnist_client_8_test.json',
    'mnist_client_9_test.json'
];

// Add sample partitions to dropdown
samplePartitions.forEach(file => {
    const option = document.createElement('option');
    option.value = file;
    option.textContent = file;
    fileDropdown.appendChild(option);
});

Object.keys(supportedDatasets).forEach(dataset => {
    const option = document.createElement('option');
    option.value = dataset;
    option.textContent = dataset;
    datasetDropdown.appendChild(option);
});

let config = {
    signaling_server: signalingServer,
    algos: {node_0: {topology: "ring"}},
    num_users: MAX_CLIENTS,
    session_id: SESSION_ID,
    epochs: 10,
    num_collaborators: 1,
    joinActiveSession: false, // Default to false
    seed: 2,
};

let datasetType = '';
let trainDataset = null;
let testDataset = null;

function disableButtons() {
    trainDataInput.disabled = true;
    testDataInput.disabled = true;
    fileDropdown.disabled = true;
    startButton.disabled = true;
    saveConfigButton.disabled = true;
    fileDropdown.disabled = true;
    datasetDropdown.disabled = true;
}

function enableButtons() {
    trainDataInput.disabled = false;
    testDataInput.disabled = false;
    fileDropdown.disabled = false;
    startButton.disabled = trainDataset === null; // Only enable if training data exists
    saveConfigButton.disabled = false;
    fileDropdown.disabled = false;
    datasetDropdown.disabled = false;
}

export function displayMessage(message) {
    const newLog = document.createElement("div");
    newLog.textContent = message;
    consoleOutput.appendChild(newLog);

    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

saveConfigButton.addEventListener('click', function() {
    config.algos.node_0.topology = document.getElementById('topology').value;
    config.signaling_server = document.getElementById('signaling_server').value;
    config.num_users = document.getElementById('num_users').value;
    config.session_id = document.getElementById('session_id').value;
    config.epochs = document.getElementById('epochs').value;
    config.num_collaborators = document.getElementById('num_collaborators').value;
    config.joinActiveSession = document.getElementById('join_active_session').checked;
    
    // Save to localStorage for persistence
    localStorage.setItem('topology', config.algos.node_0.topology);
    localStorage.setItem('signalingServer', config.signaling_server);
    localStorage.setItem('numUsers', config.num_users);
    localStorage.setItem('sessionId', config.session_id);
    localStorage.setItem('epochs', config.epochs);
    localStorage.setItem('numCollaborators', config.num_collaborators);
    localStorage.setItem('joinActiveSession', config.joinActiveSession);
    
    displayMessage('Config Saved:');
    displayMessage(JSON.stringify(config, null, 2));
    
    // Enable the Connect button if training data is loaded
    if (trainDataset) {
        startButton.disabled = false;
    }
});

trainDataInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        displayMessage('Loading training file: ' + file.name);
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const rawData = JSON.parse(e.target.result);
                trainDataset = processData(rawData);
                displayMessage('Successfully loaded training data');
                enableButtons(); // Enable start button when training data is loaded
                
                // Reset dropdown
                fileDropdown.value = '';
            } catch (error) {
                displayMessage('Error loading training data: ' + error.message);
            }
        };
        reader.readAsText(file);
    }
});

testDataInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        displayMessage('Loading test file: ' + file.name);
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const rawData = JSON.parse(e.target.result);
                testDataset = processData(rawData);
                displayMessage('Successfully loaded test data');
                fileDropdown.value = '';
            } catch (error) {
                displayMessage('Error loading test data: ' + error.message);
            }
        };
        reader.readAsText(file);
    }
});

// Handler for the sample partitions dropdown
fileDropdown.addEventListener('change', async (e) => {
    const filename = e.target.value;
    if (!filename) return;
  
    try {
        const res = await fetch(`/datasets/imgs/${filename.split('_')[0]}/${filename}`);
        const json = await res.json();
        trainDataset = processData(json);
        enableButtons(); // Enable start button when training data is loaded
        displayMessage('Successfully loaded sample partition.');
        
        // Clear file inputs
        trainDataInput.value = '';
        testDataInput.value = '';
        
        testDataset = null;
    } catch (err) {
        displayMessage(`Error loading file: ${err.message}`);
    }
});

datasetDropdown.addEventListener('change', (e) => {
    datasetType = e.target.value;
    displayMessage('Selected dataset: ' + datasetType)
});

// Helper function to split a dataset into training and testing portions
function splitDataset(dataset, trainRatio = 0.8) {
    // Create copies of the arrays to avoid modifying the original
    const images = [...dataset.images];
    const labels = [...dataset.labels];
    
    // Shuffle the arrays together (maintaining corresponding indices)
    for (let i = images.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        // Swap images
        [images[i], images[j]] = [images[j], images[i]];
        // Swap corresponding labels
        [labels[i], labels[j]] = [labels[j], labels[i]];
    }
    
    // Calculate split index
    const splitIndex = Math.floor(images.length * trainRatio);
    
    // Create training and testing datasets
    const trainData = {
        images: images.slice(0, splitIndex),
        labels: labels.slice(0, splitIndex)
    };
    
    const testData = {
        images: images.slice(splitIndex),
        labels: labels.slice(splitIndex)
    };
    
    return { trainData, testData };
}

startButton.addEventListener('click', function() {
    disableButtons();
    
    if (!trainDataset) {
        displayMessage('Error: Training dataset not loaded');
        enableButtons();
        return;
    }

    if (!datasetType) {
        displayMessage('Error: No dataset type selected');
        enableButtons();
        return;
    }
    
    let finalTrainDataset = trainDataset;
    let finalTestDataset = testDataset;
    
    // If only training data is available, split it
    if (!testDataset) {
        displayMessage('No separate test dataset provided. Splitting training data 80/20...');
        const { trainData, testData } = splitDataset(trainDataset);
        finalTrainDataset = trainData;
        finalTestDataset = testData;
        displayMessage(`Split complete: Training data has ${finalTrainDataset.images.length} samples, Test data has ${finalTestDataset.images.length} samples`);
    }
    
    displayMessage(`Starting training with ${finalTrainDataset.images.length} training samples and ${finalTestDataset.images.length} testing samples`);
    const node = new WebRTCCommUtils(config, datasetType, finalTrainDataset, finalTestDataset);
});