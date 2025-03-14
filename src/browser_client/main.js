import { data } from '@tensorflow/tfjs';
import { processData, WebRTCCommUtils } from './client.js'

const dataInput = document.getElementById('data-input');
const startButton = document.getElementById('start-button');
const consoleOutput = document.getElementById('console-output');
const saveConfigButton = document.getElementById('save-config-button');

// ** Set your session parameters here **
const SESSION_ID = 1111; // Change this to a fixed or generated session ID
const MAX_CLIENTS = 3;
const IS_CREATOR = false; // Set to true if this should create a session

// ** Start WebRTC Comm Utils **
const signalingServer = 'ws://localhost:8765'; // Your WebSocket server

let config = {
    signaling_server: signalingServer,
    num_users: MAX_CLIENTS,
    session_id: SESSION_ID,
    epochs: 10
};

let dataset = null;


function disableButtons() {
    dataInput.disabled = true;
    startButton.disabled = true;
    saveConfigButton.disabled = true;
}

function enableButtons() {
    dataInput.disabled = false;
    startButton.disabled = false;
    saveConfigButton.disabled = false;
}

function displayMessage(message) {
    const newLog = document.createElement("div");
    newLog.textContent = message;
    consoleOutput.appendChild(newLog);
}

saveConfigButton.addEventListener('click', function() {
    config.signaling_server = document.getElementById('signaling_server').value;
    config.num_users = document.getElementById('num_users').value;
    config.session_id = document.getElementById('session_id').value;
    config.epochs = document.getElementById('epochs').value;
    displayMessage('Config Saved:');
    displayMessage(JSON.stringify(config, null, 2));
});

dataInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        displayMessage('Loading file: ' + file.name);
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const rawData = JSON.parse(e.target.result);
                dataset = processData(rawData);
                displayMessage('Successfully loaded data');
                enableButtons();
            } catch (error) {
                displayMessage('Error loading data');
            }
        };
        reader.readAsText(file)
    }
});

startButton.addEventListener('click', function() {
    disableButtons();
    const node = new WebRTCCommUtils(config, dataset);
});