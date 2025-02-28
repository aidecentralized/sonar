const { ResNet10 } = require('./model.js');
// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');


// dataset loading helper
async function loadDataset(jsonFile) {
	try {
		const response = await fetch(jsonFile)
		const jsonData = await response.json()
		return processData(jsonData)
	} catch (error) {
		console.error('Error loading data:', error)
	}
}

// data processing helper
function processData(jsonData) {
	const images = jsonData.map(item => item.image)
	const labels = jsonData.map(item => item.label)

	return {
		images: images,
		labels: labels
	}
}

// ** Set your session parameters here **
const SESSION_ID = 1111; // Change this to a fixed or generated session ID
const MAX_CLIENTS = 3;
const IS_CREATOR = false; // Set to true if this should create a session

// const sessionInfo = {
//     type: IS_CREATOR ? 'create_session' : 'join_session',
//     sessionId: SESSION_ID,
//     maxClients: MAX_CLIENTS,
//     clientType: 'javascript'
// };

// ** Start WebRTC Comm Utils **
const signalingServer = 'ws://localhost:8765'; // Your WebSocket server

// TODO: fill in config
let config = {
    epochs: 10,
    batchSize: 256,
    validationSplit: 0.2,
    signaling_server: signalingServer,
    num_users: MAX_CLIENTS,
    session_id: SESSION_ID,
    shuffle: true,
    verbose: 1,
}

async function main() {
	const model = new ResNet10();
	const comms = new WebRTCCommUtils(config, model)
	console.log('model initialized... loading dataset...');
	// const dataset = await loadDataset('./datasets/imgs/bloodmnist/bloodmnist_test.json');

	const filePath = path.resolve(__dirname, './datasets/imgs/bloodmnist/bloodmnist_test.json');
	const rawData = fs.readFileSync(filePath, 'utf8');
	const data = JSON.parse(rawData);
	const dataset = processData(data);

	console.log('dataset loaded... training model...');

	for (let i = 0; i < config.epochs; i++) {
		await model.local_train_one(dataset)
		console.log('finished training');
		const peer_weights = comms.receive()
		// todo: perform fed avg
	}

	// console.log('dataset loaded... training model...');
	// await model.train(dataset);
	// console.log('finished training');
}

// Call the main function
main().catch(error => console.error('Error in main function:', error));

// const node = new WebRTCCommUtils(config);
