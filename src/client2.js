import * as tf from '@tensorflow/tfjs'

const WebSocket = require('ws');
const wrtc = require('wrtc');  // Import WebRTC for Node.js
// TODO: this can be replaced by just the browser-side without wrtc once we use browser

class TorusNode {
    constructor(signalingServer, sessionInfo) {
        this.signalingServer = signalingServer;
        this.sessionInfo = sessionInfo;
        this.connections = new Map();
        this.dataChannels = new Map();
        this.rank = null;
        this.neighbors = null;
        this.connectedPeers = new Set();
        this.pendingConnections = new Set();
        this.expectedConnections = 0;
        this.connectionRetries = new Map();
        this.MAX_RETRIES = 3;
        this.RETRY_DELAY = 2000;
        this.sessionId = null;

        this.log('Initializing TorusNode...');
        this.connect()
    }

    log(message, type = 'info') {
        // const logDiv = document.getElementById('log');
        // const entry = document.createElement('div');
        // entry.className = `log-entry log-${type}`;
        // entry.textContent = `${new Date().toISOString()} - ${message}`;
        // logDiv.appendChild(entry);
        // logDiv.scrollTop = logDiv.scrollHeight;

        if (type === 'info') {
            console.log(message);
        } else {
            console.error(message);
        }
    }

    updateStatus(message) {
        // document.getElementById('status').textContent = `Status: ${message}`;
        console.log(`Status: ${message}`)
    }

    async connect(sessionInfo) {
        try {
            this.ws = new WebSocket(this.signalingServer);
            this.ws.onmessage = this.handleWsMessage.bind(this);
            this.ws.onopen = () => {
                this.log('Connected to signaling server');
                this.ws.send(JSON.stringify({
                    // type: 'session_action',
                    type: this.sessionInfo.type,
                    sessionId: this.sessionInfo.sessionId,
                    maxClients: this.sessionInfo.maxClients,
                    clientType: 'javascript'
                }));
            };
        } catch (error) {
            this.log(`WebSocket connection error: ${error}`, 'error');
        }
    }

    async handleWsMessage(event) {
        const data = JSON.parse(event.data);
        this.log(`Received ${data.type} message`);

        switch (data.type) {
            case 'session_created':
                this.updateStatus(`Session created, rank ${data.rank}`);
                this.sessionId = data.sessionId;
                break;
            case 'session_joined':
                this.updateStatus(`Joined session, rank ${data.rank}`);
                this.sessionId = data.sessionId;
                break;
            case 'session_ready':
                this.updateStatus('Session ready! Establishing connections...');
                break;
            case 'session_error':
                this.updateStatus(`Session Error: ${data.message}`);
                this.log(data.message, 'error');
                break;
            case 'topology':
                await this.handleTopology(data);
                break;
            case 'signal':
                await this.handleSignaling(data);
                break;
            case 'network_ready':
                this.updateStatus('Network Ready');
                break;
        }
    }

    async handleTopology(data) {
        this.rank = data.rank;
        const newNeighbors = data.neighbors;
        this.log(`Received topology. Rank: ${this.rank}, Neighbors: ${JSON.stringify(newNeighbors)}`);

        if (this.neighbors) {
            const oldNeighbors = new Set(Object.values(this.neighbors));
            const newNeighborSet = new Set(Object.values(newNeighbors));
            for (const rank of oldNeighbors) {
                if (!newNeighborSet.has(rank)) {
                    await this.cleanupConnection(rank);
                }
            }
        }

        this.neighbors = newNeighbors;
        this.expectedConnections = Object.keys(newNeighbors).length;
        this.updateStatus(`Joined session, rank is ${this.rank}`);

        // Initiate connections to higher-ranked neighbors
        for (const neighborRank of Object.values(newNeighbors)) {
            if (neighborRank > this.rank && 
                !this.connections.has(neighborRank) && 
                !this.pendingConnections.has(neighborRank)) {
                this.log(`Initiating connection to ${neighborRank}`);
                this.pendingConnections.add(neighborRank);
                this.initiateConnection(neighborRank);
            }
        }
    }

    createPeerConnection() {
        const config = {
            iceServers: [{
                urls: [
                    'stun:stun.l.google.com:19302',
                    'stun:stun1.l.google.com:19302'
                ]
            }]
        };

        const pc = new wrtc.RTCPeerConnection(config);
        
        pc.oniceconnectionstatechange = () => {
            this.log(`ICE connection state: ${pc.iceConnectionState}`);
        };

        pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.log('ICE candidate generated');
            }
        };

        return pc;
    }

    async initiateConnection(targetRank) {
        try {
            const pc = this.createPeerConnection();
            this.connections.set(targetRank, pc);

            // Create data channel
            const channel = pc.createDataChannel(`chat-${this.rank}-${targetRank}`);
            this.setupDataChannel(channel, targetRank);

            // Create and set local description
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Wait for ICE gathering
            await new Promise(resolve => {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        resolve();
                    } else {
                        setTimeout(checkState, 1000);
                    }
                };
                checkState();
            });

            // Send offer
            await this.sendSignaling(targetRank, {
                type: 'offer',
                sdp: pc.localDescription.sdp
            });

        } catch (error) {
            this.log(`Failed to initiate connection to ${targetRank}: ${error}`, 'error');
            await this.handleConnectionFailure(targetRank);
        }
    }

    setupDataChannel(channel, peerRank) {
        this.dataChannels.set(peerRank, channel);

        channel.onopen = () => {
            this.log(`Data channel opened with peer ${peerRank}`);
            this.onPeerConnected(peerRank);
            this.startPingLoop(peerRank);
        };

        channel.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log(`Received message from ${peerRank}: ${data.type}`)
                if (data.type === 'ping') {
                    this.log(`Received ping from ${peerRank}`);
                    channel.send(JSON.stringify({
                        type: 'pong',
                        timestamp: data.timestamp,
                        respondedAt: Date.now()
                    }));
                } else if (data.type === 'pong') {
                    const rtt = Date.now() - data.timestamp;
                    this.log(`Received pong from ${peerRank}, RTT: ${rtt}ms`);
                }
            } catch (error) {
                this.log(`Failed to parse message from ${peerRank}: ${error}`, 'error');
            }
        };
    }

    startPingLoop(peerRank) {
        const sendPing = () => {
            const channel = this.dataChannels.get(peerRank);
            if (channel && channel.readyState === 'open') {
                channel.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now()
                }));
            }
        };

        setInterval(sendPing, 5000);
    }

    async handleSignaling(message) {
        const senderRank = message.senderRank;
        const data = message.data;

        try {
            let pc = this.connections.get(senderRank);
            if (!pc) {
                pc = this.createPeerConnection();
                this.connections.set(senderRank, pc);

                pc.ondatachannel = (event) => {
                    this.setupDataChannel(event.channel, senderRank);
                };
            }

            if (data.type === 'offer') {
                await pc.setRemoteDescription(new wrtc.RTCSessionDescription({
                    type: 'offer',
                    sdp: data.sdp
                }));
                const answer = await pc.createAnswer();
                await pc.setLocalDescription(answer);
                await this.sendSignaling(senderRank, {
                    type: 'answer',
                    sdp: answer.sdp,
                });
            } else if (data.type === 'answer') {
                await pc.setRemoteDescription(new wrtc.RTCSessionDescription({
                    type: 'answer',
                    sdp: data.sdp
                }));
            } else if (data.type === 'candidate') {
                await pc.addIceCandidate({
                    candidate: data.candidate,
                    sdpMLineIndex: 0,
                    sdpMid: '0'
                });
            }
        } catch (error) {
            this.log(`Error handling signaling message: ${error}`, 'error');
        }
    }

    async sendSignaling(targetRank, data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            await this.ws.send(JSON.stringify({
                type: 'signal',
                targetRank: targetRank,
                data: data,
                sessionId: this.sessionId
            }));
        }
    }

    onPeerConnected(peerRank) {
        this.connectedPeers.add(peerRank);
        this.pendingConnections.delete(peerRank);
        this.log(`Connected to peer ${peerRank}. Connected: ${this.connectedPeers.size}/${this.expectedConnections}`);

        this.ws.send(JSON.stringify({
            type: 'connection_established',
            peerRank: peerRank,
            sessionId: this.sessionId
        }));
    }

    async handleConnectionFailure(targetRank) {
        const retryCount = this.connectionRetries.get(targetRank) || 0;
        if (retryCount < this.MAX_RETRIES) {
            this.connectionRetries.set(targetRank, retryCount + 1);
            await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY * (retryCount + 1)));
            if (!this.connectedPeers.has(targetRank)) {
                await this.cleanupConnection(targetRank);
                this.initiateConnection(targetRank);
            }
        } else {
            this.log(`Max retries reached for ${targetRank}`, 'error');
            await this.cleanupConnection(targetRank);
        }
    }

    async cleanupConnection(rank) {
        try {
            const pc = this.connections.get(rank);
            if (pc) {
                const channel = this.dataChannels.get(rank);
                if (channel) {
                    channel.close();
                    this.dataChannels.delete(rank);
                }
                pc.close();
                this.connections.delete(rank);
            }

            this.pendingConnections.delete(rank);
            this.connectedPeers.delete(rank);
            this.log(`Cleaned up connection to peer ${rank}`);
        } catch (error) {
            this.log(`Error during connection cleanup for peer ${rank}: ${error}`, 'error');
        }
    }

    async tensorToSerializable(obj) {
        if (obj instanceof tf.Tensor) {
            return {
                "__tensor__": true,
                "data": await obj.array(),
                "dtype": obj.dtype,
                "shape": obj.shape
            };
        } else if (Array.isArray(obj)) {
            return Promise.all(obj.map(item => this.tensorToSerializable(item)));
        } else if (typeof obj === "object" && obj !== null) {
            const entries = await Promise.all(
                Object.entries(obj).map(async ([key, value]) => [key, await this.tensorToSerializable(value)])
            );
            return Object.fromEntries(entries);
        }
        return obj;
    }
    
    serializableToTensor(obj) {
        if (typeof obj === "object" && obj !== null) {
            if ("__tensor__" in obj) {
                return tf.tensor(obj.data, obj.shape, obj.dtype);
            }
            return Object.fromEntries(Object.entries(obj).map(([key, value]) => [key, this.serializableToTensor(value)]));
        } else if (Array.isArray(obj)) {
            return obj.map(item => this.serializableToTensor(item));
        }
        return obj;
    }
    
    async serializeMessage(message) {
        const serializableDict = await this.tensorToSerializable(message);
        return JSON.stringify(serializableDict, null, 2);
    }
    
    deserializeMessage(jsonStr) {
        const serializableDict = JSON.parse(json);
        return this.serializableToTensor(serializableDict);
    }
}

class Model {
	constructor() {
	}

	summary() {
		this.model.summary()
	}

	forward(x, shape) {
		const tensor = (x instanceof tf.Tensor) ? x : tf.tensor2d([x], shape)
		const output = this.model.predict(tensor)
		// addLog('Output:', output.arraySync())
		return output
	}
}

// resnet
export class ResNet10 extends Model {
	constructor() {
		super()
		// addLog("Initializing ResNet10 instance...")
		this.model = this.buildModel()
	}

	// Build the model
	buildModel() {
		const inputs = tf.input({ shape: [2352] });

		let x = tf.layers.reshape({ targetShape: [28, 28, 3] }).apply(inputs);

		// Initial Conv Layer
		x = tf.layers.conv2d({
			filters: 64,
			kernelSize: 3,
			strides: 1,
			padding: 'same',
			useBias: false
		}).apply(x);

		x = tf.layers.batchNormalization().apply(x);
		x = tf.layers.reLU().apply(x);

		// Residual Blocks
		x = this.residualBlock(x, 64);
		x = this.residualBlock(x, 128, true);
		x = this.residualBlock(x, 256, true);
		x = this.residualBlock(x, 512, true);

		// Global Average Pooling
		x = tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }).apply(x);

		// Fully Connected Layer
		x = tf.layers.dense({ units: 8, activation: 'softmax' }).apply(x);

		const model = tf.model({ inputs, outputs: x });

		model.compile({
			optimizer: 'adam',
			loss: 'categoricalCrossentropy',
			metrics: ['accuracy']
		})

		// addLog('model initialized.')

		return model;
	}

	// Function to create a residual block
	residualBlock(x, filters, downsample = false) {
		let shortcut = x;

		if (downsample) {
			shortcut = tf.layers.conv2d({
				filters: filters,
				kernelSize: 1,
				strides: 2,
				padding: 'same',
				useBias: false
			}).apply(shortcut);
			
			shortcut = tf.layers.batchNormalization().apply(shortcut);
		}

		let out = tf.layers.conv2d({
			filters: filters,
			kernelSize: 3,
			strides: downsample ? 2 : 1,
			padding: 'same',
			useBias: false
		}).apply(x);
		
		out = tf.layers.batchNormalization().apply(out);
		out = tf.layers.reLU().apply(out);

		out = tf.layers.conv2d({
			filters: filters,
			kernelSize: 3,
			strides: 1,
			padding: 'same',
			useBias: false
		}).apply(out);
		
		out = tf.layers.batchNormalization().apply(out);
		
		// Add the shortcut connection
		out = tf.layers.add().apply([out, shortcut]);
		out = tf.layers.reLU().apply(out);

		return out;
	}

	forward(x) {
		return super.forward(x, [1, 2352])
	}

	async train(dataSet, config = {
		epochs: 2,
		batchSize: 16,
		validationSplit: 0.2,
		shuffle: true,
		verbose: 1
	}) {
		// take raw array of values and turn to tensor
		const images = tf.tensor2d(dataSet.images, [dataSet.images.length, 2352])

		const labels = tf.oneHot(tf.tensor1d(dataSet.labels, 'int32'), 8)

		// create config object
		const trainingConfig = {
			epochs: config.epochs,
			batchSize: config.batchSize,
			validationSplit: config.validationSplit,
			shuffle: config.shuffle,
			verbose: config.verbose,
			callbacks: {
				// callback in between epochs
				onEpochEnd: (epoch, logs) => {
					// addLog(`Epoch ${epoch + 1}`)
					// addLog(`Loss: ${logs.loss.toFixed(4)}`)
					// addLog(`Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
					// if (logs.val_loss) {
					// 	addLog(`  Validation Loss: ${logs.val_loss.toFixed(4)}`)
					// 	addLog(`  Validation Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`)
					// }
				}
			}
		}

		try {
			// addLog(`Beginning training...`)
			const history = await this.model.fit(images, labels, trainingConfig)
			// addLog(`Training completed`)

			images.dispose()
			labels.dispose()

			return history
		} catch (error) {
			console.error('Error during training: ', error)

			images.dispose()
			labels.dispose()
			throw error
		}
	}
}


// let node = null;

// function createSession() {
//     const maxClients = document.getElementById('maxClients').value;
//     const sessionId = document.getElementById('sessionId').value || generateSessionId();
//     document.getElementById('sessionId').value = sessionId;

//     node = new TorusNode('ws://localhost:8765');
//     node.connect({
//         type: 'create_session',
//         sessionId: sessionId,
//         maxClients: parseInt(maxClients)
//     });
// }

// function joinSession() {
//     const sessionId = document.getElementById('joinSessionId').value;
//     if (!sessionId) {
//         alert('Please enter a session code');
//         return;
//     }

//     node = new TorusNode('ws://localhost:8765');
//     node.connect({
//         type: 'join_session',
//         sessionId: sessionId
//     });
// }

function generateSessionId() {
    return Math.random().toString(36).substring(2, 8).toUpperCase();
}

// Hide the forms once connected
// function hideSessionForms() {
//     document.getElementById('sessionForm').style.display = 'none';
//     document.getElementById('joinForm').style.display = 'none';
// }

// ** Set your session parameters here **
const SESSION_ID = 1111; // Change this to a fixed or generated session ID
const MAX_CLIENTS = 3;
const IS_CREATOR = false; // Set to true if this should create a session

const sessionInfo = {
    type: IS_CREATOR ? 'create_session' : 'join_session',
    sessionId: SESSION_ID,
    maxClients: MAX_CLIENTS,
    clientType: 'javascript'
};

// ** Start the TorusNode client **
const signalingServer = 'ws://localhost:8765'; // Your WebSocket server
const node = new TorusNode(signalingServer, sessionInfo);