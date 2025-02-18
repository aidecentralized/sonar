const WebSocket = require('ws');

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
                this.updateStatus(`Session created. Waiting for ${data.remainingClients} more clients...`);
                this.sessionId = data.sessionId;
                break;
            case 'session_joined':
                this.updateStatus(`Joined session. Waiting for ${data.remainingClients} more clients...`);
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
        this.updateStatus(`Connected (Rank ${this.rank})`);

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

        const pc = new RTCPeerConnection(config);
        
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
                await pc.setRemoteDescription(new RTCSessionDescription({
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
                await pc.setRemoteDescription(new RTCSessionDescription({
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