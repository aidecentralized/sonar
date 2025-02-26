const WebSocket = require('ws');
const wrtc = require('wrtc');  // Import WebRTC for Node.js
const { ResNet10 } = require('./model.js');
// TODO: this can be replaced by just the browser-side without wrtc once we use browser
// TODO: I awaited the initiateConnection, but it was originally unecessary (and still might not be necessary), you only need a sufficient timeout

const model = new ResNet10()

function tensorToSerializable(obj) {
    // If it's a "tensor-like" object, convert it into a serializable structure.
    if (obj && obj.__isTensor) {
      return {
        __tensor__: true,
        data: Array.from(obj.data), // Convert typed array to a regular Array
        dtype: obj.dtype,
        shape: obj.shape
      };
    } 
    // If it's a plain object, recursively process each value
    else if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
      const result = {};
      for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
          result[key] = tensorToSerializable(obj[key]);
        }
      }
      return result;
    } 
    // If it's an array, recursively process each element
    else if (Array.isArray(obj)) {
      return obj.map(item => tensorToSerializable(item));
    }
  
    // Otherwise, return as is (number, string, etc.)
    return obj;
}

  function serializableToTensor(obj) {
    // If the object has the "__tensor__" marker, convert it back to a "tensor-like" object.
    if (obj && typeof obj === 'object' && obj.__tensor__) {
      return {
        __isTensor: true,
        data: new Float32Array(obj.data), // or use appropriate typed array based on `obj.dtype`
        dtype: obj.dtype,
        shape: obj.shape
      };
    }
    // If it's a plain object, recursively reconstruct
    else if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
      const result = {};
      for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
          result[key] = serializableToTensor(obj[key]);
        }
      }
      return result;
    }
    // If it's an array, recursively reconstruct
    else if (Array.isArray(obj)) {
      return obj.map(item => serializableToTensor(item));
    }
  
    // Otherwise, return as is
    return obj;
  }
  
  function serializeMessage(message) {
    const serializableDict = tensorToSerializable(message);
    return JSON.stringify(serializableDict);
  }
  
  function deserializeMessage(jsonStr) {
    try {
      const parsed = JSON.parse(jsonStr);
      return serializableToTensor(parsed);
    }
    catch (error) {
      console.error(`Error deserializing message: ${error} (input: ${
        jsonStr.substring(0, 300)}...)`);
      return null;
    } 
  }

  /**
 * chunkTensor takes a "tensor-like" object, flattens its data
 * (which we assume is already 1D or typed array), and yields
 * smaller pieces of size 'chunkSize'.
 */
function chunkTensor(tensor, chunkSize) {
    const originalShape = tensor.shape;
    // const totalElements = tensor.data.length;
    const rawArray = tensor.dataSync()
    const totalElements = rawArray.length
    const numChunks = Math.ceil(totalElements / chunkSize);
    
    const chunks = [];
    
    for (let i = 0; i < numChunks; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, totalElements);
      const chunkData = rawArray.slice(start, end); // typedArray.slice or array.slice
      
      chunks.push({
        chunk: {
          __isTensor: true,
          data: chunkData,
          dtype: tensor.dtype,
          shape: [chunkData.length]  // for the chunk, shape is simply the number of elements
        },
        numChunks,
        originalShape
      });
    }
    
    return chunks;
  }

/**
 * Enum-like states, analogous to the Python NodeState enum.
 */
const NodeState = {
    CONNECTING: 1,
    READY: 2,
    DISCONNECTING: 3,
};

class WebRTCCommUtils {
    constructor(config) {
        this.config = config || {};
        this.signalingServer = this.config.signaling_server || 'ws://localhost:8765';
    
        // Networking & session references
        this.ws = null;                               // WebSocket connection
        this.sessionId = this.config.sessionId || 1111;
        this.rank = null;
        this.size = this.config.num_users || 2;
        this.expectedConnections = 0;
    
        // WebRTC connections
        this.connections = new Map();                 // peerRank -> RTCPeerConnection
        this.dataChannels = new Map();                // peerRank -> RTCDataChannel
        this.connectedPeers = new Set();
        this.pendingConnections = new Set();
    
        // Connection management
        this.connectionRetries = new Map();           // peerRank -> retryCount
        this.MAX_RETRIES = 3;
        this.RETRY_DELAY = 15000;
        this.ICE_GATHERING_TIMEOUT = 10000;
    
        // State
        this.state = NodeState.CONNECTING;
    
        // Extra placeholders for distributed training logic
        this.currentRound = 0;
        this.peer_rounds = new Map();
        this.peer_weights = {};
        this.clear_peer_weights = false;
    
        // Communication cost counters
        this.comm_cost_sent = 0;
        this.comm_cost_received = 0;
    
        // Simple logging
        this.log(`[constructor] RTCCommUtilsJS created with config: ${JSON.stringify(config)}`);
        this.connect()
    }

    // ---------------------- Basic Logging & State Helpers ----------------------

    log(msg) {
        console.log(`[RTCCommUtilsJS] ${msg}`);
    }

    setState(newState) {
        this.state = newState;
        this.log(`State changed to: ${Object.keys(NodeState)[newState - 1]}`);
    }

    async connect() {
        try {
          this.ws = new WebSocket(this.signalingServer);
    
          this.ws.onopen = () => {
            this.log('Connected to signaling server');
            if (this.config.create) {
              // Create a session
              this.ws.send(JSON.stringify({
                type: 'create_session',
                maxClients: this.size,
                clientType: 'javascript'
              }));
            } else {
              // Join an existing session
              this.ws.send(JSON.stringify({
                type: 'join_session',
                sessionId: this.sessionId,
                clientType: 'javascript',
                maxClients: this.size
              }));
            }
          };
    
          this.ws.onmessage = async (event) => {
            const data = JSON.parse(event.data);
            this.log(`Received message from server: ${data.type}`);
    
            switch (data.type) {
              case 'session_created':
                this.sessionId = data.sessionId;
                this.rank = data.rank;
                this.log(`Session created. ID=${this.sessionId}, rank=${this.rank}`);
                break;
    
              case 'session_joined':
                this.sessionId = data.sessionId;
                this.rank = data.rank;
                this.log(`Joined session. ID=${this.sessionId}, rank=${this.rank}`);
                break;
    
              case 'session_ready':
                this.log('Session is ready. Waiting for topology...');
                break;
    
              case 'session_error':
                this.log(`Session error: ${data.message}`);
                break;
    
              case 'topology':
                await this.handleTopology(data);
                break;
    
              case 'signal':
                await this.handleSignalingMessage(data);
                break;
    
              case 'network_ready':
                this.log('Network ready. All peers connected!');
                this.setState(NodeState.READY);
                break;
    
              case 'connection_established':
                this.log(`Peer ${data.peerRank} signaled connection_established.`);
                break;
            }
          };
    
          this.ws.onerror = (err) => {
            this.log(`WebSocket error: ${err}`);
          };
    
          this.ws.onclose = () => {
            this.log('WebSocket disconnected.');
            this.setState(NodeState.DISCONNECTING);
          };
    
        } catch (error) {
          this.log(`connect() error: ${error}`);
        }
      }

  /**
   * handleTopology - Receives neighbors and attempts to connect or remove stale connections.
   */
    async handleTopology(data) {
        this.log(`Handling topology... rank=${data.rank}, neighbors=${JSON.stringify(data.neighbors)} type ${typeof data.neighbors}`);

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

        // If we have zero neighbors, we can signal "node_ready" right away
        if (this.expectedConnections === 0) {
            this.broadcastNodeReady();

        }

        // Initiate connections to higher-ranked neighbors
        for (const neighborRank of Object.values(newNeighbors)) {
          // TODO: uncomment this condition later
            // if (neighborRank > this.rank && 
            //     !this.connections.has(neighborRank) && 
            //     !this.pendingConnections.has(neighborRank)) {
            //     this.log(`Initiating connection to ${neighborRank}`);
            //     this.pendingConnections.add(neighborRank);
            //     this.initiateConnection(neighborRank);
            // }
            this.log(`Initiating connection to ${neighborRank}`);
            this.pendingConnections.add(neighborRank);
            this.initiateConnection(neighborRank);
        }
    }

  // ------------------------ WebRTC Peer Connection ------------------------

  /**
   * createPeerConnection - Creates an RTCPeerConnection with the STUN servers.
   */
    createPeerConnection(otherRank) {
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
            this.log(`ICE state change for ${otherRank}: ${pc.iceConnectionState}`);
            if (pc.iceConnectionState === 'failed') {
              this.log('ICE failed. You may want to handle retries here.');
            }
          };

        pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.log('ICE candidate generated');
            }
        };

        return pc;
    }

  /**
   * initiateConnection - We (the higher rank, or the first mover) create an offer
   * and set up a data channel. Then we send that to the peer via signaling server.
   */
    async initiateConnection(targetRank) {
        try {
            const pc = this.createPeerConnection(targetRank);
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
                        setTimeout(checkState, 10000);
                    }
                };
                checkState();
            });
            // await this.waitForIceGathering(pc, this.ICE_GATHERING_TIMEOUT);

            // Send offer
            await this.sendSignalingMessage(targetRank, {
                type: 'offer',
                sdp: pc.localDescription.sdp
            });
            this.log(`Sent offer to ${targetRank}`);

        } catch (error) {
            this.log(`Failed to initiate connection to ${targetRank}: ${error}`, 'error');
            await this.handleConnectionFailure(targetRank);
        }
    }
    /**
     * Wait for the ICE gathering to complete or time out.
     */
    waitForIceGathering(pc, timeoutMs) {
        return new Promise((resolve, reject) => {
        if (pc.iceGatheringState === 'complete') {
            resolve();
        } else {
            let timedOut = false;
            const timeout = setTimeout(() => {
            timedOut = true;
            reject('ICE gathering timed out');
            }, timeoutMs);

            pc.onicegatheringstatechange = () => {
            if (!timedOut && pc.iceGatheringState === 'complete') {
                clearTimeout(timeout);
                resolve();
            }
            };
        }
        });
    }

  /**
   * setupDataChannel - Called when we create the channel, or when the peer
   *   fires `ondatachannel`.
   */
    setupDataChannel(channel, peerRank) {
        this.dataChannels.set(peerRank, channel);
        this.peer_rounds.set(peerRank, 0);
        this.log(`Setting up data channel: ${JSON.stringify([...this.dataChannels.entries()])}`);

        channel.onopen = () => {
            this.log(`Data channel opened with peer ${peerRank}`);
            this.onPeerConnected(peerRank);
        };

        let messageBuffer = '';

        channel.onmessage = (event) => {
            try {
                // Append the incoming data to the buffer
                messageBuffer += event.data;

                // Try to parse the buffer as JSON
                const data = JSON.parse(messageBuffer);

                // If successful, handle the complete message
                console.log(`Received message from ${peerRank}: ${data.type}`);
                this.handleDataChannelMessage(peerRank, data);

                // Clear the buffer after successful parsing
                messageBuffer = '';
            } catch (error) {
                // If parsing fails, log the error and keep the buffer for further data
                if (error instanceof SyntaxError) {
                    // This is expected if the message is incomplete
                    console.log(`Waiting for more data to complete the message from ${peerRank}`);
                } else {
                    // Log other types of errors
                    this.log(`Failed to parse message from ${peerRank}: ${error}, data: ${messageBuffer.substring(0, 100)} ... ${messageBuffer.substring(messageBuffer.length-30)}`, 'error');
                    // Clear the buffer if it's a different error
                    messageBuffer = '';
                }
            }
        };

        channel.onclose = () => {
            this.log(`Data channel closed with peer ${peerRank}`);
            this.dataChannels.delete(peerRank);
            this.connectedPeers.delete(peerRank);
        };

        channel.onerror = (error) => {
            this.log(`Data channel error with peer ${peerRank}: ${error}`);
        }
    }

    /**
   * onPeerConnected - When the data channel opens, we count the connection as established.
   */
  onPeerConnected(peerRank) {
    this.pendingConnections.delete(peerRank);
    this.connectedPeers.add(peerRank);

    this.log(`Node ${this.rank} connected to peer ${peerRank}. ` +
             `Connected: ${this.connectedPeers.size}/${this.expectedConnections}`);

    // If we've reached the expected number, let the server know
    // if (this.connectedPeers.size === this.expectedConnections) {
    if (this.connectedPeers >= 2) {
      this.broadcastNodeReady();
    }

    // Let the server know
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'connection_established',
        peerRank: peerRank,
        sessionId: this.sessionId,
      }));
    }
  }

    /**
   * handleDataChannelMessage - Similar to Python's `on_message` callback.
   *   - Parse the message
   *   - Handle logic for pings, receiving weights, etc.
   */
  handleDataChannelMessage(peerRank, data) {
    try {
      // this.log(`Received message from peer ${peerRank}: ${data.type}`);

      switch (data.type) {

        case 'weights_request':
          this.log(`Received weights request from peer ${peerRank}`);

          // TODO: delete later
          // send a model request to see if it works
        //   for (const neighbor of this.neighbors){
            this.log(`Sending weights request for neighbor ${this.neighbors.neighbor1}`);
            this.sendToPeer(this.neighbors.neighbor1, {
                type: "weights_request",
                request_id: 1
            })
        //   }

          // TODO: figure out the rest of this part
        //   let currRound = model.round;
            let currRound = 1
          // If you want to send your model, chunk it up here:

          const chunk_size = 2000 // around 15 kb / 4 bytes per float32
            // model is an object: { layerName: tensor, ... }

            const weights = model.model.getWeights();
            this.log("Sending model weights. Keys:", Object.keys(weights));

            
            for (const [layerName, tensor] of Object.entries(weights)) {
                this.log(`Layer: ${layerName}, dtype: ${tensor.dtype}, shape: [${tensor.shape.join(', ')}]`);
            
                const chunks = chunkTensor(tensor, chunk_size);
                this.log(`numChunks = ${chunks.length} for layer: ${layerName}`);

                for (const { chunk, numChunks, originalShape } of chunks) {
                    const serializableChunk = serializeMessage({
                        layer_name: layerName,
                        chunk: chunk,
                        num_chunks: numChunks,
                        original_shape: originalShape
                    });
                
                    // Construct the message to send via WebRTC
                    const response = {
                        type: "weights_response",
                        weights: serializableChunk,
                        round: currRound,
                        request_id: data.request_id
                    };
                    // sendToPeer is your custom function to send data over the data channel
                    this.log(`Sending chunk of size: ${JSON.stringify(response).length} bytes`);
                    this.sendToPeer(peerRank, response);

                }
            }

            finishedMessage = {
                type: "weights_finished",
                round: currRound,
                request_id: data.request_id
            }
            this.sendToPeer(peerRank, finishedMessage)
          break;

        case 'weights_response':
          // Reassemble chunk. Increase comm_cost_received, etc.
          if (this.clear_peer_weights){
            this.peer_weights = {}
            this.clear_peer_weights = false
          }

            // 1. Deserialize the chunk
            const chunkData = deserializeMessage(data.weights);
            
            const layerName = chunkData.layer_name;
            const chunk = chunkData.chunk; // this is still a "tensor-like" object
            const numChunks = chunkData.num_chunks;
            const originalShape = chunkData.original_shape;
            
            // 2. Store the chunk data
            if (!this.peer_weights[layerName]) {
                this.peer_weights[layerName] = [];
            }
            this.peer_weights[layerName].push(chunk);
            
            // 3. Check if all chunks are received
            if (this.peer_weights[layerName].length === numChunks) {
                // Concatenate all chunk data
                let fullArray = [];
                for (let partialTensor of this.peer_weights[layerName]) {
                // partialTensor.data might be a typed array, so convert to normal array or push directly
                fullArray.push(...partialTensor.data);
                }
                // or if these are typed arrays, you could do something like:
                //   const totalLength = peerWeights[layerName].reduce((acc, t) => acc + t.data.length, 0);
                //   let fullTypedArray = new Float32Array(totalLength);
                //   // copy chunk by chunk ...
            
                // If you want a typed array again:
                const fullTypedArray = new Float32Array(fullArray);
            
                // This is your final reassembled tensor
                const reassembledTensor = {
                    __isTensor: true,
                    data: fullTypedArray,
                    dtype: chunk.dtype,
                    shape: originalShape
                };
            
                // Store it or use it:
                this.peer_weights[layerName] = reassembledTensor;
                
                this.log(`Reassembled layer ${layerName}: shape [${originalShape.join(', ')}]`);
            } else {
              this.log(`Received ${this.peer_weights[layerName].length} / ${numChunks} chunks for ${layerName}`);
            }

          break;

        case 'weights_finished':
          this.log(`Peer ${peerRank} finished sending weights.`);
          break;

        case 'round_update':
            this.log(`Received round update from peer ${peerRank}`);
            this.sendToPeer(peerRank, {
                type: "round_update_response",
                round: this.currentRound
            })
            break;
        
        case 'round_update_response':
            this.log(`Received round update response from peer ${peerRank}`);
            this.peer_rounds.set(peerRank, data.round)
            break;

        // Add other message types (round_update, etc.) similarly
      }
    } catch (err) {
      this.log(`handleDataChannelMessage() parse error: ${err}`);
    }
  }

  /**
   * Send a dictionary or object to a specific peer via data channel.
   */
  sendToPeer(peerRank, obj) {
    this.log(`Sending message to peer ${peerRank} (typeof ${typeof peerRank}): ${obj.type}`);
    const channel = this.dataChannels.get(peerRank);
    if (!channel || channel.readyState !== 'open') {
      this.log(`Channel to peer ${peerRank} not open or not found; dataChannels=${JSON.stringify([...this.dataChannels.entries()])} (typeof ${typeof this.dataChannels.keys()})`, 'error');
      this.log(`Channel state: ${channel ? channel.readyState : 'N/A'}`);
      return;
    }
    const msgString = JSON.stringify(obj);
    // channel.send(msgString);

    const MAX_BUFFER = 65535; // Adjust based on testing
    if (channel.bufferedAmount > MAX_BUFFER) {
        setTimeout(() => this.sendToPeer(peerRank, obj), 100);
    } else {
        channel.send(msgString);
    }

    // Update "communication cost sent" if relevant
    this.comm_cost_sent += msgString.length;
  }

  // -------------------------- Signaling & ICE Handling --------------------------

  /**
   * handleSignalingMessage - React to "offer", "answer", or "candidate" from the server.
   */
    async handleSignalingMessage(message) {
        const senderRank = message.senderRank;
        const data = message.data;
        let pc = this.connections.get(senderRank);
        this.log(`Received signaling message from ${senderRank}: ${data.type}`);
        try {
            // If we don't have a PeerConnection yet, create one (the "answerer" side).
            if (!pc) {
                this.log(`Creating new PeerConnection for ${senderRank}`);
                pc = this.createPeerConnection(senderRank);
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
                await this.sendSignalingMessage(senderRank, {
                    type: 'answer',
                    sdp: answer.sdp,
                });

                ///////////////////////////////////////////
                // await this.waitForIceGathering(pc, this.ICE_GATHERING_TIMEOUT);
                // Send answer back
                // this.sendSignalingMessage(senderRank, {
                //     type: 'answer',
                //     // sdp: pc.localDescription.sdp,
                //     sdp: answer.sdp
                // });
                ///////////////////////////////////////////

            } else if (data.type === 'answer') {
                await pc.setRemoteDescription(new wrtc.RTCSessionDescription({
                    type: 'answer',
                    sdp: data.sdp
                }));
            } else if (data.type === 'candidate') {
                this.log(`?? should we get here??? Adding ICE candidate for ${senderRank}`);
                await pc.addIceCandidate({
                    candidate: data.candidate,
                    sdpMLineIndex: 0,
                    sdpMid: '0'
                });
            }
        } catch (error) {
            this.log(`handleSignalingMessage error: ${error}`);
        }
    }

    async sendSignalingMessage(targetRank, data) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.log('Cannot sendSignalingMessage: WebSocket is not open.');
            return;
        }
        this.ws.send(JSON.stringify({
            type: 'signal',
            targetRank: targetRank,
            data: data,
            sessionId: this.sessionId
        }));
    }

    /**
     * broadcastNodeReady - Notifies the signaling server that we've set up all channels.
     */
    broadcastNodeReady() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
            type: 'node_ready',
            sessionId: this.sessionId,
            rank: this.rank
        }));
        }
    }
  // --------------------- Error Handling & Cleanup ---------------------

  /**
   * handleConnectionFailure - Retry if we fail, up to MAX_RETRIES
   */
    async handleConnectionFailure(targetRank) {
        const retryCount = this.connectionRetries.get(targetRank) || 0;
        if (retryCount < this.MAX_RETRIES) {
            this.connectionRetries.set(targetRank, retryCount + 1);
            this.log(`Retrying connection to ${targetRank}, attempt #${retryCount + 1}`);
            await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY));
            if (!this.connectedPeers.has(targetRank)) {
                await this.cleanupConnection(targetRank);
                this.initiateConnection(targetRank);
            }
        } else {
            this.log(`Max retries reached for ${targetRank}`, 'error');
            await this.cleanupConnection(targetRank);
        }
    }

  /**
   * cleanupConnection - Close data channel & PeerConnection, remove references.
   */
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

    // sendModelWeights(model, chunkSize, sendToPeer) {
    //     // model is an object: { layerName: tensor, ... }
    //     console.log("Sending model weights. Keys:", Object.keys(model));
      
    //     for (const [layerName, tensor] of Object.entries(model)) {
    //       console.log(`Layer: ${layerName}, dtype: ${tensor.dtype}, shape: [${tensor.shape.join(', ')}]`);
      
    //       const chunks = chunkTensor(tensor, chunkSize);
    //       for (const { chunk, numChunks, originalShape } of chunks) {
    //         const serializableChunk = serializeMessage({
    //           layer_name: layerName,
    //           chunk: chunk,
    //           num_chunks: numChunks,
    //           original_shape: originalShape
    //         });
      
    //         // Construct the message to send via WebRTC
    //         const response = {
    //           type: "weights_response",
    //           weights: serializableChunk,
    //           // ... include other metadata you need, e.g. round or request_id
    //         };
    //         // sendToPeer is your custom function to send data over the data channel
    //         this.sendToPeer(response);
    //       }
    //     }
    //   }
    
    /**
     * handleReceivedChunk is called whenever a "weights_response" message arrives.
     */
    // function handleReceivedChunk(data) {
    //   // 1. Deserialize the chunk
    //   const chunkData = deserializeMessage(data.weights);
      
    //   const layerName = chunkData.layer_name;
    //   const chunk = chunkData.chunk; // this is still a "tensor-like" object
    //   const numChunks = chunkData.num_chunks;
    //   const originalShape = chunkData.original_shape;
    
    //   // 2. Store the chunk data
    //   if (!peerWeights[layerName]) {
    //     peerWeights[layerName] = [];
    //   }
    //   peerWeights[layerName].push(chunk);
    
    //   // 3. Check if all chunks are received
    //   if (peerWeights[layerName].length === numChunks) {
    //     // Concatenate all chunk data
    //     let fullArray = [];
    //     for (let partialTensor of peerWeights[layerName]) {
    //       // partialTensor.data might be a typed array, so convert to normal array or push directly
    //       fullArray.push(...partialTensor.data);
    //     }
    //     // or if these are typed arrays, you could do something like:
    //     //   const totalLength = peerWeights[layerName].reduce((acc, t) => acc + t.data.length, 0);
    //     //   let fullTypedArray = new Float32Array(totalLength);
    //     //   // copy chunk by chunk ...
    
    //     // If you want a typed array again:
    //     const fullTypedArray = new Float32Array(fullArray);
    
    //     // This is your final reassembled tensor
    //     const reassembledTensor = {
    //       __isTensor: true,
    //       data: fullTypedArray,
    //       dtype: chunk.dtype,
    //       shape: originalShape
    //     };
    
    //     // Store it or use it:
    //     peerWeights[layerName] = reassembledTensor;
        
    //     console.log(`Reassembled layer ${layerName}: shape [${originalShape.join(', ')}]`);
    //   }
    // }
      
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
    signaling_server: signalingServer,
    num_users: MAX_CLIENTS,
    session_id: SESSION_ID
}
const node = new WebRTCCommUtils(config)

module.exports = { WebRTCCommUtils };