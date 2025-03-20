const WebSocket = require('ws');
const wrtc = require('wrtc');  // Import WebRTC for Node.js
// const { ResNet10 } = require('./browser_client/model.js');
const { ResNet10 } = require('./model_archive.js');
const path = require('path');
const fs = require('fs');const tf = require('@tensorflow/tfjs-node');
// TODO: this can be replaced by just the browser-side without wrtc once we use browser
function processData(jsonData) {
	const images = jsonData.map(item => item.image)
	const labels = jsonData.map(item => item.label)

	return {
		images: images,
		labels: labels
	}
}

/**
 * Convert TensorFlow.js NHWC weights to TensorFlow (Python) NCHW format.
 * Also handles linear (dense) layers by transposing weight matrices.
 * @param {tf.Tensor} weightTensor - The weight tensor from TensorFlow.js.
 * @returns {tf.Tensor} - Transposed tensor in NCHW or correct format for linear layers.
 */
function convertTfjsToTf(weightTensor) {
  if (weightTensor.shape.length === 4) {
      return tf.transpose(weightTensor, [3, 2, 0, 1]); // NHWC -> NCHW
  } else if (weightTensor.shape.length === 2) {
      return tf.transpose(weightTensor, [1, 0]); // Transpose linear layer weights
  }
  return weightTensor; // Return as is if not 2D or 4D
}

/**
* Convert TensorFlow (Python) NCHW weights back to TensorFlow.js NHWC format.
* Also handles linear (dense) layers by transposing weight matrices.
* Includes robust error handling and shape verification.
* @param {tf.Tensor} weightTensor - The weight tensor from Python.
* @param {tf.Tensor} [targetTensor=null] - Optional target tensor to match shape with.
* @param {Function} [logFunc=console.log] - Logging function.
* @returns {tf.Tensor|null} - Transposed tensor in correct format, or null if conversion failed.
*/
function convertTfToTfjs(weightTensor, targetTensor = null, logFunc = console.log) {
  try {
    // If we have a target tensor, verify element count matches
    if (targetTensor !== null) {
      const targetElements = targetTensor.shape.reduce((a, b) => a * b, 1);
      const sourceElements = weightTensor.shape.reduce((a, b) => a * b, 1);
      
      if (targetElements !== sourceElements) {
        logFunc(`Element count mismatch: target has ${targetElements}, source has ${sourceElements}`);
        return null;
      }
    }
    
    let result;
    
    // Standard conversion for common tensor shapes
    if (weightTensor.shape.length === 4) {
      // Convert NCHW to NHWC format for conv layers
      result = tf.transpose(weightTensor, [2, 3, 1, 0]);
    } else if (weightTensor.shape.length === 2) {
      // Transpose linear layer weights
      result = tf.transpose(weightTensor, [1, 0]);
    } else {
      // For other shapes, return as is initially
      result = weightTensor;
    }
    
    // If we have a target shape and need to reshape further
    if (targetTensor !== null && !arraysEqual(result.shape, targetTensor.shape)) {
      logFunc(`Shape still mismatched after standard conversion. ` +
              `Source: ${result.shape}, Target: ${targetTensor.shape}`);
      
      try {
        // Try direct reshape
        const reshaped = result.reshape(targetTensor.shape);
        result.dispose(); // Clean up the intermediate tensor
        result = reshaped;
      } catch (reshapeError) {
        logFunc(`Direct reshape failed: ${reshapeError.message}`);
        
        // Try flatten and reshape as fallback
        try {
          const flattened = result.flatten();
          result.dispose(); // Clean up the intermediate tensor
          
          const reshaped = flattened.reshape(targetTensor.shape);
          flattened.dispose(); // Clean up the flattened tensor
          
          result = reshaped;
          logFunc(`Fallback reshape succeeded`);
        } catch (fallbackError) {
          logFunc(`Fallback reshape also failed: ${fallbackError.message}`);
          result.dispose(); // Clean up
          return null; // Signal conversion failure
        }
      }
    }
    
    return result;
  } catch (error) {
    logFunc(`Error in convertTfToTfjs: ${error.message}`);
    return null;
  }
}

// Helper function to compare arrays for equality (for tensor shape comparison)
function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

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
    constructor(config, trainDataset, testDataset = null) {
        this.startTime = Date.now();
        this.model = new ResNet10();
        this.config = config || {};
        this.signalingServer = this.config.signaling_server || 'ws://localhost:8765';
        this.trainDataset = trainDataset;
        this.testDataset = testDataset;
    
        // Networking & session references
        this.ws = null;                               // WebSocket connection
        this.sessionId = this.config.sessionId || 1111;
        this.rank = null;
        this.size = this.config.num_users || 2;
        this.num_collaborators = this.config.num_collaborators || 1;
        this.expectedConnections = 0;
    
        // WebRTC connections
        this.connections = new Map();                 // peerRank -> RTCPeerConnection
        this.dataChannels = new Map();                // peerRank -> RTCDataChannel
        this.connectedPeers = new Set();
        this.pendingConnections = new Set();
    
        // Connection management
        this.connectionRetries = new Map();           // peerRank -> retryCount
        this.MAX_RETRIES = 3;
        this.RETRY_DELAY = 30000;                    // Increased from 15000 to 30000
        this.ICE_GATHERING_TIMEOUT = 20000;          // Increased from 10000 to 20000
        this.weightReceiptTimeout = 3 * 60 * 1000; // 3 minutes timeout
    
        // State
        this.state = NodeState.CONNECTING;
        
        // Extra placeholders for distributed training logic
        this.currentRound = 0;
        this.peer_rounds = new Map();
        this.peer_weights = new Map(); // Changed from object to Map to track weights by peer rank
        this.clear_peer_weights = false;
        this.weights_finished = false;
        this.expectedLayers = new Set(); // Track which layers we expect to receive
        this.receivedWeightsFrom = new Set(); // Track which peers we've received weights_finished from
        
        // Tracking chunks and completion per peer
        this.layerChunkTracker = new Map(); // Map of peer_rank -> { layerName: { expected, received } }
        this.receivedWeightsFinished = new Map(); // Map of peer_rank -> boolean
    
        // Communication cost counters
        this.comm_cost_sent = 0;
        this.comm_cost_received = 0;
        this.bytesReceived = 0;
        this.bytesSent = 0;
    
        // Simple logging
        this.allLogsPath = path.join('logs', `${this.startTime}`, `all_logs.log`);
        if (!fs.existsSync(path.dirname(this.allLogsPath))) {
          fs.mkdirSync(path.dirname(this.allLogsPath), { recursive: true });
        }
        fs.writeFileSync(this.allLogsPath, '');
        this.log(`[constructor] RTCCommUtilsJS created with config: ${JSON.stringify(config)}`);
        this.connect()

        this.js2pythonMapping = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'js2python.json'), 'utf8'));
    }

    // ---------------------- Basic Logging & State Helpers ----------------------

    log(msg) {
        console.log(`${msg}`);
        // Append to log file
        fs.appendFileSync(this.allLogsPath, msg + '\n');
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
                
                // Initialize metrics logger now that we have the rank
                this.metricsLogger = new MetricsLogger(path.join('logs', `${this.startTime}`, `node_${this.rank}`));
                
                // Initialize metrics files
                ['test_acc', 'test_loss', 'test_time', 
                 'train_acc', 'train_loss', 'train_time',
                 'time_elapsed', 'bytes_sent', 'bytes_received',
                 'peak_dram', 'peak_gpu', 'neighbors', 'test_acc_post_agg'].forEach(metric => {
                    this.metricsLogger.initializeMetric(metric);
                });
                
                this.log(`Initialized metrics logging for client ${this.rank}`);
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
                this.startTraining()
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
        for (const neighborList of Object.values(this.neighbors)) {
          // TODO: uncomment this condition later
            // if (neighborRank > this.rank && 
            //     !this.connections.has(neighborRank) && 
            //     !this.pendingConnections.has(neighborRank)) {
            //     this.log(`Initiating connection to ${neighborRank}`);
            //     this.pendingConnections.add(neighborRank);
            //     this.initiateConnection(neighborRank);
            // }
            for (const neighbor of neighborList) {
              this.log(`Initiating connection to ${neighbor}`);
              this.pendingConnections.add(neighbor);
              this.initiateConnection(neighbor);
            }
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
    setupDataChannel(channel, peerRankStr) {
        const peerRank = Number(peerRankStr);
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
                // console.log(`Received message from ${peerRank}: ${data.type}`);
                this.handleDataChannelMessage(peerRank, data);

                // Clear the buffer after successful parsing
                messageBuffer = '';
            } catch (error) {
                // If parsing fails, log the error and keep the buffer for further data
                if (error instanceof SyntaxError) {
                    // This is expected if the message is incomplete
                    // console.log(`Waiting for more data to complete the message from ${peerRank}`);
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
  onPeerConnected(peerRankStr) {
    const peerRank = Number(peerRankStr);
    this.pendingConnections.delete(peerRank);
    this.connectedPeers.add(peerRank);

    this.log(`Node ${this.rank} connected to peer ${peerRank}. ` +
             `Connected: ${this.connectedPeers.size}/${this.expectedConnections}`);

    // If we've reached the expected number, let the server know
    if (this.connectedPeers.size >= this.expectedConnections) {
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
  handleDataChannelMessage(peerRankStr, data) {
    try {
      const peerRank = parseInt(peerRankStr);
      // Track bytes received (approximately) - using the stringified data size
      const dataSize = JSON.stringify(data).length;
      this.bytesReceived += dataSize;
      // Don't log every message, we'll log the total at the end of the round
      
      // this.log(`Received message from peer ${peerRank}: ${data.type}`);

      switch (data.type) {

        case 'weights_request':
          this.log(`Received weights request from peer ${peerRank}`);

          const currRound = 1;
          const chunk_size = 2000;
          const weights = this.model.model.getWeights();
          const layers = this.model.model.layers;

          for (let i = 0; i < layers.length; i++) {
              const layer = layers[i];
              const layerWeights = layer.getWeights();

              for (let j = 0; j < layerWeights.length; j++) {
                  let weightTensor = layerWeights[j];

                  // Convert to TensorFlow (Python) format before sending
                  weightTensor = convertTfjsToTf(weightTensor);

                  // Rename the layer and map to Python layer name
                  const layerName = `${layer.name}_weight_${j}`;
                  const pythonLayerName = this.js2pythonMapping[layerName];

                  const chunks = chunkTensor(weightTensor, chunk_size);
                  // this.log(`Layer ${i}_${j}: ${layerName} -> ${pythonLayerName}, dtype: ${weightTensor.dtype}, shape: [${weightTensor.shape.join(', ')}], numChunks: ${chunks.length}`);

                  let chunkIdx = 0; // Add chunk index counter
                  for (const { chunk, numChunks, originalShape } of chunks) {
                      const serializableChunk = serializeMessage({
                          layer_name: pythonLayerName,
                          chunk: chunk,
                          chunk_idx: chunkIdx, // Add chunk index to serialized data
                          num_chunks: numChunks,
                          original_shape: originalShape
                      });

                      const response = {
                          type: "weights_response",
                          weights: serializableChunk,
                          round: currRound,
                          request_id: data.request_id
                      };

                      this.sendToPeer(peerRank, response);
                      chunkIdx++; // Increment chunk index
                  }
              }
          }

          const finishedMessage = {
              type: "weights_finished",
              round: currRound,
              request_id: data.request_id
          };
          this.sendToPeer(peerRank, finishedMessage);
          break;
        case 'weights_response': {
          try {
            // Initialize tracking structures for this peer if needed
            if (!this.layerChunkTracker.has(peerRank)) {
              this.layerChunkTracker.set(peerRank, {});
            }
            
            if (!this.peer_weights.has(peerRank)) {
              this.peer_weights.set(peerRank, {});
            }
            
            // 1. Deserialize the chunk
            const chunkData = deserializeMessage(data.weights);
            
            let chunk = chunkData.chunk;
            chunk = convertTfToTfjs(chunk); // Convert to TensorFlow.js format
            
            const layerName = chunkData.layer_name;
            const chunkIdx = chunkData.chunk_idx; // Get the chunk index
            const numChunks = chunkData.num_chunks;
            const originalShape = chunkData.original_shape;
            
            // Get this peer's weights and layer trackers
            const peerWeights = this.peer_weights.get(peerRank);
            const peerLayerTrackers = this.layerChunkTracker.get(peerRank);
            
            // Initialize the layer tracking for this peer if needed
            if (!peerWeights[layerName]) {
              // Initialize with array of proper size filled with nulls
              peerWeights[layerName] = Array(numChunks).fill(null);
              
              // Track this layer
              peerLayerTrackers[layerName] = {
                expected: numChunks,
                received: 0
              };
              
              // this.log(`New layer ${layerName} from peer ${peerRank}, expecting ${numChunks} chunks`);
            }

            // Store the chunk at correct position using its index
            peerWeights[layerName][chunkIdx] = chunk;
            peerLayerTrackers[layerName].received++;
            
            // Log progress periodically
            const tracker = peerLayerTrackers[layerName];
            if (tracker.received === tracker.expected || 
                tracker.received === 1 || 
                tracker.received % 10 === 0) {
              // this.log(`Layer ${layerName} from peer ${peerRank}: ${tracker.received}/${tracker.expected} chunks`);
            }
            
            // If we've received all chunks for this layer, reconstruct it
            if (tracker.received === tracker.expected) {
              // Check that all chunks are received (no null values)
              if (!peerWeights[layerName].includes(null)) {
                // this.log(`✓ Layer ${layerName} from peer ${peerRank} complete: all ${numChunks} chunks received`);
                
                // Concatenate all chunk data
                let fullArray = [];
                for (let partialTensor of peerWeights[layerName]) {
                  fullArray.push(...Array.from(partialTensor.data));
                }
                
                // Create the reassembled tensor
                const reassembledTensor = {
                  __isTensor: true,
                  data: new Float32Array(fullArray),
                  dtype: chunk.dtype,
                  shape: originalShape
                };
                
                // Replace the array of chunks with the complete tensor
                peerWeights[layerName] = reassembledTensor;
                
                // Check if we can move on now
                if (this.canMoveOn()) {
                  this.log(`All weights are now complete from all peers!`);
                }
              } else {
                // Some chunks are missing despite the counter reaching expected count
                this.log(`Warning: Layer ${layerName} from peer ${peerRank} has missing chunks despite count reaching expected`, 'warning');
                // Decrement the counter since we can't consider it complete
                peerLayerTrackers[layerName].received--;
              }
            }
            
          } catch (error) {
            this.log(`Error processing weights_response: ${error.stack}`, 'error');
          }
          break;
        }

        case 'weights_finished': {
          this.log(`Received weights_finished from peer ${peerRank}`);
          this.receivedWeightsFinished.set(peerRank, true);
          
          // Check if we're still missing any chunks from this peer
          if (this.layerChunkTracker.has(peerRank)) {
            const peerLayerTrackers = this.layerChunkTracker.get(peerRank);
            const incompleteLayers = Object.entries(peerLayerTrackers)
              .filter(([_, info]) => info.received < info.expected)
              .map(([name, info]) => `${name} (${info.received}/${info.expected})`)
              .join(', ');
            
            if (incompleteLayers) {
              this.log(`Received weights_finished from peer ${peerRank} but still waiting for chunks: ${incompleteLayers}`);
            } else if (Object.keys(peerLayerTrackers).length === 0) {
              this.log(`Received weights_finished from peer ${peerRank} but no layers have been registered yet`);
            } else {
              this.log(`✓ Received weights_finished from peer ${peerRank} and all tracked layers are complete`);
            }
          } else {
            this.log(`Received weights_finished from peer ${peerRank} but no layer tracking data exists`);
          }
          
          // Check if we can move on now
          if (this.canMoveOn()) {
            this.log(`All weights are now complete from all peers!`);
          }
          
          break;
        }

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
    // this.log(`Sending message to peer ${peerRank} (typeof ${typeof peerRank}): ${obj.type}`);
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

  /**
   * Single helper function to check if we've received all chunks from all layers
   * @returns {boolean} True if we can move on, false if we need to wait
   */
  canMoveOn() {
    // If we haven't received any weights finished signals or have no neighbors, we can't move on
    // if (this.receivedWeightsFinished.size === 0 || this.connectedPeers.size === 0) {
    //   return false;
    // }
    
    // Check if we've received weights_finished from all neighbors we're connected to
    for (const peerRank of this.collaborator_list) {
      if (!this.receivedWeightsFinished.has(peerRank) || !this.receivedWeightsFinished.get(peerRank)) {
        return false;
      }
    }
    
    // If we have no layers being tracked for any peer, we can't move on yet
    // if (this.layerChunkTracker.size === 0) {
    //   return false;
    // }
    
    // Check if all tracked layers for all peers have received all their expected chunks
    for (const [peerRank, layerTrackers] of this.layerChunkTracker.entries()) {
      for (const layerName in layerTrackers) {
        const tracker = layerTrackers[layerName];
        if (tracker.received < tracker.expected) {
          return false; // This layer is still missing chunks for this peer
        }
      }
    }
    
    // All checks passed
    return true;
  }

  async receive() {
    
    // Reset our tracking state
    this.clear_peer_weights = true;
    this.layerChunkTracker = new Map(); // Reset per-peer layer tracking
    this.receivedWeightsFinished = new Map(); // Reset per-peer completion status
    this.peer_weights = new Map(); // Reset peer weights map
    this.weightReceiptTimeout = 3 * 60 * 1000; // 3 minutes timeout
    
    // Create a promise that will resolve when all weights are received
    const waitForWeights = new Promise((resolve, reject) => {
      // Set timeout to prevent indefinite waiting
      const timeout = setTimeout(() => {
        // Log what we're still waiting for if timeout occurs
        let incompleteInfo = [];
        
        for (const [peerRank, layerTrackers] of this.layerChunkTracker.entries()) {
          const incompleteLayerNames = Object.entries(layerTrackers)
            .filter(([_, info]) => info.received < info.expected)
            .map(([name, info]) => `${name} (${info.received}/${info.expected})`)
            .join(', ');
            
          if (incompleteLayerNames) {
            incompleteInfo.push(`Peer ${peerRank}: ${incompleteLayerNames}`);
          }
        }
          
        reject(new Error(`Weights receipt timed out after ${this.weightReceiptTimeout/1000} seconds. Still waiting for: ${incompleteInfo.join('; ')}`));
      }, this.weightReceiptTimeout);
      
      // Create a check function that periodically checks if we can move on
      const checkComplete = () => {
        if (this.canMoveOn()) {
          this.log("✓ All expected layers and chunks received from all peers, resolving promise");
          clearTimeout(timeout);
          resolve();
        } else {
          setTimeout(checkComplete, 100); // Check again in 100ms
        }
      };
      
      // Start checking
      checkComplete();
    });
    
    // Send weight requests to all neighbors
    for (const neighborRank of this.collaborator_list) {
      this.log(`Sending weights request for neighbor ${neighborRank}`);
      this.sendToPeer(neighborRank, {
        type: "weights_request",
        request_id: Date.now() // Use timestamp as unique request ID
      });
    }
    
    try {
      // Wait for the weights to be received
      await waitForWeights;
      
      this.log(`✓ Successfully received weights from ${this.peer_weights.size} peers`);
      
      // Convert Map to array of objects to maintain compatibility with the existing aggregate function
      const peerWeightsArray = [];
      for (const [peerRank, weights] of this.peer_weights.entries()) {
        peerWeightsArray.push({
          sender: peerRank,
          model: weights
        });
      }
      
      return peerWeightsArray;
    } catch (error) {
      this.log(`Error in receive(): ${error.message}`);
      // Return whatever weights we have so far as an array of objects
      const peerWeightsArray = [];
      for (const [peerRank, weights] of this.peer_weights.entries()) {
        peerWeightsArray.push({
          sender: peerRank,
          model: weights
        });
      }
      return peerWeightsArray;
    }
  }

  async aggregate(peer_weights) {
    this.log("Aggregating model weights");
    
    try {
      // Count how many peer models we're aggregating with
      const peerModelCount = peer_weights.length;
      
      this.log(`Peer weights received: ${peerModelCount} entries`);
      
      // Get current model weights and layers
      const layers = this.model.model.layers;
      this.log(`Model has ${layers.length} layers`);
      
      // Make a clone of all current weights to avoid modifying the original tensors directly
      const originalWeights = [];
      const layerMap = {};
      let weightsList = [];
      
      // First, get all the current weights in order and clone them
      for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        const layerWeights = layer.getWeights();
        
        for (let j = 0; j < layerWeights.length; j++) {
          const jsLayerName = `${layer.name}_weight_${j}`;
          // this.log(`AGG2: Layer ${i}_${j}: ${jsLayerName} has ${layerWeights[j].shape}`);
          
          // Clone the weight tensor to avoid modifying the original
          const clonedWeight = layerWeights[j].clone();
          
          layerMap[jsLayerName] = weightsList.length;
          weightsList.push(clonedWeight);
          originalWeights.push(layerWeights[j]); // Keep reference to originals
        }
      }
      
      // Create a reverse mapping from Python layer names to JS layer names
      const python2jsMapping = {};
      for (const [jsName, pythonName] of Object.entries(this.js2pythonMapping)) {
        python2jsMapping[pythonName] = jsName;
      }      
      // Count how many peer models we're aggregating with
      // TODO: make this flexible
      // const peerModelCount = 1
      
      this.log(`Found ${peerModelCount} peer models for aggregation`);
      
      // Only proceed if we have peers to aggregate with
      if (peerModelCount > 0) {
        this.log("Starting weight aggregation with peers");
        
        // Process each peer weight tensor
        for (const { sender, model } of peer_weights) {
          for (const [pythonLayerName, weightTensor] of Object.entries(model)) {
            // Skip if this isn't a weight tensor or if we don't have a mapping for it
            if (!weightTensor || !weightTensor.__isTensor || !python2jsMapping[pythonLayerName]) {
              this.log(`Skipping ${pythonLayerName}: is tensor? ${weightTensor?.__isTensor}, has mapping? ${Boolean(python2jsMapping[pythonLayerName])}`);
              continue;
            }
            
            const jsLayerName = python2jsMapping[pythonLayerName];
            const weightIndex = layerMap[jsLayerName];
            
            // Skip if we don't have this layer in our model
            if (weightIndex === undefined) {
              this.log(`Warning: No matching weight index for ${jsLayerName} (python: ${pythonLayerName})`);
              continue;
            }
            
            // this.log(`Processing ${pythonLayerName} -> ${jsLayerName} at index ${weightIndex}`);
            
            // Get the current weight tensor for this layer (the cloned one)
            const currentWeight = weightsList[weightIndex];
            
            // Convert torch dtype to TF.js compatible dtype
            let tensorDtype = weightTensor.dtype;
            if (tensorDtype && tensorDtype.startsWith('torch.')) {
              tensorDtype = tensorDtype.replace('torch.', '');
            }
            
            const incomingData = Array.from(weightTensor.data);
            
            // Create the incoming tensor with the original shape from PyTorch
            const incomingTensor = tf.tensor(
              incomingData,
              weightTensor.shape,
              tensorDtype
            );
            
            // Get the expected shape from the current model weights
            const expectedShape = currentWeight.shape;
            // this.log(`Expected tensor shape: ${expectedShape}, incoming shape: ${incomingTensor.shape}`);
            
            // Use our enhanced convertTfToTfjs function to handle reshaping and conversion
            const convertedTensor = convertTfToTfjs(incomingTensor, currentWeight, this.log.bind(this));
            
            // Skip if conversion failed
            if (!convertedTensor) {
              this.log(`Conversion failed for ${jsLayerName}, skipping`);
              incomingTensor.dispose();
              continue;
            }
            
            // Add to current weights and create a new tensor to store the result
            const updatedTensor = currentWeight.add(convertedTensor);
            // this.log(`Updated tensor created for ${jsLayerName}`);
            
            // Replace in our weights list
            weightsList[weightIndex].dispose(); // Dispose the old cloned tensor
            weightsList[weightIndex] = updatedTensor;
            
            // Clean up tensors to avoid memory leaks
            incomingTensor.dispose();
            convertedTensor.dispose();
          }
        }
        
        // Average the weights (divide by total number of models)
        const totalModels = 1 + peerModelCount; // Current model + peer models
        this.log(`Averaging weights across ${totalModels} models (1 local + ${peerModelCount} peers)`);
        const scalar = tf.scalar(1 / totalModels);
        
        // Scale each weight tensor
        for (let i = 0; i < weightsList.length; i++) {
          const tensor = weightsList[i];
          
          // Create a new scaled tensor
          const scaledTensor = tensor.mul(scalar);
          
          // Dispose the old tensor and replace with the scaled one
          tensor.dispose();
          weightsList[i] = scaledTensor;
        }
        
        scalar.dispose(); // Clean up the scalar
        
        // Verify shapes match the original model before setting weights
        const finalWeightsList = [];
        
        for (let i = 0; i < weightsList.length; i++) {
          const aggregatedWeight = weightsList[i];
          const originalShape = originalWeights[i].shape;
          
          if (!arraysEqual(originalShape, aggregatedWeight.shape)) {
            this.log(`SHAPE MISMATCH for tensor ${i}: expected ${originalShape}, got ${aggregatedWeight.shape}`);
            
            try {
              // Try to reshape
              this.log(`Attempting to reshape tensor ${i} from ${aggregatedWeight.shape} to ${originalShape}`);
              const reshapedWeight = aggregatedWeight.reshape(originalShape);
              this.log(`Reshape succeeded!`);
              finalWeightsList.push(reshapedWeight);
              aggregatedWeight.dispose();
            } catch (error) {
              this.log(`Error reshaping tensor ${i}: ${error.message}`);
              
              // Additional logging and diagnostic information
              const originalElements = originalShape.reduce((a, b) => a * b, 1);
              const aggregatedElements = aggregatedWeight.shape.reduce((a, b) => a * b, 1);
              this.log(`Element counts - Original: ${originalElements}, Aggregated: ${aggregatedElements}`);
              
              if (originalElements === aggregatedElements) {
                this.log(`Element counts match, but reshape failed. Attempting to flatten and reshape.`);
                try {
                  const flattened = aggregatedWeight.flatten();
                  const reshaped = flattened.reshape(originalShape);
                  this.log(`Flatten and reshape succeeded!`);
                  finalWeightsList.push(reshaped);
                  flattened.dispose();
                  aggregatedWeight.dispose();
                } catch (secondError) {
                  this.log(`Flatten and reshape also failed: ${secondError.message}`);
                  this.log(`Falling back to original weight.`);
                  // Clone the original weight to ensure we don't modify it
                  finalWeightsList.push(originalWeights[i].clone());
                  aggregatedWeight.dispose();
                }
              } else {
                this.log(`Element counts don't match. Using original weight.`);
                // Clone the original weight to ensure we don't modify it
                finalWeightsList.push(originalWeights[i].clone());
                aggregatedWeight.dispose();
              }
            }
          } else {
            // this.log(`Shapes match for tensor ${i}`);
            finalWeightsList.push(aggregatedWeight);
          }
        }
        
        // Set the aggregated weights back to the model
        // this.log(`Setting ${finalWeightsList.length} weights back to model`);
        
        // LAST RESORT: Instead of trying to set all weights at once, set them layer by layer
        try {
          this.log("Attempting to set weights layer by layer instead of all at once");
          let weightIndex = 0;
          
          for (let i = 0; i < layers.length; i++) {
            const layer = layers[i];
            const layerWeights = layer.getWeights();
            
            if (layerWeights.length > 0) {
              // Extract just the weights needed for this layer
              const weightsForLayer = [];
              for (let j = 0; j < layerWeights.length; j++) {
                if (weightIndex < finalWeightsList.length) {
                  weightsForLayer.push(finalWeightsList[weightIndex]);
                  // this.log(`Setting for layer ${layer.name} (${i}): weight ${j}, shape ${finalWeightsList[weightIndex].shape}`);
                  weightIndex++;
                }
              }
              
              // Set weights just for this specific layer
              if (weightsForLayer.length === layerWeights.length) {
                try {
                  layer.setWeights(weightsForLayer);
                } catch (layerError) {
                  this.log(`Error setting weights for layer ${layer.name}: ${layerError.message}`);
                  
                  // If this specific layer fails, use its original weights
                  this.log(`Falling back to original weights for layer ${layer.name}`);
                  const originalLayerWeights = layerWeights.map(w => w.clone());
                  layer.setWeights(originalLayerWeights);
                  
                  // Skip ahead in the index
                  weightIndex -= weightsForLayer.length;
                  weightIndex += layerWeights.length;
                }
              }
            }
          }
          
          this.log("Layer-by-layer weight setting completed");
        } catch (layeredError) {
          this.log(`Error in layer-by-layer approach: ${layeredError.message}`);
          this.log("Falling back to original weights for the entire model");
          
          // Clean up final weights
          finalWeightsList.forEach(w => {
            if (w && !w.isDisposed) {
              w.dispose();
            }
          });
          
          // Set original weights back to the model
          const safeOriginalWeights = originalWeights.map(w => w.clone());
          this.model.model.setWeights(safeOriginalWeights);
        }
      } else {
        this.log("No peer tensors found for aggregation, keeping original weights");
        
        // Clean up our cloned weights
        weightsList.forEach(w => w.dispose());
      }
      
      this.log("Model weights aggregated successfully");
    } catch (error) {
      this.log(`Error in aggregate: ${error.message}`);
      this.log(error.stack);
      
      // Additional error info
      this.log("Error context:");
      try {
        this.log(`Model defined: ${Boolean(this.model && this.model.model)}`);
        this.log(`Peer weights type: ${typeof peer_weights}`);
        if (peer_weights) {
          this.log(`Peer weights keys: ${Object.keys(peer_weights).join(', ')}`);
        }
      } catch (e) {
        this.log(`Error while logging debug info: ${e.message}`);
      }
    }
  }
  
  /**
   * Helper method to scale a weight tensor by a coefficient
   * @param {Object} weight - The weight tensor object
   * @param {number} coeff - The coefficient to scale by
   * @returns {Object} The scaled weight tensor
   */
  scaleWeight(weight, coeff) {
    // Handle tensor.js objects
    if (weight.__isTensor) {
      const scaledData = new Float32Array(weight.data.length);
      for (let i = 0; i < weight.data.length; i++) {
        scaledData[i] = weight.data[i] * coeff;
      }
      return {
        __isTensor: true,
        data: scaledData,
        dtype: weight.dtype,
        shape: weight.shape
      };
    }
    // Handle tf.js tensors
    else if (weight instanceof tf.Tensor) {
      return weight.mul(coeff);
    }
    // Handle plain arrays or TypedArrays
    else {
      const result = new Float32Array(weight.length);
      for (let i = 0; i < weight.length; i++) {
        result[i] = weight[i] * coeff;
      }
      return result;
    }
  }
  
  /**
   * Helper method to add two weight tensors
   * @param {Object} weight1 - The first weight tensor
   * @param {Object} weight2 - The second weight tensor
   * @returns {Object} The sum of the two weight tensors
   */
  addWeights(weight1, weight2) {
    // Handle tensor.js objects
    if (weight1.__isTensor && weight2.__isTensor) {
      const sumData = new Float32Array(weight1.data.length);
      for (let i = 0; i < weight1.data.length; i++) {
        sumData[i] = weight1.data[i] + weight2.data[i];
      }
      return {
        __isTensor: true,
        data: sumData,
        dtype: weight1.dtype,
        shape: weight1.shape
      };
    }
    // Handle tf.js tensors
    else if (weight1 instanceof tf.Tensor && weight2 instanceof tf.Tensor) {
      return weight1.add(weight2);
    }
    // Handle plain arrays or TypedArrays
    else {
      const result = new Float32Array(weight1.length);
      for (let i = 0; i < weight1.length; i++) {
        result[i] = weight1[i] + weight2[i];
      }
      return result;
    }
  }

  async startTraining() {
    this.log('started training, loading dataset...');

    // DATASET HACK START
    const filePath = path.resolve(__dirname, `./browser_client/public/datasets/imgs/cifar10_iid_split10/cifar10_client_${this.rank - 1}_train.json`);
    this.log(`Loading training dataset from ${filePath}`);
    const rawData = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(rawData);
    const dataset = processData(data);

    const { trainData, testData } = splitDataset(dataset);
    this.trainDataset = trainData;
    this.testDataset = testData;

    // DATASET HACK END

    this.log(`dataset loaded... training model for ${this.config.epochs} epochs...`);
    
    // Record start time for overall training
    const overallStartTime = Date.now();

    // randomly choose num_collaborators from connectedPeers
    this.collaborator_list = [...this.connectedPeers.keys()].sort(() => Math.random() - 0.5).slice(0, this.num_collaborators);
    
    for (let i = 0; i < this.config.epochs; i++) {
      this.currentRound = i;
      this.log(`Starting round ${i} of training`);
      
      // Initialize byte counters for this round
      this.bytesReceived = 0;
      this.bytesSent = 0;
      
      // Record start time for this training round
      const roundStartTime = Date.now();
      
      // Train for one epoch and get the history object
      const history = await this.model.local_train_one(this.trainDataset, this.testDataset, undefined, this.log.bind(this));
      
      // Calculate training time
      const trainTime = Date.now() - roundStartTime;
      
      // Log training metrics
      const trainLoss = history.history.loss[0];
      const trainAcc = history.history.acc[0];
      
      this.metricsLogger.logMetric('train_time', i, trainTime);
      this.metricsLogger.logMetric('train_loss', i, trainLoss);
      this.metricsLogger.logMetric('train_acc', i, trainAcc);
      
      this.log(`Round ${i}: Training completed - Loss: ${trainLoss.toFixed(4)}, Accuracy: ${(trainAcc * 100).toFixed(2)}%`);
      
      // Log test metrics if validation data was used
      if (history.history.val_loss && history.history.val_acc) {
        const testLoss = history.history.val_loss[0];
        const testAcc = history.history.val_acc[0];
        const testStartTime = Date.now();
        
        // Evaluate on test set to get test metrics
        const evalResult = await this.model.model.evaluate(
          tf.tensor2d(this.testDataset.images, [this.testDataset.images.length, 3 * 32 * 32]),
          tf.oneHot(tf.tensor1d(this.testDataset.labels, 'int32'), 10)
        );
        
        const testTime = Date.now() - testStartTime;
        
        this.metricsLogger.logMetric('test_time', i, testTime);
        this.metricsLogger.logMetric('test_loss', i, testLoss);
        this.metricsLogger.logMetric('test_acc', i, testAcc);
        
        this.log(`Round ${i}: Test metrics - Loss: ${testLoss.toFixed(4)}, Accuracy: ${(testAcc * 100).toFixed(2)}%`);
        
        // Log elapsed time for this round
        const timeElapsed = Date.now() - overallStartTime;
        this.metricsLogger.logMetric('time_elapsed', i, timeElapsed);
        
        // Log memory usage metrics
        const memoryInfo = tf.memory();
        this.metricsLogger.logMetric('peak_dram', i, memoryInfo.numBytes);
        
        // Can't get GPU usage directly in Node.js, just log 0 as placeholder
        this.metricsLogger.logMetric('peak_gpu', i, 0);
        
        // Log the total bytes sent and received for this round
        this.metricsLogger.logMetric('bytes_sent', i, this.bytesSent);
        this.metricsLogger.logMetric('bytes_received', i, this.bytesReceived);
        
        this.log(`finished round ${i} training`);

        // Reset tracking for this round
        this.layerChunkTracker = new Map();
        this.receivedWeightsFinished = new Map();
        this.weightReceiptTimeout = 3 * 60 * 1000; // 5 minutes timeout
        const sendStartTime = Date.now();
        
        // randomly choose num_collaborators from connectedPeers
        this.collaborator_list = [...this.connectedPeers.keys()].sort(() => Math.random() - 0.5).slice(0, this.num_collaborators);
        
        this.metricsLogger.logMetric('neighbors', i, JSON.stringify(this.collaborator_list));
        
        const peer_weights = await this.receive();
        this.log(`Round ${i}: Received weights from peers, performing aggregation...`);
        
        // Perform federated averaging with peer_weights
        await this.aggregate(peer_weights);
        this.log(`Round ${i}: Completed aggregation of model weights`);

        const testResult = await this.model.local_test(this.testDataset);
        
        // Log the test accuracy post aggregation
        if (testResult && testResult.testAcc !== undefined) {
          this.metricsLogger.logMetric('test_acc_post_agg', i, testResult.testAcc);
          this.log(`Test Accuracy Post Aggregation: ${(testResult.testAcc * 100).toFixed(2)}%`);
        }
        
        this.currentRound = i + 1;
    }

    this.log("finished training");
    
    // Log final metrics
    const totalTime = Date.now() - overallStartTime;
    this.log(`Total training time: ${totalTime / 1000} seconds: RoundTrainTime: ${trainTime / 1000} seconds, RoundSendTime: ${(Date.now() - roundStartTime) / 1000} seconds`);
  }
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
        // Ensure targetRank is a number
        const targetRankNum = Number(targetRank);
        
        // Log the type of targetRank for debugging
        this.log(`handleConnectionFailure called with targetRank=${targetRank} (type: ${typeof targetRank}), converted to ${targetRankNum} (type: ${typeof targetRankNum})`, 'debug');
        
        const retryCount = this.connectionRetries.get(targetRankNum) || 0;
        if (retryCount < this.MAX_RETRIES) {
            this.connectionRetries.set(targetRankNum, retryCount + 1);
            this.log(`Retrying connection to ${targetRankNum}, attempt #${retryCount + 1}`);
            await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY));
            if (!this.connectedPeers.has(targetRankNum)) {
                await this.cleanupConnection(targetRankNum);
                this.initiateConnection(targetRankNum);
            }
        } else {
            this.log(`Max retries reached for ${targetRankNum}`, 'error');
            await this.cleanupConnection(targetRankNum);
        }
    }

  /**
   * cleanupConnection - Close data channel & PeerConnection, remove references.
   */
    async cleanupConnection(rankStr) {
      const rank = Number(rank)
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
    //       console.log(`numChunks = ${chunks.length} for layer: ${layerName}`);

    //       for (const { chunk, numChunks, originalShape } of chunks) {
    //         const serializableChunk = serializeMessage({
    //           layer_name: layerName,
    //           chunk: chunk,
    //           num_chunks: numChunks,
    //           original_shape: originalShape
    //         });

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

// Logger utility for metrics
class MetricsLogger {
  constructor(logDir = 'logs') {
    this.logDir = logDir;
    this.metrics = new Map();
    this.ensureLogDirectoryExists();
  }

  ensureLogDirectoryExists() {
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
      console.log(`Created log directory: ${this.logDir}`);
    }
  }

  initializeMetric(metricName) {
    const filePath = path.join(this.logDir, `${metricName}.csv`);
    
    // Create or overwrite the file with header
    fs.writeFileSync(filePath, 'iteration,value\n');
    console.log(`Initialized metric log: ${metricName}.csv`);
    
    // Add to our metrics map
    this.metrics.set(metricName, filePath);
  }

  logMetric(metricName, iteration, value) {
    // Make sure the metric is initialized
    if (!this.metrics.has(metricName)) {
      this.initializeMetric(metricName);
    }

    const filePath = this.metrics.get(metricName);
    const logLine = `${iteration},${value}\n`;
    
    // Append the data to the file
    fs.appendFileSync(filePath, logLine);
  }
}

// ** Set your session parameters here **
const SESSION_ID = 1111; // Change this to a fixed or generated session ID
const MAX_CLIENTS = 10;
const IS_CREATOR = false; // Set to true if this should create a session

// ** Start WebRTC Comm Utils **
const signalingServer = 'ws://localhost:8765'; // Your WebSocket server

// TODO: fill in config
let config = {
    signaling_server: signalingServer,
    num_users: MAX_CLIENTS,
    session_id: SESSION_ID,
    epochs: 200,
    num_collaborators: 1,
}

// const filePath = path.resolve(__dirname, './datasets/imgs/cifar10/cifar10_test_small.json');
// const filePath = path.resolve(__dirname, './browser_client/public/datasets/imgs/cifar10_iid/cifar10_client_0_test.json');
// const rawData = fs.readFileSync(filePath, 'utf8');
// const data = JSON.parse(rawData);
// const dataset = processData(data);

// // Helper function to split a dataset into training and testing portions
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

// const { trainData, testData } = splitDataset(dataset);

// const node = new WebRTCCommUtils(config, trainData, testData)
const node = new WebRTCCommUtils(config, null, null)

async function testSendWeights() {
    // Initialize the ResNet10 model
    const resNet10Model = new ResNet10();

    // Load the js2python mapping
    const js2pythonMapping = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'js2python.json'), 'utf8'));

    // Get the weights and layers from the model
    const weights = resNet10Model.model.getWeights();
    const layers = resNet10Model.model.layers;

    const chunk_size = 16000; // Define a chunk size for testing
    console.log(layers.length);

    // Collect information to log
    const logData = [];

    for (const [layerInd, tensor] of Object.entries(weights)) {
        // const layer = layers[layerInd];
        // const layerName = `${layer.name}_weight_${layerInd}`;
        const pythonLayerName = js2pythonMapping[layerInd];

        console.log(`Layer: ${layerInd}, dtype: ${tensor.dtype}, shape: [${tensor.shape.join(', ')}]`);

        // Collect data for logging
        logData.push({
            layerInd: layerInd,
            layerName: pythonLayerName,
            dtype: tensor.dtype,
            shape: tensor.shape
        });

        const chunks = chunkTensor(tensor, chunk_size);
        console.log(`numChunks = ${chunks.length} for layer: ${pythonLayerName}`);

        for (const { chunk, numChunks, originalShape } of chunks) {
            const serializableChunk = serializeMessage({
                layer_name: pythonLayerName,
                chunk: chunk,
                num_chunks: numChunks,
                original_shape: originalShape
            });

            const response = {
                type: "weights_response",
                weights: serializableChunk,
                round: 1, // Example round number
                request_id: 123 // Example request ID
            };

            // Log the response instead of sending it
            // console.log(JSON.stringify(response, null, 2));
        }
    }

    // Write the collected log data to a file
    const logFilePath = path.resolve(__dirname, 'weights_seq_log.json');
    fs.writeFileSync(logFilePath, JSON.stringify(logData, null, 2));
    console.log(`Log data written to ${logFilePath}`);
}

// Run the test
// testSendWeights();

async function testSendWeights2() {
    // Initialize the ResNet10 model
    const resNet10Model = new ResNet10();

    // Load the js2python mapping
    const js2pythonMapping = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'js2python.json'), 'utf8'));

    // Prepare the JSON structure to store the required information
    const logData = [];

    const layers = resNet10Model.model.layers;
    console.log(`Model layers: ${layers.length}`);

    for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        const layerWeights = layer.getWeights();

        for (let j = 0; j <layerWeights.length; j++) {
            const weightTensor = layerWeights[j];

            // Rename the layer and map to Python layer name
            const layerName = `${layer.name}_weight_${j}`;
            const pythonLayerName = js2pythonMapping[layerName]; // Fallback to original if not found

            // Collect data for logging
            logData.push({
                layerInd: `${i}_${j}`,
                layerName: pythonLayerName,
                dtype: weightTensor.dtype,
                shape: weightTensor.shape
            });
        }
    }

    // Write the collected log data to a file
    const logFilePath = path.resolve(__dirname, 'weights_mapped_log.json');
    fs.writeFileSync(logFilePath, JSON.stringify(logData, null, 2));
    console.log(`Mapped weights data written to ${logFilePath}`);
}

// Run the test
// testSendWeights2();

module.exports = { WebRTCCommUtils, MetricsLogger };