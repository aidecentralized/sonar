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
        this.expName = "non_iid_iso"
        this.startTime = Date.now();
        this.model = new ResNet10();
        this.config = config || {};
        this.signalingServer = this.config.signaling_server || 'ws://localhost:8765';
        this.trainDataset = trainDataset;
        this.testDataset = testDataset;

        // Validation Data
        const filePath = path.resolve(__dirname, `./browser_client/public/datasets/imgs/cifar10_non_iid_unique_labels/cifar10_client_0_test.json`);
        const rawData = fs.readFileSync(filePath, 'utf8');
        const data = JSON.parse(rawData);
        this.valDataset = processData(data);
    
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
        this.allLogsPath = path.join('logs', `${this.expName}`, `all_logs.log`);
        if (!fs.existsSync(path.dirname(this.allLogsPath))) {
          fs.mkdirSync(path.dirname(this.allLogsPath), { recursive: true });
        }
        fs.writeFileSync(this.allLogsPath, '');

        // Initialize metrics logger now that we have the rank
        this.metricsLogger = new MetricsLogger(path.join('logs', `${this.expName}`, `node_0`));

        // Initialize metrics files
        ['test_acc', 'test_loss', 'test_time', 
            'train_acc_noniid', 'train_loss', 'train_time',
            'time_elapsed', 'bytes_sent', 'bytes_received',
            'peak_dram', 'peak_gpu', 'neighbors', 'test_acc_iid'].forEach(metric => {
            this.metricsLogger.initializeMetric(metric);
        });

        this.log(`[constructor] RTCCommUtilsJS created with config: ${JSON.stringify(config)}`);

        this.startTraining()

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
  

  async startTraining() {
    this.log('started training, loading dataset...');

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
        
        // Evaluate on validation set to get IID metrics
        const evalResult = await this.model.model.evaluate(
          tf.tensor2d(this.valDataset.images, [this.valDataset.images.length, 3 * 32 * 32]),
          tf.oneHot(tf.tensor1d(this.valDataset.labels, 'int32'), 10)
        );
        
        // Extract IID accuracy from evalResult
        const iidAcc = evalResult[1].dataSync()[0];
        
        const testTime = Date.now() - testStartTime;
        
        this.metricsLogger.logMetric('test_time', i, testTime);
        this.metricsLogger.logMetric('test_loss', i, testLoss);
        this.metricsLogger.logMetric('test_acc_noniid', i, testAcc);
        this.metricsLogger.logMetric('test_acc_iid', i, iidAcc);
        
        this.log(`Round ${i}: Test metrics - Loss: ${testLoss.toFixed(4)}, Accuracy: ${(testAcc * 100).toFixed(2)}%`);
        this.log(`Round ${i}: IID Test Accuracy: ${(iidAcc * 100).toFixed(2)}%`);
        
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
        // this.layerChunkTracker = new Map();
        // this.receivedWeightsFinished = new Map();
        // this.weightReceiptTimeout = 3 * 60 * 1000; // 5 minutes timeout
        const sendStartTime = Date.now();
        
        // randomly choose num_collaborators from connectedPeers
        // this.collaborator_list = [...this.connectedPeers.keys()].sort(() => Math.random() - 0.5).slice(0, this.num_collaborators);
        
        // this.metricsLogger.logMetric('neighbors', i, JSON.stringify(this.collaborator_list));
        
        // const peer_weights = await this.receive();
        // this.log(`Round ${i}: Received weights from peers, performing aggregation...`);
        
        // Perform federated averaging with peer_weights
        // await this.aggregate(peer_weights);
        // this.log(`Round ${i}: Completed aggregation of model weights`);

        // We've already logged test_acc_iid from evalResult, so we can remove this
        // const testResult = await this.model.local_test(this.valDataset);
        // if (testResult && testResult.testAcc !== undefined) {
        //   this.metricsLogger.logMetric('test_acc_iid', i, testResult.testAcc);
        //   this.log(`Test Accuracy On Noniid Data: ${(testResult.testAcc * 100).toFixed(2)}%`);
        // }
        
        this.currentRound = i + 1;
    }

    this.log("finished training");
    
    // Log final metrics
    const totalTime = Date.now() - overallStartTime;
    this.log(`Total training time: ${totalTime / 1000} seconds: RoundTrainTime: ${trainTime / 1000} seconds, RoundSendTime: ${(Date.now() - roundStartTime) / 1000} seconds`);
  }
}
      
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

// const trainFilePath = path.resolve(__dirname, './browser_client/public/datasets/imgs/cifar10_non_iid_10clients_2classes/cifar10_client_0_train.json');
// const trainRawData = fs.readFileSync(trainFilePath, 'utf8');
// const trainData = JSON.parse(trainRawData);
// const trainDataset = processData(trainData);

// const testFilePath = path.resolve(__dirname, './browser_client/public/datasets/imgs/cifar10_non_iid_10clients_2classes/cifar10_client_0_test_noniid.json');
// const testRawData = fs.readFileSync(testFilePath, 'utf8');
// const testData = JSON.parse(testRawData);
// const testDataset = processData(testData);

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

const filePath = path.resolve(__dirname, `./browser_client/public/datasets/imgs/cifar10_non_iid_unique_labels/cifar10_client_0_train.json`);
const rawData = fs.readFileSync(filePath, 'utf8');
const data = JSON.parse(rawData);
const dataset = processData(data);

const { trainData, testData } = splitDataset(dataset);

const node = new WebRTCCommUtils(config, trainData, testData)
// const node = new WebRTCCommUtils(config, null, null)

module.exports = { WebRTCCommUtils, MetricsLogger };