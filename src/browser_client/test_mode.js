import { ResNet10 as OriginalResNet10 } from './model.js'
import { MiniResNet20 } from './mini_model.js'
import { MiniResNet, MediumResNet, ResNet10 } from './mini_model2.js'
import * as tf from '@tensorflow/tfjs'
import JSZip from 'jszip';

// Initialize model variable
let model;

/**
 * Initialize the model with the specified model type and dataset
 * @param {string} modelType - The model type to initialize (e.g., 'MiniResNet', 'MediumResNet', 'ResNet10')
 * @param {string} dataset - The dataset to use (e.g., 'cifar10', 'mnist')
 * @returns {Promise<void>}
 */
async function initializeModel(modelType = 'MiniResNet', dataset = 'cifar10') {
  try {
    // Initialize the appropriate model based on selection
    switch (modelType) {
      case 'MiniResNet':
        model = new MiniResNet();
        break;
      case 'MediumResNet':
        model = new MediumResNet();
        break;
      case 'ResNet10':
        model = new ResNet10();
        break;
      default:
        model = new MiniResNet();
    }
    
    console.log(`${modelType} initialized successfully`);
    document.getElementById('model-status').textContent = `${modelType} initialized successfully`;
    document.getElementById('download-btn').disabled = false;
    return model;
  } catch (error) {
    console.error("Error initializing model:", error);
    document.getElementById('model-status').textContent = `Error: ${error.message}`;
    return null;
  }
}

/**
 * Download the model when the button is clicked
 * @returns {Promise<void>}
 */
async function downloadModel() {
  if (!model) {
    
    console.error("Model not initialized");
    document.getElementById('model-status').textContent = 'Error: Model not initialized';
    return;
  }

  try {
    const numTrainableParams = model.model.trainableWeights
    .map(w => w.shape.reduce((a, b) => a * b, 1))
    .reduce((a, b) => a + b, 0);

    console.log(`Trainable parameters: ${numTrainableParams}`);

    document.getElementById('model-status').textContent = 'Downloading model...';
    document.getElementById('download-btn').disabled = true;
    
    // Save the model to downloads with the model's name
    const modelName = model.name.toLowerCase().replace(/\s+/g, '_');
    await model.model.save(`downloads://${modelName}`);
    
    document.getElementById('model-status').textContent = 'Model downloaded successfully';
    document.getElementById('download-btn').disabled = false;
  } catch (error) {
    console.error("Error downloading model:", error);
    document.getElementById('model-status').textContent = `Error downloading: ${error.message}`;
    document.getElementById('download-btn').disabled = false;
  }
}

/**
 * Test local training with the MiniResNet20 model
 * @returns {Promise<void>}
 */
async function testTrain() {
  if (!model) {
    console.error("Model not initialized");
    document.getElementById('model-status').textContent = 'Error: Model not initialized';
    return;
  }

  try {
    document.getElementById('model-status').textContent = 'Loading dataset for training...';
    document.getElementById('test-train-btn').disabled = true;
    
    // Get the dataset type from the model
    const datasetType = model.imageClasses === 10 ? 'cifar10' : 'mnist';
    
    // Create a small training dataset
    const trainSize = 100; // Small dataset for testing
    const images = [];
    const labels = [];
    
    // Generate random data for testing
    // This is just for testing the shape compatibility
    for (let i = 0; i < trainSize; i++) {
      // Create a random flat array with the correct size
      const randomImage = Array(model.imageFlattenSize).fill(0).map(() => Math.random());
      const randomLabel = Math.floor(Math.random() * model.imageClasses);
      
      images.push(randomImage);
      labels.push(randomLabel);
    }
    
    const trainSet = {
      images: images,
      labels: labels
    };
    
    // Log model summary
    console.log("Model summary:");
    model.summary();
    document.getElementById('model-status').textContent = 'Training model locally...';
    
    // Train the model for one epoch
    const trainMetrics = await model.local_train_one(trainSet, null, {
      epochs: 1,
      batchSize: 16,
      validationSplit: 0.2,
      shuffle: true,
      verbose: 1
    });
    
    // Display results
    const statusElement = document.getElementById('model-status');
    statusElement.innerHTML = `
      <strong>Training Complete!</strong><br>
      Accuracy: ${(trainMetrics.trainAcc * 100).toFixed(2)}%<br>
      Loss: ${trainMetrics.trainLoss.toFixed(4)}<br>
      Time: ${trainMetrics.trainTime.toFixed(2)} seconds
    `;
    
    document.getElementById('test-train-btn').disabled = false;
    
  } catch (error) {
    console.error("Error during test training:", error);
    document.getElementById('model-status').textContent = `Error during training: ${error.message}`;
    document.getElementById('test-train-btn').disabled = false;
  }
}

// Export functions as named exports for ES modules
export { initializeModel, downloadModel, testTrain };
