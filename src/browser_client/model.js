import * as tf from '@tensorflow/tfjs'

// BLOODMNIST
// const imageShape = [28, 28, 3];
// const imageFlattenSize = 2352;
// const imageClasses = 8;

// CIFAR10
const imageShape = [32, 32, 3];
const imageFlattenSize = 3072;
const imageClasses = 10;


class Model {
	constructor() {
	}

	summary() {
		this.model.summary()
	}

	forward(x, shape) {
		const tensor = (x instanceof tf.Tensor) ? x : tf.tensor2d([x], shape)
		const output = this.model.predict(tensor)
		console.log('Output:', output.arraySync())
		return output
	}
}

// resnet
export class ResNet10 extends Model {
	constructor() {
		super()
		console.log("Initializing ResNet10 instance...")
		this.model = this.buildModel()
	}

	// Build the model
	buildModel() {

		const inputs = tf.input({ shape: [imageFlattenSize] });

		let x = tf.layers.reshape({ targetShape: imageShape }).apply(inputs);

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

		x = tf.layers.dense({ units: imageClasses, activation: 'softmax' }).apply(x);

		const model = tf.model({ inputs, outputs: x });

		model.compile({
			optimizer: 'adam',
			loss: 'categoricalCrossentropy',
			metrics: ['accuracy']
		})

		console.log('model initialized.')

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
		return super.forward(x, [1, imageShape])
	}

	async train(dataSet, config = {
		epochs: 2,
		batchSize: 16,
		validationSplit: 0.2,
		shuffle: true,
		verbose: 1
	}) {
		// take raw array of values and turn to tensor
		const images = tf.tensor2d(dataSet.images, [dataSet.images.length, imageFlattenSize])

		const labels = tf.oneHot(tf.tensor1d(dataSet.labels, 'int32'), imageClasses)

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
					console.log(`Epoch ${epoch + 1}`)
					console.log(`Loss: ${logs.loss.toFixed(4)}`)
					console.log(`Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
					// if (logs.val_loss) {
					// 	addLog(`  Validation Loss: ${logs.val_loss.toFixed(4)}`)
					// 	addLog(`  Validation Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`)
					// }

					// TODO: should I add sending here?
				}
			}
		}

		try {
			console.log(`Beginning training...`)
			const history = await this.model.fit(images, labels, trainingConfig)
			console.log(`Training completed`)

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

	async local_train_one(dataSet, config = {
		epochs: 1,
		batchSize: 64,
		validationSplit: 0.2,
		shuffle: true,
		verbose: 1
	}, logFunc = console.log) {
		// First check if this is a dataset that needs internal splitting
		const needsCustomSplit = dataSet.isCustomSplit === true;
		
		// Always create a deep copy of the training data to prevent mutation issues
		let trainingData = {
			images: [...dataSet.train.images],
			labels: [...dataSet.train.labels]
		};

		// Log dataset dimensions for debugging
		logFunc(`Training data dimensions: images=${trainingData.images.length}, labels=${trainingData.labels.length}`);
		if (trainingData.images.length > 0) {
			logFunc(`First image dimensions: ${trainingData.images[0].length}`);
		}

		// Get validation data
		let validationData = null;
		
		// Handle custom splitting if needed (when only training data is provided)
		if (needsCustomSplit) {
			// Only execute this split once and save the indices for consistency across training rounds
			if (!dataSet.splitIndices) {
				// Create a validation split that will remain consistent
				const dataSize = trainingData.images.length;
				const valSize = Math.floor(dataSize * config.validationSplit);
				
				// Create an array of indices and shuffle it
				const indices = Array.from({ length: dataSize }, (_, i) => i);
				// Use a seed-based shuffle for consistency
				const shuffledIndices = this.seedShuffle([...indices]);
				
				// Split into training and validation indices
				const trainIndices = shuffledIndices.slice(valSize);
				const valIndices = shuffledIndices.slice(0, valSize);
				
				// Store the indices for consistent use across training rounds
				dataSet.splitIndices = {
					train: trainIndices,
					validation: valIndices
				};
				
				logFunc(`Created consistent dataset split: ${trainIndices.length} training samples, ${valIndices.length} validation samples`);
			}
			
			// Extract the validation data using stored indices
			const valIndices = dataSet.splitIndices.validation;
			const valImages = valIndices.map(i => dataSet.train.images[i]);
			const valLabels = valIndices.map(i => dataSet.train.labels[i]);
			
			// Log validation dimensions
			logFunc(`Validation data dimensions: images=${valImages.length}, labels=${valLabels.length}`);
			
			// Create validation tensors
			const valImagesTensor = tf.tensor2d(valImages, [valImages.length, imageFlattenSize]);
			const valLabelsTensor = tf.oneHot(tf.tensor1d(valLabels, 'int32'), imageClasses);
			validationData = [valImagesTensor, valLabelsTensor];
			
			// Extract the training data using stored indices
			const trainIndices = dataSet.splitIndices.train;
			const trainImages = trainIndices.map(i => dataSet.train.images[i]);
			const trainLabels = trainIndices.map(i => dataSet.train.labels[i]);
			
			// Use a fresh copy to avoid mutations
			trainingData = {
				images: trainImages,
				labels: trainLabels
			};
			
			// Set validation split to 0 since we're using external validation data
			config.validationSplit = 0;
			
			logFunc(`Using ${trainImages.length} samples for training and ${valImages.length} samples for validation`);
		}
		// Handle provided test data as validation
		else if (dataSet.test && dataSet.test.images.length > 0) {
			// Use the pre-split test set as validation data
			const testImages = tf.tensor2d(dataSet.test.images, [dataSet.test.images.length, imageFlattenSize]);
			const testLabels = tf.oneHot(tf.tensor1d(dataSet.test.labels, 'int32'), imageClasses);
			validationData = [testImages, testLabels];
			
			// Adjust validation split to 0 since we're using external validation data
			config.validationSplit = 0;
			
			logFunc(`Using separate test set with ${dataSet.test.images.length} samples for validation`);
		}
		
			// Convert training data to tensors
			const images = tf.tensor2d(trainingData.images, [trainingData.images.length, imageFlattenSize]);
			const labels = tf.oneHot(tf.tensor1d(trainingData.labels, 'int32'), imageClasses);
			
			// create config object
			const trainingConfig = {
				epochs: 1,
				batchSize: config.batchSize,
				validationSplit: config.validationSplit,
				shuffle: config.shuffle,
				verbose: config.verbose,
				callbacks: {
					// callback in between epochs
					onEpochEnd: (epoch, logs) => {
						const epochLog = `Epoch ${epoch + 1}`;
						const lossLog = `Loss: ${logs.loss.toFixed(4)}`;
						const accLog = `Accuracy: ${(logs.acc * 100).toFixed(2)}%`;
						
						// Use standard console.log for development visibility
						console.log(epochLog);
						console.log(lossLog);
						console.log(accLog);
						
						// Use the custom log function if provided
						if (logFunc && typeof logFunc === 'function') {
							logFunc(epochLog);
							logFunc(lossLog);
							logFunc(accLog);
							
							// Log validation metrics if available
							if (logs.val_loss) {
								const valLossLog = `Validation Loss: ${logs.val_loss.toFixed(4)}`;
								const valAccLog = `Validation Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`;
								console.log(valLossLog);
								console.log(valAccLog);
								logFunc(valLossLog);
								logFunc(valAccLog);
							}
						}
					}
				}
			};

		try {
			console.log(`Beginning training...`);
			
			// If we have separate validation data, include it in the fit call
			const history = validationData 
				? await this.model.fit(images, labels, { ...trainingConfig, validationData })
				: await this.model.fit(images, labels, trainingConfig);
				
			console.log(`Training completed`);

			// Clean up tensors
			images.dispose();
			labels.dispose();
			
			if (validationData) {
				validationData[0].dispose();
				validationData[1].dispose();
			}

			return history;
		} catch (error) {
			console.error('Error during training: ', error);
			logFunc(`Training error: ${error.message}`);

			// Attempt to clean up any tensors that might have been created
			try {
				tf.disposeVariables();
			} catch (e) {
				console.log('Error during tensor cleanup:', e);
			}
			
			throw error;
		}
	}

	seedShuffle(array) {
		const seed = 42;
		const shuffled = [...array];
		for (let i = shuffled.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
		}
		return shuffled;
	}
}
