import * as tf from '@tensorflow/tfjs'

export const supportedDatasets = {
	'cifar10': {
		'imageShape': [32, 32, 3],
		'imageClasses': 10
	},
	// 'bloodmnist': {
	// 	'imageShape': [28, 28, 3],
	// 	'imageClasses': 8
	// },
	'mnist': {
		'imageShape': [28, 28, 1],
		'imageClasses': 10
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
		console.log('Output:', output.arraySync())
		return output
	}
}

// resnet
export class ResNet10 extends Model {
	constructor(dataset) {
		super()
		console.log("Initializing ResNet10 instance...")
		if (!(dataset in supportedDatasets)) {
			throw new Error('Dataset not supported.')
		}
		this.imageShape = supportedDatasets[dataset]['imageShape']
		this.imageClasses = supportedDatasets[dataset]['imageClasses']
		this.imageFlattenSize = this.imageShape.reduce((prod, num) => prod * num, 1)

		this.model = this.buildModel()
	}

	// Build the model
	buildModel() {

		const inputs = tf.input({ shape: [this.imageFlattenSize] });

		let x = tf.layers.reshape({ targetShape: this.imageShape }).apply(inputs);

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

		x = tf.layers.dense({ units: this.imageClasses, activation: 'softmax' }).apply(x);

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
		return super.forward(x, [1, this.imageShape])
	}

	async train(dataSet, config = {
		epochs: 2,
		batchSize: 16,
		validationSplit: 0.2,
		shuffle: true,
		verbose: 1
	}) {
		// take raw array of values and turn to tensor
		const images = tf.tensor2d(dataSet.images, [dataSet.images.length, this.imageFlattenSize])

		const labels = tf.oneHot(tf.tensor1d(dataSet.labels, 'int32'), this.imageClasses)

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

	async local_train_one(trainDataSet, testDataSet = null, config = {
		epochs: 1,
		batchSize: 64,
		validationSplit: 0.2,
		shuffle: true,
		verbose: 1
	}, logFunc = console.log) {
		// take raw array of values and turn to tensor
		const trainImages = tf.tensor2d(trainDataSet.images, [trainDataSet.images.length, this.imageFlattenSize])
		const trainLabels = tf.oneHot(tf.tensor1d(trainDataSet.labels, 'int32'), this.imageClasses)
		
		// prepare test data if provided
		let testImages = null;
		let testLabels = null;
		
		// create config object
		const trainingConfig = {
			epochs: 1,
			batchSize: config.batchSize,
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

					// TODO: should I add sending here?
				}
			}
		}
		
		// If testDataSet is provided, use it as validation data instead of using validationSplit
		if (testDataSet) {
			testImages = tf.tensor2d(testDataSet.images, [testDataSet.images.length, this.imageFlattenSize]);
			testLabels = tf.oneHot(tf.tensor1d(testDataSet.labels, 'int32'), this.imageClasses);
			
			// Remove validationSplit since we're using separate validation data
			delete trainingConfig.validationSplit;
		} else {
			// Use validationSplit parameter when no separate test data is provided
			trainingConfig.validationSplit = config.validationSplit;
		}

		try {
			console.log(`Beginning training...`);
			const startTime = performance.now();
			let history;
			
			if (testDataSet) {
				// Use separate validation data
				history = await this.model.fit(
					trainImages, 
					trainLabels, 
					{
						...trainingConfig,
						validationData: [testImages, testLabels]
					}
				);
			} else {
				// Use validation split
				history = await this.model.fit(trainImages, trainLabels, trainingConfig);
			}
			
			const endTime = performance.now();
			const trainingTime = (endTime - startTime) / 1000; // Convert to seconds
			console.log(`Training completed in ${trainingTime.toFixed(2)} seconds`);

			// Clean up tensors
			trainImages.dispose();
			trainLabels.dispose();
			
			if (testImages) testImages.dispose();
			if (testLabels) testLabels.dispose();

			// Extract metrics
			const metrics = {
				trainAcc: history.history.acc[0],
				trainLoss: history.history.loss[0],
				trainTime: trainingTime
			};
			
			// If validation data was used, include validation metrics
			if (history.history.val_acc) {
				metrics.testAcc = history.history.val_acc[0];
				metrics.testLoss = history.history.val_loss[0];
			}
			
			return metrics;
		} catch (error) {
			console.error('Error during training: ', error);

			// Clean up tensors even if there's an error
			trainImages.dispose();
			trainLabels.dispose();
			
			if (testImages) testImages.dispose();
			if (testLabels) testLabels.dispose();
			
			throw error;
		}
	}

	async local_test(testDataSet, logFunc = console.log) {
		logFunc("local_test called");
		if (!testDataSet) {
			console.error("No test dataset provided.");
			return { testAcc: 0, testLoss: 0, testTime: 0 };
		}
	
		// Convert test data to tensors
		const testImages = tf.tensor2d(testDataSet.images, [testDataSet.images.length, this.imageFlattenSize]);
		const testLabels = tf.oneHot(tf.tensor1d(testDataSet.labels, 'int32'), this.imageClasses);
	
		try {
			console.log("Evaluating model on test data...");
			const startTime = performance.now();
			const evalResult = await this.model.evaluate(testImages, testLabels);
			const endTime = performance.now();
			const testTime = (endTime - startTime) / 1000; // Convert to seconds
			
			// Extract loss and accuracy values
			const testLoss = evalResult[0].dataSync()[0];
			const testAccuracy = evalResult[1].dataSync()[0];
	
			// Log results
			const lossLog = `Test Loss (After Aggregation): ${testLoss.toFixed(4)}`;
			const accLog = `Test Accuracy (After Aggregation): ${(testAccuracy * 100).toFixed(2)}%`;
			const timeLog = `Test Time: ${testTime.toFixed(2)} seconds`;
	
			console.log(lossLog);
			console.log(accLog);
			console.log(timeLog);
	
			if (logFunc && typeof logFunc === 'function') {
				logFunc(lossLog);
				logFunc(accLog);
				logFunc(timeLog);
			}
			
			// Return metrics
			return {
				testAcc: testAccuracy,
				testLoss: testLoss,
				testTime: testTime
			};
		} catch (error) {
			console.error("Error during evaluation:", error);
			return { testAcc: 0, testLoss: 0, testTime: 0 };
		} finally {
			// Dispose tensors
			testImages.dispose();
			testLabels.dispose();
		}
	}
}
