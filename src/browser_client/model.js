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

	async local_train_one(trainDataSet, testDataSet = null, config = {
		epochs: 1,
		batchSize: 64,
		validationSplit: 0.2,
		shuffle: true,
		verbose: 1
	}, logFunc = console.log) {
		// take raw array of values and turn to tensor
		const trainImages = tf.tensor2d(trainDataSet.images, [trainDataSet.images.length, imageFlattenSize])
		const trainLabels = tf.oneHot(tf.tensor1d(trainDataSet.labels, 'int32'), imageClasses)
		
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
			testImages = tf.tensor2d(testDataSet.images, [testDataSet.images.length, imageFlattenSize]);
			testLabels = tf.oneHot(tf.tensor1d(testDataSet.labels, 'int32'), imageClasses);
			
			// Remove validationSplit since we're using separate validation data
			delete trainingConfig.validationSplit;
		} else {
			// Use validationSplit parameter when no separate test data is provided
			trainingConfig.validationSplit = config.validationSplit;
		}

		try {
			console.log(`Beginning training...`);
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
			
			console.log(`Training completed`);

			// Clean up tensors
			trainImages.dispose();
			trainLabels.dispose();
			
			if (testImages) testImages.dispose();
			if (testLabels) testLabels.dispose();

			return history;
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
}
