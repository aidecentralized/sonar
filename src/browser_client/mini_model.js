import * as tf from '@tensorflow/tfjs';

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

/**
 *  MiniResNet‑20  (≈0.56 M params ≈2.2 MB float32)
 */
export class MiniResNet20 extends Model {
    constructor(dataset) {
        super()
        console.log("Initializing MiniResNet20 instance...")
        if (!(dataset in supportedDatasets)) {
            throw new Error('Dataset not supported.')
        }
        this.imageShape = supportedDatasets[dataset]['imageShape']
        this.imageClasses = supportedDatasets[dataset]['imageClasses']
        this.imageFlattenSize = this.imageShape.reduce((prod, num) => prod * num, 1)

        this.model = this.buildModel()
    }

  // ---------- Model definition ----------
  buildModel() {
    // identical 1‑D input spec as before → keeps your data pipeline unchanged
    const inputs = tf.input({shape: [this.imageFlattenSize]});
    let x = tf.layers.reshape({targetShape: this.imageShape}).apply(inputs);

    // initial 3×3 conv, 16 filters
    x = tf.layers.conv2d({filters: 16, kernelSize: 3, padding: 'same', useBias: false}).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.reLU().apply(x);

    // stage configuration: [filters, nBlocks, firstBlockStride]
    const cfg = [
      [16,  3, 1],
      [32,  3, 2],
      [64,  3, 2],
      [128, 3, 2],
    ];

    cfg.forEach(([filters, n, stride]) => {
      // first block (may down‑sample)
      x = this._resBlock(x, filters, stride);
      // remaining blocks
      for (let i = 1; i < n; i++) x = this._resBlock(x, filters, 1);
    });

    x = tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }).apply(x);
    const outputs = tf.layers.dense({units: this.imageClasses, activation: 'softmax'}).apply(x);

    const model = tf.model({inputs, outputs});
    model.compile({
      optimizer: tf.train.adam(),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    return model;
  }

  // ---------- Residual block ----------
  _resBlock(inp, filters, stride = 1) {
    // -------- decide if we need projection on the shortcut ----------
    // • Projection is required if we down‑sample (stride ≠ 1)
    // • or if #channels changes
    //   – But `inp.shape` can be null on symbolic tensors, so guard it.
    let inChannels = null;
    if (inp && Array.isArray(inp.shape) && inp.shape.length > 0) {
      inChannels = inp.shape[inp.shape.length - 1];   // e.g. 16, 32 …
    }
    const useProj = (stride !== 1) || (inChannels !== null && inChannels !== filters);
  
    // ---------------- main path ----------------
    let x = tf.layers.conv2d({
      filters, kernelSize: 3, strides: stride,
      padding: 'same', useBias: false
    }).apply(inp);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.reLU().apply(x);          // same API as before
  
    x = tf.layers.conv2d({
      filters, kernelSize: 3, strides: 1,
      padding: 'same', useBias: false
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
  
    // ---------------- shortcut path ------------
    let shortcut = inp;
    if (useProj) {
      shortcut = tf.layers.conv2d({
        filters, kernelSize: 1, strides: stride,
        padding: 'same', useBias: false
      }).apply(shortcut);
      shortcut = tf.layers.batchNormalization().apply(shortcut);
    }
  
    // ---------------- merge --------------------
    x = tf.layers.add().apply([x, shortcut]);
    return tf.layers.reLU().apply(x);
  }

  /** keep your original forward helper */
  forward(x) { return super.forward(x, [1, this.imageShape]); }

      async local_train_one(trainDataSet, testDataSet = null, config = {
        epochs: 1,
        batchSize: 16,
        validationSplit: 0.2,
        shuffle: true,
        verbose: 1
    }, logFunc = console.log) {
        // take raw array of values and turn to tensor
        const [trainImages, trainLabels] = tf.tidy(() => {
            const img = tf.tensor2d(trainDataSet.images, [trainDataSet.images.length, this.imageFlattenSize]);
            const lbl = tf.oneHot(tf.tensor1d(trainDataSet.labels, 'int32'), this.imageClasses);
            return [img, lbl];
          });
        // const trainImages = tf.tensor2d(trainDataSet.images, [trainDataSet.images.length, this.imageFlattenSize])
        // const trainLabels = tf.oneHot(tf.tensor1d(trainDataSet.labels, 'int32'), this.imageClasses)
        
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
            const [testImages, testLabels] = tf.tidy(() => {
                const img = tf.tensor2d(testDataSet.images, [testDataSet.images.length, this.imageFlattenSize]);
                const lbl = tf.oneHot(tf.tensor1d(testDataSet.labels, 'int32'), this.imageClasses);
                return [img, lbl];
              });
            // testImages = tf.tensor2d(testDataSet.images, [testDataSet.images.length, this.imageFlattenSize]);
            // testLabels = tf.oneHot(tf.tensor1d(testDataSet.labels, 'int32'), this.imageClasses);
            
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
  
  
            history = null; // free references early
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
        const [testImages, testLabels] = tf.tidy(() => {
            const img = tf.tensor2d(testDataSet.images, [testDataSet.images.length, this.imageFlattenSize]);
            const lbl = tf.oneHot(tf.tensor1d(testDataSet.labels, 'int32'), this.imageClasses);
            return [img, lbl];
        });
        // const testImages = tf.tensor2d(testDataSet.images, [testDataSet.images.length, this.imageFlattenSize]);
        // const testLabels = tf.oneHot(tf.tensor1d(testDataSet.labels, 'int32'), this.imageClasses);
    
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
