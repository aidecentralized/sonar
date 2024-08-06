import { Button, Container, Grid, Link, TextField, Switch, FormControlLabel, Table, TableHead, TableBody, TableRow, TableCell, TableContainer } from '@mui/material';
import React from 'react';
import './App.css';
import Plot from 'react-plotly.js';
import * as ort from 'onnxruntime-web/training';
import { XSumData } from './xsum';
import { Summary } from './Summary';

function App() {
    const lossNodeName = "onnx::loss::8";
    const logIntervalMs = 1000;
    const waitAfterLoggingMs = 500;
    let lastLogTime = 0;
    let messagesQueue: string[] = [];

    const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState<number>(320);
    const [maxNumTestSamples, setMaxNumTestSamples] = React.useState<number>(160);
    const [batchSize, setBatchSize] = React.useState<number>(XSumData.BATCH_SIZE);
    const [numEpochs, setNumEpochs] = React.useState<number>(3);
    const [trainingLosses, setTrainingLosses] = React.useState<number[]>([]);
    const [testAccuracies, setTestAccuracies] = React.useState<number[]>([]);
    const [summaries, setSummaries] = React.useState<{ text: string, summary: string }[]>([]);
    const [summaryPredictions, setSummaryPredictions] = React.useState<string[]>([]);
    const [isTraining, setIsTraining] = React.useState<boolean>(false);
    const [moreInfoIsCollapsed, setMoreInfoIsCollapsed] = React.useState<boolean>(true);
    const [enableLiveLogging, setEnableLiveLogging] = React.useState<boolean>(false);
    const [statusMessage, setStatusMessage] = React.useState("");
    const [errorMessage, setErrorMessage] = React.useState("");
    const [messages, setMessages] = React.useState<string[]>([]);

    const numCols = 1024; // Example value

    function toggleMoreInfoIsCollapsed() {
        setMoreInfoIsCollapsed(!moreInfoIsCollapsed);
    }

    function showStatusMessage(message: string) {
        console.log(message);
        setStatusMessage(message);
    }

    function showErrorMessage(message: string) {
        console.error(message);
        setErrorMessage(message);
    }

    function addMessages(messagesToAdd: string[]) {
        setMessages(messages => [...messages, ...messagesToAdd]);
    }

    function addMessageToQueue(message: string) {
        messagesQueue.push(message);
    }

    function clearOutputs() {
        setTrainingLosses([]);
        setTestAccuracies([]);
        setMessages([]);
        setStatusMessage("");
        setErrorMessage("");
        messagesQueue = [];
    }

    async function logMessage(message: string) {
        console.log(message);
        messagesQueue.push(message);
        if (Date.now() - lastLogTime > logIntervalMs) {
            showStatusMessage(message);
            setMessages(messages => [...messages, ...messagesQueue]);
            messagesQueue = [];
            await new Promise(r => setTimeout(r, waitAfterLoggingMs));
            lastLogTime = Date.now();
        }
    }

    async function loadTrainingSession(): Promise<ort.TrainingSession> {
        showStatusMessage('Attempting to load training session...');
        const chkptPath = 'checkpoint';
        const trainingPath = 'training_model.onnx';
        const optimizerPath = 'optimizer_model.onnx';
        const evalPath = 'eval_model.onnx';

        const createOptions: ort.TrainingSessionCreateOptions = {
            checkpointState: chkptPath,
            trainModel: trainingPath,
            evalModel: evalPath,
            optimizerModel: optimizerPath
        };

        try {
            const session = await ort.TrainingSession.create(createOptions);
            showStatusMessage('Training session loaded');
            return session;
        } catch (err) {
            showErrorMessage('Error loading the training session: ' + err);
            console.error("Error loading the training session: " + err);
            throw err;
        }
    };

    async function updateSummaryPredictions(session: ort.TrainingSession) {
        const input = new Float32Array(summaries.length * numCols);
        const batchShape = [summaries.length, numCols];
        const labels: string[] = [];
        for (let i = 0; i < summaries.length; ++i) {
            const textTokens = summaries[i].text.split(' ').map(word => word.length); // Example tokenization method
            for (let j = 0; j < textTokens.length; ++j) {
                input[i * numCols + j] = textTokens[j];
            }
            labels.push(summaries[i].summary);
        }        
        const feeds = {
            input: new ort.Tensor('float32', input, batchShape),
            labels: new ort.Tensor('string', labels, [summaries.length])
        };
        const results = await session.runEvalStep(feeds);
        const predictions = getPredictions(results['output']);
        setSummaryPredictions(predictions.slice(0, summaries.length));
    }

    function getPredictions(results: ort.Tensor): string[] {
        const predictions: string[] = [];
        const [batchSize, numTokens] = results.dims;
    
        for (let i = 0; i < batchSize; ++i) {
            const tokens = Array.from(results.data.slice(i * numTokens, (i + 1) * numTokens) as Float32Array);
            const prediction = tokens.map(t => String.fromCharCode(Math.round(t))).join(''); // might need to change
            predictions.push(prediction);
        }
        return predictions;
    }
    
    function countCorrectPredictions(output: ort.Tensor, labels: ort.Tensor): number {
        let result = 0;
        const predictions = getPredictions(output);
        const labelsArray = Array.from(labels.data as Iterable<string>).map(String);
        for (let i = 0; i < predictions.length; ++i) {
            if (predictions[i] === labelsArray[i].toString()) {
                ++result;
            }
        }
        return result;
    }

    async function runTrainingEpoch(session: ort.TrainingSession, dataSet: XSumData, epoch: number) {
        let batchNum = 0;
        let totalNumBatches = dataSet.getNumTrainingBatches();
        const epochStartTime = Date.now();
        let iterationsPerSecond = 0;
    
        await logMessage(`TRAINING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Starting training...`);
        for await (const batch of dataSet.trainingBatches()) {
            ++batchNum;
    
            const batchSize = batch.data.dims[0];
            const input = new Float32Array(batchSize * numCols);
            const labels: string[] = [];
            for (let i = 0; i < batchSize; ++i) {
                const textTokens = batch.data.data.slice(i * numCols, (i + 1) * numCols) as Float32Array;
                for (let j = 0; j < textTokens.length; ++j) {
                    input[i * numCols + j] = textTokens[j];
                }
                labels.push(batch.labels.data[i].toString());
            }
    
            const feeds = {
                input: new ort.Tensor('float32', input, [batchSize, numCols]),
                labels: new ort.Tensor('string', labels, [batchSize])
            };
    
            const results = await session.runTrainStep(feeds);
            const loss = parseFloat(results[lossNodeName.toString()].data[0].toString());
            setTrainingLosses(losses => losses.concat(loss));
    
            iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
            const message = `TRAINING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Loss: ${loss.toFixed(4)} | ${iterationsPerSecond.toFixed(2)} it/s`;
            await logMessage(message);
    
            await session.runOptimizerStep();
            await session.lazyResetGrad();
            await updateSummaryPredictions(session);
        }
        return iterationsPerSecond;
    }
    
    async function runTestingEpoch(session: ort.TrainingSession, dataSet: XSumData, epoch: number): Promise<number> {
        let batchNum = 0;
        let totalNumBatches = dataSet.getNumTestBatches();
        let numCorrect = 0;
        let testSummariesSoFar = 0;
        let accumulatedLoss = 0;
        const epochStartTime = Date.now();
        await logMessage(`TESTING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Starting testing...`);
        for await (const batch of dataSet.testBatches()) {
            ++batchNum;
    
            const batchSize = batch.data.dims[0];
            const input = new Float32Array(batchSize * numCols);
            const labels: string[] = [];
            for (let i = 0; i < batchSize; ++i) {
                const textTokens = batch.data.data.slice(i * numCols, (i + 1) * numCols) as Float32Array;
                for (let j = 0; j < textTokens.length; ++j) {
                    input[i * numCols + j] = textTokens[j];
                }
                labels.push(batch.labels.data[i].toString());
            }
    
            const feeds = {
                input: new ort.Tensor('float32', input, [batchSize, numCols]),
                labels: new ort.Tensor('string', labels, [batchSize])
            };
    
            const results = await session.runEvalStep(feeds);
            const loss = parseFloat(results[lossNodeName].data[0].toString());
            accumulatedLoss += loss;
            testSummariesSoFar += batchSize;
            numCorrect += countCorrectPredictions(results['output'], new ort.Tensor('string', labels, [batchSize]));
            const iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
            const message = `TESTING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Average test loss: ${(accumulatedLoss / batchNum).toFixed(2)} | Accuracy: ${numCorrect}/${testSummariesSoFar} (${(100 * numCorrect / testSummariesSoFar).toFixed(2)}%) | ${iterationsPerSecond.toFixed(2)} it/s`;
            await logMessage(message);
        }
        const avgAcc = numCorrect / testSummariesSoFar;
        setTestAccuracies(accs => accs.concat(avgAcc));
        return avgAcc;
    }
    
    async function train() {
        console.log("Training init");
        setIsTraining(true);
        if (maxNumTrainSamples > XSumData.MAX_NUM_TRAIN_SAMPLES || maxNumTestSamples > XSumData.MAX_NUM_TEST_SAMPLES) {
            showErrorMessage(`Max number of training samples (${maxNumTrainSamples}) or test samples (${maxNumTestSamples}) exceeds the maximum allowed (${XSumData.MAX_NUM_TRAIN_SAMPLES} and ${XSumData.MAX_NUM_TEST_SAMPLES}, respectively). Please try again.`);
            return;
        }
    
        console.log("Loading training session...");
        const trainingSession = await loadTrainingSession();
        console.log("Training session loaded");
    
        const dataSet = new XSumData(batchSize, maxNumTrainSamples, maxNumTestSamples);
        console.log("Dataset initialized");
    
        lastLogTime = Date.now();
        const startTrainingTime = Date.now();
        showStatusMessage("Training started" + dataSet.batchSize);
        let itersPerSecCumulative = 0;
        let testAcc = 0;
    
        for (let epoch = 0; epoch < numEpochs; epoch++) {
            console.log(`Starting epoch ${epoch + 1}`);
            itersPerSecCumulative += await runTrainingEpoch(trainingSession, dataSet, epoch);
            testAcc = await runTestingEpoch(trainingSession, dataSet, epoch);
        }
    
        const trainingTimeMs = Date.now() - startTrainingTime;
        showStatusMessage(`Training completed. Final test set accuracy: ${(100 * testAcc).toFixed(2)}% | Total training time: ${trainingTimeMs / 1000} seconds | Average iterations / second: ${(itersPerSecCumulative / numEpochs).toFixed(2)}`);
        setIsTraining(false);
    }
    

    function renderPlots() {
        const margin = { t: 20, r: 25, b: 25, l: 40 };
        return (<div className="section">
            <h3>Plots</h3>
            <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                    <h4>Training Loss</h4>
                    <Plot
                        data={[
                            {
                                x: trainingLosses.map((_, i) => i),
                                y: trainingLosses,
                                type: 'scatter',
                                mode: 'lines',
                            }
                        ]}
                        layout={{ margin, width: 400, height: 320 }}
                    />
                </Grid><Grid item xs={12} md={6}>
                    <h4>Test Accuracy (%)</h4>
                    <Plot
                        data={[
                            {
                                x: testAccuracies.map((_, i) => i + 1),
                                y: testAccuracies.map(a => 100 * a),
                                type: 'scatter',
                                mode: 'lines+markers',
                            }
                        ]}
                        layout={{ margin, width: 400, height: 320 }}
                    />
                </Grid>
            </Grid>
        </div>);
    }

    function renderSummaries() {
        return (<div className="section">
            <h4>Test Summaries</h4>
            <Grid container spacing={2}>
                {summaries.map((summary, summaryIndex) => {
                    const { text, summary: trueSummary } = summary;
                    return (<Grid key={summaryIndex} item xs={12}>
                        <Summary text={text} summary={trueSummary} prediction={summaryPredictions[summaryIndex]} />
                    </Grid>);
                })}
            </Grid>
        </div>);
    }

    const loadSummaries = React.useCallback(async () => {
        const maxNumSummaries = 10;
        const dataSet = new XSumData();
        dataSet.maxNumTestSamples = 2 * dataSet.batchSize;
        const summaries: { text: string; summary: string }[] = [];
        for await (const testBatch of dataSet.testBatches()) {
            const { data, labels } = testBatch;
            const batchSize = labels.dims[0];
            for (let i = 0; summaries.length < maxNumSummaries && i < batchSize; ++i) {
                const text = data.data[i].toString();
                const summary = labels.data[i].toString();
                summaries.push({ text, summary });
            }
            if (summaries.length >= maxNumSummaries) {
                break;
            }
        }
        setSummaries(summaries);
    }, []);

    React.useEffect(() => {
        loadSummaries();
    }, [loadSummaries]);

    return (
        <Container className="App">
            <div className="section">
                <h2>ONNX Runtime Web Training Demo</h2>
                <p>
                    This demo showcases using <Link href="https://onnxruntime.ai/docs/">ONNX Runtime Training for Web</Link> to train a simple neural network for text summarization using the XSum dataset.
                </p>
            </div>
            <div className="section">
                <h3>Background</h3>
                <p>
                    Based on: <Link href="https://github.com/microsoft/onnxruntime-training-examples/tree/master/on_device_training/web">ONNX Runtime Training Web Example</Link>
                </p>
            </div>
            
            <div className="section">
                <h3>Training</h3>
                <Grid container spacing={{ xs: 1, md: 2 }}>
                    <Grid item xs={12} md={4} >
                        <TextField label="Number of epochs"
                            type="number"
                            value={numEpochs}
                            onChange={(e) => setNumEpochs(Number(e.target.value))}
                        />
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <TextField label="Batch size"
                            type="number"
                            value={batchSize}
                            onChange={(e) => setBatchSize(Number(e.target.value))}
                        />
                    </Grid>
                </Grid>
            </div>
            <div className="section">
                <Grid container spacing={{ xs: 1, md: 2 }}>
                    <Grid item xs={12} md={4} >
                        <TextField type="number"
                            label="Max number of training samples"
                            value={maxNumTrainSamples}
                            onChange={(e) => setMaxNumTrainSamples(Number(e.target.value))}
                        />
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <TextField type="number"
                            label="Max number of test samples"
                            value={maxNumTestSamples}
                            onChange={(e) => setMaxNumTestSamples(Number(e.target.value))}
                        />
                    </Grid>
                </Grid>
            </div>
            <div className="section">
                <FormControlLabel
                    control={<Switch
                        checked={enableLiveLogging}
                        onChange={(e) => setEnableLiveLogging(!enableLiveLogging)} />}
                    label='Log all batch results as they happen. Can slow down training.' />
            </div>
            <div className="section">
                <Button onClick={train}
                    disabled={isTraining}
                    variant='contained'>
                    Train
                </Button>
                <br></br>
            </div>
            <pre>{statusMessage}</pre>
            {errorMessage &&
                <p className='error'>
                    {errorMessage}
                </p>}

            {renderPlots()}

            {renderSummaries()}

            {messages.length > 0 &&
                <div>
                    <h3>Logs:</h3>
                    <pre>
                        {messages.map((m, i) => (<React.Fragment key={i}>
                            {m}
                            <br />
                        </React.Fragment>))}
                    </pre>
                </div>}
        </Container>
    );
}

export default App;
