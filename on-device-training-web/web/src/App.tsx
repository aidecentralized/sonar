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

    const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState<number>(5000);
    const [maxNumTestSamples, setMaxNumTestSamples] = React.useState<number>(1000);
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
        const labels: bigint[] = [];
        for (let i = 0; i < summaries.length; ++i) {
            const text = summaries[i].text.split(' ').map(parseFloat);
            for (let j = 0; j < text.length; ++j) {
                input[i * text.length + j] = text[j];
            }
            labels.push(BigInt(summaries[i].summary.split(' ').map(parseFloat).join(' ')));
        }

        const feeds = {
            input: new ort.Tensor('float32', input, batchShape),
            labels: new ort.Tensor('int64', new BigInt64Array(labels), [summaries.length])
        };

        const results = await session.runEvalStep(feeds);
        const predictions = getPredictions(results['output']);
        setSummaryPredictions(predictions.slice(0, summaries.length));
    }

    function getPredictions(results: ort.Tensor): string[] {
        const predictions: string[] = [];
        const dataArray = Array.from(results.data as Iterable<number | bigint>) as (number | bigint)[];
        dataArray.forEach(d => predictions.push(d.toString()));
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
            const feeds = {
                input: batch.data,
                labels: batch.labels
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
    
            const feeds = {
                input: batch.data,
                labels: batch.labels
            };
    
            const results = await session.runEvalStep(feeds);
            const loss = parseFloat(results[lossNodeName].data[0].toString());
            accumulatedLoss += loss;
            testSummariesSoFar += batch.data.dims[0];
            numCorrect += countCorrectPredictions(results['output'], batch.labels);
            const iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
            const message = `TESTING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Average test loss: ${(accumulatedLoss / batchNum).toFixed(2)} | Accuracy: ${numCorrect}/${testSummariesSoFar} (${(100 * numCorrect / testSummariesSoFar).toFixed(2)}%) | ${iterationsPerSecond.toFixed(2)} it/s`;
            await logMessage(message);
        }
        const avgAcc = numCorrect / testSummariesSoFar;
        setTestAccuracies(accs => accs.concat(avgAcc));
        return avgAcc;
    }

    async function train() {
        console.log("Training started");
        setIsTraining(true);
        if (maxNumTrainSamples > XSumData.MAX_NUM_TRAIN_SAMPLES || maxNumTestSamples > XSumData.MAX_NUM_TEST_SAMPLES) {
            showErrorMessage(`Max number of training samples (${maxNumTrainSamples}) or test samples (${maxNumTestSamples}) exceeds the maximum allowed (${XSumData.MAX_NUM_TRAIN_SAMPLES} and ${XSumData.MAX_NUM_TEST_SAMPLES}, respectively). Please try again.`);
            return;
        }

        const trainingSession = await loadTrainingSession();
        const dataSet = new XSumData(batchSize, maxNumTrainSamples, maxNumTestSamples);

        lastLogTime = Date.now();
        const startTrainingTime = Date.now();
        showStatusMessage('Training started');
        let itersPerSecCumulative = 0;
        let testAcc = 0;
        for (let epoch = 0; epoch < numEpochs; epoch++) {
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
                    ONNX Runtime Training for Web is a new feature in ORT 1.17.0 that enables developers to train machine learning models in the browser using CPU and WebAssembly.
                </p>
                <p>
                    This in-browser training capability is specifically designed to support federated learning scenarios, where multiple devices can collaborate to train a model without sharing data with each other.
                    This approach enhances privacy and security while still allowing for effective machine learning.
                </p>
                <p>
                    If you're interested in learning more about ONNX Runtime Training for Web and its potential applications, be sure to check out our blog coming out soon.
                </p>
                <p>
                    For more information on how to use ONNX Runtime Web for training, please refer to <Link href="https://onnxruntime.ai/docs/">ONNX Runtime documentation</Link> or
                    the <Link href="https://github.com/microsoft/onnxruntime-training-examples">ONNX Runtime Training Examples code</Link>.
                </p>
            </div>
            <div className="section">
                <h3>Training Metrics</h3>
                <TableContainer sx={{ width: '50%' }}>
                    <Table size='small'>
                        <TableHead>
                            <TableRow>
                                <TableCell sx={{ fontWeight: 'bold' }}>Browser</TableCell>
                                <TableCell sx={{ fontWeight: 'bold' }}>Heap usage in MB</TableCell>
                                <TableCell sx={{ fontWeight: 'bold' }}>it/s</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell>Chrome</TableCell>
                                <TableCell>25.2</TableCell>
                                <TableCell>54.30</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>Edge</TableCell>
                                <TableCell>24.2</TableCell>
                                <TableCell>55.48</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
                <div className="section moreInfo">
                    <Button onClick={toggleMoreInfoIsCollapsed}>{moreInfoIsCollapsed ? 'Expand' : 'Collapse'} more info</Button>
                    {!moreInfoIsCollapsed && <div>
                        <p>
                            The above measurements were obtained on a Windows PC in a window with a single tab open.
                        </p>
                        <p>
                            Measuring memory usage and performance in the browser is difficult because things such as screen resolution, window size, OS and OS version of the host machine, the number of tabs or windows open, the number of extensions installed, and more can affect memory usage.
                            Thus, the above results may be difficult to replicate. The above numbers are meant to reflect that training in the browser does not have to be compute- or memory-intensive when using the ORT Web for Training framework.
                        </p>
                    </div>}
                </div>
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
