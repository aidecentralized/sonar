import * as ort from 'onnxruntime-web/training';

export class XSumData {
    static readonly BATCH_SIZE = 32;
    static readonly MAX_NUM_TRAIN_SAMPLES = 204045;
    static readonly MAX_NUM_TEST_SAMPLES = 11332;

    constructor(
        public batchSize = XSumData.BATCH_SIZE,
        public maxNumTrainSamples = XSumData.MAX_NUM_TRAIN_SAMPLES,
        public maxNumTestSamples = XSumData.MAX_NUM_TEST_SAMPLES,
    ) {
        if (batchSize <= 0) {
            throw new Error("batchSize must be > 0");
        }
    }

    public getNumTrainingBatches(): number {
        return Math.floor(this.maxNumTrainSamples / this.batchSize);
    }

    public getNumTestBatches(): number {
        return Math.floor(this.maxNumTestSamples / this.batchSize);
    }

    private *batches(data: ort.Tensor[], labels: ort.Tensor[]) {
        for (let batchIndex = 0; batchIndex < data.length; ++batchIndex) {
            yield {
                data: data[batchIndex],
                labels: labels[batchIndex],
            };
        }
    }

    public async *trainingBatches() {
        const trainingData = await this.getData('data/xsum-train.json');
        yield* this.batches(trainingData.data, trainingData.labels);
    }

    public async *testBatches() {
        const testData = await this.getData('data/xsum-test.json');
        yield* this.batches(testData.data, testData.labels);
    }

    private async getData(url: string): Promise<{ data: ort.Tensor[], labels: ort.Tensor[] }> {
        console.debug(`Loading data from "${url}".`);
        const response = await fetch(url);
        const jsonData = await response.json();
        const data = jsonData.map((item: any) => new ort.Tensor('float32', new Float32Array(item.text.split(' ').map(parseFloat)), [item.text.split(' ').length]));
        const labels = jsonData.map((item: any) => new ort.Tensor('float32', new Float32Array(item.summary.split(' ').map(parseFloat)), [item.summary.split(' ').length]));
        return { data, labels };
    }
}
