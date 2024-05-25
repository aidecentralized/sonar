import grpc
import fl_pb2
import fl_pb2_grpc
import numpy as np

def run_client():
    # Simulate local data
    local_data = np.random.random(100)

    # Compute local average
    local_average = np.mean(local_data)
    print(f"Local average: {local_average}")

    # Connect to server and send local average
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = fl_pb2_grpc.FederatedLearningStub(channel)
        response = stub.SendLocalAverage(fl_pb2.AverageRequest(local_average=local_average))
        print(f"Global average from server: {response.global_average}")

if __name__ == '__main__':
    run_client()
