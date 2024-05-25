from concurrent import futures
import grpc
import fl_pb2
import fl_pb2_grpc

class FederatedLearningServicer(fl_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.local_averages = []

    def SendLocalAverage(self, request, context):
        self.local_averages.append(request.local_average)
        return fl_pb2.AverageResponse(global_average=self._compute_global_average())

    def GetGlobalAverage(self, request, context):
        return fl_pb2.AverageResponse(global_average=self._compute_global_average())

    def _compute_global_average(self):
        if not self.local_averages:
            return 0
        return sum(self.local_averages) / len(self.local_averages)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fl_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
