import asyncio
import json
import logging
from typing import Any, Dict, OrderedDict
import torch
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer

from algos.base_class import BaseFedAvgClient
from algos.topologies.collections import select_topology

class FedRTCNode(BaseFedAvgClient):
    """
    Federated Learning Node using WebRTC for communication
    """

    def __init__(
        self, 
        config: Dict[str, Any], 
        signaling_server: str = "ws://localhost:8080"
    ) -> None:
        # Communication setup
        self.signaling_server = signaling_server
        self.websocket = None
        self.session_id = None
        self.rank = None
        
        # Peer connection management
        self.connections: Dict[int, RTCPeerConnection] = {}
        self.data_channels: Dict[int, RTCDataChannel] = {}
        self.neighbors: Dict[int, int] = {}
        
        # Federated learning setup
        super().__init__(config, comm_utils=None)  # Override comm_utils
        
        # Topology setup
        self.topology = select_topology(config, self.node_id)
        self.topology.initialize()
        
        # Logging
        self.logger = logging.getLogger(f"FedRTCNode-{self.node_id}")
        self.logger.setLevel(logging.INFO)

    def create_peer_connection(self) -> RTCPeerConnection:
        """Create a WebRTC peer connection"""
        config = RTCConfiguration([
            RTCIceServer(urls=[
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302"
            ])
        ])
        return RTCPeerConnection(configuration=config)

    async def setup_data_channel(self, channel: RTCDataChannel, peer_rank: int):
        """Setup data channel for model updates"""
        self.data_channels[peer_rank] = channel

        @channel.on("open")
        def on_open():
            self.logger.info(f"Data channel opened with peer {peer_rank}")

        @channel.on("message")
        def on_message(message):
            try:
                # Parse and handle model updates
                data = json.loads(message)
                if data['type'] == 'model_update':
                    self.handle_model_update(data['payload'], peer_rank)
            except Exception as e:
                self.logger.error(f"Error processing message from {peer_rank}: {e}")

    def handle_model_update(self, payload: Dict, sender_rank: int):
        """Process received model updates"""
        # Convert payload to model update representation
        model_update = self.deserialize_model_update(payload)
        self.comm_utils.received_updates[sender_rank] = model_update

    def serialize_model_update(self, model_update: OrderedDict) -> Dict:
        """Serialize model update for transmission"""
        serialized = {}
        for name, param in model_update.items():
            serialized[name] = param.cpu().numpy().tolist()
        return serialized

    def deserialize_model_update(self, payload: Dict) -> OrderedDict:
        """Deserialize received model update"""
        deserialized = OrderedDict()
        for name, param_list in payload.items():
            deserialized[name] = torch.tensor(param_list)
        return deserialized

    async def connect_to_signaling_server(self):
        """Establish WebSocket connection to signaling server"""
        try:
            self.websocket = await websockets.connect(self.signaling_server)
            
            # Create or join session
            await self.websocket.send(json.dumps({
                'type': 'create_session',
                'maxClients': self.num_collaborators,
                'clientType': 'federated'
            }))
            
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data['type'] == 'session_created':
                self.session_id = data['sessionId']
                self.rank = data['rank']
                self.logger.info(f"Created session {self.session_id}")
        except Exception as e:
            self.logger.error(f"Signaling server connection error: {e}")

    async def handle_topology_and_connections(self):
        """Handle network topology and peer connections"""
        async def process_messages():
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data['type'] == 'topology':
                    await self.process_topology(data)
                elif data['type'] == 'signal':
                    await self.handle_signaling_message(data)

        await process_messages()

    async def process_topology(self, data):
        """Process received network topology"""
        self.rank = data["rank"]
        new_neighbors = data["neighbors"]
        self.logger.info(f"Node {self.rank} received topology. Neighbors: {new_neighbors}")
        
        # Update neighbors
        self.neighbors = new_neighbors
        
        # Initiate connections to neighbors
        for neighbor_rank in self.neighbors.values():
            if neighbor_rank > self.rank:
                await self.initiate_peer_connection(neighbor_rank)

    async def initiate_peer_connection(self, target_rank: int):
        """Initiate WebRTC peer connection"""
        try:
            pc = self.create_peer_connection()
            self.connections[target_rank] = pc
            
            # Create data channel
            channel = pc.createDataChannel(f"federated-{self.rank}-{target_rank}")
            await self.setup_data_channel(channel, target_rank)
            
            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            # Send offer via signaling server
            await self.send_signaling(target_rank, {
                "type": "offer",
                "sdp": pc.localDescription.sdp
            })
        except Exception as e:
            self.logger.error(f"Connection initiation to {target_rank} failed: {e}")

    async def handle_signaling_message(self, message: Dict[str, Any]) -> None:
        """
        Handles signaling messages for WebRTC connections.

        Args:
            message (Dict[str, Any]): The signaling message received.
        """
        try:
            sender_rank: int = message["senderRank"]
            data: Dict[str, Any] = message["data"]

            if sender_rank not in self.connections:
                pc = RTCPeerConnection(configuration=RTCConfiguration(
                    iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
                ))
                self.connections[sender_rank] = pc

                @pc.on("datachannel")
                def on_datachannel(channel: Any) -> None:
                    asyncio.create_task(self.setup_data_channel(channel, sender_rank))

                @pc.on("icecandidate")
                async def on_icecandidate(candidate: Any) -> None:
                    if candidate:
                        await self.send_signaling(sender_rank, {
                            "type": "candidate",
                            "candidate": candidate.sdp
                        })

            pc: RTCPeerConnection = self.connections[sender_rank]

            if data["type"] == "offer":
                if pc.signalingState != "stable":
                    await pc.setLocalDescription(await pc.createAnswer())
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"],
                        type="offer"
                    ))
                else:
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"],
                        type="offer"
                    ))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await self.send_signaling(sender_rank, {
                        "type": "answer",
                        "sdp": answer.sdp
                    })

            elif data["type"] == "answer":
                if pc.signalingState != "stable":
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"],
                        type="answer"
                    ))

            elif data["type"] == "candidate" and pc.remoteDescription:
                await pc.addIceCandidate({
                    "sdp": data["candidate"],
                    "sdpMLineIndex": 0,
                    "sdpMid": "0"
                })
        except Exception as e:
            logging.error(f"Error handling signaling message from {sender_rank}: {e}")

    def run_protocol(self) -> None:
        """Runs the federated learning protocol"""
        asyncio.run(self.run_async_protocol())

    async def run_async_protocol(self):
        """Async version of run_protocol"""
        await self.connect_to_signaling_server()
        
        # Parallel tasks for signaling and FL protocol
        signaling_task = asyncio.create_task(self.handle_topology_and_connections())
        
        # Main federated learning protocol
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)
        
        for it in range(start_round, total_rounds):
            self.round_init()
            
            # Local training
            # self.local_train(it, epochs_per_round)
            
            # Prepare model update for transmission
            model_update = self.get_model_update()
            serialized_update = self.serialize_model_update(model_update)
            
            # Send model update to neighbors via WebRTC
            for neighbor_rank, channel in self.data_channels.items():
                channel.send(json.dumps({
                    'type': 'model_update',
                    'payload': serialized_update
                }))
            
            # Receive and aggregate updates
            neighbors = list(self.neighbors.values())
            self.receive_and_aggregate(neighbors)
            
            self.local_test()
            self.round_finalize()
        
        # Cleanup
        await signaling_task

class FedRTCServer(BaseFedAvgClient):
    """
    Federated RTC Server Class. It does not do anything.
    It just exists to keep the code compatible across different algorithms.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        pass

    def run_protocol(self) -> None:
        pass
