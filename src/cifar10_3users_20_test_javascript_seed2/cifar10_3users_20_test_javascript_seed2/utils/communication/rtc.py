import asyncio
import json
import logging
import sys
from typing import Dict, Any, List, OrderedDict
import torch
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from utils.communication.interface import CommunicationInterface

class RTCCommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.config = config
        self.comm = None
        self.rank = None
        self.size = None
        
        # WebRTC specific attributes
        self.signaling_server = config.get("signaling_server", "ws://localhost:8080")
        self.websocket = None
        self.session_id = None
        self.connections: Dict[int, RTCPeerConnection] = {}
        self.data_channels: Dict[int, RTCDataChannel] = {}
        self.received_updates: Dict[int, OrderedDict] = {}
        
        # Setup enhanced logging
        self._setup_logging()
        
        # Event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger.info("RTCCommUtils initialized with config: %s", config)

    def _setup_logging(self):
        """Setup detailed logging configuration"""
        self.logger = logging.getLogger("RTCCommUtils")
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # File handler
        file_handler = logging.FileHandler('rtc_comm.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Log initial setup
        self.logger.info("Logging system initialized")

    def initialize(self):
        """Initialize WebRTC connections"""
        self.logger.info("Starting RTCCommUtils initialization")
        try:
            self.loop.run_until_complete(self._connect_to_signaling_server())
            self.logger.info("RTCCommUtils initialization completed successfully")
        except Exception as e:
            self.logger.error("Failed to initialize RTCCommUtils: %s", str(e), exc_info=True)
            raise

    async def _connect_to_signaling_server(self):
        """Connect to signaling server and setup initial session"""
        self.logger.info("Attempting to connect to signaling server at %s", self.signaling_server)
        try:
            self.websocket = await websockets.connect(self.signaling_server)
            self.logger.debug("WebSocket connection established")
            
            join_message = {
                'type': 'join_session',
                'clientType': 'federated',
                'maxClients': self.config.get("num_users"),
                "sessionId": 1111, # TODO: Remove hardcoded session ID
            }
            self.logger.debug("Sending join session message: %s", join_message)
            await self.websocket.send(json.dumps(join_message))
            
            response = await self.websocket.recv()
            data = json.loads(response)
            self.logger.debug("Received session response: %s", data)
            
            if data['type'] == 'session_joined':
                self.session_id = data['sessionId']
                self.rank = data['rank']
                self.size = self.config.get("num_users")
                self.logger.info("Successfully joined session %s as rank %d (total clients: %d)", 
                               self.session_id, self.rank, self.size)
                
                # Start background task for handling signaling messages
                asyncio.create_task(self._handle_signaling())
            else:
                self.logger.error("Unexpected response type: %s", data['type'])
        except Exception as e:
            self.logger.error("Failed to connect to signaling server: %s", str(e), exc_info=True)
            raise

    def register_self(self, obj: Any):
        """Register the federated learning node"""
        self.logger.info("Registering federated learning node")
        self.comm = obj
        self.logger.debug("Node registration complete: %s", type(obj).__name__)

    def get_rank(self) -> int:
        """Get the rank of current node"""
        rank = self.rank if self.rank is not None else 0
        self.logger.debug("Returning rank: %d", rank)
        return rank

    def send(self, dest: str | int, data: Any):
        """Send data to a specific destination"""
        if isinstance(dest, str):
            dest = int(dest)
        
        self.logger.info("Preparing to send data to destination %d", dest)
        
        if dest not in self.data_channels:
            self.logger.error("No connection to destination %d", dest)
            return
            
        try:
            # Log data size for debugging
            data_size = len(str(data)) if data else 0
            self.logger.debug("Serializing data of size %d bytes", data_size)
            
            serialized_data = self._serialize_data(data)
            message = json.dumps({
                'type': 'model_update',
                'payload': serialized_data
            })
            
            self.logger.debug("Sending message to destination %d", dest)
            self.loop.run_until_complete(self._send_async(dest, message))
            self.logger.info("Successfully sent data to destination %d", dest)
        except Exception as e:
            self.logger.error("Error sending data to %d: %s", dest, str(e), exc_info=True)

    async def _send_async(self, dest: int, message: str):
        """Async helper for sending data"""
        channel = self.data_channels[dest]
        self.logger.debug("Checking channel state for destination %d: %s", dest, channel.readyState)
        
        if channel.readyState == "open":
            channel.send(message)
            self.logger.debug("Message sent through channel to destination %d", dest)
        else:
            self.logger.error("Channel to destination %d not open (state: %s)", dest, channel.readyState)

    def receive(self, node_ids: List[int]) -> Any:
        """Receive data from specified nodes"""
        self.logger.info("Attempting to receive data from nodes: %s", node_ids)
        try:
            updates = {}
            for node_id in node_ids:
                if node_id in self.received_updates:
                    self.logger.debug("Retrieved update from node %d", node_id)
                    updates[node_id] = self.received_updates[node_id]
                    del self.received_updates[node_id]
                else:
                    self.logger.debug("No update available from node %d", node_id)
            
            self.logger.info("Received updates from %d nodes", len(updates))
            return updates
        except Exception as e:
            self.logger.error("Error receiving data: %s", str(e), exc_info=True)
            return {}

    def broadcast(self, data: Any):
        """Broadcast data to all connected peers"""
        self.logger.info("Broadcasting data to %d peers", len(self.data_channels))
        for dest in self.data_channels.keys():
            self.logger.debug("Broadcasting to destination %d", dest)
            self.send(dest, data)

    def all_gather(self):
        """Gather data from all nodes"""
        self.logger.info("Gathering data from all %d nodes", self.size)
        return self.receive(list(range(self.size)))

    def finalize(self):
        """Clean up WebRTC connections"""
        self.logger.info("Starting cleanup of WebRTC connections")
        try:
            # Close all data channels
            for peer_id, channel in self.data_channels.items():
                self.logger.debug("Closing data channel for peer %d", peer_id)
                channel.close()
            
            # Close all peer connections
            for peer_id, pc in self.connections.items():
                self.logger.debug("Closing peer connection for peer %d", peer_id)
                pc.close()
            
            # Close websocket
            if self.websocket:
                self.logger.debug("Closing websocket connection")
                self.loop.run_until_complete(self.websocket.close())
            
            # Clean up event loop
            self.logger.debug("Closing event loop")
            self.loop.close()
            
            self.logger.info("Successfully finalized all connections")
        except Exception as e:
            self.logger.error("Error in finalize: %s", str(e), exc_info=True)

    async def _handle_signaling(self):
        """Handle incoming signaling messages"""
        self.logger.info("Started signaling message handler")
        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                self.logger.debug("Received signaling message: %s", data['type'])
                
                if data['type'] == 'ice_candidate':
                    await self._handle_ice_candidate(data)
                elif data['type'] == 'offer':
                    await self._handle_offer(data)
                elif data['type'] == 'answer':
                    await self._handle_answer(data)
        except Exception as e:
            self.logger.error("Error in signaling handler: %s", str(e), exc_info=True)

    async def _handle_ice_candidate(self, data: Dict):
        """Handle ICE candidate message"""
        sender = data['sender']
        candidate = data['candidate']
        self.logger.debug("Handling ICE candidate from peer %d", sender)
        
        if sender in self.connections:
            pc = self.connections[sender]
            await pc.addIceCandidate(candidate)
            self.logger.debug("Added ICE candidate for peer %d", sender)
        else:
            self.logger.warning("Received ICE candidate for unknown peer %d", sender)

    async def _handle_offer(self, data: Dict):
        """Handle WebRTC offer"""
        sender = data['sender']
        offer = data['offer']
        self.logger.info("Handling WebRTC offer from peer %d", sender)
        
        pc = RTCPeerConnection()
        self.connections[sender] = pc
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            self.logger.debug("Data channel opened from peer %d", sender)
            self.data_channels[sender] = channel
            self._setup_data_channel(channel, sender)
        
        await pc.setRemoteDescription(RTCSessionDescription(**offer))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        response = {
            'type': 'answer',
            'target': sender,
            'answer': {'sdp': answer.sdp, 'type': answer.type}
        }
        self.logger.debug("Sending answer to peer %d", sender)
        await self.websocket.send(json.dumps(response))

    async def _handle_answer(self, data: Dict):
        """Handle WebRTC answer"""
        sender = data['sender']
        answer = data['answer']
        self.logger.info("Handling WebRTC answer from peer %d", sender)
        
        if sender in self.connections:
            pc = self.connections[sender]
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
            self.logger.debug("Set remote description for peer %d", sender)
        else:
            self.logger.warning("Received answer from unknown peer %d", sender)

    def _setup_data_channel(self, channel: RTCDataChannel, peer_id: int):
        """Setup handlers for data channel"""
        self.logger.info("Setting up data channel for peer %d", peer_id)
        
        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                self.logger.debug("Received message from peer %d: %s", peer_id, data['type'])
                
                if data['type'] == 'model_update':
                    update = self._deserialize_data(data['payload'])
                    self.received_updates[peer_id] = update
                    self.logger.info("Processed model update from peer %d", peer_id)
            except Exception as e:
                self.logger.error("Error processing message from peer %d: %s", 
                                peer_id, str(e), exc_info=True)

    def _serialize_data(self, data: Any) -> Dict:
        """Serialize data for transmission"""
        self.logger.debug("Serializing data")
        try:
            if isinstance(data, OrderedDict):
                return {name: param.cpu().numpy().tolist() for name, param in data.items()}
            return data
        except Exception as e:
            self.logger.error("Error serializing data: %s", str(e), exc_info=True)
            raise

    def _deserialize_data(self, data: Dict) -> Any:
        """Deserialize received data"""
        self.logger.debug("Deserializing data")
        try:
            if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                return OrderedDict({name: torch.tensor(param) for name, param in data.items()})
            return data
        except Exception as e:
            self.logger.error("Error deserializing data: %s", str(e), exc_info=True)
            raise


# from typing import Dict, Any, List
# from utils.communication.interface import CommunicationInterface


# class RTCCommUtils(CommunicationInterface):
#     def __init__(self, config: Dict[str, Dict[str, Any]]):
#         self.comm = None
#         self.rank = None
#         self.size = None

#     def initialize(self):
#         print("RTCCommUtils initialize called")
#         pass

#     def register_self(self, obj: Any):
#         print("RTCCommUtils register_self called")
#         pass

#     def get_rank(self) -> int:
#         print("RTCCommUtils get_rank called")
#         return 0

#     def send(self, dest: str | int, data: Any):
#         print("RTCCommUtils send called")

#     def receive(self, node_ids: List[int]) -> Any:
#         print("RTCCommUtils receive called")

#     def broadcast(self, data: Any):
#         print("RTCCommUtils broadcast called")

#     def all_gather(self):
#         """
#         This function is used to gather data from all the nodes.
#         """
#         print("RTCCommUtils all_gather called")

#     def finalize(self):
#         print("RTCCommUtils finalize called")
