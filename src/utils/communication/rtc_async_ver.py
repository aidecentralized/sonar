import threading
import asyncio
import json
import os
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
import logging
from collections import defaultdict, OrderedDict
from typing import Dict, Set, List, Any, Optional
from enum import Enum
import time
import argparse
import numpy as np
import random
from queue import Queue, Empty
from utils.communication.interface import CommunicationInterface
import torch
import io

class NodeState(Enum):
    CONNECTING = 1
    READY = 2
    DISCONNECTING = 3

logging.basicConfig(level=logging.INFO)

import json
import torch
import numpy as np
from typing import Dict, Any

def tensor_to_serializable(obj: Any) -> Any:
    """Convert PyTorch tensors to serializable format."""
    if isinstance(obj, torch.Tensor):
        # Convert tensor to numpy array and then to list
        return {
            "__tensor__": True,
            "data": obj.cpu().detach().numpy().tolist(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape)
        }
    elif isinstance(obj, dict):
        return {key: tensor_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(item) for item in obj]
    return obj

def serializable_to_tensor(obj: Any) -> Any:
    """Convert serializable format back to PyTorch tensors."""
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            # Convert back to tensor
            data = torch.tensor(obj["data"], dtype=getattr(torch, obj["dtype"].split('.')[-1]))
            return data.reshape(obj["shape"])
        return {key: serializable_to_tensor(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serializable_to_tensor(item) for item in obj]
    return obj

def serialize_message(message: Dict[str, Any]) -> str:
    """Serialize PyTorch dictionary to JSON string."""
    serializable_dict = tensor_to_serializable(message)
    return json.dumps(serializable_dict)

def deserialize_message(json_str: str) -> Dict[str, Any]:
    """Deserialize JSON string back to PyTorch dictionary."""
    serializable_dict = json.loads(json_str)
    return serializable_to_tensor(serializable_dict)

class RTCCommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.config = config
        self.signaling_server = config.get("signaling_server", "ws://localhost:8765")
        self.websocket = None
        self.connections: Dict[int, RTCPeerConnection] = {}
        self.data_channels: Dict[int, RTCDataChannel] = {}
        self.rank = None
        self.size: int = int(config.get("num_users", 2))
        self.session_id = 1111
        self.neighbors = None
        self.state = NodeState.CONNECTING
        self.state_lock = asyncio.Lock()
        self.connection_queue = asyncio.Queue()
        self.connection_retries = defaultdict(int)
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 30  # Increased from 2 to 30 seconds
        self.pending_connections: Set[int] = set()
        self.connected_peers: Set[int] = set()
        self.expected_connections = 0
        self.connection_timeout = 60  # Increased from 30 to 60 seconds
        self.ice_gathering_timeout = 20  # Increased from 10 to 20 seconds
        self.logger = self.setup_logger()

        # Training attributes
        self.current_round = 0
        self.clear_peer_weights = False # boolean to clear
        # self.model_weights = np.zeros(10)  # Example model weights
        self.model_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.peer_rounds: Dict[int, int] = {}
        self.peer_weights: Dict[Any, Any] = {}
        self.message_callbacks: Dict[str, Any] = {}
        self.rounds_match_events: Dict[int, threading.Event] = {}

        # adding in send and message queues
        self.send_queue = Queue()     # For outgoing messages (Main -> WebRTC)
        self.message_queue = Queue()  # For received messages (WebRTC -> Main)
        self.stop_workers = False

        # communication cost
        self.comm_cost_sent = 0
        self.comm_cost_received = 0

        # Add tracking for expected chunks
        self.expected_chunks = {}  # {layer_name: expected_num_chunks}
        self.received_chunks = {}  # {layer_name: current_num_chunks} 
        self.all_chunks_received = False

    def setup_logger(self) -> logging.Logger:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create a logger for this instance
        logger = logging.getLogger(f"Node")  # Will be updated with rank later
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # We'll add the file handler when we get our rank
        return logger

    def setup_file_logging(self):
        if self.rank is not None:
            # Create file handler
            fh = logging.FileHandler(f"logs/client_{self.rank}_new.log", mode='w')
            fh.setLevel(logging.INFO)
            
            # Create formatter
            # formatter = logging.Formatter(
            #     '%(asctime)s - %(levelname)s - %(message)s'
            # )
            # fh.setFormatter(formatter)
            
            # Add rank to logger name
            self.logger = logging.getLogger(f"Node-{self.rank}")
            
            # Add handler
            self.logger.handlers.clear()  # Remove any existing handlers
            self.logger.addHandler(fh)
            
            # Add console handler only if not already added
            if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):
                ch = logging.StreamHandler()
                # ch.setFormatter(formatter)
                self.logger.addHandler(ch)

    async def change_state(self, new_state: NodeState):
        async with self.state_lock:
            self.state = new_state
            self.logger.info(f"Node {self.rank} state changed to {new_state}")

    async def get_state(self) -> NodeState:
        async with self.state_lock:
            return self.state

    def register_self(self, obj: "BaseNode"):
        self.base_node = obj

    async def setup_data_channel(self, channel: RTCDataChannel, peer_rank: int):
        self.logger.info(f"setup_data_channel() called for peer {peer_rank}")
        self.data_channels[peer_rank] = channel
        self.peer_rounds[peer_rank] = 0

        @channel.on("open")
        def on_open():
            self.logger.info(f"Data channel opened with peer {peer_rank}")
            # Note: this is only called on the offerer side
            asyncio.create_task(self.on_peer_connected(peer_rank))
            # asyncio.create_task(self.ping_loop(peer_rank))

                # Sending messages from the send queue
        async def send_messages():
            print("[WebRTC] send_messages() started")
            while True:
                try:
                    message = self.send_queue.get_nowait()  # Get message from the queue
                    [peer_rank, data] = message
                    if peer_rank in self.data_channels:
                        channel = self.data_channels[peer_rank]
                        if channel.readyState == "open":
                            message = json.dumps(data)
                            channel.send(message)
                            # self.logger.info(f"Successfully sent message to peer {peer_rank}")
                        else:
                            self.logger.error(f"Channel to peer {peer_rank} not open")
                    else:
                        self.logger.error(f"No channel to peer {peer_rank}")
                    # print(f"[WebRTC] Sent a message to peer {peer_rank}")
                except Empty:
                    # print("[WebRTC] No message in send_queue, sleeping...")
                    await asyncio.sleep(0.05)  # Reduced sleep duration for faster checks

        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                # self.logger.info("[WebRTC] Putting into message_queue message from peer " + str(peer_rank) + ' of type ' + str(data["type"])) # + str(data))
                self.message_queue.put([peer_rank, data])
                # if data["type"] == "ping":
                #     self.logger.info(f"{self.rank} Received ping from {peer_rank}")
                #     channel.send(json.dumps({
                #         "type": "pong",
                #         "timestamp": data["timestamp"],
                #         "respondedAt": time.time() * 1000
                #     }))
                # elif data["type"] == "pong":
                #     rtt = time.time() * 1000 - data["timestamp"]
                #     self.logger.info(f"{self.rank} Received pong from {peer_rank}, RTT: {rtt:.2f}ms")
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse message from {peer_rank}: {message}")
            except Exception as e:
                self.logger.error(f"Failed to handle message from {peer_rank}: {e}")

        asyncio.create_task(send_messages())

        @channel.on("close")
        def on_close():
            self.logger.info(f"Data channel with peer {peer_rank} closed")
            if peer_rank in self.data_channels:
                del self.data_channels[peer_rank]
            if peer_rank in self.connected_peers:
                self.connected_peers.remove(peer_rank)

    async def on_peer_connected(self, peer_rank: int):
        self.connected_peers.add(peer_rank)
        self.pending_connections.discard(peer_rank)
        self.logger.info(f"Node {self.rank} connected to peer {peer_rank}. "
                    f"Connected: {len(self.connected_peers)}/{self.expected_connections + 1}")
        
        await self.websocket.send(json.dumps({
            "type": "connection_established",
            "peerRank": peer_rank,
            "sessionId": self.session_id,
        }))

        if len(self.connected_peers) == self.expected_connections:
            self.logger.info(f"Node {self.rank} broadcasting node ready")
            await self.broadcast_node_ready()

    async def ping_loop(self, peer_rank: int):
        while self.state != NodeState.DISCONNECTING:
            if peer_rank in self.data_channels:
                channel = self.data_channels[peer_rank]
                if channel.readyState == "open":
                    channel.send(json.dumps({
                        "type": "ping",
                        "timestamp": time.time() * 1000
                    }))
            await asyncio.sleep(5)

    async def connection_worker(self):
        while True:
            try:
                target_rank = await self.connection_queue.get()
                self.logger.info(f"Node {self.rank} worker processing connection to {target_rank}")
                retry_count = self.connection_retries[target_rank]
                
                try:
                    await asyncio.wait_for(
                        self.initiate_connection(target_rank),
                        timeout=self.connection_timeout
                    )
                    self.logger.info(f"Node {self.rank} successfully initiated connection to {target_rank}")
                    self.connection_retries[target_rank] = 0
                except asyncio.TimeoutError:
                    logging.error(f"Connection timeout to {target_rank}")
                    await self.handle_connection_failure(target_rank)
                except Exception as e:
                    logging.error(f"Connection attempt to {target_rank} failed: {e}")
                    await self.handle_connection_failure(target_rank)
                finally:
                    self.connection_queue.task_done()
            except asyncio.CancelledError:
                break

    def create_peer_connection(self) -> RTCPeerConnection:
        config = RTCConfiguration([
            # For local testing, prioritize host candidates
            RTCIceServer(urls=[
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302"
            ])
        ])
        # Create peer connection with the configuration
        pc = RTCPeerConnection(configuration=config)
        return pc

    async def initiate_connection(self, target_rank: int):
        try:
            pc = self.create_peer_connection()
            self.connections[target_rank] = pc
            
            @pc.on("iceconnectionstatechange")
            async def on_ice_connection_state_change():
                self.logger.info(f"ICE connection state to {target_rank}: {pc.iceConnectionState}")
                if pc.iceConnectionState == "failed":
                    logging.error(f"ICE connection to {target_rank} failed")
                    await self.handle_connection_failure(target_rank)
                elif pc.iceConnectionState == "connected":
                    self.logger.info(f"ICE connection to {target_rank} established")

            @pc.on("icegatheringstatechange")
            async def on_ice_gathering_state_change():
                self.logger.info(f"ICE gathering state for {target_rank}: {pc.iceGatheringState}")

            # Create data channel first
            channel = pc.createDataChannel(f"chat-{self.rank}-{target_rank}")
            await self.setup_data_channel(channel, target_rank)
            
            # Create and set local description
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            # Wait for ICE gathering or timeout
            gathering_complete = asyncio.Event()
            
            @pc.on("icegatheringstatechange")
            async def on_gathering_complete():
                if pc.iceGatheringState == "complete":
                    gathering_complete.set()
            
            try:
                await asyncio.wait_for(gathering_complete.wait(), self.ice_gathering_timeout)
            except asyncio.TimeoutError:
                logging.warning(f"ICE gathering timed out for {target_rank}")
                
            # Send offer
            await self.send_signaling(target_rank, {
                "type": "offer",
                "sdp": pc.localDescription.sdp
            })

        except Exception as e:
            logging.error(f"Failed to initiate connection to {target_rank}: {e}")
            await self.cleanup_connection(target_rank)
            raise

    async def cleanup_connection(self, rank: int):
        try:
            if rank in self.connections:
                pc = self.connections[rank]
                
                # Close data channel first
                if rank in self.data_channels:
                    channel = self.data_channels[rank]
                    if channel and channel.readyState != "closed":
                        channel.close()
                    del self.data_channels[rank]
                
                # Stop all transceivers
                for transceiver in pc.getTransceivers():
                    await transceiver.stop()
                
                # Close the peer connection
                await pc.close()
                
                # Remove from connections
                del self.connections[rank]
            
            if rank in self.pending_connections:
                self.pending_connections.remove(rank)
            if rank in self.connected_peers:
                self.connected_peers.remove(rank)
                
            self.logger.info(f"Cleaned up connection to peer {rank}")
            
        except Exception as e:
            logging.error(f"Error during connection cleanup for peer {rank}: {e}")

    async def handle_connection_failure(self, target_rank):
        retry_count = self.connection_retries[target_rank]
        if retry_count < self.MAX_RETRIES:
            self.connection_retries[target_rank] = retry_count + 1
            await asyncio.sleep(self.RETRY_DELAY * (retry_count + 1))
            if target_rank not in self.connected_peers:
                await self.cleanup_connection(target_rank)  # Clean up before retrying
                await self.connection_queue.put(target_rank)
                self.logger.info(f"Retrying connection to {target_rank}, attempt {retry_count + 1}")
        else:
            logging.error(f"Max retries reached for {target_rank}")
            await self.cleanup_connection(target_rank)

    async def handle_signaling_message(self, message):
        try:
            sender_rank = message["senderRank"]
            data = message["data"]
            
            if sender_rank not in self.connections:
                pc = RTCPeerConnection(configuration=RTCConfiguration(
                    iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
                ))
                self.connections[sender_rank] = pc
                
                @pc.on("datachannel")
                def on_datachannel(channel):
                    asyncio.create_task(self.setup_data_channel(channel, sender_rank))
                    
                @pc.on("icecandidate")
                async def on_icecandidate(candidate):
                    if candidate:
                        await self.send_signaling(sender_rank, {
                            "type": "candidate",
                            "candidate": candidate.sdp
                        })
            
            pc = self.connections[sender_rank]
            
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
                    "sdpMLineIndex": data["sdpMLineIndex"],
                    "sdpMid": data["sdpMid"]
                })
        except Exception as e:
            logging.error(f"Error handling signaling message from {sender_rank}: {e}")

    async def create_session(self, max_clients: int):
        try:
            await self.websocket.send(json.dumps({
                'type': 'create_session',
                'maxClients': max_clients,
                'clientType': 'python'
            }))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if (data['type'] == 'session_created'):
                self.session_id = data['sessionId']
                self.rank = data['rank']
                self.logger.info(f"Created session {self.session_id}")
                print(f"Share this session code with other clients: {self.session_id}")
            else:
                self.logger.error(f"Failed to create session: {data.get('message', 'Unknown error')}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return False

    async def join_session(self, session_id: str):
        try:
            await self.websocket.send(json.dumps({
                'type': 'join_session',
                'sessionId': session_id,
                'clientType': 'python',
                'maxClients': self.config.get("num_users", 2),
                'config': self.config
            }))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data['type'] == 'session_joined':
                self.session_id = data['sessionId']
                self.rank = data['rank']
                self.logger.info(f"Joined session {self.session_id} with rank {self.rank}")
                return True
            else:
                self.logger.error(f"Failed to join session: {data.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            self.logger.error(f"Error joining session: {e}")
            return False

    async def connect(self, create: bool = False, session_id: str = None, max_clients: int = None):
        max_clients = self.size
        session_id = self.session_id

        network_ready_event = asyncio.Event()  # Create an event to wait for network readiness

        try:
            self.websocket = await websockets.connect(self.signaling_server)
            await self.change_state(NodeState.CONNECTING)
            
            if create:
                if not max_clients:
                    raise ValueError("max_clients is required when creating a session")
                if not await self.create_session(max_clients):
                    return
            else:
                if not session_id:
                    raise ValueError("session_id is required when joining a session")
                if not await self.join_session(session_id):
                    return

            workers = [self.connection_worker() for _ in range(3)]
            worker_tasks = [asyncio.create_task(w) for w in workers]
            
            try:
                async def process_messages():
                    while True:
                        message = await self.websocket.recv()
                        data = json.loads(message)
                        self.logger.info(f"Node received message: {data['type']}")
                        
                        if data['type'] == 'session_ready':
                            self.logger.info(data['message'])
                        elif data['type'] == 'topology':
                            await self.handle_topology(data)
                        elif data['type'] == 'signal':
                            await self.handle_signaling_message(data)
                        elif data['type'] == 'network_ready':
                            await self.change_state(NodeState.READY)
                            self.logger.info("All connections established!")
                            network_ready_event.set()  # Set the event when network is ready
                            return
                
                await process_messages()
                await network_ready_event.wait()  # Wait for the network to be ready
                self.logger.info("finished network ready wait, waiting forever")
                while True:
                    await asyncio.sleep(1)
                return True
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                return False
            finally:
                for task in worker_tasks:
                    task.cancel()
                await self.change_state(NodeState.DISCONNECTING)

        except Exception as e:
            self.logger.error(f"Connection error: {e}")

    async def broadcast_node_ready(self):
        if self.websocket:
            await self.websocket.send(json.dumps({
                "type": "node_ready",
                "sessionId": self.session_id,
                "rank": self.rank
            }))
    async def handle_topology(self, data):
        self.rank = data["rank"]
        self.setup_file_logging()
        new_neighbors: List[int] = data["neighbors"][f"neighbor{self.rank}"]
        self.logger.info(f"Node {self.rank} received topology. Neighbors: {new_neighbors}")

        if self.neighbors:
            removed = set(self.neighbors) - set(new_neighbors)
            for rank in removed:
                await self.cleanup_connection(rank)

        self.neighbors = new_neighbors
        self.expected_connections = len(new_neighbors)


        if self.expected_connections == 0:
            print(f"Node {self.rank} broadcasting node ready")
            await self.broadcast_node_ready()

        # Only initiate connections to higher-ranked neighbors
        connection_tasks = []
        for neighbor_rank in self.neighbors:
            if neighbor_rank is None:
                continue
            # if neighbor_rank > self.rank:
            if (neighbor_rank not in self.connections and 
                neighbor_rank not in self.pending_connections):
                self.logger.info(f"Node {self.rank} queueing connection to {neighbor_rank}")
                await self.connection_queue.put(neighbor_rank)
                self.pending_connections.add(neighbor_rank)

    async def send_signaling(self, target_rank: int, data: dict):
        if self.websocket:
            await self.websocket.send(json.dumps({
                "type": "signal",
                "targetRank": target_rank,
                "data": data,
                "sessionId": self.session_id,
            }))

    def chunk_tensor(self, tensor, chunk_size):
        """Yield successive chunks from a tensor, including its original shape."""
        original_shape = tensor.shape
        num_chunks = (tensor.numel() + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            yield tensor.flatten()[i * chunk_size:(i + 1) * chunk_size].reshape(-1), num_chunks, original_shape

    # TODO: Here are all the functions that are not going to be on the listening thread
    def get_peer_weights(self, peer_rank: int, max_retries: int = 3) -> Optional[np.ndarray]:
        requests = set()
        for attempt in range(max_retries):
            try:
                request_id = f"weights_request_{time.time()}_{random.randint(0, 1000)}"
                requests.add(request_id)

                # Create an event for synchronization
                # response_event = threading.Event()
                # response_data = []

                # def callback(data):
                #     response_data.append(data)
                #     response_event.set()

                # self.message_callbacks[request_id] = callback

                self.send_to_peer(peer_rank, {
                    "type": "weights_request",
                    "request_id": request_id
                })

                return # for testing

                # Wait for response with timeout
                # if response_event.wait(timeout=5.0):
                #     return np.array(response_data[0]["weights"])
                message = self.message_queue.get(timeout=5.0)
                self.logger.info('########after message_queue.get', message)
                self.logger.info(f"[Main Thread] Popped from message_queue, reading...")

                if (message):
                    self.logger.info(f"[Main Thread] Received from peer {message[0]} of type {message[1]['type']}")
                    peer_rank = message[0]
                    data = message[1]
                    self.handle_data_channel_message(peer_rank, data)
                    if (data["type"] == "weights_response" and data["request_id"] in requests):
                        self.logger.info(f"request id matched!")
                        # reconstructed_weights = OrderedDict({k: torch.tensor(v) if isinstance(v, list) else v for k, v in data["weights"].items()})
                        sender = peer_rank
                        curr_round = data["round"]
                        deserialized_weights = deserialize_message(data["weights"])
                        return {'sender': sender, 'round': curr_round, 'model': deserialized_weights}

                self.logger.warning(f"Timeout getting weights from peer {peer_rank}, attempt {attempt + 1}")

            except Exception as e:
                self.logger.error(f"Error getting weights from peer {peer_rank}: {e}")

            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))

        return None

    def receive(self, node_ids: List[int]) -> List[OrderedDict[str, Any]]:
        finished = False
        items = []
        # for peer_rank in node_ids: # TODO: change this back, small change for testing
        for peer_rank in self.neighbors:
            # self.wait_until_rounds_match(peer_rank)
            model_data = self.get_peer_weights(peer_rank)
            if (model_data):
                self.logger.info(f"get_peer_weights() returned {model_data.keys()}")
                items.append(model_data)

        # Process messages with timeout
        timestamp = time.time()
        timeout = 300
        
        while not (self.all_chunks_received and finished):
            if time.time() - timestamp > timeout:
                self.logger.error("Timeout waiting for all chunks to arrive")
                break
            
            try:
                message = self.message_queue.get(timeout=0.1)
                if message:
                    peer_rank = message[0]
                    data = message[1]
                    self.handle_data_channel_message(peer_rank, data)
                    if data["type"] == "weights_finished":
                        finished = True
                    # self.logger.info(f"[Main Thread] Popped from message queue type {data['type']} from peer {peer_rank}")
            except Empty:
                pass
        self.logger.info(f"ALL CHUNKS RECEIVED")

        # log the time it took to send in a nice format, and the keys of the peer_weights
        self.logger.info(f"Time to receive: {time.strftime('%M:%S', time.gmtime(time.time() - timestamp))}")
        # try:
            # for key in self.peer_weights:               
                # self.logger.info(f"Layer: {key}, dtype: {self.peer_weights[key].dtype}, size: {self.peer_weights[key].size()}")
        # except Exception as e:
            # self.logger.error(f"Error logging peer weights: {e}, keys: {self.peer_weights.keys()}, types: {type(self.peer_weights[key])}")
        

        incomplete_layers = [
            layer_name for layer_name in self.expected_chunks
            if self.received_chunks[layer_name] < self.expected_chunks[layer_name]
        ]
        if incomplete_layers:
            self.logger.error(f"Incomplete layers after timeout: {incomplete_layers}")

        self.clear_peer_weights = True
        return [{'model': self.peer_weights}]

    def send_to_peer(self, peer_rank: int, data: dict):
        # self.logger.info(f"[Main Thread] Putting message of type {data['type']} in send queue for peer {peer_rank}")
        self.send_queue.put([peer_rank, data])

    def handle_data_channel_message(self, peer_rank: int, message: str):
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message  # Assume it's already a dictionary
            msg_type = data["type"]

            if msg_type == "round_update":
                self.logger.info(f"Received round update request from peer {peer_rank}")
                current_round = self.current_round

                self.send_to_peer(peer_rank, {
                    "type": "round_update_response",
                    "round": current_round
                })

            elif msg_type == "round_update_response":
                peer_round = data.get("round")
                if peer_round is not None:
                    self.peer_rounds[peer_rank] = peer_round
                    self.logger.info(f"Received round update response from peer {peer_rank}: round {peer_round}")
                else:
                    self.logger.warning(f"Invalid 'round_update_response' from peer {peer_rank}: {message}")

            elif msg_type == "weights_request":
                self.logger.info(f"Received weights request from peer {peer_rank}")
                node_data = self.base_node.get_model_weights()
                curr_round = node_data["round"]
                # serializable_weights = serialize_message(node_data['model'])
                # total_size = sum(tensor.numel() * tensor.element_size() for tensor in node_data['model'].values())
                # first_layer = OrderedDict({list(node_data['model'].items())[0]})
                # serializable_weights = serialize_message(first_layer)
                # first_layer_size = sum(tensor.numel() * tensor.element_size() for tensor in first_layer.values())
                # self.logger.info(f"First layer size: {first_layer_size} / {total_size} bytes")
                # dummy_weights = self.model_weights.tolist()
                # dummy_weights = OrderedDict({
                #     'layer1.weight': torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
                #     'layer2.weight': torch.tensor([[[0.9, 1.0], [1.1, 1.2]], [[1.3, 1.4], [1.5, 1.6]]])
                # })
                # self.logger.info(f"dummy_weights: {dummy_weights}")
                # serializable_weights = serialize_message(dummy_weights)

                chunk_size = 15000 # supposedly 16kb is limit
                # self.logger.info(f"Sending model weights. Keys: {node_data['model'].keys()}")
                for layer_name, tensor in node_data['model'].items():
                    # self.logger.info(f"Layer: {layer_name}, dtype: {tensor.dtype}, size: {tensor.size()}")
                    for chunk, num_chunks, original_shape in self.chunk_tensor(tensor, chunk_size):
                        size_sent = chunk.numel() * chunk.element_size()
                        self.comm_cost_sent += size_sent

                        serializable_chunk = serialize_message({'layer_name': layer_name, 'chunk': chunk, 'num_chunks': num_chunks, 'original_shape': original_shape})
                        response = {
                            "type": "weights_response",
                            "weights": serializable_chunk,
                            "round": curr_round,
                            "request_id": data["request_id"]
                        }
                        self.send_to_peer(peer_rank, response)

                finished_message = {
                    "type": "weights_finished",
                    "request_id": data["request_id"],
                    "round": curr_round
                }
                self.send_to_peer(peer_rank, finished_message)

            elif msg_type == "weights_response":
                if self.clear_peer_weights:
                    self.peer_weights = {}
                    self.expected_chunks = {}  # Reset tracking
                    self.received_chunks = {}
                    self.all_chunks_received = False
                    self.clear_peer_weights = False

                chunk_data = deserialize_message(data["weights"])
                layer_name = chunk_data["layer_name"]
                chunk = chunk_data["chunk"]
                num_chunks = chunk_data["num_chunks"]
                original_shape = chunk_data["original_shape"]

                # Track expected chunks for this layer
                if layer_name not in self.expected_chunks:
                    self.expected_chunks[layer_name] = num_chunks
                    self.received_chunks[layer_name] = 0

                if layer_name not in self.peer_weights:
                    self.peer_weights[layer_name] = []
                    # self.logger.info(f"Received initial {layer_name}")

                self.peer_weights[layer_name].append(chunk)
                self.received_chunks[layer_name] += 1

                size_received = chunk.numel() * chunk.element_size()
                self.comm_cost_received += size_received

                # Check if all chunks are received for this layer
                if self.received_chunks[layer_name] == self.expected_chunks[layer_name]:
                    full_tensor = torch.cat(self.peer_weights[layer_name])
                    self.peer_weights[layer_name] = full_tensor.reshape(original_shape)
                    # self.logger.info(f"Reconstructed full tensor for {layer_name}")
                # else:
                #     self.logger.info(f"Received {len(self.peer_weights[layer_name])} / {num_chunks} chunks for {layer_name}")

                # Check if all layers are complete
                self.all_chunks_received = all(
                    self.received_chunks.get(layer, 0) == self.expected_chunks.get(layer)
                    for layer in self.expected_chunks.keys()
                )

            elif msg_type == "weights_finished":
                self.logger.info(f"Received finished message for weights from peer {peer_rank}")
                
                # Wait for a short time to allow any in-flight chunks to arrive
                # if not self.all_chunks_received:
                #     self.logger.warning("Received finished message but not all chunks have arrived yet")
                #     # Optional: Add a small delay here if needed
                #     await asyncio.sleep(2)
                
                # Log incomplete layers
                # for layer_name in self.expected_chunks:
                #     if self.received_chunks[layer_name] < self.expected_chunks[layer_name]:
                #         self.logger.warning(
                #             f"Layer {layer_name} is incomplete: "
                #             f"received {self.received_chunks[layer_name]} / {self.expected_chunks[layer_name]} chunks"
                #         )

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse message from {peer_rank}: {message.keys()}")
        except Exception as e:
            self.logger.error(f"Error handling message from {peer_rank}: {e}, message: {message.keys()}")

    def send(self, dest: str | int, data: Any):
        print("RTCCommUtils send called")

    def broadcast(self, data: Any):
        print("RTCCommUtils broadcast called")

    def all_gather(self):
        """
        This function is used to gather data from all the nodes.
        """
        print("RTCCommUtils all_gather called")

    def finalize(self):
        self.logger.info("Finalizing RTCCommUtils")
        # self.loop.call_soon_threadsafe(self.loop.stop)
        # self.loop_thread.join()
        self.stop_workers = True
        if self.websocket:
            self.websocket.close()
        for rank in list(self.connections.keys()):
            self.cleanup_connection(rank)
        self.logger.info("RTCCommUtils finalized")

    def get_comm_cost(self):
        return self.comm_cost_received, self.comm_cost_sent

    def set_is_working(self, is_working: bool):
        self.is_working = is_working
        self.logger.info("RTCCommUtils set_is_working called")

    def run_webrtc(self):
        async def main(self):
            await self.connect()
        asyncio.run(main(self))

    def initialize(self):
        self.webrtc_thread = threading.Thread(target=self.run_webrtc)
        self.webrtc_thread.start()



async def main():
    parser = argparse.ArgumentParser(description='Torus Node Client')
    parser.add_argument('--create', action='store_true', help='Create a new session')
    parser.add_argument('--join', type=str, help='Join an existing session with session ID')
    parser.add_argument('--max-clients', type=int, help='Maximum number of clients for new session')
    args = parser.parse_args()

    node = TorusNode("ws://localhost:8080")
    if args.create:
        if not args.max_clients:
            print("Error: --max-clients is required when creating a session")
            return
        await node.connect(create=True, max_clients=args.max_clients)
    elif args.join:
        await node.connect(create=False, session_id=args.join)
    else:
        print("Error: Must specify either --create or --join")
        return

if __name__ == "__main__":
    asyncio.run(main())