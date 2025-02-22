import threading
import queue
import json
import os
import websocket
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer, RTCIceCandidate
import logging
from collections import defaultdict
from typing import Dict, Set, List, Any, Optional
from enum import Enum
import time
import numpy as np
import random
from utils.communication.interface import CommunicationInterface

class NodeState(Enum):
    CONNECTING = 1
    READY = 2
    TRAINING = 3
    EXCHANGING = 4
    DISCONNECTING = 5

class RTCCommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.config = config
        self.signaling_server = config.get("signaling_server", "ws://localhost:8765")
        self.websocket = None
        self.connections: Dict[int, RTCPeerConnection] = {}
        self.data_channels: Dict[int, RTCDataChannel] = {}
        self.rank = None  # Will be set upon session creation/joining
        self.size: int = int(config.get("num_users", 2))
        self.session_id = 1111  # Will be set upon session creation/joining
        self.neighbors = None
        self.state = NodeState.CONNECTING
        self.state_lock = threading.Lock()
        self.connection_queue = queue.Queue()
        self.connection_retries = defaultdict(int)
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2
        self.pending_connections: Set[int] = set()
        self.connected_peers: Set[int] = set()
        self.expected_connections = 0
        self.connection_timeout = 30
        self.ice_gathering_timeout = 30
        self.logger = self.setup_logger()
        
        # Training attributes
        self.current_round = 0
        self.model_weights = np.zeros(10)  # Example model weights
        self.peer_rounds: Dict[int, int] = {}
        self.peer_weights: Dict[int, Any] = {}
        self.message_callbacks: Dict[str, Any] = {}
        self.rounds_match_events: Dict[int, threading.Event] = {}

        # Queues for sending and receiving messages
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()

        # Event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.start_event_loop)
        # self.loop_thread = threading.Thread(target=self.loop.run_forever)
        self.loop_thread.start()

        self.stop_workers = False

    def setup_logger(self) -> logging.Logger:
        os.makedirs("logs", exist_ok=True)
        logger = logging.getLogger(f"Node")
        # logger.setLevel(logging.INFO)
        logging.getLogger('aioice').setLevel(logging.WARNING)
        logging.getLogger('aiortc').setLevel(logging.WARNING)
        logger.handlers.clear()
        return logger

    def setup_file_logging(self):
        if self.rank is not None:
            fh = logging.FileHandler(f"logs/client_{self.rank}.log")
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger = logging.getLogger(f"Node-{self.rank}")
            self.logger.handlers.clear()
            self.logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def change_state(self, new_state: NodeState):
        with self.state_lock:
            self.state = new_state
            self.logger.info(f"Node {self.rank} state changed to {new_state}")
    
    def get_state(self) -> NodeState:
        with self.state_lock:
            return self.state

    def register_self(self, obj: "BaseNode"):
        self.base_node = obj

    def run_coro_sync(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def setup_data_channel(self, channel: RTCDataChannel, peer_rank: int):
        self.logger.info(f"Setting up data channel with peer {peer_rank}")
        self.data_channels[peer_rank] = channel
        self.peer_rounds[peer_rank] = 0

        @channel.on("open")
        def on_open():
            self.logger.info(f"Data channel opened with peer {peer_rank}, channel: {channel}")
            self.on_peer_connected(peer_rank)

            # Send a test message to confirm communication
            test_message = json.dumps({"type": "test_message", "content": f"Hello from Node {self.rank}"})
            channel.send(test_message)
            # self.send_queue.put(test_message)
            self.logger.info(f"Test message sent to peer {peer_rank}: {test_message}")

        @channel.on("message")
        def on_message(message):
            self.logger.info(f"Received message from peer {peer_rank}")
            # self.receive_queue.put(message)
            self.handle_data_channel_message(peer_rank, message)

        @channel.on("close")
        def on_close():
            self.logger.info(f"Data channel with peer {peer_rank} closed")
            if peer_rank in self.data_channels:
                del self.data_channels[peer_rank]
            if peer_rank in self.connected_peers:
                self.connected_peers.remove(peer_rank)

        self.logger.info(f"Data channel setup complete for peer {peer_rank}")

    def on_peer_connected(self, peer_rank: int):
        self.connected_peers.add(peer_rank)
        self.pending_connections.discard(peer_rank)
        self.logger.info(f"Node {self.rank} connected to peer {peer_rank}. "
                         f"Connected: {len(self.connected_peers)}/{self.expected_connections}")

        if self.websocket:
            self.websocket.send(json.dumps({
                "type": "connection_established",
                "peerRank": peer_rank,
                "sessionId": self.session_id,
            }))

        if len(self.connected_peers) == self.expected_connections:
            print(f"Node {self.rank} broadcasting node ready")
            self.broadcast_node_ready()

    def connection_worker(self):
        while not self.stop_workers:
            try:
                target_rank = self.connection_queue.get(timeout=1)
                self.logger.info(f"Node {self.rank} worker processing connection to {target_rank}")
                retry_count = self.connection_retries[target_rank]
                
                try:
                    self.initiate_connection(target_rank)
                    self.logger.info(f"Node {self.rank} successfully initiated connection to {target_rank}")
                    self.connection_retries[target_rank] = 0
                except Exception as e:
                    self.logger.error(f"Connection attempt to {target_rank} failed: {e}")
                    self.handle_connection_failure(target_rank)
                finally:
                    self.connection_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")

    def create_peer_connection(self) -> RTCPeerConnection:
        config = RTCConfiguration([
            RTCIceServer(urls=[
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302"
            ])
        ])
        pc = RTCPeerConnection(configuration=config)
        return pc

    def initiate_connection(self, target_rank: int):
        try:
            self.run_coro_sync(self.initiate_connection_async(target_rank))
        except Exception as e:
            self.logger.error(f"Failed to initiate connection to {target_rank}: {e}")
            self.cleanup_connection(target_rank)
            raise

    async def initiate_connection_async(self, target_rank: int):
        try:
            pc = self.create_peer_connection()
            self.connections[target_rank] = pc

            def on_ice_connection_state_change():
                state = pc.iceConnectionState
                self.logger.info(f"ICE connection state to {target_rank}: {state}")
                if state == "failed":
                    self.logger.error(f"ICE connection to {target_rank} failed")
                    self.handle_connection_failure(target_rank)
                elif state == "connected":
                    self.logger.info(f"ICE connection to {target_rank} established")

            def on_ice_gathering_state_change():
                self.logger.info(f"ICE gathering state for {target_rank}: {pc.iceGatheringState}")

            pc.on("iceconnectionstatechange", on_ice_connection_state_change)
            pc.on("icegatheringstatechange", on_ice_gathering_state_change)

            channel = pc.createDataChannel(f"chat-{self.rank}-{target_rank}")
            self.setup_data_channel(channel, target_rank)

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            gathering_complete = asyncio.Event()

            def on_gathering_complete():
                if pc.iceGatheringState == "complete":
                    self.loop.call_soon_threadsafe(gathering_complete.set)

            pc.on("icegatheringstatechange", on_gathering_complete)

            try:
                await asyncio.wait_for(gathering_complete.wait(), timeout=self.ice_gathering_timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"ICE gathering timed out for {target_rank}")

            self.send_signaling(target_rank, {
                "type": "offer",
                "sdp": pc.localDescription.sdp
            })

        except Exception as e:
            self.logger.error(f"Failed to initiate connection to {target_rank}: {e}")
            self.cleanup_connection(target_rank)
            raise

    def cleanup_connection(self, rank: int):
        try:
            if rank in self.connections:
                pc = self.connections[rank]

                if rank in self.data_channels:
                    channel = self.data_channels[rank]
                    if channel and channel.readyState != "closed":
                        # self.run_coro_sync(channel.close())
                        channel.close()
                    del self.data_channels[rank]

                for transceiver in pc.getTransceivers():
                    transceiver.stop()

                self.run_coro_sync(pc.close())
                del self.connections[rank]

            if rank in self.pending_connections:
                self.pending_connections.remove(rank)
            if rank in self.connected_peers:
                self.connected_peers.remove(rank)

            self.logger.info(f"Cleaned up connection to peer {rank}")

        except Exception as e:
            self.logger.error(f"Error during connection cleanup for peer {rank}: {e}")

    def handle_connection_failure(self, target_rank):
        retry_count = self.connection_retries[target_rank]
        if retry_count < self.MAX_RETRIES:
            self.connection_retries[target_rank] = retry_count + 1
            time.sleep(self.RETRY_DELAY * (retry_count + 1))
            if target_rank not in self.connected_peers:
                self.cleanup_connection(target_rank)
                self.connection_queue.put(target_rank)
                self.logger.info(f"Retrying connection to {target_rank}, attempt {retry_count + 1}")
        else:
            self.logger.error(f"Max retries reached for {target_rank}")
            self.cleanup_connection(target_rank)

    def handle_signaling_message(self, message):
        try:
            sender_rank = message["senderRank"]
            data = message["data"]

            if sender_rank not in self.connections:
                pc = self.create_peer_connection()
                self.connections[sender_rank] = pc

                @pc.on("datachannel")
                def on_datachannel(channel):
                    self.logger.info(f"Data channel created by peer {sender_rank}")
                    self.setup_data_channel(channel, sender_rank)

                @pc.on("icecandidate")
                async def on_icecandidate(candidate):
                    if candidate:
                        self.send_signaling(sender_rank, {
                            "type": "candidate",
                            "candidate": candidate.sdp
                        })

            pc = self.connections[sender_rank]

            self.run_coro_sync(self.handle_signaling_message_async(pc, sender_rank, data))

        except Exception as e:
            self.logger.error(f"Error handling signaling message from {sender_rank}: {e}")

    async def handle_signaling_message_async(self, pc, sender_rank, data):
        try:
            if data["type"] == "offer":
                self.logger.info(f"Received offer from {sender_rank}")
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=data["sdp"],
                    type="offer"
                ))

                # Attach ICE callback:
                @pc.on("icecandidate")
                async def on_icecandidate(candidate):
                    if candidate:
                        self.send_signaling(sender_rank, {
                            "type": "candidate",
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        })

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                self.send_signaling(sender_rank, {
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                })

            elif data["type"] == "answer":
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=data["sdp"],
                    type="answer"
                ))

                # ALSO attach ICE callback if not already:
                @pc.on("icecandidate")
                async def on_icecandidate(candidate):
                    if candidate:
                        self.send_signaling(sender_rank, {
                            "type": "candidate",
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        })


            elif data["type"] == "candidate" and pc.remoteDescription:
                candidate = RTCIceCandidate(
                    sdpMid=data.get("sdpMid"),
                    sdpMLineIndex=data.get("sdpMLineIndex"),
                    candidate=data["candidate"]
                )
                await pc.addIceCandidate(candidate)
        except Exception as e:
            self.logger.error(f"Error in signaling message async handler for {sender_rank}: {e}")

    def create_session(self, max_clients: int) -> bool:
        try:
            self.websocket.send(json.dumps({
                'type': 'create_session',
                'maxClients': max_clients,
                'clientType': 'python'
            }))
            response = self.websocket.recv()
            data = json.loads(response)

            if data['type'] == 'session_created':
                self.session_id = data['sessionId']
                self.rank = data['rank']
                self.logger.info(f"Created session {self.session_id}")
                print(f"Share this session code with other clients: {self.session_id}")
                return True
            else:
                self.logger.error(f"Failed to create session: {data.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return False

    def join_session(self, session_id: str) -> bool:
        try:
            self.websocket.send(json.dumps({
                'type': 'join_session',
                'sessionId': session_id,
                'clientType': 'python',
                'maxClients': self.config.get("num_users", 2)
            }))
            response = self.websocket.recv()
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

    def initialize(self, create: bool = False, session_id: str = None, max_clients: int = None) -> bool:
        """
        Initialize the communication utilities by connecting to the signaling server,
        establishing peer connections, and waiting for the network to be ready.
        """
        print("Initializing rtc node")
        # Set the variables to class variables
        max_clients = self.size
        session_id = self.session_id
        print("before initializing, rank is", self.rank)

        try:
            # Initialize the websocket connection
            self.websocket = websocket.create_connection(self.signaling_server)
            self.change_state(NodeState.CONNECTING)

            # Validate input parameters
            if create and not self.size:
                raise ValueError("max_clients is required when creating a session")
            if not create and not session_id:
                raise ValueError("session_id is required when joining a session")

            # Create or join session
            session_success = self.create_session(self.size) if create else self.join_session(session_id)
            if not session_success:
                self.logger.error("Failed to create/join session")
                return False

            # Start connection workers in separate threads
            worker_count = 3  # Number of concurrent connection workers
            worker_threads = []
            for _ in range(worker_count):
                thread = threading.Thread(target=self.connection_worker)
                thread.daemon = True
                thread.start()
                worker_threads.append(thread)

            # Wait for network setup and connections
            network_ready = self.wait_for_network_ready()
            if not network_ready:
                raise RuntimeError("Network initialization failed")

            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            # Cleanup any partial initialization
            if self.websocket:
                self.websocket.close()
            for rank in list(self.connections.keys()):
                self.cleanup_connection(rank)
            return False

    def wait_for_network_ready(self) -> bool:
        print("Waiting for network to be ready")
        try:
            start_time = time.time()
            timeout = 600.0  # 600 second timeout

            while time.time() - start_time < timeout:
                try:
                    message = self.websocket.recv()
                    data = json.loads(message)
                    self.logger.info(f"Received message: {data['type']}")

                    if data['type'] == 'topology':
                        print("Received topology")
                        self.handle_topology(data)
                    elif data['type'] == 'signal':
                        print("Received signaling message")
                        self.handle_signaling_message(data)
                    elif data['type'] == 'network_ready':
                        self.change_state(NodeState.READY)
                        self.logger.info("Network initialization complete!")
                        return True
                    elif data['type'] == 'error':
                        self.logger.error(f"Received error: {data.get('message')}")
                        return False
                except websocket.WebSocketTimeoutException:
                    continue
            self.logger.error("Network initialization timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error during network initialization: {e}")
            return False

    def handle_topology(self, data):
        self.rank = data["rank"]
        self.setup_file_logging()
        new_neighbors = data["neighbors"]
        self.logger.info(f"Node {self.rank} received topology. Neighbors: {new_neighbors}")

        if self.neighbors:
            removed = set(self.neighbors.values()) - set(new_neighbors.values())
            for rank in removed:
                self.cleanup_connection(rank)

        self.neighbors = new_neighbors
        self.expected_connections = len(new_neighbors)
        print("Expected connections:", self.expected_connections)

        if self.expected_connections == 0:
            print(f"Node {self.rank} broadcasting node ready")
            self.broadcast_node_ready()

        # Only initiate connections to higher-ranked neighbors
        for neighbor_rank in self.neighbors.values():
            if neighbor_rank is None:
                continue
            if (neighbor_rank not in self.connections and
                neighbor_rank not in self.pending_connections):
                self.logger.info(f"Node {self.rank} queueing connection to {neighbor_rank}")
                self.connection_queue.put(neighbor_rank)
                self.pending_connections.add(neighbor_rank)

    def send_signaling(self, target_rank: int, data: dict):
        if self.websocket:
            self.websocket.send(json.dumps({
                "type": "signal",
                "targetRank": target_rank,
                "senderRank": self.rank,
                "data": data,
                "sessionId": self.session_id,
            }))

    def handle_data_channel_message(self, peer_rank: int, message: str):
        self.logger.info(f"Raw data channel message from peer {peer_rank}, message: {message}")
        try:
            data = json.loads(message)
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
                response = {
                    "type": "weights_response",
                    "weights": self.model_weights.tolist(),
                    "round": self.current_round,
                    "request_id": data["request_id"]
                }
                self.send_to_peer(peer_rank, response)

            elif msg_type == "weights_response":
                self.logger.info(f"Received weights response from peer {peer_rank}")
                request_id = data["request_id"]
                if request_id in self.message_callbacks:
                    callback = self.message_callbacks.pop(request_id)
                    callback(data)

            # Handle test messages
            elif msg_type == "test_message":
                self.logger.info(f"Test message received from peer {peer_rank}: {data['content']}")
                return

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse message from {peer_rank}: {message}")
        except Exception as e:
            self.logger.error(f"Error handling message from {peer_rank}: {e}, message: {message}")

    def broadcast_node_ready(self):
        if self.websocket:
            self.websocket.send(json.dumps({
                "type": "node_ready",
                "sessionId": self.session_id,
                "rank": self.rank
            }))

    def wait_until_rounds_match(self, peer_rank: int, timeout: float = 5.0, check_interval: float = 0.5):
        """
        Wait until the peer's round matches or exceeds the local round.
        Poll the peer's round periodically within the timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.peer_rounds.get(peer_rank, 0) >= self.current_round:
                self.logger.info(f"Rounds match with peer {peer_rank}")
                return  # Synchronization complete

            self.send_to_peer(peer_rank, {
                "type": "round_update",
                "round": self.current_round
            })

            time.sleep(check_interval)

        self.logger.warning(f"Timeout waiting for round match with peer {peer_rank}")

    def receive(self, node_ids: List[int]) -> List[Any]:
        items = []
        # for peer_rank in node_ids: # TODO: change this back, small change for testing
        peer_rank = self.neighbors['neighbor1']
        if peer_rank is not None:
            print("Peer rank sending from RTC server is ", peer_rank)
            self.wait_until_rounds_match(peer_rank)
            weights = self.get_peer_weights(peer_rank)
            items.append(weights)
        return items

    def get_peer_weights(self, peer_rank: int, max_retries: int = 3) -> Optional[np.ndarray]:
        # TODO: remove later, only for testing
        if peer_rank != 2:
            # non blocking wait for 30 seconds
            self.logger.info(f"Waiting for 30 seconds from {self.rank}")
            start = time.time()
            while time.time() - start < 30:
                pass
            self.logger.info(f"Done waiting for 30 seconds from {self.rank}")
            return self.model_weights.tolist()
        
        for attempt in range(max_retries):
            try:
                request_id = f"weights_request_{self.rank}_{time.time()}_{random.randint(0, 1000)}"

                # Create an event for synchronization
                response_event = threading.Event()
                response_data = []

                def callback(data):
                    response_data.append(data)
                    response_event.set()

                self.message_callbacks[request_id] = callback

                self.send_to_peer(peer_rank, {
                    "type": "weights_request",
                    "request_id": request_id,
                    "content": "Hi"
                })

                # Wait for response with timeout
                if response_event.wait(timeout=30.0):
                    return np.array(response_data[0]["weights"])

                self.logger.warning(f"Timeout getting weights from peer {peer_rank}, attempt {attempt + 1}")

            except Exception as e:
                self.logger.error(f"Error getting weights from peer {peer_rank}: {e}")

            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))

        return None

    def send_to_peer(self, peer_rank: int, data: dict):
        if peer_rank in self.data_channels:
            channel = self.data_channels[peer_rank]
            self.logger.info(f"Channel state to peer {peer_rank}: state {channel.readyState}, sending {data}")
            try:
                if channel.readyState == "open":
                    message = json.dumps(data)
                    # self.run_coro_sync(channel.send(message))
                    channel.send(message)
                    self.logger.info(f"Sent message to peer {peer_rank}: {data}, data channel: {channel}")
                else:
                    self.logger.error(f"Channel to peer {peer_rank} not open")
            except Exception as e:
                self.logger.error(f"Error sending to peer {peer_rank}: {e}")
        else:
            self.logger.error(f"No channel to peer {peer_rank}")

    def run_training(self, num_rounds: int = 5):
        self.logger.info("Starting training")
        if self.state != NodeState.READY:
            self.logger.error("Node not ready for training")
            return

        for round in range(num_rounds):
            self.change_state(NodeState.TRAINING)
            self.train_local_model()

            self.current_round += 1

            self.change_state(NodeState.EXCHANGING)
            neighbor_ranks = list(self.neighbors.values())
            weights_list = self.receive(neighbor_ranks)

            # Average weights
            valid_weights = [w for w in weights_list if w is not None]
            if valid_weights:
                all_weights = [self.model_weights] + valid_weights
                self.model_weights = np.mean(all_weights, axis=0)

            self.logger.info(f"Completed round {round + 1}/{num_rounds}")

        self.logger.info("Training completed")

    def train_local_model(self):
        # Placeholder for actual training logic
        self.model_weights += np.random.randn(*self.model_weights.shape) * 0.01
        self.logger.info(f"Trained local model for round {self.current_round}")

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
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
        self.stop_workers = True
        if self.websocket:
            self.websocket.close()
        for rank in list(self.connections.keys()):
            self.cleanup_connection(rank)
        self.logger.info("RTCCommUtils finalized")

    def set_is_working(self, is_working: bool):
        self.is_working = is_working
        print("RTCCommUtils set_is_working called")

    def start_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def send_message(self):
        while True:
            message = await asyncio.to_thread(self.send_queue.get)
            for peer_rank, channel in self.data_channels.items():
                if channel.readyState == "open":
                    await channel.send(message)

    def start(self):
        asyncio.run_coroutine_threadsafe(self.send_message(), self.loop)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
        self.stop_workers = True
        if self.websocket:
            self.websocket.close()
        for rank in list(self.connections.keys()):
            self.cleanup_connection(rank)
        self.logger.info("RTCCommUtils finalized")
