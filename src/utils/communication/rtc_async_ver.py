import threading
import asyncio
import json
import os
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
import logging
from collections import defaultdict
from typing import Dict, Set, List, Any, Optional
from enum import Enum
import time
import argparse
import numpy as np
import random
from queue import Queue, Empty
from utils.communication.interface import CommunicationInterface


class NodeState(Enum):
    CONNECTING = 1
    READY = 2
    DISCONNECTING = 3

logging.basicConfig(level=logging.INFO)

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
        self.RETRY_DELAY = 2
        self.pending_connections: Set[int] = set()
        self.connected_peers: Set[int] = set()
        self.expected_connections = 0
        self.connection_timeout = 30
        self.ice_gathering_timeout = 10
        self.logger = self.setup_logger()

        # Training attributes
        self.current_round = 0
        self.model_weights = np.zeros(10)  # Example model weights
        self.peer_rounds: Dict[int, int] = {}
        self.peer_weights: Dict[int, Any] = {}
        self.message_callbacks: Dict[str, Any] = {}
        self.rounds_match_events: Dict[int, threading.Event] = {}

        # adding in send and message queues
        self.send_queue = Queue()     # For outgoing messages (Main -> WebRTC)
        self.message_queue = Queue()  # For received messages (WebRTC -> Main)
        self.stop_workers = False

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
            fh = logging.FileHandler(f"logs/client_{self.rank}.log")
            fh.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            
            # Add rank to logger name
            self.logger = logging.getLogger(f"Node-{self.rank}")
            
            # Add handler
            self.logger.handlers.clear()  # Remove any existing handlers
            self.logger.addHandler(fh)
            
            # Add console handler as well
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
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
        self.data_channels[peer_rank] = channel
        self.peer_rounds[peer_rank] = 0

        @channel.on("open")
        def on_open():
            self.logger.info(f"Data channel opened with peer {peer_rank}")
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
                        else:
                            self.logger.error(f"Channel to peer {peer_rank} not open")
                    else:
                        self.logger.error(f"No channel to peer {peer_rank}")
                    print(f"[WebRTC] Sent: {message}")
                except Empty:
                    # print("[WebRTC] No message in send_queue, sleeping...")
                    await asyncio.sleep(0.05)  # Reduced sleep duration for faster checks

        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                self.logger.info("Received message: " + str(data))
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
                    f"Connected: {len(self.connected_peers)}/{self.expected_connections}")
        
        await self.websocket.send(json.dumps({
            "type": "connection_established",
            "peerRank": peer_rank,
            "sessionId": self.session_id,
        }))

        if len(self.connected_peers) == self.expected_connections:
            self.logger.info(f"Node {self.rank} broadcasting node ready")
            self.broadcast_node_ready()

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
            
            if data['type'] == 'session_created':
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
                'maxClients': self.config.get("num_users", 2)
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
                            return
                
                await process_messages()
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

    def broadcast_node_ready(self):
        if self.websocket:
            self.websocket.send(json.dumps({
                "type": "node_ready",
                "sessionId": self.session_id,
                "rank": self.rank
            }))
    async def handle_topology(self, data):
        self.rank = data["rank"]
        self.setup_file_logging()
        new_neighbors = data["neighbors"]
        self.logger.info(f"Node {self.rank} received topology. Neighbors: {new_neighbors}")

        if self.neighbors:
            removed = set(self.neighbors.values()) - set(new_neighbors.values())
            for rank in removed:
                await self.cleanup_connection(rank)

        self.neighbors = new_neighbors
        self.expected_connections = len(new_neighbors)


        if self.expected_connections == 0:
            print(f"Node {self.rank} broadcasting node ready")
            self.broadcast_node_ready()

        # Only initiate connections to higher-ranked neighbors
        connection_tasks = []
        for neighbor_rank in self.neighbors.values():
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


    # TODO: Here are all the functions that are not going to be on the listening thread
    def get_peer_weights(self, peer_rank: int, max_retries: int = 3) -> Optional[np.ndarray]:
        for attempt in range(max_retries):
            try:
                request_id = f"weights_request_{time.time()}_{random.randint(0, 1000)}"

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

                # Wait for response with timeout
                # if response_event.wait(timeout=5.0):
                #     return np.array(response_data[0]["weights"])
                message = self.message_queue.get(timeout=5.0)
                print(f"[Main Thread] Received: {message}")

                if (message):
                    self.handle_data_channel_message(message[0], message[1])

                self.logger.warning(f"Timeout getting weights from peer {peer_rank}, attempt {attempt + 1}")

            except Exception as e:
                self.logger.error(f"Error getting weights from peer {peer_rank}: {e}")

            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))

        return None

    def receive(self, node_ids: List[int]) -> List[Any]:
        items = []
        for peer_rank in node_ids:
            # self.wait_until_rounds_match(peer_rank)
            weights = self.get_peer_weights(peer_rank)
            items.append(weights)
        return items
    
    def send_to_peer(self, peer_rank: int, data: dict):
        self.logger.info(f"[Main Thread] Putting message in send queue for peer {peer_rank}: {data}")
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
                response = {
                    "type": "weights_response",
                    "weights": self.model_weights.tolist(),
                    "round": self.current_round,
                    "request_id": data["request_id"]
                }
                self.send_to_peer(peer_rank, response)

            elif msg_type == "weights_response":
                request_id = data["request_id"]
                self.logger.info(f"Received weights response from peer {peer_rank}: {data}")
                # if request_id in self.message_callbacks:
                #     callback = self.message_callbacks.pop(request_id)
                #     callback(data)

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse message from {peer_rank}: {message}")
        except Exception as e:
            self.logger.error(f"Error handling message from {peer_rank}: {e}, message: {message}")

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