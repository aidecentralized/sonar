import asyncio
import json
import math
import secrets
import logging
from dataclasses import dataclass
from typing import Dict, Set
from collections import defaultdict

import websockets

logging.basicConfig(level=logging.INFO)

@dataclass
class ClientInfo:
    rank: int
    client_type: str
    session_id: str
    ready: bool = False
    connected_peers: Set[int] = None
    
    def __post_init__(self):
        self.connected_peers = set()

@dataclass
class SessionInfo:
    session_id: str
    max_clients: int
    # Remove deprecated WebSocketServerProtocol from type hints; just use "object" or no annotation
    clients: Dict[object, ClientInfo] = None
    num_ready: int = 0
    
    def __post_init__(self):
        self.clients = {}

class SignalingServer:
    def __init__(self):
        self.sessions: Dict[str, SessionInfo] = {}
        # Locks to prevent concurrency issues when multiple connections are established simultaneously
        self.connection_locks = defaultdict(asyncio.Lock)

    def calculate_grid_size(self, session: SessionInfo) -> int:
        """
        (Optional) Example helper: if you want to display or use a grid-based layout.
        """
        return math.floor(math.sqrt(len(session.clients)))

    def get_neighbor_ranks(self, session: SessionInfo, rank: int) -> dict:
        """
        RING-BASED TOPOLOGY:
        For N total clients, each node has two neighbors:
            - (rank - 1) % N
            - (rank + 1) % N
        If N=1, no neighbors; if N=2, each sees the other as both 'prev' and 'next'.
        """
        total = len(session.clients)
        if total <= 1:
            return {}

        prev_rank = (rank - 1) % total
        next_rank = (rank + 1) % total
        return {
            'prev': prev_rank,
            'next': next_rank
        }

    async def handle_client(self, websocket):
        """
        Main handler for each new client connection.
        1) Reads initial "create_session" or "join_session".
        2) Assigns rank, sets up session info.
        3) Listens for subsequent messages in a loop.
        """
        try:
            # Wait for the first message to determine session creation/join
            initial_message = await websocket.recv()
            data = json.loads(initial_message)

            # 1. CREATE SESSION
            if data['type'] == 'create_session':
                session_id = data.get('sessionId', secrets.token_hex(3))
                max_clients = int(data['maxClients'])

                # Create new session
                self.sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    max_clients=max_clients
                )

                # Add first client
                self.sessions[session_id].clients[websocket] = ClientInfo(
                    rank=0,
                    client_type=data.get('clientType', 'javascript'),
                    session_id=session_id
                )

                # Respond with session info
                await websocket.send(json.dumps({
                    'type': 'session_created',
                    'sessionId': session_id,
                    'rank': 0
                }))
                logging.info(f"Created session {session_id} for {max_clients} clients")

            # 2. JOIN SESSION
            elif data['type'] == 'join_session':
                session_id = data['sessionId']
                
                # If session doesn't exist, create it on the fly
                if session_id not in self.sessions:
                    max_clients = int(data['maxClients'])
                    self.sessions[session_id] = SessionInfo(
                        session_id=session_id,
                        max_clients=max_clients
                    )
                    rank = 0
                    self.sessions[session_id].clients[websocket] = ClientInfo(
                        rank=0,
                        client_type=data.get('clientType', 'javascript'),
                        session_id=session_id
                    )
                    session = self.sessions[session_id]
                else:
                    session = self.sessions[session_id]

                    # Check if session is at capacity
                    if len(session.clients) >= session.max_clients:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Session is full'
                        }))
                        return
                    
                    # Otherwise assign next rank to new client
                    rank = len(session.clients)
                    session.clients[websocket] = ClientInfo(
                        rank=rank,
                        client_type=data.get('clientType', 'javascript'),
                        session_id=session_id
                    )

                # Acknowledge join
                await websocket.send(json.dumps({
                    'type': 'session_joined',
                    'sessionId': session_id,
                    'rank': rank
                }))
                logging.info(f"Client joined session {session_id} with rank {rank}")

            else:
                # Invalid initial message type
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid initial message type.'
                }))
                return

            # After the initial session handling, listen for further messages
            async for msg in websocket:
                data = json.loads(msg)
                session_id = data.get('sessionId')
                if not session_id or session_id not in self.sessions:
                    # Possibly unknown session
                    continue

                session = self.sessions[session_id]
                client_info = session.clients.get(websocket)

                if data['type'] == 'signal':
                    # 3. SIGNAL MESSAGES
                    sender_rank = client_info.rank
                    target_rank = data['targetRank']

                    # Use a lock to avoid concurrency for the same pair
                    lock_key = f"{min(sender_rank, target_rank)}-{max(sender_rank, target_rank)}"
                    async with self.connection_locks[lock_key]:
                        target_ws = next(
                            (ws for ws, info in session.clients.items() if info.rank == target_rank),
                            None
                        )
                        if target_ws:
                            await target_ws.send(json.dumps({
                                'type': 'signal',
                                'senderRank': sender_rank,
                                'senderType': client_info.client_type,
                                'data': data['data']
                            }))

                elif data['type'] == 'node_ready':
                    # 4. READY MESSAGES
                    session.num_ready += 1
                    logging.info(f"Session {session_id} now has {session.num_ready} ready clients.")
                    await self.check_session_ready(session)

        except websockets.exceptions.ConnectionClosed:
            # Handle client disconnection
            for sess in self.sessions.values():
                if websocket in sess.clients:
                    await self.handle_disconnect(websocket, sess)
                    break

    async def handle_disconnect(self, websocket, session: SessionInfo):
        """
        Remove disconnected client from session,
        re-broadcast updated topology if needed.
        """
        if websocket in session.clients:
            disconnected_rank = session.clients[websocket].rank
            del session.clients[websocket]

            # Remove from connected peers
            for _, info in session.clients.items():
                if disconnected_rank in info.connected_peers:
                    info.connected_peers.remove(disconnected_rank)
                    info.ready = False

            # If session is empty, remove entirely
            if not session.clients:
                del self.sessions[session.session_id]
                logging.info(f"Session {session.session_id} removed.")
            else:
                # Re-broadcast new ring topology to remaining clients
                await self.broadcast_topology(session)

    async def broadcast_topology(self, session: SessionInfo):
        """
        Send each client the ring neighbors after any change.
        """
        grid_size = self.calculate_grid_size(session)  # if you need it
        for ws, info in session.clients.items():
            try:
                neighbors = self.get_neighbor_ranks(session, info.rank)
                await ws.send(json.dumps({
                    'type': 'topology',
                    'rank': info.rank,
                    'neighbors': neighbors,
                    'gridSize': grid_size,
                    'totalClients': len(session.clients)
                }))
            except websockets.exceptions.ConnectionClosed:
                logging.error(f"Failed to send topology to client {info.rank}")

    async def broadcast_session_ready(self, session: SessionInfo):
        """
        Send a session_ready message to all clients.
        """
        message = json.dumps({
            'type': 'session_ready',
            'message': f"All {len(session.clients)} clients connected"
        })
        await asyncio.gather(*[
            ws.send(message) for ws in session.clients
        ])

    async def check_session_ready(self, session: SessionInfo):
        """
        If all clients in the session are marked ready,
        broadcast 'network_ready'.
        """
        all_ready = session.num_ready == len(session.clients)
        if all_ready:
            logging.info(f"All nodes in session {session.session_id} reported ready!")
            await self.broadcast_network_ready(session)

    async def broadcast_network_ready(self, session: SessionInfo):
        """
        Notify all clients that the network is ready.
        """
        message = json.dumps({'type': 'network_ready'})
        await asyncio.gather(*[
            ws.send(message) for ws in session.clients
        ])

async def main():
    server = SignalingServer()
    async with websockets.serve(server.handle_client, "localhost", 8765):
        logging.info("Signaling server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())