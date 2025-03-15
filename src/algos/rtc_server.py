import asyncio
import json
import logging
import math
import secrets
from dataclasses import dataclass
from typing import Dict, Set
from collections import defaultdict

import websockets

from algos.topologies.collections import select_topology

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
    clients: Dict[websockets.WebSocketServerProtocol, ClientInfo] = None
    num_ready: int = 0
    config: dict = None  

    def __post_init__(self):
        self.clients = {}
        if self.config is None:
            self.config = {}

class SignalingServer:
    def __init__(self):
        self.sessions: Dict[str, SessionInfo] = {}
        self.connection_locks = defaultdict(asyncio.Lock)

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        try:
            message = await websocket.recv()
            data = json.loads(message)
            # Handle session creation
            if data['type'] == 'create_session':
                session_id = data.get('sessionId', secrets.token_hex(3))
                max_clients = int(data['maxClients'])
                default_config = {"topology_name": "ring", "seed": 42}
                client_config = data.get("config", {})
                default_config.update(client_config)
                config = default_config
                config["num_users"] = max_clients
                self.sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    max_clients=max_clients,
                    config=config
                )
                # First client gets rank 0
                rank = 0
                self.sessions[session_id].clients[websocket] = ClientInfo(
                    rank=rank,
                    client_type=data.get('clientType', 'javascript'),
                    session_id=session_id
                )
                logging.info(f"[RTC] Created session {session_id} for {max_clients} clients")
                await websocket.send(json.dumps({
                    'type': 'session_created',
                    'sessionId': session_id,
                    'rank': rank
                }))
            # Handle joining a session
            elif data['type'] == 'join_session':
                session_id = data.get('sessionId')
                if session_id not in self.sessions:
                    max_clients = int(data['maxClients'])
                    default_config = {"topology_name": "ring", "seed": 42}
                    client_config = data.get("config", {})
                    default_config.update(client_config)
                    config = default_config
                    config["num_users"] = max_clients
                    self.sessions[session_id] = SessionInfo(
                        session_id=session_id,
                        max_clients=max_clients,
                        config=config
                    )
                    rank = 0
                    self.sessions[session_id].clients[websocket] = ClientInfo(
                        rank=rank,
                        client_type=data.get('clientType', 'javascript'),
                        session_id=session_id
                    )
                    session = self.sessions[session_id]
                else:
                    session = self.sessions[session_id]
                    if len(session.clients) >= session.max_clients:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Session is full'
                        }))
                        return
                    rank = len(session.clients)
                    session.clients[websocket] = ClientInfo(
                        rank=rank,
                        client_type=data.get('clientType', 'javascript'),
                        session_id=session_id
                    )
                logging.info(f"[RTC] Client joined session {session_id} with rank {rank}")
                await websocket.send(json.dumps({
                    'type': 'session_joined',
                    'sessionId': session_id,
                    'rank': rank
                }))
                # When session is full, broadcast topology
                if len(session.clients) == session.max_clients:
                    logging.info(f"[RTC] Session {session_id} is full, broadcasting topology")
                    await self.broadcast_session_ready(session)
                    await self.broadcast_topology(session)
            async for message in websocket:
                data = json.loads(message)
                session = self.sessions.get(data['sessionId'])
                if not session:
                    continue
                if data['type'] == 'signal':
                    sender_rank = session.clients[websocket].rank
                    target_rank = data['targetRank']
                    async with self.connection_locks[f"{min(sender_rank, target_rank)}-{max(sender_rank, target_rank)}"]:
                        target_ws = next(
                            (ws for ws, info in session.clients.items() if info.rank == target_rank),
                            None
                        )
                        if target_ws:
                            await target_ws.send(json.dumps({
                                'type': 'signal',
                                'senderRank': sender_rank,
                                'senderType': session.clients[websocket].client_type,
                                'data': data['data']
                            }))
                elif data['type'] == 'node_ready':
                    session.num_ready += 1
                    print(f"All ready: {session.num_ready} / {len(session.clients)}")
                    await self.check_session_ready(session)
        except websockets.exceptions.ConnectionClosed:
            for session in self.sessions.values():
                if websocket in session.clients:
                    await self.handle_disconnect(websocket, session)
                    break

    async def handle_disconnect(self, websocket: websockets.WebSocketServerProtocol, session: SessionInfo):
        if websocket in session.clients:
            disconnected_rank = session.clients[websocket].rank
            del session.clients[websocket]
            for _, info in session.clients.items():
                if disconnected_rank in info.connected_peers:
                    info.connected_peers.remove(disconnected_rank)
                    info.ready = False
            if not session.clients:
                del self.sessions[session.session_id]
                logging.info(f"[RTC] Session {session.session_id} removed")
            else:
                await self.broadcast_topology(session)

    async def broadcast_topology(self, session: SessionInfo):
        config = session.config
        for ws, info in session.clients.items():
            rank = info["rank"]
            topology = select_topology(config, rank)
            topology.initialize()
            neighbors = topology.get_all_neighbours()
            try:
                await ws.send(json.dumps({
                    'type': 'topology',
                    'rank': rank,
                    'neighbors': neighbors,
                    'totalClients': len(session.clients)
                }))
            except websockets.exceptions.ConnectionClosed:
                logging.error(f"[RTC] Failed to send topology to client {info.rank}")

    async def broadcast_session_ready(self, session: SessionInfo):
        message = json.dumps({
            'type': 'session_ready',
            'message': f'All {len(session.clients)} clients connected'
        })
        await asyncio.gather(*[ws.send(message) for ws in session.clients])
            
    async def check_session_ready(self, session: SessionInfo):
        all_ready = session.num_ready == len(session.clients)
        print(f"All ready: {all_ready}: {session.num_ready} / {len(session.clients)}")
        if all_ready:
            logging.info(f"[RTC] All nodes in session {session.session_id} are connected!")
            await self.broadcast_network_ready(session)
            
    async def broadcast_network_ready(self, session: SessionInfo):
        message = json.dumps({'type': 'network_ready'})
        await asyncio.gather(*[ws.send(message) for ws in session.clients])

async def main():
    server = SignalingServer()
    async with websockets.serve(server.handle_client, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
