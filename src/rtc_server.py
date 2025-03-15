import asyncio
import json
import websockets
import math
from dataclasses import dataclass
from typing import Dict, Set, Optional, Any
import logging
from collections import defaultdict
import secrets
from algos.topologies.base import BaseTopology
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
    config: Dict[str, Any]
    clients: Dict[websockets.WebSocketServerProtocol, ClientInfo] = None
    num_ready: int = 0
    
    def __post_init__(self):
        self.clients = {}

class SignalingServer:
    def __init__(self):
        self.sessions: Dict[str, SessionInfo] = {}
        self.connection_locks = defaultdict(asyncio.Lock)
        
    def calculate_grid_size(self, session: SessionInfo) -> int:
        return math.floor(math.sqrt(len(session.clients)))
        
    def get_neighbor_ranks(self, session: SessionInfo, rank: int) -> dict:
        # grid_size = self.calculate_grid_size(session)
        # if grid_size < 2:
        #     return {}
            
        # row = rank // grid_size
        # col = rank % grid_size
        
        # return {
        #     'north': ((row - 1 + grid_size) % grid_size) * grid_size + col,
        #     'south': ((row + 1) % grid_size) * grid_size + col,
        #     'west': row * grid_size + ((col - 1 + grid_size) % grid_size),
        #     'east': row * grid_size + ((col + 1) % grid_size)
        # }
        neighbor = rank + 1 if rank + 1 < len(session.clients) else 1
        if rank == 0: return {}
        return {'neighbor1': neighbor}
        # return {'neighbor': (rank + 1) % len(session.clients)}

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        try:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data['type'] == 'create_session':
                # Generate a unique 6-character session ID if there was no given session id
                session_id = data.get('sessionId', secrets.token_hex(3)) # 6 characters
                # session_id = secrets.token_hex(3)  # 6 characters
                max_clients = int(data['maxClients'])
                
                # Create new session
                self.sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    max_clients=max_clients
                )
                
                # Add first client to session
                self.sessions[session_id].clients[websocket] = ClientInfo(
                    rank=0,
                    client_type=data.get('clientType', 'javascript'),
                    session_id=session_id
                )
                
                # Send session ID back to creator
                await websocket.send(json.dumps({
                    'type': 'session_created',
                    'sessionId': session_id,
                    'rank': 0
                }))
                
                logging.info(f"Created session {session_id} for {max_clients} clients")
                
            elif data['type'] == 'join_session':
                session_id = data['sessionId']
                if session_id not in self.sessions:
                    # await websocket.send(json.dumps({
                    #     'type': 'error',
                    #     'message': 'Invalid session ID'
                    # }))
                    # return

                    # Create new session if it doesn't exist
                    max_clients = int(data['maxClients'])
                    rank = 0
                    # get topology to send to server
                    config = data["config"]
                    # topology_config = {
                    #     "topology": {"name": config["algos"]["node_0"]["topology"]["name"]},
                    #     "num_users": config["num_users"],
                    #     "seed": config["seed"]
                    # }
                    # topology = select_topology(topology_config, rank+1)
                    self.sessions[session_id] = SessionInfo(
                        session_id=session_id,
                        max_clients=max_clients,
                        config = config
                    )

                    # Add first client to session
                    self.sessions[session_id].clients[websocket] = ClientInfo(
                        rank=0,
                        client_type=data.get('clientType', 'javascript'),
                        session_id=session_id,
                    )
                    session = self.sessions[session_id]


                else:
                    session = self.sessions[session_id]
                    if len(session.clients) > session.max_clients:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Session is full'
                        }))
                        return
                    
                    # Add client to session
                    rank = len(session.clients)
                    session.clients[websocket] = ClientInfo(
                        rank=rank,
                        client_type=data.get('clientType', 'javascript'),
                        session_id=session_id
                    )
                
                await websocket.send(json.dumps({
                    'type': 'session_joined',
                    'sessionId': session_id,
                    'rank': rank
                }))
                
                logging.info(f"Client joined session {session_id} with rank {rank}")
                
                # If session is full, broadcast topology to all clients
                if len(session.clients) == session.max_clients + 1:
                    logging.info(f"Session {session_id} is full, broadcasting topology")
                    await self.broadcast_session_ready(session)
                    await self.broadcast_topology(session)
            
            async for message in websocket:
                data = json.loads(message)
                session = self.sessions[data['sessionId']]
                
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
                elif data['type'] == 'connection_established':
                    # peer_rank = data['peerRank']
                    # session.clients[websocket].connected_peers.add(peer_rank)
                    
                    # Check if all connections in the session are established
                    # await self.check_session_ready(session)
                    pass
                elif data['type'] == "node_ready":
                    session.num_ready += 1
                    print("Updating num_ready to ", session.num_ready)
                    await self.check_session_ready(session)


        except websockets.exceptions.ConnectionClosed:
            # Find and clean up the client's session
            for session in self.sessions.values():
                if websocket in session.clients:
                    await self.handle_disconnect(websocket, session)
                    break

    async def handle_disconnect(self, websocket: websockets.WebSocketServerProtocol, session: SessionInfo):
        if websocket in session.clients:
            disconnected_rank = session.clients[websocket].rank
            del session.clients[websocket]
            
            # Reset connection state for affected neighbors
            for _, info in session.clients.items():
                if disconnected_rank in info.connected_peers:
                    info.connected_peers.remove(disconnected_rank)
                    info.ready = False
            
            # If session is empty, remove it
            if not session.clients:
                del self.sessions[session.session_id]
                logging.info(f"Session {session.session_id} removed")
            else:
                await self.broadcast_topology(session)

    async def broadcast_topology(self, session: SessionInfo):
        grid_size = self.calculate_grid_size(session)
        for ws, info in session.clients.items():
            try:
                # neighbors = self.get_neighbor_ranks(session, info.rank)

                # this assumes that max_clients number of clients are connected
                # list of ints
                if info.rank == 0:
                    neighbor_dict = {}
                    print(neighbor_dict)
                else:
                    print(info.rank)
                    topology_config = {
                        "topology": {"name": session.config["algos"]["node_0"]["topology"]["name"]},
                        "num_users": session.config["num_users"],
                        "seed": session.config["seed"]
                    }
                    print(topology_config)
                    topology = select_topology(topology_config, info.rank)
                    topology.initialize()
                    # do we only want 1 neighbor?
                    neighbors = topology.sample_neighbours(session.config["num_users"]) #type: ignore
                    neighbor_dict = {f"neighbor{info.rank}": neighbors}
                    print(neighbor_dict)
                await ws.send(json.dumps({
                    'type': 'topology',
                    'rank': info.rank,
                    'neighbors': neighbor_dict,
                    'gridSize': grid_size,
                    'totalClients': len(session.clients)
                }))
            except websockets.exceptions.ConnectionClosed:
                logging.error(f"Failed to send topology to client {info.rank}")
                pass
    
    async def broadcast_session_ready(self, session: SessionInfo):
        message = json.dumps({
            'type': 'session_ready',
            'message': f'All {len(session.clients)} clients connected'
        })
        await asyncio.gather(*[
            ws.send(message) for ws in session.clients
        ])
            
    async def check_session_ready(self, session: SessionInfo):
        # expected_connections = 2  # Each node should connect to 2 neighbors
        # all_ready = all(
        #     len(info.connected_peers) == expected_connections 
        #     for info in session.clients.values()
        # )

        all_ready = session.num_ready == len(session.clients)
        print(f"All ready: {all_ready}: {session.num_ready} / {len(session.clients)}")

        if all_ready:
            logging.info(f"All nodes in session {session.session_id} are connected!")
            await self.broadcast_network_ready(session)
            
    async def broadcast_network_ready(self, session: SessionInfo):
        message = json.dumps({'type': 'network_ready'})
        await asyncio.gather(*[
            ws.send(message) for ws in session.clients
        ])

async def main():
    server = SignalingServer()
    async with websockets.serve(server.handle_client, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
