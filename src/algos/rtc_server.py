import asyncio
import json
import logging
import secrets
import websockets

from algos.topologies.collections import select_topology

logging.basicConfig(level=logging.DEBUG)

class SignalingServer:
    def __init__(self):
        self.sessions = {}  
    async def handle_client(self, websocket):
        try:
            message = await websocket.recv()
            data = json.loads(message)
            # When a client creates a session:
            if data['type'] == 'create_session':
                session_id = data.get('sessionId', secrets.token_hex(3))
                max_clients = int(data.get('maxClients', 2))
                config = data.get("config", {})
                config["num_users"] = max_clients
                self.sessions[session_id] = {
                    "session_id": session_id,
                    "max_clients": max_clients,
                    "clients": {},
                    "config": config
                }
                # First client gets rank 0.
                rank = 0
                self.sessions[session_id]["clients"][websocket] = {"rank": rank}
                logging.info(f"[RTC] Created session {session_id} for {max_clients} clients")
                await websocket.send(json.dumps({
                    "type": "session_created",
                    "sessionId": session_id,
                    "rank": rank
                }))
            # When a client joins an existing session:
            elif data['type'] == 'join_session':
                session_id = data.get("sessionId")
                if session_id not in self.sessions:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid session ID"
                    }))
                    return
                session = self.sessions[session_id]
                if len(session["clients"]) >= session["max_clients"]:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Session is full"
                    }))
                    return
                rank = len(session["clients"])
                session["clients"][websocket] = {"rank": rank}
                logging.info(f"[RTC] Client joined session {session_id} with rank {rank}")
                await websocket.send(json.dumps({
                    "type": "session_joined",
                    "sessionId": session_id,
                    "rank": rank
                }))
                # When the session is full, generate and broadcast the topology.
                if len(session["clients"]) == session["max_clients"]:
                    logging.info(f"[RTC] Session {session_id} is full, generating topology")
                    await self.broadcast_topology(session_id)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def broadcast_topology(self, session_id):
        session = self.sessions[session_id]
        config = session["config"]
        for ws, info in session["clients"].items():
            rank = info["rank"]
            topology = select_topology(config, rank)
            topology.initialize()
            neighbors = topology.get_all_neighbours()
            try:
                await ws.send(json.dumps({
                    "type": "topology",
                    "rank": rank,
                    "neighbors": neighbors
                }))
            except websockets.exceptions.ConnectionClosed:
                pass

async def main():
    server = SignalingServer()
    async with websockets.serve(server.handle_client, "localhost", 8888):
        logging.info("[RTC] Server started on ws://localhost:8888")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
