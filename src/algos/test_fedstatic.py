import asyncio
import json
import websockets
import logging
import secrets

logging.basicConfig(level=logging.DEBUG)

class SignalingServer:
    def __init__(self):
        self.sessions = {}  # Store active sessions

    async def handle_client(self, websocket):
        try:
            message = await websocket.recv()
            data = json.loads(message)

            if data['type'] == 'create_session':
                session_id = secrets.token_hex(3)
                max_clients = int(data['maxClients'])
                self.sessions[session_id] = {
                    "session_id": session_id,
                    "max_clients": max_clients,
                    "clients": {}
                }

                rank = 0
                self.sessions[session_id]["clients"][websocket] = {"rank": rank}

                logging.info(f"[RTC] Created session {session_id} for {max_clients} clients")

                await websocket.send(json.dumps({
                    'type': 'session_created',
                    'sessionId': session_id,
                    'rank': rank
                }))

            elif data['type'] == 'join_session':
                session_id = data.get("sessionId")

                if session_id not in self.sessions:
                    logging.error(f"[ERROR] Invalid session ID: {session_id}")
                    await websocket.send(json.dumps({'type': 'error', 'message': 'Invalid session ID'}))
                    return

                session = self.sessions[session_id]
                if len(session["clients"]) >= session["max_clients"]:
                    logging.error(f"[ERROR] Session {session_id} is full")
                    await websocket.send(json.dumps({'type': 'error', 'message': 'Session is full'}))
                    return

                rank = len(session["clients"])
                session["clients"][websocket] = {"rank": rank}

                logging.info(f"[RTC] Client joined session {session_id} with rank {rank}")

                await websocket.send(json.dumps({
                    'type': 'session_joined',
                    'sessionId': session_id,
                    'rank': rank
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
