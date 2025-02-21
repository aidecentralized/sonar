import asyncio
import json
import websockets
import logging
import secrets
import os

logging.basicConfig(level=logging.DEBUG)

class SignalingServer:
    def __init__(self):
        self.sessions = {}  # Store active sessions

    async def handle_client(self, websocket):
        try:
            async for message in websocket:
                logging.debug(f"Received message: {message}")
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logging.error("Invalid JSON received")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))
                    continue

                msg_type = data.get('type')
                if msg_type == 'create_session':
                    session_id = secrets.token_hex(3)
                    try:
                        max_clients = int(data['maxClients'])
                    except (KeyError, ValueError):
                        logging.error("Invalid or missing 'maxClients' in create_session")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Invalid or missing maxClients'
                        }))
                        continue

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

                elif msg_type == 'join_session':
                    session_id = data.get("sessionId")
                    if session_id not in self.sessions:
                        logging.error(f"[ERROR] Invalid session ID: {session_id}")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Invalid session ID'
                        }))
                        continue

                    session = self.sessions[session_id]
                    if len(session["clients"]) >= session["max_clients"]:
                        logging.error(f"[ERROR] Session {session_id} is full")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Session is full'
                        }))
                        continue

                    rank = len(session["clients"])
                    session["clients"][websocket] = {"rank": rank}

                    logging.info(f"[RTC] Client joined session {session_id} with rank {rank}")
                    await websocket.send(json.dumps({
                        'type': 'session_joined',
                        'sessionId': session_id,
                        'rank': rank
                    }))

                else:
                    logging.error(f"[ERROR] Unknown message type: {msg_type}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Unknown message type'
                    }))
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

async def main():
    server = SignalingServer()
    # Bind to port 8888 (or choose port 0 for dynamic assignment)
    async with websockets.serve(server.handle_client, "localhost", 8888) as ws_server:
        # If you use port 0, uncomment the next two lines to get the dynamically assigned port
        # actual_port = ws_server.sockets[0].getsockname()[1]
        # logging.info(f"[RTC] Server started on ws://localhost:{actual_port}")
        logging.info("[RTC] Server started on ws://localhost:8888")
        
        # Write the port to server_port.txt in the same directory as this script
        port_to_write = 8888  # Change to actual_port if using dynamic assignment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        port_file = os.path.join(current_dir, "server_port.txt")
        with open(port_file, "w") as f:
            f.write(str(port_to_write))
        logging.info(f"Port {port_to_write} written to {port_file}")
        
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
