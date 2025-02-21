import asyncio
import json
import websockets
import os

async def test():
    # Locate server_port.txt in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    port_file = os.path.join(current_dir, "server_port.txt")
    
    try:
        with open(port_file, "r") as f:
            actual_port = f.read().strip()
    except FileNotFoundError:
        print("Error: server_port.txt not found. Make sure the server is running and the file is in the same directory.")
        return

    uri = f"ws://localhost:{actual_port}"
    print(f"Connecting to {uri}")
    async with websockets.connect(uri) as ws:
        # Send a create_session request
        await ws.send(json.dumps({
            'type': 'create_session',
            'maxClients': 3
        }))
        response = await ws.recv()
        print("Server response:", response)
        session_data = json.loads(response)
        session_id = session_data.get("sessionId")

        if session_id:
            # Send a join_session request using the session ID from the response
            await ws.send(json.dumps({
                'type': 'join_session',
                'sessionId': session_id
            }))
            join_response = await ws.recv()
            print("Join response:", join_response)

asyncio.run(test())
