# test_fedstatic.py

import os

# Set environment variables before importing any modules that use wand
os.environ['MAGICK_HOME'] = "/opt/homebrew/Cellar/imagemagick/7.1.1-43"
os.environ['DYLD_LIBRARY_PATH'] = "/opt/homebrew/Cellar/imagemagick/7.1.1-43/lib:" + os.environ.get('DYLD_LIBRARY_PATH', '')

import asyncio
import json
import websockets
from algos.fl_static import FedStaticNode

# Placeholder config for FedStaticNode
config = {
    "rounds": 2,                # number of training rounds
    "start_round": 0,
    "epochs_per_round": 1,
    # ... add other config options you need ...
}

# If you have a real CommunicationManager, import it.
# For now, we can pass None if your code doesn't strictly require it.
comm_utils = None

# Create our FL node
node = FedStaticNode(config, comm_utils)

async def test_fedstatic(action="create_session", session_id=None, max_clients=2):
    """
    Connect to the signaling server, create or join a session,
    receive the ring topology, then run the FL protocol.
    """
    uri = "ws://localhost:8765"   # same as your server's host & port

    async with websockets.connect(uri) as websocket:
        # 1) Create or join the session
        if action == "create_session":
            msg = {
                "type": "create_session",
                "maxClients": max_clients,
                "clientType": "python_test"
            }
        else:  # "join_session"
            msg = {
                "type": "join_session",
                "sessionId": session_id,   # must match an existing session
                "clientType": "python_test"
            }

        await websocket.send(json.dumps(msg))
        print(f"Sent {msg}")

        # 2) Listen for server messages
        while True:
            try:
                resp = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                break

            data = json.loads(resp)
            print(f"Received: {data}")

            # Handle responses
            msg_type = data.get("type")

            if msg_type == "session_created":
                # Assign node_id from rank, if needed
                node.node_id = data["rank"]
                print(f"Session created with rank {node.node_id}")

            elif msg_type == "session_joined":
                node.node_id = data["rank"]
                print(f"Joined session with rank {node.node_id}")

            elif msg_type == "topology":
                # For ring-based neighbors: data["neighbors"] might be {"prev": 0, "next": 2}
                node.store_server_neighbors(data["neighbors"])
                print(f"Node {node.node_id} got neighbors: {node.server_neighbors}")

                # Once we have neighbors, we can run the FL protocol
                # Note: run_protocol() is synchronous, so it will block.
                # If you want asynchronous, use run_async_protocol() instead.
                node.run_protocol()

            elif msg_type == "session_ready":
                print("Session is ready for FL.")
                # Possibly wait for "topology" or do other logic

            elif msg_type == "network_ready":
                print("All nodes reported ready. (Optional)")

            elif msg_type == "error":
                print(f"Error from server: {data.get('message')}")
                break

async def main():
    # Example: create a session with max 2 clients
    await test_fedstatic(action="create_session", max_clients=2)

    # If you want a second client to join:
    # await test_fedstatic(action="join_session", session_id="the_session_id_from_above")

if __name__ == "__main__":
    asyncio.run(main())