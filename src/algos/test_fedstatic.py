import asyncio
import json
import websockets
import logging
import os

logging.basicConfig(level=logging.DEBUG)

async def node_behavior(node_id, uri, create=False, session_id=None):
    """
    Simulate the behavior of a federated learning node.
    
    Args:
        node_id (int): Unique identifier for the node.
        uri (str): WebSocket URI for the signaling server.
        create (bool): If True, this node creates the session.
        session_id (str): The session ID to join if create is False.
    
    Returns:
        The session_id used (if node created the session, it is returned).
    """
    async with websockets.connect(uri) as ws:
        if create:
            # Node creates a session
            await ws.send(json.dumps({
                'type': 'create_session',
                'maxClients': 3
            }))
            response = await ws.recv()
            data = json.loads(response)
            session_id = data.get("sessionId")
            rank = data.get("rank")
            logging.info(f"Node {node_id} created session {session_id} with rank {rank}")
        else:
            # Node joins an existing session
            if not session_id:
                logging.error(f"Node {node_id}: No session_id provided for join.")
                return None
            await ws.send(json.dumps({
                'type': 'join_session',
                'sessionId': session_id
            }))
            response = await ws.recv()
            data = json.loads(response)
            rank = data.get("rank")
            logging.info(f"Node {node_id} joined session {session_id} with rank {rank}")
        
        # Simulate a federated learning training round
        logging.info(f"Node {node_id} starting training round...")
        await asyncio.sleep(1)  # simulate training time delay
        logging.info(f"Node {node_id} completed training round.")
        
        return session_id

async def main():
    # Read the server port from server_port.txt in the same directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    port_file = os.path.join(current_dir, "server_port.txt")
    try:
        with open(port_file, "r") as f:
            port = f.read().strip()
    except FileNotFoundError:
        logging.error("server_port.txt not found. Make sure the signaling server is running.")
        return

    uri = f"ws://localhost:{port}"
    logging.info(f"FL nodes will connect to {uri}")
    
    # Node 0 creates the session
    session_id = await node_behavior(0, uri, create=True)
    if not session_id:
        logging.error("Failed to create session.")
        return
    
    # Other nodes join the session
    join_tasks = []
    for i in range(1, 3):  # e.g., node 1 and node 2 join
        join_tasks.append(asyncio.create_task(node_behavior(i, uri, create=False, session_id=session_id)))
    
    # Wait for all nodes to complete their simulated training round
    await asyncio.gather(*join_tasks)

if __name__ == "__main__":
    asyncio.run(main())
