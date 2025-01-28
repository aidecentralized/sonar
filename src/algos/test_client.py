import asyncio
import json
import websockets

async def test_client():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to the signaling server.")

            # 1. Send a 'create_session' message to the server
            create_msg = {
                "type": "create_session",
                "maxClients": 2,    # or any integer
                "clientType": "python_test"
            }
            await websocket.send(json.dumps(create_msg))
            print(">>> Sent:", create_msg)

            # 2. Keep listening for messages until the server or client closes connection
            while True:
                response = await websocket.recv()  # waits for server message
                print("<<< Received:", response)

    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

async def main():
    await test_client()

if __name__ == "__main__":
    asyncio.run(main())