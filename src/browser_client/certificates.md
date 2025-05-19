# Certificates

We need certificates to utilize https / wss, which is necessary for cross-device communication.

## Generating Certs

1. Install mkcert:

brew install mkcert       # macOS

sudo apt install mkcert   # Ubuntu

choco install mkcert      # Windows (via Chocolatey)


2. Install local CA:

mkcert -install


3. Generate cert in working directory:

mkcert {space-separated list of IP addresses}

You need to specify all addresses from which you may be accessing this client/server.

## Using Certs
1. To make the browser client run on https and use wss, add certificates to src/browser_client/certs/ and add the cert & key filepaths to src/browser_client/vite.config.js (or use hosted browser clients onRender)

2. To make the python signalling server use wss, add certificates to src/certs/ and add the cert & key filepaths to main() in src/rtc_server.py

3. The python clients don't need their own certs, but for them to validate these certs and communicate with the signaling server / browser clients, add the CA Root filepath (directory is found using mkcert -CAROOT, filename is typically 'rootCA.pem') to the cafile argument in ssl.create_default_context() in src/utils/communication/rtc_async_ver.py

4. Make sure all websocket addresses (src/configs/sys_config.py, config form in browser client) use wss:// not ws:// 

5. To run any client on a different device, you need to install the CA Root certificate (found at mkcert -CAROOT under rootCA.pem) on each device for it to authenticate the generated certificates above. For windows you need to rename it from .pem to .crt