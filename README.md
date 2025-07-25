![Warning](https://img.shields.io/badge/Warning-This%20project%20is%20in%20beta-yellow)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/markdown-guide/badge/?version=latest)](https://aidecentralized.github.io/sonar/)

# Project SONAR Web - Self-Organizing Network of Aggregated Representations

![Architecture Diagram](https://github.com/aidecentralized/sonar/blob/main/docs/arch.png)

Documentation: https://aidecentralized.github.io/sonar/

A modular framework for decentralized training of neural networks across Python, web, and mobile platforms, enabling real-time peer-to-peer model training without centralized infrastructure.

## Overview

SONAR Web introduces a modular framework for decentralized training of neural networks across Python, web, and mobile platforms. We develop and release a cross-platform open-source implementation with TensorFlow.js-based browser and Node.js clients, enabling real-time peer-to-peer model training without centralized infrastructure.

SONAR Web demonstrates practical, cross-platform decentralized training across real-world devices, benchmarking the viability of on-device collaborative learning in constrained and heterogeneous settings. It takes a step towards enabling decentralized learning over resource-constrained devices on the edge, contributing to a more inclusive, heterogeneous AI ecosystem.

## System Design

SONAR Web is designed as a modular framework for real-time, privacy-preserving, and fully decentralized collaborative learning across heterogeneous environments—including mobile devices, web browsers, and Python-based clients. Our system supports dynamic, on-the-fly participation from diverse clients with minimal setup and strict data locality.

### Core Components

SONAR Web is composed of four core modules:
1. **Lightweight peer registration and discovery mechanism**
2. **Unified communication layer** abstracting platform differences
3. **Platform-agnostic configuration interface**
4. **Training and monitoring framework** for collaborative learning across devices

### Design Goals

- **Modularity:** Each component is decoupled and independently replaceable, enabling flexible experimentation and extensibility
- **Interoperability:** Real-time communication and learning across diverse platforms, including web browsers, mobile devices, and Python environments
- **Decentralization:** No centralized coordinator or aggregator, relying instead on peer-to-peer communication and local control
- **Minimal Setup and Accessibility:** Participation requires no specialized infrastructure, lowering the barrier to entry

## Quick Start

### Server Setup
```bash
pip install -r requirements.txt
python src/rtc_server.py
```

### Client Setup
```bash
cd src/browser_client
npm install
npm run dev
```
Then navigate to the provided link in your browser.

## Features

- Real-time peer-to-peer model training
- Cross-platform support (Web, Node.js, Python)
- TensorFlow.js integration for browser-based learning
- Privacy-preserving decentralized architecture
- Minimal setup requirements
- Dynamic peer discovery and registration

## Contributing

This project is actively developed and welcomes contributions. Please check the documentation for development guidelines and contribution instructions.
