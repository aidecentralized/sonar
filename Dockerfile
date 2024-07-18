# Use the latest LTS (Long-Term Support) version of Ubuntu
FROM ubuntu:latest

# Set working directory to /sonar
WORKDIR /sonar

#Install base utilities and Python with required dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    build-essential \
    wget \
    python3.11 \
    python3-pip \
    python3-venv \
    openmpi-bin \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python3 -m venv env && \
    . env/bin/activate && \
    env/bin/pip install -r requirements.txt

# To activate the virtual environment by default, use ENTRYPOINT
ENTRYPOINT ["/bin/bash", "-c", "source /sonar/env/bin/activate && exec \"$@\"", "--"]