# Use the latest LTS (Long-Term Support) version of Ubuntu
FROM ubuntu:latest

# Set working directory to /sonar
WORKDIR /sonar

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa

# Install base utilities
#RUN apt-get install -y \
#    build-essential \
#    wget \
#    python3.11 \  
#    python3-pip \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to the working directory
#COPY requirements.txt .

# Install Python development tools (needed for creating virtual environments)
#RUN apt-get update && apt-get install -y python3-venv

#RUN apt-get update && apt-get install -y openmpi-bin

#RUN apt-get update && apt-get install -y libopenmpi-dev

# Create a virtual environment
#RUN python3 -m venv env

# Activate the virtual environment (source this command in subsequent RUN steps)
#RUN . env/bin/activate

# Install dependencies from requirements.txt within the virtual environment
#RUN pip3 install -r requirements.txt
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