FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Build args
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV MODEL_DIR=/app/model

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir /app
WORKDIR /app

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Install some basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo gcc build-essential openssh-client && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
                && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
ENV SHELL=/bin/bash

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    rm /app/requirements.txt

# Fetch the model
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs
RUN git lfs install
RUN git clone https://huggingface.co/coqui/XTTS-v2 ${MODEL_DIR}

# Add src files (Worker Template)
ADD src /app

ENV RUNPOD_DEBUG_LEVEL=INFO

CMD python3 -u /app/rp_handler.py --model-dir="${MODEL_DIR}"
