FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy

WORKDIR /workspace
