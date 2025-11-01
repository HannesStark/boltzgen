# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    HF_HOME=/cache
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    build-essential \
    git \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    libgl1 \
    libhdf5-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip setuptools setuptools_scm wheel

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e /app

ARG DOWNLOAD_WEIGHTS=false
ARG HF_TOKEN=""
RUN mkdir -p "${HF_HOME}" && \
    if [ "${DOWNLOAD_WEIGHTS}" = "true" ]; then \
        HF_TOKEN="${HF_TOKEN}" boltzgen download --cache "${HF_HOME}" --force_download; \
    fi

ARG USERNAME=boltzgen
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME}

RUN mkdir -p "${HF_HOME}" && chown -R ${USER_UID}:${USER_GID} "${HF_HOME}"

USER ${USERNAME}
WORKDIR /workspace

ENTRYPOINT ["boltzgen"]
CMD ["--help"]
