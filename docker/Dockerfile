FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=keysg
ARG USERNAME=keysg
ARG USER_UID=1000
ARG USER_GID=1000

ENV CUDA_HOME=/usr/local/cuda \
    CONDA_DIR=${CONDA_DIR} \
    ENV_NAME=${ENV_NAME} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    gnupg \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libglu1-mesa-dev \
    libsm6 \
    libusb-1.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniconda.sh && \
    conda config --system --remove channels defaults || true && \
    conda config --system --remove-key default_channels || true && \
    conda config --system --remove-key custom_channels || true && \
    conda config --system --add channels conda-forge && \
    conda config --system --set channel_priority strict && \
    conda clean -afy

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm --version

SHELL ["/bin/bash", "-lc"]

WORKDIR /tmp/keysg-build

COPY environment.yaml pyproject.toml README.md ./

RUN python - <<'PY'
from pathlib import Path

src = Path("/tmp/keysg-build/environment.yaml")
dst = Path("/tmp/keysg-build/environment.docker.yaml")

lines = []
for line in src.read_text().splitlines():
    if line.strip() == "- -e .":
        continue
    lines.append(line)

dst.write_text("\n".join(lines) + "\n")
PY

RUN conda env create -n "${ENV_NAME}" -f /tmp/keysg-build/environment.docker.yaml && \
    conda clean -afy

RUN conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir matplotlib==3.10.8

RUN npm install -g @openai/codex@latest @anthropic-ai/claude-code@latest && \
    codex --version && \
    claude --version

WORKDIR /workspace/keysg
COPY . /workspace/keysg

RUN conda run -n "${ENV_NAME}" python -m pip install --no-deps -e /workspace/keysg && \
    conda run -n "${ENV_NAME}" python -c "import cv2, faiss, keysg, matplotlib, open3d, open_clip, sam2, torch, transformers, viser; print('keysg docker environment ready')"

RUN groupadd --gid "${USER_GID}" "${USERNAME}" && \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" && \
    mkdir -p /workspace/datasets /workspace/keysg/.cache /workspace/keysg/checkpoints /workspace/keysg/output && \
    chown -R "${USERNAME}:${USERNAME}" /home/"${USERNAME}" /workspace /opt/conda

RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/"${USERNAME}"/.bashrc && \
    echo "conda activate ${ENV_NAME}" >> /home/"${USERNAME}"/.bashrc

USER ${USERNAME}

ENV HOME=/home/${USERNAME} \
    HF_HOME=/workspace/keysg/.cache/huggingface \
    TORCH_HOME=/workspace/keysg/.cache/torch \
    XDG_CACHE_HOME=/workspace/keysg/.cache \
    MPLCONFIGDIR=/workspace/keysg/.cache/matplotlib \
    PYTHONPATH=/workspace/keysg

WORKDIR /workspace/keysg

CMD ["/bin/bash"]
