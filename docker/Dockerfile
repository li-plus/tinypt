ARG CUDA=1

FROM ubuntu:20.04 as cuda-0

FROM nvidia/cuda:11.1.1-devel-ubuntu20.04 as cuda-1

FROM cuda-${CUDA} as final

RUN \
    sed -e "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" \
        -e "s/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" -i /etc/apt/sources.list && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -yq --no-install-recommends \
        sudo zsh vim tmux gdb git clang-format wget unzip gcc g++ cmake make \
        libeigen3-dev libopencv-dev libgflags-dev libgoogle-glog-dev \
        python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --upgrade pip
