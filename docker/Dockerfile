ARG BASE_IMAGE=ubuntu:20.04

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN \
    sed -e "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" \
        -e "s/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" -i /etc/apt/sources.list && \
    apt update && \
    apt install -yq --no-install-recommends \
        sudo vim tmux gdb git clang-format wget unzip gcc g++ cmake make \
        libeigen3-dev libopencv-dev libgflags-dev libgoogle-glog-dev \
        python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --upgrade pip

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
