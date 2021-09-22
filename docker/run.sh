#!/usr/bin/env bash

set -e

function show_help() {
    cat << EOF
USAGE: bash $0 [-h|--help] [--rebuild] [--restart] [--cuda]
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
    --rebuild)  rebuild=1; shift 1;;
    --restart)  restart=1; shift 1;;
    --cuda)     cuda=1; shift 1;;
    -h|--help)  show_help; exit 0;;
    *)          echo "Unexpected argument $1"; show_help; exit 1;;
    esac
done

image_name=tinypt:latest
image_id=$(docker image ls -q $image_name)
if [[ -z $image_id || $rebuild -eq 1 ]]; then
    # build image
    if [[ $cuda -eq 1 ]]; then
        BASE_IMAGE=nvidia/cuda:11.1.1-devel-ubuntu20.04
    else
        BASE_IMAGE=ubuntu:20.04
    fi
    build_dir=$(dirname $(realpath $0))
    docker build $build_dir -t $image_name --build-arg BASE_IMAGE=$BASE_IMAGE
fi

container_name=tinypt-dev
container_id=$(docker ps -a -q -f name=$container_name)
running_id=$(docker ps -q -f name=$container_name)

if [[ $restart -eq 1 && -n $container_id ]]; then
    docker stop $container_id
    docker rm $container_id
    container_id=""
    running_id=""
fi

if [[ -z $container_id ]]; then
    # create container
    run_args=""
    if [[ $cuda -eq 1 ]]; then
        run_args="$run_args --runtime nvidia"
    fi

    docker run -it -d $run_args \
        -v $HOME:$HOME \
        -w $HOME \
        -e uid=$(id -u) \
        -e gid=$(id -g) \
        -e user=$USER \
        -e group=$(id -ng) \
        -e home=$HOME \
        -v /etc/timezone:/etc/timezone:ro \
        -v /etc/localtime:/etc/localtime:ro \
        --name $container_name \
        $image_name
elif [[ -z $running_id ]]; then
    # start container
    docker start $container_name
fi

docker exec -it -u $(id -u):$(id -g) $container_name bash
