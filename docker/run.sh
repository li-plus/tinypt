#!/usr/bin/env bash

set -e

function show_help() {
    cat << EOF
USAGE: bash $0 [-h|--help] [-r|--restart] IMAGE
EOF
}

positional_args=()
while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)      show_help; exit 0;;
    -r|--restart)   restart=1; shift;;
    -*)             echo "Unknown option $1"; show_help; exit 1;;
    *)              positional_args+=("$1"); shift;;
    esac
done

set -- "${positional_args[@]}"
if [[ $# -ne 1 ]]; then
    show_help
    exit 1
fi

image_name=$1
container_name=$image_name-dev
container_name=${container_name/:/-}

uid=$(id -u)
user=$(id -nu)
gid=$(id -g)
group=$(id -ng)
shell=${SHELL:-bash}

container_id=$(docker ps -a -q -f name=$container_name)
running_id=$(docker ps -q -f name=$container_name)

if [[ $restart -eq 1 && -n $container_id ]]; then
    docker stop $container_id
    docker rm $container_id
    container_id=
    running_id=
fi

if [[ -z $container_id ]]; then
    echo "Launching new container ..."
    docker_exec=docker
    if [[ -x $(command -v nvidia-docker) ]]; then
        docker_exec=nvidia-docker
    fi
    $docker_exec run -it -d \
        -v $HOME:$HOME \
        -w $HOME \
        --network=host \
        -v /etc/timezone:/etc/timezone:ro \
        -v /etc/localtime:/etc/localtime:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /etc/shadow:/etc/shadow:ro \
        --name $container_name \
        $image_name \
        bash -c "echo '$user   ALL=(ALL:ALL) NOPASSWD: ALL' >> /etc/sudoers && exec su $user"
elif [[ -z $running_id ]]; then
    echo "Restarting stopped container ..."
    docker start $container_name
fi

exec docker exec -it -u $(id -u):$(id -g) $container_name $shell
