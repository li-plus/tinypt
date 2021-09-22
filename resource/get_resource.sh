#!/usr/bin/env bash

set -ex

resource_dir=$(dirname $(realpath $0))
mkdir -p ${resource_dir}

function download() {
    url=$1
    name=$2
    target_dir=${resource_dir}/${name}
    if [[ ! -d ${target_dir} ]]; then
        echo "[INFO]: Downloading ${name} ..."
        wget ${url} -O resource.zip
        mkdir -p ${target_dir}
        unzip resource.zip -d ${target_dir}
        rm resource.zip
    else
        echo "[Warning]: Skipping existing resource ${name}"
    fi
}

if [[ ! -d ${resource_dir}/breakfast_room ]]; then
    download https://casual-effects.com/g3d/data10/research/model/breakfast_room/breakfast_room.zip breakfast_room
    patch ${resource_dir}/breakfast_room/breakfast_room.mtl ${resource_dir}/breakfast_room.mtl.patch
fi
download https://casual-effects.com/g3d/data10/research/model/bunny/bunny.zip bunny
download https://casual-effects.com/g3d/data10/common/model/CornellBox/CornellBox.zip CornellBox
download https://casual-effects.com/g3d/data10/common/model/crytek_sponza/sponza.zip crytek_sponza
download https://casual-effects.com/g3d/data10/research/model/dabrovic_sponza/sponza.zip dabrovic_sponza
download https://casual-effects.com/g3d/data10/research/model/dragon/dragon.zip dragon
download https://casual-effects.com/g3d/data10/research/model/fireplace_room/fireplace_room.zip fireplace_room
download https://casual-effects.com/g3d/data10/research/model/living_room/living_room.zip living_room
download https://casual-effects.com/g3d/data10/research/model/rungholt/rungholt.zip rungholt
download https://casual-effects.com/g3d/data10/research/model/salle_de_bain/salle_de_bain.zip salle_de_bain
download https://casual-effects.com/g3d/data10/research/model/sibenik/sibenik.zip sibenik

# environment texture
envmap_dir=${resource_dir}/envmap
mkdir -p ${envmap_dir}
if [[ ! -f ${envmap_dir}/venice_sunset_4k.hdr ]]; then
    wget https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/venice_sunset_4k.hdr -O ${envmap_dir}/venice_sunset_4k.hdr
fi
