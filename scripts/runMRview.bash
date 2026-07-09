#!/usr/bin/env bash
# =============================================================================
# runMRview.bash: part of nano-eptk package.
#
# Run mrview inside Docker container.
#
# =============================================================================

# Required arguments
# Host directoriy
host_workdir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
data_dir="data"

# =============================================================================
# Run Docker application
# =============================================================================
# Docker application image name
docker_img="aboutill/nano-eptk:latest"

# Docker directory
docker_workdir="/usr/src/app/"

# Enable xserver GUI connection
xhost +local:docker

# Launch docker
docker run \
  --user $(id -u):$(id -g) \
  -v "${host_workdir}/${data_dir}":"${docker_workdir}/${data_dir}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --gpus all \
  --security-opt apparmor=unconfined \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e QT_QPA_PLATFORM=xcb \
  -it \
  --rm \
  $docker_img \
  mrview 

# Disable xserver GUI connection
xhost -local:docker
