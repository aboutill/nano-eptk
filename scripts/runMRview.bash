#!/usr/bin/env bash
# =============================================================================
# runMRview.bash: part of nano-eptk package.
#
# Run MRtrix mrview GUI via the nano-eptk Docker image.
# This script is experimental.
#
# Configuration is set in the USER INPUTS block below.
# =============================================================================

set -euo pipefail

# =============================================================================
# USER INPUTS
# =============================================================================
# Data directory, relative to the app root
data_dir="data"

# =============================================================================
# Do not edit
# =============================================================================
readonly HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
readonly DOCKER_DIR="/usr/src/app"
readonly DOCKER_IMG="aboutill/nano-eptk:latest"

die() { echo "Error: $*" >&2; exit 1; }

# =============================================================================
# Validation
# =============================================================================
command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
[[ -e "${HOST_DIR}/${data_dir}" ]] || die "data not found: ${HOST_DIR}/${data_dir}"
  
# =============================================================================
# Build command
# =============================================================================
app_cmd=( mrview )

docker_flags=( 
  --rm 
  --user "$(id -u):$(id -g)"
  --volume "${HOST_DIR}/${data_dir}":"${DOCKER_DIR}/${data_dir}"
  --volume /tmp/.X11-unix:/tmp/.X11-unix 
  --gpus all
  --security-opt apparmor=unconfined
  --ipc=host
  --env DISPLAY=$DISPLAY
  --env QT_X11_NO_MITSHM=1
  --env LIBGL_ALWAYS_SOFTWARE=1
  --env QT_QPA_PLATFORM=xcb
)
[[ -t 0 && -t 1 ]] && docker_flags+=( -it )

# =============================================================================
# Run Docker application
# =============================================================================
echo "Running mrview via ${DOCKER_IMG} ..."

xhost +local:docker

docker run \
  "${docker_flags[@]}" \
  "${DOCKER_IMG}" \
  "${app_cmd[@]}"
  
xhost -local:docker

echo "Done!"
