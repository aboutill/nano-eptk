#!/usr/bin/env bash
# =============================================================================
# runNotebook.bash: part of nano-eptk package.
#
# Run Jupyter notebooks inside Docker container interactively.
#
# =============================================================================

# Host directories
host_workdir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
cfg_dir="cfg"
data_dir="data"
notebooks_dir="notebooks"
results_dir="results"

# Docker application image name
docker_img="aboutill/nano-eptk:latest"

# Docker directory
docker_workdir="/usr/src/app"

# Port (host and container must match Jupyter's --port)
port=8008

# Launch docker
docker run \
  --user $(id -u):$(id -g) \
  -v "${host_workdir}/${cfg_dir}":"${docker_workdir}/${cfg_dir}" \
  -v "${host_workdir}/${data_dir}":"${docker_workdir}/${data_dir}" \
  -v "${host_workdir}/${notebooks_dir}":"${docker_workdir}/${notebooks_dir}" \
  -v "${host_workdir}/${results_dir}":"${docker_workdir}/${results_dir}" \
  -p "${port}":"${port}" \
  -it \
  --rm \
  "$docker_img" \
  jupyter notebook \
    --ip=0.0.0.0 \
    --port="${port}" \
    --no-browser

# Cmd display
echo "Docker container stopped!"
