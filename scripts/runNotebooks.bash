#!/usr/bin/env bash
# =============================================================================
# runNotebook.bash: part of nano-eptk package.
#
# Run the Jupyter notebooks interactively via the nano-eptk  Docker image.
#
# Configuration is set in the USER INPUTS block below.
# =============================================================================

set -euo pipefail

# =============================================================================
# USER INPUTS
# =============================================================================
cfg_dir="cfg"             # Configuration directory, relative to the app root
data_dir="data"           # Data directory, relative to the app root
notebooks_dir="notebooks" # Notebooks directory, relative to the app root
results_dir="results"     # Results directory, relative to the app root

# =============================================================================
# Do not edit
# =============================================================================
readonly HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
readonly DOCKER_DIR="/usr/src/app"
readonly DOCKER_IMG="aboutill/nano-eptk:latest"
readonly PORT=8888

die() { echo "Error: $*" >&2; exit 1; }

# =============================================================================
# Validation
# =============================================================================
command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
[[ -d "${HOST_DIR}/${cfg_dir}" ]] || die "cfg directory not found: ${HOST_DIR}/${cfg_dir}"
[[ -d "${HOST_DIR}/${data_dir}" ]] || die "data directory not found: ${HOST_DIR}/${data_dir}"
[[ -d "${HOST_DIR}/${notebooks_dir}" ]] || die "notebooks directory not found: ${HOST_DIR}/${notebooks_dir}"

mkdir -p "${HOST_DIR}/${results_dir}"

# =============================================================================
# Build command
# =============================================================================
app_cmd=(
  jupyter notebook
  --ip=0.0.0.0
  --port="${PORT}"
  --no-browser
)

docker_flags=( 
  --rm 
  --user "$(id -u):$(id -g)" 
  -p "${PORT}":"${PORT}" 
  --volume "${HOST_DIR}/${cfg_dir}:${DOCKER_DIR}/${cfg_dir}"
  --volume "${HOST_DIR}/${data_dir}:${DOCKER_DIR}/${data_dir}"
  --volume "${HOST_DIR}/${notebooks_dir}:${DOCKER_DIR}/${notebooks_dir}"
  --volume "${HOST_DIR}/${results_dir}:${DOCKER_DIR}/${results_dir}"
)
[[ -t 0 && -t 1 ]] && docker_flags+=( -it )

# =============================================================================
# Run Docker application
# =============================================================================
echo "Running notebooks via ${DOCKER_IMG} ..."

docker run \
  "${docker_flags[@]}" \
  "${DOCKER_IMG}" \
  "${app_cmd[@]}"
  
echo "Done!"
