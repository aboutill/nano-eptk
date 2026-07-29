#!/usr/bin/env bash
# =============================================================================
# runPipeline_dHCP_EPI_POCR.bash: part of nano-eptk package.
#
# Run the dHCP EPI pipeline based on Phase-Only Convection-Reaction method 
# (Gurler and Ider, 2017).
#
# Required arguments:
#  - Input magnitude and phase images (NIFTI format).
#  - Output conductivity.
#
# Configuration is set in the USER INPUTS block below.
# =============================================================================

set -euo pipefail

# =============================================================================
# USER INPUTS
# Data and configuration directories, relative to app root
data_dir="data/dHCP/example_subject/EPI"
cfg_dir="cfg"

# Input nifti files, relative to data_dir
mag="mag.nii.gz"
pha="pha.nii.gz"
mask="derivatives/mask.nii.gz"
dhcp_labels9="derivatives/dhcp_labels9.nii.gz"

# Output nifti and report, relative to data_dir
sig="ep_recon_pocr/sig.nii.gz" 
ep_metric="ep_recon_pocr/ep_metrics.json"

# Configuration file, relative to cfg_dir
cfg="dhcp_epi_pocr.yaml"

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
[[ -e "${HOST_DIR}/${cfg_dir}" ]] || die "data not found: ${HOST_DIR}/${cfg_dir}"

# =============================================================================
# Build command
# =============================================================================
app_cmd=(
  dhcp_epi_pocr \
  --mag "${DOCKER_DIR}/${data_dir}/${mag}"
  --pha "${DOCKER_DIR}/${data_dir}/${pha}"
  --mask "${DOCKER_DIR}/${data_dir}/${mask}"
  --sig "${DOCKER_DIR}/${data_dir}/${sig}"
  --ep_metric "${DOCKER_DIR}/${data_dir}/${ep_metric}"
  --dhcp_labels9 "${DOCKER_DIR}/${data_dir}/${dhcp_labels9}"
  --cfg "${DOCKER_DIR}/${cfg_dir}/${cfg}"
)

docker_flags=( 
  --rm 
  --user "$(id -u):$(id -g)"
  --volume "${HOST_DIR}/${data_dir}:${DOCKER_DIR}/${data_dir}"
  --volume "${HOST_DIR}/${cfg_dir}:${DOCKER_DIR}/${cfg_dir}"
)
[[ -t 0 && -t 1 ]] && docker_flags+=( -it )

# =============================================================================
# Run Docker application
# =============================================================================
echo "Running dHCP EPI POCR pipeline via ${DOCKER_IMG} ..."

docker run \
  "${docker_flags[@]}" \
  "${DOCKER_IMG}" \
  "${app_cmd[@]}"
  
echo "Done!"
