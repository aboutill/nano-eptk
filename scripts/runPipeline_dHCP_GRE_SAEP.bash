#!/usr/bin/env bash
# =============================================================================
# runPipeline_dHCP_GRE_SAEP.bash: part of nano-eptk package.
#
# Run the dHCP GRE pipeline based on Single-Acquisition Electrical Properties 
# (SAEP) method (Marques et al., 2015).
#
# Required arguments:
#  - Input magnitude and phase images (NIFTI format).
#  - Output conductivity and permittivity.
#
# =============================================================================


# =============================================================================
# USER INPUTS
# =============================================================================
# Required arguments
# Host directories
host_workdir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
cfg_dir="cfg"
data_dir="data/dHCP/example_subject/GRE" # in host_workdir

# Input images (in data_dir)
mag="mag.nii.gz"
pha="pha.nii.gz"
mask="derivatives/mask.nii.gz"
dhcp_labels9="derivatives/dhcp_labels9.nii.gz"

# Input configuration (in cfg_dir)
cfg="dhcp_gre_saep.yaml"

# Output images (in data_dir)
sig="ep_recon_saep/sig.nii.gz" 
eps="ep_recon_saep/eps.nii.gz"

# Output report (in data_dir)
ep_metric="ep_recon_saep/ep_metrics.json"


# =============================================================================
# Run Docker application
# =============================================================================
# Docker application image name
docker_img="aboutill/nano-eptk:latest"

# Docker directory
docker_workdir="/usr/src/app"

# SAEP command
cmd="dhcp_gre_saep \
--mag ${docker_workdir}/${data_dir}/${mag} \
--pha ${docker_workdir}/${data_dir}/${pha} \
--mask ${docker_workdir}/${data_dir}/${mask} \
--sig ${docker_workdir}/${data_dir}/${sig} \
--eps ${docker_workdir}/${data_dir}/${eps} \
--ep_metric ${docker_workdir}/${data_dir}/${ep_metric} \
--dhcp_labels9 ${docker_workdir}/${data_dir}/${dhcp_labels9} \
--cfg ${docker_workdir}/${cfg_dir}/${cfg}"

# Launch docker
docker run \
  --user $(id -u):$(id -g) \
  -v "${host_workdir}/${data_dir}:${docker_workdir}/${data_dir}" \
  -v "${host_workdir}/${cfg_dir}:${docker_workdir}/${cfg_dir}" \
  -it \
  --rm \
  $docker_img \
  $cmd

# Cmd display
echo "Docker container stopped!"

