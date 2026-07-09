# nano-eptk: Electrical Properties Tomography Toolkit

**nano-eptk** provides the processing pipelines, Electrical Properties Tomography (EPT) reconstruction algorithms, and statistical analysis notebooks used to investigate electrical conductivity and its relationship with brain development in neonates.

The package includes:

- **Reconstruction algorithms** for electrical conductivity ($\sigma$) and relative permittivity ($\epsilon_r$).
- **Processing pipelines** for dHCP-style acquisitions (GRE, EPI, TSE).
- **Tabulated EP measurements** ($\sigma$ and $\epsilon_r$) with relevant covariates for each study cohort (neonates, infants, and children).
- **A week-by-week neonatal brain conductivity template.**
- **Phantom data** and **raw scans** from a single neonatal subject.
- **Analysis notebooks** reproducing the statistical results.

*Developed by Arnaud Boutillon — arnaud.boutillon@kcl.ac.uk*

---

## Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Docker (recommended)](#docker-recommended)
  - [Local installation](#local-installation)
- [Usage](#usage)
  - [Running the EPT tools](#running-the-ept-tools)
  - [Running the notebooks](#running-the-notebooks)
- [Citation](#citation)
- [References](#references)
- [Contact](#contact)

---

## Prerequisites

nano-eptk relies on the following neuroimaging packages, which must be available on your system for the reconstruction pipelines to run:

| Software  | Link |
|----------|------|
| MRtrix3 (developer build) | https://mrtrix.readthedocs.io/ |
| FSL | https://fsl.fmrib.ox.ac.uk/ |
| MIRTK | https://mirtk.github.io/ |

Please refer to each project's website for installation instructions. **These dependencies are bundled in the Docker image**, so the Docker route below requires no separate installation.

Python dependencies are listed in `requirements.txt`.

---

## Installation

### Docker (recommended)

The Docker image ships with all required software (MRtrix3, FSL, and MIRTK) and Python dependencies pre-installed:

```bash
docker pull aboutill/nano-eptk:latest
```

### Local installation

Local installation requires MRtrix3, FSL, and MIRTK to be installed separately (see [Prerequisites](#prerequisites)). To install the package and its Python dependencies:

```bash
pip install -r requirements.txt
pip install .
```

---

## Usage

### Running the EPT tools

The reconstruction algorithms and processing pipelines are exposed as command-line tools, available both in Docker and in a local installation.

Start an interactive container:

```bash
docker run -it --rm aboutill/nano-eptk:latest
```

List all available commands:

```bash
nano-eptk
```

Display the help for a specific command — for example, the Single-Acquisition Electrical Properties (SAEP) method (Marques et al., 2015):

```bash
nano-eptk saep -h
```

We provide helper scripts to run the EPT pipelines inside docker:

```bash
./scripts/runPipeline_dHCP_GRE_SAEP.bash
./scripts/runPipeline_dHCP_EPI_POCR.bash
```

### Running MRtrix3, FSL and MIRTK tools

All MRtrix3, FSL and MIRTK command-line tools and utilities are available both in Docker and in a local installation.

For example, display the help for a specific MRtrix3 command:

```bash
mrconvert -h
```

We provide a helper script to run MRtrix3 Graphical User Interface (MRview) inside docker:

```bash
./scripts/runMRview.bash
```

### Running the notebooks

The analysis notebooks can be run locally with:

```bash
jupyter notebook
```

Or inside Docker using the provided helper script:

```bash
./scripts/runNotebooks.bash
```





---

## Citation

If you use nano-eptk in your work, please cite the reference below (Boutillon et al., *Scientific Reports*, 2026).

---

## References

1. A. Boutillon *et al.*, "Radiofrequency electrical conductivity reveals a distinct dimension of early human brain development," *Scientific Reports*, 2026. DOI: [10.1038/s41598-026-61290-3](https://doi.org/10.1038/s41598-026-61290-3)
2. A. Boutillon *et al.*, "Investigating the association between conductivity and brain development over the neonatal period," *ISMRM*, 2024. DOI: [10.58530/2025/3266](https://doi.org/10.58530/2025/3266)

---

## Contact

Arnaud Boutillon — arnaud.boutillon@kcl.ac.uk
