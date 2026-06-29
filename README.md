# nano-eptk: Electrical properties tomography ToolKit (EPTK)
This package provides the processing pipelines, Electrical Properties Tomography (EPT) reconstruction algorithms and statistical analysis notebooks used for investigating electrical conductivity and its relationship with brain development in neonates. It contains tabulated data of all extracted EP measurements (conductivity $\sigma$ and permittivity $\epsilon_r$), along with relevant covariates for each study cohort (neonates, infants and children). Additionally, it includes the week-by-week neonatal brain conductivity template, phantom data, and raw scans from a single neonatal subject.

nano-eptk package by Arnaud Boutillon (arnaud.boutillon@kcl.ac.uk)

## Installation
This package relies on MRtrix (https://mrtrix.readthedocs.io/, developer build), FSL (https://fsl.fmrib.ox.ac.uk/) and MIRTK (https://mirtk.github.io/) softwares. Please refer to their respective websites for proper installation.

To install this package and its Python dependencies:\
`pip install -r requirements.txt`\
`pip install .`

## References
A. Boutillon et al. "Radiofrequency electrical conductivity reveals a distinct dimension of early human brain development", Preprint
A. Boutillon et al. "Investigating the association between conductivity and brain development over the neonatal period", ISMRM 2024, DOI: https://doi.org/10.58530/2025/3266
