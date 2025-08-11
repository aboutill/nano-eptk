# nano-eptk: Electrical Properties ToolKit (EPTK)
This package provides the processing pipelines, EPT reconstruction algorithms and statistical analysis notebooks used for investigating electrical conductivity and its relationship with brain development in neonates. It contains tabulated data of all extracted EP measurements (conductivity $\sigma$ and permittivity $\epsilon_r$), along with relevant covariates for each study cohort (neonates, infants and children). In addition, the week-by-week neonatal brain conductivity template is included. 

eptk-nano package by Arnaud Boutillon (arnaud.boutillon@kcl.ac.uk)

## Installation
This package relies on MRtrix (https://mrtrix.readthedocs.io/, developer build), FSL (https://fsl.fmrib.ox.ac.uk/) and MIRTK (https://mirtk.github.io/) softwares. Please refer to their respective websites for proper installation.

To install this package and its Python dependencies:\
`pip install -r requirements.txt`\
`pip install .`
