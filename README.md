# MS
This repository contains the material used in my MS thesis.

## Topic of thesis - the abstract
Describe the topic of this thesis.

## Structure
The notebooks for conducting the experiments can be found in MS/notebooks/experiments.
### download_reanalysis_data
Instructions for downloading reanalyse data. Inspired by ECMWF own description.
### notebooks 
Conatins notebooks used to perform experiments.
### sclouds
Source code for project.



## Project enviornment and friend repositories

```bash
# clone code repository and the supplementary material in the same directory.
git clone https://github.com/hannasv/MS.git # code
git clone https://github.com/hannasv/MS-suppl.git # supplementary material
cd MS # move to code repository
conda env create -f environment.yml # create sciclouds enviornment on unix-system
conda activate sciclouds # activate the enviorment
python setup.py install # (develop) install sciclouds as a python package
```
