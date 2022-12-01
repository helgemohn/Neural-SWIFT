# Neural-SWIFT
#  Simulating the stratospheric ozone chemistry
#  Related to SWIFT-AI-DS, a comprehensive benchmark dataset to learn 24-hour ozone tendencies

Code repository related to the benchmark dataset:
Mohn, H., Kreyling, D., Wohltmann, I., Lehmann, R., Rex, M., 2021. Benchmark dataset for 24-hour stratospheric ozone tendencies. https://doi.org/10.1594/PANGAEA.939121

## Abstract
SWIFT-AI-DS is a benchmark dataset that consists of samples that have been derived from two simulation runs (each 2.5 years long) of the chemistry and transport model ATLAS (Wohltmann and Rex, 2009; Wohltmann et al., 2010). This data set of nearly 200 million samples meets the requirements of a labelled data set and is ideally suited for training and testing of a machine learning based surrogate model. 

The dataset covers the entire Earth geographically, but is vertically restricted to the altitudes of the lower to middle stratosphere, for which the SWIFT (Rex et al., 2014; Kreyling et. al, 2017; Wohltmann et al., 2017) approach of 24-hour ozone tendencies can be applied. Applicability was determined in terms of the chemical lifetime of stratospheric ozone, which is a function of solar irradiance and altitude. It can be described by a dynamic upper bound [Kreyling et. Al, 2017]. Within the range where the chemical lifetime is longer than 14 days, ozone is not in quasi-chemical equilibrium. Moreover, this data set focuses on the region of the lower to middle stratosphere because it is the region with the largest contribution to the total ozone column.

State-of-the-art physical process models for stratospheric chemistry require enormous computational time. Our research is focused on developing much faster, yet accurate, surrogate models for computing the 24-hour tendencies of stratospheric ozone. Much faster models of stratospheric ozone provide a new application area such as for climate models. These surrogate models benefit greatly from the methodological and hardware improvements of the last decade. 

Each simulation run uses the full stratospheric chemistry model to solve a system of differential equations involving 47 chemical species and 171 chemical reactions at a very high (<< seconds) and variable temporal resolution. The ATLAS model is driven by ECMWF reanalysis data (either ERA-I or ERA5). The air parcel state has been sampled at a 24-hour time step (00:00 UTC model time). During postprocessing some variables are stored as 24-hour averages, as 24-hour tendencies or as the state at the beginning of the 24-hour time step. The dataset is stored in 12 monthly netCDF-files.

## Dataset Meta data
### source
Chemistry and transport model ATLAS (Wohltmann and Rex, 2009; Wohltmann et al., 2010)
### Simulation time periods
- November 1998 to March 2001
- November 2004 to March 2007
### Variables
See [Description_Variables.pdf](https://github.com/helgemohn/SWIFT-AI-DS/tree/main/Description_Variables.pdf)

## Installation
We provide an environment file environment.yml or requirements.txt containing the required dependencies. Clone the repo and run the following command in the root of this directory:

- conda
`conda env create -f environment.yml`
- pip
```python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Repository Structure

This repository is organized as follows:

- [data](https://github.com/helgemohn/SWIFT-AI-DS/tree/main/data): download the files of this benchmark dataset to this directory. The followings scripts ecpects the data to be downloaded upfront.
- [tools](https://github.com/helgemohn/SWIFT-AI-DS/tree/main/tools):  contains Python files to open and process the dataset
- [main.ipynb](https://github.com/helgemohn/SWIFT-AI-DS/tree/main/main.ipynb): provides vanilla code to explore the dataset

## Using the code

All experiments are run with 'main.ipynb'.

## Cite

If you found this benchmark dataset useful in your research, please consider citing:

@misc{mohn2021bdfd,
 author={Helge {Mohn} and Daniel {Kreyling} and Ingo {Wohltmann} and Ralph {Lehmann} and Markus {Rex}},
 title={{Benchmark dataset for 24-hour tendencies of stratospheric ozone}},
 year={2021},
 doi={10.1594/PANGAEA.939121},
 url={https://doi.org/10.1594/PANGAEA.939121},
 type={data set},
 publisher={PANGAEA}
}

## Acknowledgements

The first author was supported by grants from the Helmholtz School for Marine Data Science (MarDATA) (HIDSS-0005) and partly co-funded as part of the PhD program of the Alfred Wegener Institute for Polar and Marine Research.
The authors gratefully acknowledge the Earth System Modelling Project (ESM) for funding this work by providing computing time on the ESM partition of the supercomputer JUWELS at the Jülich Supercomputing Centre (JSC).
