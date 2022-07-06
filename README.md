# SWIFT-AI-DS: 
#  Simulating the stratospheric ozone layer: a comprehensive benchmark dataset to learn 24-hour ozone tendencies

Code repository related to the benchmark dataset (ToDo).

## Abstract
ToDo

## Dataset Meta data
### source
Two simulations (each 2.5 years) of the Lagrangian chemistry and transport model ATLAS using the full stratospheric chemistry model. Storage time step: each 24h at 00:00UTC modeltime. Version 2020. Reaction Rates from NASA Chemical Kinetics and Photochemical Data (2015).

## Installation

# conda
We provide an environment file; environment.yml containing the required dependencies. Clone the repo and run the following command in the root of this directory:

conda env create -f environment.yml

## Repository Structure

This repository is organized as follows:

- [data](#real-cool-heading) download the files of this benchmark dataset to this directory. The followings scripts ecpects the data to be downloaded upfront.
- 'tools' contains Python files to open and process the dataset
- 'main.ipynb' provides vanilla code to explore the dataset

## Using the code

All experiments are run with 'main.ipynb'.

## Reproducing figures

Please see the figures README for details on reproducing the paper's figures.

## Cite

If you found this benchmark dataset useful in your research, please consider citing:
<! ---
@misc{mohn2021bdfd,
 author={Helge {Mohn} and Daniel {Kreyling} and Ingo {Wohltmann} and Ralph {Lehmann} and Markus {Rex}},
 title={{Benchmark dataset for 24h tendencies of stratospheric ozone}},
 year={2021},
 doi={10.1594/PANGAEA.939121},
 url={https://doi.org/10.1594/PANGAEA.939121},
 type={data set},
 publisher={PANGAEA}
}
--- !>

## Acknowledgements

The first author was supported by grants from the Helmholtz School for Marine Data Science (MarDATA) (HIDSS-0005) and partly co-funded as part of the PhD program of the Alfred Wegener Institute for Polar and Marine Research.
The authors gratefully acknowledge the Earth System Modelling Project (ESM) for funding this work by providing computing time on the ESM partition of the supercomputer JUWELS at the J\"ulich Supercomputing Centre (JSC).
