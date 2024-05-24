# Table of contents

- [**Introduction**](#introduction)
- [**Installation**](#installation)
- [**Quick usage examples**](#quick-usage-examples)
  * [Predicting promoter activity](##predicting-promoter-activity)
  * [Running _in-silico_ mutagenesis](##running-in-silico-mutagenesis)
  * [Plotting results of _in-silico_ mutagenesis](#plotting-results-of-in-silico-mutagenesis)
- [**Output files**](#output-files)
- [**Citation**](#citation)
- [**License**](#license)

# Introduction

# Installation
To install **PARM**, first clone this repository in your machine. Make sure you have conda installed:

```sh
git clone git@github.com:vansteensellab/PARM.git
```

Then, create the environment with all **PARM**'s requirements:

```sh
conda create -n parm -c bioconda -c conda-forge -c nvidia -c pytorch pytorch=2.1.1 biopython=1.78 numpy=1.26.4 pandas=2.2.2 matplotlib=3.7.3 logomaker=0.8 tqdm=4.64.0 seaborn=0.13.0 einops=0.4.1 -y
conda activate parm
```

After this, you can install **PARM** with pip:

```sh
pip install ./PARM
```

You can now check if everything went fine with:
```sh
PARM -h
```


# Quick usage examples

## Predicting promoter activity

## Running _in-silico_ mutagenesis

## Plotting results of _in-silico_ mutagenesis

## Output files

## Citation

## License
