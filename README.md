# Table of contents

- [**Introduction**](#introduction)
- [**Installation**](#installation)
- [**Quick usage examples**](#quick-usage-examples)
  * [Predicting promoter activity](#predicting-promoter-activity)
  * [Running _in-silico_ mutagenesis](#running-in-silico-mutagenesis)
  * [Plotting results of _in-silico_ mutagenesis](#plotting-results-of-in-silico-mutagenesis)

# Introduction

PARM (Promoter Activity Regulatory Model) is a deep learning model that predicts the promoter activity from the DNA sequence itself.
As a convolution neural network trained on MPRA data, **PARM** is very lightweight and produces predictions in a cell-type-specific manner.

With the `PARM predict` tool, you can get predictions for any sequence that you want for K562, HepG2, MCF7, LNCaP, or HCT116 cells. 

With `PARM mutagenesis`, in addition to simple promoter activity scores, **PARM** can also produce the so-called _in-silico_ mutagenesis plot.
This is useful for predicting which TFs are regulating (activating or repressing) your sequence. (read more on [Running _in-silico_ mutagenesis](#running-in-silico-mutagenesis)).

# Installation

> Soon, **PARM** will be available as a Bioconda package and these steps won't be necessary. 

To install **PARM**, first clone this repository in your machine. Make sure you have conda installed:

```sh
git clone git@github.com:vansteensellab/PARM.git
```

Then, create the environment with all **PARM**'s requirements:

```sh
conda create -n parm -c bioconda -c conda-forge -c nvidia -c pytorch pytorch=2.1.1 biopython=1.78 numpy=1.26.4 pandas=2.2.2 matplotlib=3.7.3 logomaker=0.8 tqdm=4.64.0 seaborn=0.13.0 einops=0.4.1 optuna=3.6.0 scikit-lear=1.3.0 pytorch-model-summary=0.1.1 h5py=3.11.0 -y
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

# Usage examples

## Predicting promoter activity

To predict the promoter activity in K562 of every sequence in a fasta file, run:

```sh
parm predict \
  --input example_data/input.fasta \
  --output output_K562.txt \
  --model pre_trained_models/K562.parm
```

> Note that you should replace `pre_trained_models/K562.parm` with the actual path to the pre-trained models available on this page.

To perform predictions for more than one cell, you can simply provide all the paths separated by space:

```sh
parm predict \
  --input example_data/input.fasta \
  --output output_K562_HepG2_LNCaP.txt \
  --model pre_trained_models/K562.parm pre_trained_models/HepG2.parm pre_trained_models/LNCaP.parm
```

The output is a tab-separated file. 
The first and second columns contain information about the sequence (the sequence and its header).
The following column contains the predicted promoter activity for the model you have selected. 
If you performed predictions for more than one cell, more than one column will be created here.

For the command line above, you should expect the following result:

```
sequence	header	prediction_K562	prediction_HepG2	prediction_LNCaP
CTGGGAGG...	CXCR4_chr2:136875708:136875939:-	2.287095785140991	1.4889564514160156	0.2345067262649536
GCAACTAA...	MED16_chr19:893131:893362:-	2.22406268119812	2.6182565689086914	0.30299943685531616
ACGCCCAG...	TERT_chr5:1295135:1295366:-	1.993780255317688	1.474591612815857	0.11847741901874542

```

## Running _in-silico_ mutagenesis

To compute the _in-silico_ mutagenesis for every sequence in a fasta file, run:

```sh
parm mutagenesis \
  --input example_data/input.fasta \
  --output in_silico_mutagenesis_K562 \
  --model pre_trained_models/K562.parm
```

You can also run `PARM mutagenesis` for more than one cell:

```sh
parm mutagenesis \
  --input input.fasta \
  --output in_silico_mutagenesis_K562_HepG2_LNCaP \
  --model pre_trained_models/K562.parm pre_trained_models/HepG2.parm pre_trained_models/LNCaP.parm
```

For every sequence in the input fasta, **PARM** will predict the effect of every possible mutation of every single base pair.
This result is stored as a matrix, where every row is a nucleotide of the original sequence and the columns are A, C, G, and T; the values in the matrix are the predicted mutation effect.
**PARM** uses the mutagenesis matrix to scan for known transcription factor (TF) binding sites. 
(As default, **PARM** uses the core human database from HOCOOMOCOv11 as the motif dataset, but this can be changed with the `--motif_database` parameter.)

The output of `PARM mutagenesis` is a directory where, for every sequence, both the mutagenesis matrix (`mutagenesis_*.txt.gz`) and the scanned TF motifs (`hits_*txt.gz`) are stored.

## Plotting results of _in-silico_ mutagenesis

Results of _in-silico_ mutagenesis are more insightful when visualized in the following format:

<p align="center"><img src="misc/CXCR4_chr2:136875708:136875939:-.png" alt="plot example" width="100%"></p>

Where you can easily see the mutagenesis matrix plus all the scanned TF motifs.

To produce such a visualization, you can run:

```sh
parm plot \
  --input in_silico_mutagenesis_K562/CXCR4_chr2:136875708:136875939:-/
```

This will read the mutagenesis matrix and the hits for the sequence called `sequence_of_interest` and generate the plot.
By default, **PARM** stored the result plot as a PDF inside the input dir.
This can be changed using optional arguments. 
Run `parm plot --help` for additional help on that.

