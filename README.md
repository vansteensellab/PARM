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

With the `PARM predict` tool, you can get predictions for any sequence that you want for AGS, HAP1, HCT116, HEK116, HepG2, K562, LNCaP, MCF7, and U2OS cells.

With `PARM mutagenesis`, in addition to simple promoter activity scores, **PARM** can also produce the so-called _in-silico_ mutagenesis plot.
This is useful for predicting which TFs are regulating (activating or repressing) your sequence. (read more on [Running _in-silico_ mutagenesis](#running-in-silico-mutagenesis)).

# Installation

**PARM** can be easily installed with `conda`:

```sh
conda install -c anaconda -c conda-forge -c bioconda -c pytorch parm
```

# Usage examples

## Predicting promoter activity

To predict the promoter activity in K562 of every sequence in a fasta file, run:

```sh
parm predict \
  --input example_data/input.fasta \
  --output output_K562.txt \
  --model pre_trained_models/K562/
```

> Note that you should replace `pre_trained_models/K562/` with the actual path to the pre-trained models available on this page.
> Also, note that a PARM model is composed of five different folds, as each model is trained five times. If you check the content of `pre_trained_models/K562/`,
> you will see the `.parm` files there, one for each fold. Do not rename or change the files there unless you know what you are doing.

The output is a tab-separated file. 
The first and second columns contain information about the sequence (the sequence and its header).
The following column contains the predicted promoter activity for the model you have selected.

For the command line above, you should expect the following result:

| sequence    | header                           | prediction_K562   |
|-------------|----------------------------------|-------------------|
| CTGGGAGG... | CXCR4_chr2:136875708:136875939:- | 2.287095785140991 |
| GCAACTAA... | MED16_chr19:893131:893362:-      | 2.22406268119812  |
| ACGCCCAG... | TERT_chr5:1295135:1295366:-      | 1.993780255317688 |


## Running _in-silico_ mutagenesis

To compute the _in-silico_ mutagenesis for every sequence in a fasta file, run:

```sh
parm mutagenesis \
  --input example_data/input.fasta \
  --output in_silico_mutagenesis_K562 \
  --model pre_trained_models/K562/
```

For every sequence in the input fasta, **PARM** will predict the effect of every possible mutation of every single base pair.
This result is stored as a matrix, where every row is a nucleotide of the original sequence and the columns are A, C, G, and T; the values in the matrix are the predicted mutation effect.
**PARM** uses the mutagenesis matrix to scan for known transcription factor (TF) binding sites. 
(As default, **PARM** uses the core human database from HOCOOMOCOv11 as the motif dataset, which can be changed with the `--motif_database` parameter.)

The output of `PARM mutagenesis` is a directory where, for every sequence, both the mutagenesis matrix (`mutagenesis_*.txt.gz`) and the scanned TF motifs (`hits_*txt.gz`) are stored.

## Plotting results of _in-silico_ mutagenesis

Results of _in-silico_ mutagenesis are more insightful when visualized in the following format:

<p align="center"><img src="misc/CXCR4_chr2:136875708:136875939:-.png" alt="plot example" width="100%"></p>

You can easily see the mutagenesis matrix and all the scanned TF motifs.

To produce such a visualization, you can run:

```sh
parm plot \
  --input in_silico_mutagenesis_K562/CXCR4_chr2:136875708:136875939:-/
```

This will read the mutagenesis matrix and the hits for the sequence `sequence_of_interest` and generate the plot.
By default, **PARM** stored the result plot as a PDF inside the input dir.
This can be changed using optional arguments. 

Run `parm plot --help` for additional help on that.

