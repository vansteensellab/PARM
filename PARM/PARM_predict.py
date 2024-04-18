"""
Copyright 2024 PARM developers
https://github.com/vansteensellab/PARM

This file is part of PARM. PARM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. PARM is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with PARM.
If not, see <http://www.gnu.org/licenses/>.
"""

import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import os
from PARM_utils_load_model import load_PARM
import PARM_misc


def PARM_predict(input, output, model_weights, parm_version):
    """
    Reads the input fasta file and predicts promoter activity scores using the PARM models.
    Writes the output as tab-separated values.
    """
    # Load models
    PARM_misc.log("Loading models", parm_version)
    complete_models = dict()
    for model_weight in model_weights:
        model_name = os.path.basename(model_weight).split(".")[0]
        complete_models["prediction_" + model_name] = load_PARM(model_weight)
    # Iterate over sequences and predict scores
    PARM_misc.log("Making predictions", parm_version)
    output_df = pd.DataFrame()
    for record in SeqIO.parse(input, "fasta"):
        # Initiate output df
        sequence = str(record.seq).upper()
        tmp = pd.DataFrame({"sequence": [sequence], "header": [record.id]})
        # Get predictions for all models
        for model_name, model in complete_models.items():
            tmp[model_name] = get_prediction(sequence, model)
        # Store in output df
        output_df = pd.concat([output_df, tmp], axis=0, ignore_index=True)
    # Write output
    PARM_misc.log("Writing output file", parm_version)
    output_df.to_csv(output, sep="\t", index=False)


def get_prediction(sequence, complete_model):
    """
    Predicts promoter activity score for input sequence
    """
    if torch.cuda.is_available():
        complete_model = complete_model.cuda()
    onehot_fragment = torch.tensor(np.float32(sequence_to_onehot([sequence], L_max = len(sequence)))).permute(
        0, 2, 1
    )
    if torch.cuda.is_available():
        onehot_fragment = onehot_fragment.cuda()
    score = complete_model(onehot_fragment).item()

    return score


def sequence_to_onehot(sequences, L_max, for_mutagenesis=False):
    """
    Transform list of sequences to one hot. Padding is done in the middle, and the padding value is 0.
    Args:
        sequences: (list) of string sequences
        L_max: (int) Max length of sequences. Relevant for padding.

    Returns:
        X_OneHot: (np.array) Array with length (samples, L_max, 4)
    """
    # Define nucleotide to vector
    letter_to_vector = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),
        "C": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]),
        "T": np.array([0.0, 0.0, 0.0, 1.0]),
        "N": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    # get On Hot Encoding
    if for_mutagenesis is False:
        one_hot = []
        for seq in sequences:
            x = np.array([letter_to_vector[s] for s in seq])
            pw = (L_max - x.shape[0]) / 2
            PW = [int(np.ceil(pw)), int(np.floor(pw))]
            one_hot.append(np.pad(x, [PW, [0, 0]], constant_values=0))
        one_hot = np.array(one_hot)
    else:
        # for mutagenesis, the L_max is not set to 600, but is the sequence length, instead
        one_hot = []
        for seq in sequences:
            this_seq_length = len(seq)
            x = np.array([letter_to_vector[s] for s in seq])
            pw = (this_seq_length - x.shape[0]) / 2
            PW = [int(np.ceil(pw)), int(np.floor(pw))]
            one_hot.append(np.pad(x, [PW, [0, 0]], constant_values=0))
        one_hot = np.array(one_hot)
        

    return one_hot
