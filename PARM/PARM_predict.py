
import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import os
from .PARM_utils_load_model import load_PARM
from .PARM_misc import log
from tqdm import tqdm


def PARM_predict(input : str,
                 output : str, 
                 model_weights : list,
                 n_seqs_per_batch : int = 1,
                 store_sequence : bool = True,
                 filter_size : int = 125):
    """
    Reads the input (fasta file) and predicts promoter activity scores using the PARM models.
    Writes the output as tab-separated values, where each column is a model and each row is a sequence.
    
    Parameters
    ----------
    input : str
        Path to the input fasta file.
    output : str
        Path to the output file.
    model_weights : list
        List of paths to the PARM model weights. This should be a list even if there is only one model.
    n_seqs_per_batch : int
        Number of batches to use for prediction. If your GPUs runs out of memory, might be because of that. Default is 1.
    
    store_sequence : bool
        If True, the output file will contain the sequences and headers. Otherwise, only the headers and score will be saved. Default is True.
        
    Returns
    -------
    None
    
    Examples
    --------
    >>> PARM_predict("input.fasta", "output.tsv", ["model1.parm", "model2.parm"])
    """
    # Load models
    log("Loading models")
    complete_models = dict()
    for model_weight in model_weights:
        model_name = os.path.basename(model_weight).split(".")[0]
        complete_models["prediction_" + model_name] = load_PARM(model_weight, filter_size = filter_size)
    # Iterate over sequences and predict scores
    log("Making predictions")
    total_sequences = sum(1 for _ in SeqIO.parse(input, "fasta"))
    log(f"Total sequences: {total_sequences}")
    total_interactions = total_sequences * len(
        complete_models
    )
    pbar = tqdm(total=int(total_interactions/n_seqs_per_batch), ncols=80)

    i = 0
    for i_record, record in enumerate(SeqIO.parse(input, "fasta")):
        # Initiate output df
        sequence = str(record.seq).upper()
        if i ==0: tmp = pd.DataFrame({"sequence": [sequence], "header": [record.id]})
        else: tmp = pd.concat([tmp, pd.DataFrame({"sequence": [sequence], "header": [record.id]})])
        
        # Get predictions for all models
        if (i+1) == n_seqs_per_batch or (i_record == (total_sequences-1)):
            for model_name, model in complete_models.items():
                tmp[model_name] = get_prediction(tmp.sequence.to_list(), model)
                pbar.update(1)

            # Store in output df
            #IF it's the first batch, save the df with headers, otherwise, save only the scores
            if i_record < n_seqs_per_batch: 
                if store_sequence: tmp.to_csv(output, sep="\t", index=False)
                else: (tmp.drop(columns=["sequence"])).to_csv(output, sep="\t", index=False)
            else:
                if store_sequence: tmp.to_csv(output, sep="\t", index=False, mode='a', header=False)
                else: (tmp.drop(columns=["sequence"])).to_csv(output, sep="\t", index=False, mode='a', header=False)

            i = 0
            
            
        else: i += 1
    # Write output
    pbar.close()
    log("Finish output file")


def get_prediction(sequence, complete_model):
    """
    Predicts promoter activity score for input sequence
    """
    #Check if sequence is a list or not and make it a list if not
    if not isinstance(sequence, list): sequence = [sequence]

    if torch.cuda.is_available():
        complete_model = complete_model.cuda()
    onehot_fragment = torch.tensor(
        np.float32(sequence_to_onehot(sequence, L_max=len(sequence[0])))
    ).permute(0, 2, 1)
    if torch.cuda.is_available():
        onehot_fragment = onehot_fragment.cuda()
    score = complete_model(onehot_fragment).cpu().detach().numpy()
    
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
