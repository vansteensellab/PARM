import torch
import numpy as np
from Bio import SeqIO
from Bio import motifs
import pandas as pd
import os
import io
import urllib
from .PARM_utils_load_model import load_PARM
from .PARM_misc import log
from tqdm import tqdm
from matplotlib import pyplot as plt, colors
import matplotlib
import logomaker
import seaborn as sns
from pathlib import Path


def PARM_mutagenesis(
    input: str,
    model_weights: list,
    motif_database: str,
    output_directory: str,
):
    """
    Function to execute the in-silico mutageesis of a sequence using the PARM models.

    For each sequence in the fasta, it writes three outputs: the mutagenesis matrix,
    the motif hits.

    Parameters
    ----------
    input : str
        Path to the input fasta file.
    model_weights : list
        List of paths to the PARM models. This should be a list even if there's only one model.
    motif_database : str
        Path to the motif database. Usually this is the HOCOMOCO database.
    output_directory : str
        Path to the output directory where the results will be saved.

    Returns
    -------
    None

    Examples
    --------
    >>> PARM_mutagenesis("input.fasta", ["model1.parm", "model2.parm"], "motif_database.txt", "output")

    """
    parm_scores = dict()
    # Loading motif database
    log("Loading motif database")
    PFM_hocomoco_dict, _, _ = dict_jaspar(file=motif_database, reverse=True)
    # Loading models
    complete_models = dict()
    for model in model_weights:
        model_name = os.path.basename(model).split(".parm")[0]
        log(f"Loading model {model_name}")
        complete_models[model_name] = load_PARM(model)
        parm_scores[model_name] = dict()

    # ====================================================================================
    # Parsing the fasta file =============================================================
    # ====================================================================================
    log("Computing saturation mutagenesis")
    total_interactions = len(list(SeqIO.parse(input, "fasta"))) * len(model_weights)
    pbar = tqdm(total=total_interactions, ncols=80)
    for record in SeqIO.parse(input, "fasta"):
        sequence_ID = record.id
        sequence = str(record.seq).upper()
        for model_name, model in complete_models.items():
            mutagenesis_data = os.path.join(
                output_directory, sequence_ID, "mutagenesis_" + sequence_ID + ".txt.gz"
            )
            if os.path.exists(mutagenesis_data):
                log(
                    f"WARNING: mutagenesis data for {sequence_ID} already exist. Skipping..."
                )
            else:
                create_dataframe_mutation_effect(
                    name=sequence_ID,
                    strand="+",
                    model=model,
                    output_directory=output_directory,
                    PFM_hocomoco_dict=PFM_hocomoco_dict,
                    L_max=600,
                    reverse=False,
                    seq=sequence,
                    type_motif_scanning="PCC",
                )
            pbar.update(1)
        #     # save the parm score of this sequence to the predictions dict
        #     # -- this will be used later to have the predictions in the title of
        #     #    the plot
        #     parm_scores[model_name][sequence_ID] = parm_score
        #     # Now plot the data
        #     fig, _ = plot_mutagenesis_results(
        #         sequence=sequence,
        #         mutagenesis_df=mutagenesis_df,
        #         motif_scanning_results=motif_scanning_results,
        #         sequence_onehot=sequence_onehot,
        #         sequence_score=parm_score,
        #         model_name=model_name,
        #         motif_db_ICT=motif_db_ICT,
        #     )
        #     output_pdf.savefig(fig)
        #     pbar.update(1)
        # output_pdf.close()


def dict_jaspar(
    file: str = "https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt",
    reverse: bool = False,
):
    """
    Load all hocomoco motifs in a dictionary format from jaspar file.

    Parameters
    ----------
    file : str
        Path to the motif database. Usually this is the HOCOMOCO database.
    reverse : bool
        If True, the reverse complement of the motif will be computed.

    Returns
    -------
    PFM_hocomoco_dict : dict
        Dictionary with the PFM of the motifs.
    consensus_hocomoco_dict : dict
        Dictionary with the consensus sequence of the motifs. Keys are the motif names.
    ICT_hocomoco_dict : dict
        Dictionary with the Information Content of the motifs. Keys are the motif names.

    Examples
    --------
    >>> PFM_hocomoco_dict, consensus_hocomoco_dict, ICT_hocomoco_dict = dict_jaspar("motif_database.txt", reverse=True)
    """

    if "https" in file:
        handle = urllib.request.urlopen(file)
        hocomoco_jaspar = io.TextIOWrapper(handle, encoding="utf-8")

    else:
        hocomoco_jaspar = open(file)

    PWM_hocomoco_dict = dict()
    PSSM_hocomoco_dict = dict()
    PFM_hocomoco_dict = dict()
    ICT_hocomoco_dict = dict()
    consensus_hocomoco_dict = dict()

    # Loop through all motifs
    for m in motifs.parse(hocomoco_jaspar, "jaspar"):
        PWM = np.array([m.pwm[i] for i in m.pwm.keys()])
        PWM_hocomoco_dict[m.name] = PWM

        pssm = np.array([m.pssm[i] for i in m.pssm.keys()])
        pssm[np.isinf(pssm)] = pssm[~np.isinf(pssm)].min()
        PSSM_hocomoco_dict[m.name] = pssm
        counts = m.counts

        PFM = np.array([counts[i] for i in counts.keys()])
        PFM = PFM / PFM.sum(axis=0)
        PFM_hocomoco_dict[m.name] = PFM
        consensus_hocomoco_dict[m.name] = str(m.consensus)

        ICTtotal = np.log2(4)
        ICT = ICTtotal - np.sum((-PFM * np.log2(PFM + 0.00001)), axis=0)
        ICT_hocomoco_dict[m.name] = PFM * ICT

    # Compute reverse of all motifs
    if reverse:
        reverse_DNA = "".maketrans("ACGT", "TGCA")
        tf_ids = list(PFM_hocomoco_dict.keys())

        for PFM_name in tf_ids:
            motif_PWM = PFM_hocomoco_dict[PFM_name]
            PFM_hocomoco_dict[f"{PFM_name}-"] = np.flip(motif_PWM, (0, 1))

            motif_PWM = ICT_hocomoco_dict[PFM_name]
            ICT_hocomoco_dict[f"{PFM_name}-"] = np.flip(motif_PWM, (0, 1))

            cons_motif = consensus_hocomoco_dict[PFM_name]

            rev_motif = cons_motif.translate(reverse_DNA)[::-1]
            consensus_hocomoco_dict[f"{PFM_name}-"] = rev_motif

    return (PFM_hocomoco_dict, consensus_hocomoco_dict, ICT_hocomoco_dict)


def predict_fragments(seq: str, L_max: int, completemodel: torch.nn.Module):
    """
    Predicts the promoter activity score for input fragment
    Converts the sequence to one hot and then predicts the score.

    Parameters
    ----------
    seq : str
        Sequence to predict.
    L_max : int
        Max length of sequence. Necessary for padding.
    completemodel : torch.nn.Module
        Model used to compute attributions.

    Returns
    -------
    score : float
        Predicted score for that sequence.

    Examples
    --------
    >>> predict_fragments("ATCG", 600, model)

    """
    if torch.cuda.is_available():
        completemodel = completemodel.cuda()

    onehot_fragment = torch.tensor(np.float32(get_one_hot([seq], L_max))).permute(
        0, 2, 1
    )
    if torch.cuda.is_available():
        onehot_fragment = onehot_fragment.cuda()

    score = completemodel(onehot_fragment).item()

    return score


def get_one_hot(
    Seqs: list, L_max: int, padding: str = "middle", padding_value: int = 0
):
    """
    Transform list of sequences to one hot.

    Parameters
    ----------
    Seqs : list
        List of string sequences. This should be a list even if there's only one sequence.
    L_max : int
        Max length of sequences. Sequences smaller than this will be padded.
    padding : str
        Padding type. Options are 'middle', 'right', 'left'.
    padding_value : int
        Value to pad the sequences.

    Returns
    -------
    X_OneHot : np.array
        One-hot enconded array with shape (samples, L_max, 4).

    Examples
    --------
    >>> get_one_hot(["ATCG"], 600)
    """
    # Define nucleotide to vector
    letter2vector = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),
        "C": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]),
        "T": np.array([0.0, 0.0, 0.0, 1.0]),
        "N": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    # get On Hot Encoding
    X_OneHot = []
    for seq in Seqs:

        x = np.array([letter2vector[s] for s in seq])
        diff = L_max - x.shape[0]
        pw = (L_max - x.shape[0]) / 2
        PW = [int(np.ceil(pw)), int(np.floor(pw))]

        if padding == "middle":

            X_OneHot.append(np.pad(x, [PW, [0, 0]], constant_values=padding_value))

        elif padding == "right":
            X_OneHot.append(
                np.pad(x, [[0, diff], [0, 0]], constant_values=padding_value)
            )

        elif padding == "left":
            X_OneHot.append(
                np.pad(x, [[diff, 0], [0, 0]], constant_values=padding_value)
            )

        else:
            raise Exception(f"Padding option not recognised: {padding}")

    X_OneHot = np.array(X_OneHot)

    return X_OneHot


def motif_attribution(
    seq: str,
    L_max: int,
    start_motif: int,
    end_motif: int,
    completemodel: torch.nn.Module,
    ref_to_alt_attribution: bool = False,
    return_reference_score: bool = False,
    index_output: int = 0,
    alt_nt: list = ["A", "C", "G", "T"],
    window_del: bool = False,
):
    """
    Computes saturation mutagenesis of a given region of a sequence (or the complete one) to determine
        importance (or attribution) by the model.

    Basically it mutates every base to every other possible base.

    If ref_to_alt_attribution == False: The importance of each base is determined by the score of that base - the mean score of the other possible 3 bases.

    If ref_to_alt_attribution == True: The importance of each base is determined by the REF - score of that base.


    Parameters
    ----------
    seq : str or list
        Sequence (str) or list of sequences to study.
    L_max : int
        Input length sequence of the model. Necessary for padding.
    start_motif : int
        Start of the motif of interest, relative to seq.
    end_motif : int
        End of the motif of interest, relative to seq.
    completemodel : torch.nn.Module
        Model used to compute attributions.
    ref_to_alt_attribution : bool
        If True, the importance of each base is determined by the REF - score of that base.
    return_reference_score : bool
        If True, returns the prediction of the reference sequence.
    index_output : int
        Which index of the output to take, usually if the model only outputs one cell line and/or replicates there's only one output value and index is 0.
    alt_nt : list
        List of possible nucleotides to mutate to in a list. --> ['A']
    window_del : bool
        If it is not False, provide a list of integer(s) e.g. [1], or [1,2]. It will compute the effect of a deletion of the different sizes provided in the list.

    Returns
    -------
    att : torch.Tensor
        Array of shape (n_bases, length_motif).
    ref_score : int
        Returns prediction of Reference Sequence.

    Examples
    --------
    >>> seq = "AGCTAGCTAGCTAGCTTAGC"
    >>> motif_attribution(seq, 600, 0, 4, model)
    """

    # If seq is a list, we have multiple sequences or index ouputs
    if type(seq) != list:
        seq = [seq]
    if type(start_motif) != list:
        start_motif = [start_motif]
    if type(end_motif) != list:
        end_motif = [end_motif]
    if type(index_output) != list:
        index_output = [index_output]

    # If the start_motif is a list, it should have the same length as seq
    if len(seq) != len(start_motif) or len(seq) != len(end_motif):
        raise Exception("Length of start_motif should be the same as the length of seq")
    # Check if all motifs have the same length
    if len(set([end_motif[i] - start_motif[i] for i in range(len(seq))])) > 1:
        raise Exception("All motifs should have the same length")
    size_motifs = end_motif[0] - start_motif[0]

    # if thee window_del is not empty, we should also include deletions
    ## This new list will contains the original alt_nt, and de deletion information in format "-,2", were the number indicates the length of the deletion
    if window_del is not False:
        alt_nt = alt_nt + ["-," + str(wind) for wind in window_del]

    if torch.cuda.is_available():
        completemodel = completemodel.cuda()

    # Compute SuRE score of reference sequence
    reference = torch.tensor(np.float32(get_one_hot(seq, L_max))).permute(0, 2, 1)
    if torch.cuda.is_available():
        reference = reference.cuda()

    with torch.no_grad():
        if len(seq) == 1 and len(index_output) == 1:
            sure_score = completemodel(reference)[0, index_output].detach().cpu()
        else:
            sure_score = completemodel(reference)[:, index_output].detach().cpu()

    del reference  # Free memory

    # Create empty dataframe and fill it
    attribution = torch.zeros(len(alt_nt), size_motifs)

    # Store all possible mutated sequences
    seqs_position_mutation = []

    for it_seq, unique_seq in enumerate(seq):  # Loop through sequences
        for pos in range(
            start_motif[it_seq], end_motif[it_seq]
        ):  # Loop through all the positions of the motif
            seq_mutated = list(unique_seq)

            for ALT in alt_nt:  # Iterate through all possible nucleotides
                if "-" in ALT:  # If there's a deletion
                    del_size = int(ALT.split(",")[-1])
                    round_min = np.max([pos - int(np.floor(del_size / 2)), 0])
                    round_max = pos + int(np.ceil(del_size / 2))

                    ALT_seq_left = seq_mutated[:(round_min)]  # Check left sequence
                    ALT_seq_right = seq_mutated[(round_max):]  # Check right sequence
                    ALT_seq = "".join(ALT_seq_left + ALT_seq_right)
                    seqs_position_mutation.append(ALT_seq)

                else:  # A pair substitution
                    seq_mutated[pos] = ALT
                    ALT_seq = "".join(seq_mutated)
                    seqs_position_mutation.append(ALT_seq)

            # The sequences are going to be saved for each position the four nucleotides,
            # so if we compute all the possible sequences, and the model returns the predictions
            # in a flatten array we have to reconvert the array in shape base x nucleotides.

    with torch.no_grad():
        alt_reference = torch.tensor(
            np.float32(get_one_hot(seqs_position_mutation, L_max))
        ).permute(0, 2, 1)
        if torch.cuda.is_available():
            alt_reference = alt_reference.cuda()
        # We need to reshape it and then transpose it otherwise it gets wrong order
        att_per_index = []
        for idx in index_output:  # Loop if there's more than one index output
            attribution = torch.reshape(
                completemodel(alt_reference)[:, idx].detach().cpu().flatten(),
                (len(seq), size_motifs, len(alt_nt)),
            ).transpose(1, 2)
            att_per_index.append(attribution)

        del alt_reference

    if ref_to_alt_attribution:
        # If we are interested in having the REF-ALT of all bases,
        att = []
        for idx_output, attribution in enumerate(
            att_per_index
        ):  # Loop if there's more than one index output
            if it_seq != 0:
                att_per_seq = []
                for it_seq in range(len(seq)):  # Loop through sequence
                    att_per_seq.append(
                        sure_score[it_seq, idx_output] - attribution[it_seq, :, :]
                    )

                att.append(att_per_seq)
            else:
                att = sure_score[idx_output] - attribution[0, :, :]

    else:  # If we are interested in the contribution of each base as REF-mean(ALT). --> Not done for deletion

        att = []
        for idx_output, attribution in enumerate(
            att_per_index
        ):  # Loop if there's more than one index output
            att_seq = []
            for it_seq in range(len(seq)):  # Loop through sequence
                attribution_seq = attribution[it_seq, :, :]

                if window_del is not False:
                    index_no_del = np.where([sub != "-" for sub in alt_nt])[
                        0
                    ]  # Check indiced where there's no deletion
                    attribution_seq = attribution_seq[index_no_del, :]

                idx = list(range(attribution_seq.shape[0]))
                att_seq.append(
                    torch.stack(
                        [
                            attribution_seq[x]
                            - attribution_seq[idx[:x] + idx[x + 1 :]].mean(axis=0)
                            for x in idx
                        ]
                    )
                )
            att.append(att_seq)

    att = np.squeeze(np.array(att))  # Remove one size dimensions

    ##Free space
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(sure_score) == 1:
        sure_score = sure_score[0]
    if return_reference_score:
        return (att, sure_score)
    return att


def peaks_scanning(
    attribution: np.array,
    length_peaks: int = 5,
    number_vocab: int = 4,
    append: bool = True,
):
    """
    Finds four different types of peaks:
        (1) Gain_of_activator: Only one possible mutation (e.g. A-->T) has a effect on having HIGHER promoter activity, not the other possible mutations or the surronding nucleotides.
        (2) Gain_of_repressor: Only one possible mutation (e.g. A-->T) has a effect on having LOWER promoter activity, not the other possible mutations or the surronding nucleotides.
        (3) Loss_of_activator: All possible mutations (e.g. A-->T,C,G) has a effect on having LOWER promoter activity, as well as the mutations around  that base.
        (4) Loss_of_repressor: All possible mutations (e.g. A-->T,C,G) has a effect on having HIGHER promoter activity, as well as the mutations around  that base.


    Parameters
    ----------
    attribution : np.array
        Array of shape (n_bases, length_motif).
    length_peaks : int
        Length of the peaks.
    number_vocab : int
        Number of possible nucleotides.
    append : bool
        If True, the peaks will be appended with zeros.
        
    Returns
    -------
    dict_scores : dict
        Dictionary with the scores of the peaks.
        
    Examples
    --------
    >>> peaks_scanning(attribution, 5, 4, True)

    """

    if append:
        zeros_padding = np.zeros(shape=(4, 5))
        attribution = np.concatenate(
            [zeros_padding, attribution, zeros_padding], axis=1
        )

    mid_peak = int(length_peaks / 2)

    # Create Filter to scan for gain of activator (1)
    filter_gain_of_activator = np.ones((number_vocab, length_peaks), dtype=int)
    filter_gain_of_activator[0, mid_peak] = (
        -1
    ) * length_peaks  # here we assume that the peak is in the 1st base (A), so we will have to change it to the other possible bases later

    ## Attribution matrix will have to be all negative

    # Filter to scan for gain of activator (2)
    filter_gain_of_repressor = -filter_gain_of_activator
    ## Attribution matrix will have to be all positive

    # Filter to scan for loss of activator (3)
    filter_loss_of_activator = np.ones((number_vocab, length_peaks), dtype=int)

    # Filter to scan for loss of repressor (4)
    filter_loss_of_repressor = -filter_loss_of_activator

    ## START SCANNING

    (
        score_loss_of_activator,
        score_loss_of_repressor,
        score_gain_of_repressor,
        score_gain_of_activator,
    ) = ([], [], [], [])

    # Loop through sequence length
    for start_peak in range(attribution.shape[1] - length_peaks + 1):
        # Define where the end of the peak should be
        end_peak = start_peak + length_peaks
        attribution_position = attribution[:, start_peak:end_peak]

        ##Now we have to do the four possible scanns
        # (1) GAIN OF ACTIVATOR
        attribution_position_gain_activator = np.minimum(attribution_position, 0)

        score_gain_of_activator_base = np.mean(
            attribution_position_gain_activator * filter_gain_of_activator
        )

        for nt_pos in range(1, number_vocab + 1):
            # The peak might be in any possible base, so we have to permute the filter position on the nucleotides axis and do the 3 possible remaining scannns

            order = (
                list(range(number_vocab))[nt_pos:]
                + list(range(number_vocab))[:(nt_pos)]
            )

            att_score = np.mean(
                attribution_position_gain_activator * filter_gain_of_activator[order, :]
            )

            if score_gain_of_activator_base < att_score:
                score_gain_of_activator_base = (
                    att_score  # Take the one that maximizes the score
                )

        score_gain_of_activator.append(score_gain_of_activator_base)

        # (2) GAIN OF REPRESSOR

        attribution_position_gain_repressor = np.maximum(attribution_position, 0)

        score_gain_of_repressor_base = np.mean(
            attribution_position_gain_repressor * filter_gain_of_repressor
        )
        for nt_pos in range(1, number_vocab + 1):
            # The peak might be in any possible base, so we have to permute the filter position on the nucleotides axis and do the 3 possible remaining scannns

            order = (
                list(range(number_vocab))[nt_pos:]
                + list(range(number_vocab))[:(nt_pos)]
            )

            att_score = np.mean(
                attribution_position_gain_repressor * filter_gain_of_repressor[order, :]
            )

            if score_gain_of_repressor_base < att_score:
                score_gain_of_repressor_base = (
                    att_score  # Take the one that maximizes the score
                )

        score_gain_of_repressor.append(score_gain_of_repressor_base)

        # (3) LOSS OF ACTIVATOR
        score_loss_of_activator.append(
            np.mean(attribution_position * filter_loss_of_activator)
        )

    score_loss_of_activator = np.array(score_loss_of_activator)
    # (4) LOSS OF REPRESSOR
    ## Its the same but the opposite of activator
    score_loss_of_repressor = -(score_loss_of_activator)

    def find_unique_peaks_in_region(scores_computed, length_peaks):
        ##To make sure that we find the clearer peaks (center of the motif), and we find a unique peak
        ## instead of high values around
        ## we take the largest value within a small range

        step = int(length_peaks * 2)  # Window to run max_length

        start_possibles = (
            0,
            length_peaks,
        )  # Different starts so it doesnt get stuck in an edge
        for start in start_possibles:
            for ind_start in range(start, len(scores_computed) - step, step):

                # Create index where region of interest is
                index_interest = np.arange(ind_start, (ind_start + step))

                # Obtain max value in that region
                max_score_region = np.max(scores_computed[index_interest])

                # Obtain index of those which don't have the highest value
                region_no_max = index_interest[
                    (scores_computed < max_score_region)[index_interest]
                ]

                scores_computed[region_no_max] = 0

        return scores_computed

    score_loss_of_activator = find_unique_peaks_in_region(
        score_loss_of_activator, length_peaks
    )
    score_loss_of_repressor = find_unique_peaks_in_region(
        score_loss_of_repressor, length_peaks
    )
    score_gain_of_activator = np.array(score_gain_of_activator)
    score_gain_of_repressor = np.array(score_gain_of_repressor)

    if append:
        score_loss_of_repressor = score_loss_of_repressor[5:(-5)]
        score_loss_of_activator = score_loss_of_activator[5:(-5)]
        score_gain_of_activator = score_gain_of_activator[5:(-5)]
        score_gain_of_repressor = score_gain_of_repressor[5:(-5)]

    dict_scores = {
        "score_gain_of_activator": score_gain_of_activator,
        "score_gain_of_repressor": score_gain_of_repressor,
        "score_loss_of_activator": score_loss_of_activator,
        "score_loss_of_repressor": score_loss_of_repressor,
    }

    return dict_scores


def compute_attribution(
    seq: str,
    L_max : int,
    start_motif : int,
    end_motif : int,
    completemodel : torch.nn.Module,
    index_output : int = 0,
    alt_nt : list = ["A", "C", "G", "T"],
    window_del : bool = False,
):
    """
    Computes saturation mutagenesis of a given region of a sequence (or the complete one) to determine
        importance (or attribution) by the model.

    Basically it mutates every base to every other possible base. The importance of each base
    is determined by the SuRE score of that base - the mean SuRE score of the other possible 3 bases.
    
    Parameters
    ----------
    seq : str
        Sequence to study.
    L_max : int
        Input length sequence of the model. Necessary for padding.
    start_motif : int
        Start of the motif of interest, relative to seq.
    end_motif : int
        End of the motif of interest, relative to seq.
    completemodel : torch.nn.Module
        Model used to compute attributions.
    index_output : int
        Which index of the output to take, usually if the model only outputs one cell line and/or replicates there's only one output value and index is 0.
    alt_nt : list
        List of possible nucleotides to mutate to in a list. --> ['A']
    window_del : bool
        If it is not False, provide a list of integer(s) e.g. [1], or [1,2]. It will compute the effect of a deletion of the different sizes provided in the list.

    Returns
    -------
    att : torch.Tensor
        Array of shape (n_bases, length_motif).
        
    Examples
    --------
    >>> compute_attribution("ATCG", 600, 0, 4, model)

    """

    att, ref_score = motif_attribution(
        seq,
        L_max,
        start_motif,
        end_motif,
        completemodel,
        ref_to_alt_attribution=True,
        return_reference_score=True,
        index_output=index_output,
        alt_nt=alt_nt,
        window_del=window_del,
    )

    return (att, ref_score)


def compute_correlation_between_motifs(
    pfm : np.array,
    motif_attribution : np.array, 
    ICM : np.array = False, 
    return_PFM : bool = False
):
    """
    Normalizes motif attribution defined by model, and computes correlation
    between known PFM and normalized motif attribution.

    Parameters
    ----------
    pfm : np.array
        The PFM matrix of the motif database.
    motif_attribution : np.array
        The attribution scores of the motif.
    ICM : np.array
        The Information Content Matrix of the motif database.
    return_PFM : bool
        If True, returns the PFM of the motif.
        
    Returns
    -------
    rho_prob : float
        Correlation score between known PFM and the one of the attribution scores.
        
    Examples
    --------
    >>> compute_correlation_between_motifs(pfm, motif_attribution, ICM, True)
    """
    ## Compute the correlation score between known PPM and the one of the attribution scores

    ## 1. Positive attributions (activators)

    # p_mut (bases, position)
    #  CREATE PSSM (probability)
    p_mut = np.maximum(motif_attribution, 0)  # If negative attribution, set to 0
    p_mut[:, np.sum(p_mut, axis=0) == 0] = (
        0.25  # If in a position all nt are 0, set all of them to 0.25
    )
    p_mut = p_mut / np.sum(p_mut, axis=0, keepdims=True)  # Normalize
    p_mut = p_mut + 0.01  # Add pseudocount
    p_mut_pos = p_mut / np.sum(p_mut, axis=0, keepdims=True)
    rho_pos_prob = np.corrcoef([p_mut.flatten(), pfm.flatten()])[0, 1]
    rho_pos_prob = np.max([rho_pos_prob, 0])

    #  CREATE BITS, INFORMATION CONTENT MATRIX
    if ICM is not False:
        p_mut_IC_pos = np.maximum(
            motif_attribution, 0
        )  # If negative attribution, set to 0
        max = np.sum(p_mut_IC_pos, axis=0).max()
        p_mut_IC_pos = p_mut_IC_pos / max * 2
        rho_pos_ICM = np.corrcoef([p_mut_IC_pos.flatten(), ICM.flatten()])[0, 1]
        rho_pos_ICM = np.max([rho_pos_ICM, 0])

    ## 2. Negative attributions (repressors)

    # p_mut (position, bases)
    p_mut_neg = -np.minimum(motif_attribution, 0)  # If negative attribution, set to 0
    p_mut_neg[:, np.sum(p_mut_neg, axis=0) == 0] = (
        0.25  # If in a position all nt are 0, set all of them to 0.25
    )
    p_mut_neg = p_mut_neg / np.sum(p_mut_neg, axis=0, keepdims=True)  # Normalize
    p_mut_neg = p_mut_neg + 0.01  # Add pseudocount
    p_mut_neg = p_mut_neg / np.sum(p_mut_neg, axis=0, keepdims=True)
    rho_neg_prob = np.corrcoef([p_mut_neg.flatten(), pfm.flatten()])[0, 1]
    rho_neg_prob = np.max([rho_neg_prob, 0])

    #  CREATE BITS, INFORMATION CONTENT MATRIX
    if ICM is not False:
        p_mut_IC_neg = -np.minimum(
            motif_attribution, 0
        )  # If negative attribution, set to 0
        max = np.sum(p_mut_IC_neg, axis=0).max()
        p_mut_IC_neg = p_mut_IC_neg / max * 2
        rho_neg_ICM = np.corrcoef([p_mut_IC_neg.flatten(), ICM.flatten()])[0, 1]
        rho_neg_ICM = np.max([rho_neg_ICM, 0])

    # sign_attribution = np.mean(motif_attribution*pfm)

    # Return the one that has higher correlation in absolute value
    # if sign_attribution >= 0:

    if rho_pos_prob >= rho_neg_prob:
        rho_prob = rho_pos_prob
        p_mut_prob = p_mut_pos
    else:
        rho_prob = -rho_neg_prob
        p_mut_prob = p_mut_neg

    if ICM is not False:
        if rho_pos_ICM >= rho_neg_ICM:
            rho_ICM = rho_pos_ICM
            p_mut_IC = p_mut_IC_pos
        else:
            rho_ICM = -rho_neg_ICM
            p_mut_IC = p_mut_IC_neg

    if not return_PFM and ICM is False:
        return rho_prob
    elif not return_PFM:
        return rho_ICM
    else:
        return (rho_prob, p_mut_prob, rho_ICM, p_mut_IC)


def run_motif_scanning(
    known_PFM : dict,
    attribution_seq : np.array,
    threshold : float = 0.6,
    attribution : bool = False,
    append : bool = True,
    multiple_one_hot : bool = False,
    normalize : bool = False,
    split_pos_neg : bool = False,
    cutoff_att : float = 0.001,
):
    """
    Parameters
    ----------
    known_PFM : dict
        Dict of known PFM with name of PFM as keys, optionally can also be used IC instead of PFM.
    attribution_seq : np.array
        Array showing attribution of sequence obtained from in silico mutagenesis.
    threshold : float, optional
        Threshold to consider a good match between known TFBS and computed attribution. (default is 0.6)
    attribution : bool, optional
        If true returns as an extra column the mean attribution * PFM of that hit. (default is False)
    append : bool, optional
        If True, then in the attribution_seq a matrix of (4,5) is appended on the end and start of the sequence.
        In this way it allows for motifs scanning to also find hits in the edges. (default is True)
    multiple_one_hot : bool, optional
        If not False, provide one-hot encoded sequence, and the scanning will be done with the normalized sequence*one_hot.
    normalize : bool, optional
        If normalize each base to add up to 1. If IC instead of PFM provided we recommend to not use it. (default is False)
    split_pos_neg : bool, optional
        If True, split the attribution between negative and positive values. (default is False)
    cutoff_att : float, optional
        Cutoff value for small attributions. (default is 0.001)

    Returns
    -------
    hits : list
        Hits of list that contains [motif_name, correlation, start_hit, end_hit]
        
    Examples
    --------
    >>> run_motif_scanning(known_PFM, attribution_seq, 0.6, False, True, False, False, 0.001)
        
    """

    if append:
        zeros_padding = np.zeros(shape=(4, 5))
        attribution_seq = np.concatenate(
            [zeros_padding, attribution_seq, zeros_padding], axis=1
        )
        if multiple_one_hot is not False:
            multiple_one_hot = np.concatenate(
                [zeros_padding, multiple_one_hot, zeros_padding], axis=1
            )
    ###Normalize whole attribution sequence

    length_sequence = attribution_seq.shape[-1]

    if split_pos_neg:

        ## 1. Positive attributions (activators)
        p_mut_pos = np.maximum(attribution_seq, 0)  # If negative attribution, set to 0
        ## 2. Negative attributions (inhibitors)
        p_mut_neg = -np.minimum(attribution_seq, 0)  # If negative attribution, set to 0

    if normalize:
        p_mut_pos[:, np.sum(p_mut_pos, axis=0) == 0] = (
            0.25  # If in a position all nt are 0, set all of them to 0.25
        )
        p_mut_pos = p_mut_pos / np.sum(p_mut_pos, axis=0, keepdims=True)  # Normalize
        p_mut_pos = p_mut_pos + 0.01  # Add pseudocount
        p_mut_pos = p_mut_pos / np.sum(p_mut_pos, axis=0, keepdims=True)

        p_mut_neg[:, np.sum(p_mut_neg, axis=0) == 0] = (
            0.25  # If in a position all nt are 0, set all of them to 0.25
        )
        p_mut_neg = p_mut_neg / np.sum(p_mut_neg, axis=0, keepdims=True)  # Normalize
        p_mut_neg = p_mut_neg + 0.01  # Add pseudocount
        p_mut_neg = p_mut_neg / np.sum(p_mut_neg, axis=0, keepdims=True)

    # If multiple_one_hot argument is not False. We will multiple the normalized sequence by the one hot encoded sequence.
    ## In this way we will only report hits of TF in the sequence, not putative.
    if multiple_one_hot is not False:
        p_mut_neg = p_mut_neg * multiple_one_hot
        print("p_mut_neg", p_mut_neg)
        p_mut_pos = p_mut_pos * multiple_one_hot
        print("p_mut_pos", p_mut_pos)

    hits = []

    rho_pos_prob, rho_neg_prob, rho_prob = 0, 0, 0

    # Group motifs by their length
    ## this way we can save a bit of time by doing the scanning for motifs that have the same length
    unique_lengths_motifs = np.unique(
        np.array([x.shape[1] for x in known_PFM.values()])
    )
    length_to_motif_name = {len: [] for len in unique_lengths_motifs}
    for name_motif, PFM in known_PFM.items():
        length_to_motif_name[PFM.shape[1]].append(name_motif)

    # Loop through group of length motifs
    for length_PFM, name_motifs_same_length in length_to_motif_name.items():

        # Loop through sequence length
        for start_motif in range(length_sequence - length_PFM + 1):
            end_motif = start_motif + length_PFM

            if split_pos_neg:
                p_mut_short_pos = p_mut_pos[:, start_motif:end_motif]
                p_mut_short_neg = p_mut_neg[:, start_motif:end_motif]

            else:
                attribution_seq_short = attribution_seq[:, start_motif:end_motif]

            # Avoid computing hits if the attributions are super small
            if split_pos_neg:
                if (p_mut_short_pos.sum() < cutoff_att) and (
                    np.abs(p_mut_neg).sum() < (cutoff_att)
                ):
                    continue
            else:
                if np.abs(cutoff_att).sum() < cutoff_att:
                    continue

            # Loop through motifs in the same length group

            for name_motif in name_motifs_same_length:

                PFM = known_PFM[name_motif]
                sum_motif = PFM.sum().sum()

                if split_pos_neg:
                    ##Positive correlation
                    rho_pos_prob = np.corrcoef(
                        [p_mut_short_pos.flatten(), PFM.flatten()]
                    )[0, 1]
                    rho_pos_prob = np.max([rho_pos_prob, 0])

                    ##Negative correlation
                    rho_neg_prob = np.corrcoef(
                        [p_mut_short_neg.flatten(), PFM.flatten()]
                    )[0, 1]
                    rho_neg_prob = np.max([rho_neg_prob, 0])

                else:
                    rho_prob = np.corrcoef(
                        [attribution_seq_short.flatten(), PFM.flatten()]
                    )[0, 1]

                if attribution is not False:

                    ##If there's a hit --> Save the product of the attribution x PFM --> Useful to later take the longest motif
                    if (
                        (rho_pos_prob > threshold)
                        or (rho_neg_prob > threshold)
                        or (np.abs(rho_prob) > threshold)
                    ):

                        if split_pos_neg:
                            attribution_seq_short = attribution_seq[
                                :, start_motif:end_motif
                            ]
                        att_x_PFM = (
                            attribution_seq_short.flatten() * PFM.flatten()
                        ).mean()
                        att_pos = (attribution_seq_short.flatten()).sum()

                        if append:
                            start_motif_name = start_motif - 5
                            end_motif_name = end_motif - 5

                        if split_pos_neg:
                            if rho_pos_prob > threshold:
                                hits.append(
                                    [
                                        name_motif,
                                        rho_pos_prob,
                                        start_motif_name,
                                        end_motif_name,
                                        att_pos,
                                        att_x_PFM,
                                        length_PFM,
                                        sum_motif,
                                    ]
                                )

                            if rho_neg_prob > threshold:
                                hits.append(
                                    [
                                        name_motif,
                                        (-rho_neg_prob),
                                        start_motif_name,
                                        end_motif_name,
                                        att_pos,
                                        att_x_PFM,
                                        length_PFM,
                                        sum_motif,
                                    ]
                                )

                        else:
                            hits.append(
                                [
                                    name_motif,
                                    rho_prob,
                                    start_motif_name,
                                    end_motif_name,
                                    att_pos,
                                    att_x_PFM,
                                    length_PFM,
                                    sum_motif,
                                ]
                            )

                else:
                    if append:
                        start_motif_name = start_motif - 5
                        end_motif_name = end_motif - 5

                    if split_pos_neg:
                        if rho_pos_prob > threshold:
                            hits.append(
                                [
                                    name_motif,
                                    rho_pos_prob,
                                    start_motif_name,
                                    end_motif_name,
                                    length_PFM,
                                ]
                            )
                        if rho_neg_prob > threshold:
                            hits.append(
                                [
                                    name_motif,
                                    (-rho_neg_prob),
                                    start_motif_name,
                                    end_motif_name,
                                    length_PFM,
                                ]
                            )

                    else:
                        hits.append(
                            [
                                name_motif,
                                rho_prob,
                                start_motif_name,
                                end_motif_name,
                                length_PFM,
                            ]
                        )

    if attribution is not False:
        hits = pd.DataFrame(
            hits,
            columns=[
                "name_motif",
                "rho",
                "start",
                "end",
                "att",
                "att_x_PFM",
                "length_motif",
                "sum_motif",
            ],
        )

    else:
        hits = pd.DataFrame(
            hits, columns=["name_motif", "rho", "start", "end", "length_motif"]
        )

    return hits


def slide_through_attribution_and_PFM_fast(
    known_PFM : dict, 
    attribution_seq : np.array, 
    append : bool = True, 
    cutoff_att : float = 0.001
):
    """
    Slides through sequence, computes attribution and then check which motifs and where have a high correlation.

    Parameters
    ----------
    known_PFM : dict
        Dict of known PFM with name of PFM as keys, optionally can also be used IC instead of PFM.
    attribution_seq : np.array
        Array showing attribution of sequence obtained from in silico mutagenesis.
    append : bool
        If True, the peaks will be appended with zeros.
    cutoff_att : float
        Cutoff value for small attributions.
        
    Returns
    -------
    hits : pd.DataFrame
        Hits of list that contains [motif_name, correlation, start_hit, att_abs_sum, length_motif]
        
    Examples
    --------
    >>> slide_through_attribution_and_PFM_fast(known_PFM, attribution_seq, True, 0.001)
    """

    if append:
        zeros_padding = np.zeros(shape=(4, 5))
        attribution_seq = np.concatenate(
            [zeros_padding, attribution_seq, zeros_padding], axis=1
        )
    ###Normalize whole attribution sequence

    length_sequence = attribution_seq.shape[-1]

    if torch.cuda.is_available():
        attribution_seq = torch.tensor(attribution_seq).cuda()

    hits = []

    # Group motifs by their length
    ## this way we can save a bit of time by doing the scanning for motifs that have the same length
    t = time.time()
    unique_lengths_motifs = np.unique(
        np.array([x.shape[1] for x in known_PFM.values()])
    )
    length_to_motif_name = {len: [] for len in unique_lengths_motifs}
    for name_motif, PFM in known_PFM.items():
        length_to_motif_name[PFM.shape[1]].append(name_motif)
    known_PFM = {
        name_motif: torch.from_numpy(PFM.copy()).cuda()
        for name_motif, PFM in known_PFM.items()
    }

    #
    # Loop through group of length motifs
    for length_PFM, name_motifs_same_length in length_to_motif_name.items():

        # shape before: bases, length
        attribution_seq_slid = attribution_seq.permute(1, 0).unfold(
            dimension=0, size=length_PFM, step=1
        )

        # shape before: sliding_window, bases, length

        # Loop through sequence length
        for it_pos, att_slid in enumerate(attribution_seq_slid):

            att_abs_sum = att_slid.abs().sum().item()

            # Avoid computing hits if the attributions are super small
            if att_abs_sum < cutoff_att:
                continue

            # Loop through motifs in the same length group

            for name_motif in name_motifs_same_length:

                PFM = known_PFM[name_motif]
                ##Pearson correlation
                rho_prob = torch.corrcoef(
                    torch.stack([att_slid.flatten(), PFM.flatten()])
                )[0, 1].item()

                ##If there's a hit --> Save the product of the attribution x PFM --> Useful to later take the longest motif
                hits.append([name_motif, rho_prob, it_pos, att_abs_sum, length_PFM])

    hits = pd.DataFrame(
        hits, columns=["name_motif", "rho", "start", "att_abs_sum", "length_motif"]
    )

    return hits


def conv_with_PFM_slide_through_attribution(
    known_PFM : dict, 
    attribution_seq : np.array, 
    multiple_one_hot : np.array = False, 
    normalize : bool = False, 
    sequence : bool = False
):
    """
    Slides through sequence of attribution and then computes convolution for each given motif.

    Parameters
    ----------
    known_PFM : dict
        Dict of known PFM with name of PFM as keys, optionally can also be used IC instead of PFM.
    attribution_seq : np.array
        Array showing attribution of sequence obtained from in silico mutagenesis.
    multiple_one_hot : np.array
        If not False, provide one-hot encoded sequence, and the scanning will be done with the normalized sequence*one_hot.
    normalize : bool
        If normalize each base to add up to 1. If IC instead of PFM provided we recommend to not use it.
    sequence : bool
        If True, the attribution_seq is the sequence itself.
        
    Returns
    -------
    hits : list
        Hits of list that contains [motif_name, correlation, start_hit, end_hit]
        
    Examples
    --------
    >>> conv_with_PFM_slide_through_attribution(known_PFM, attribution_seq, False, False, False)
    """

    length_sequence = attribution_seq.shape[-1]

    ## 1. Positive attributions (activators)
    p_mut_pos = np.maximum(attribution_seq, 0.0)  # If negative attribution, set to 0
    ## 2. Negative attributions (inhibitors)
    p_mut_neg = -np.minimum(attribution_seq, 0.0)  # If negative attribution, set to 0

    if normalize:
        p_mut_pos[:, np.sum(p_mut_pos, axis=0) == 0] = (
            0.25  # If in a position all nt are 0, set all of them to 0.25
        )
        p_mut_pos = p_mut_pos / np.sum(p_mut_pos, axis=0, keepdims=True)  # Normalize
        p_mut_pos = p_mut_pos + 0.01  # Add pseudocount
        p_mut_pos = p_mut_pos / np.sum(p_mut_pos, axis=0, keepdims=True)

        p_mut_neg[:, np.sum(p_mut_neg, axis=0) == 0] = (
            0.25  # If in a position all nt are 0, set all of them to 0.25
        )
        p_mut_neg = p_mut_neg / np.sum(p_mut_neg, axis=0, keepdims=True)  # Normalize
        p_mut_neg = p_mut_neg + 0.01  # Add pseudocount
        p_mut_neg = p_mut_neg / np.sum(p_mut_neg, axis=0, keepdims=True)

    # If multiple_one_hot argument is not False. We will multiple the normalized sequence by the one hot encoded sequence.
    ## In this way we will only report hits of TF in the sequence, not putative.
    if multiple_one_hot is not False:
        p_mut_neg = p_mut_neg * multiple_one_hot
        print("p_mut_neg", p_mut_neg)
        p_mut_pos = p_mut_pos * multiple_one_hot
        print("p_mut_pos", p_mut_pos)

    # Now transform the attributions into torch
    p_mut_pos = torch.tensor(p_mut_pos)
    p_mut_neg = torch.tensor(p_mut_neg)

    if torch.cuda.is_available():
        p_mut_neg = p_mut_neg.cuda()
        p_mut_pos = p_mut_pos.cuda()

    hits = []

    # Group motifs by their length
    ## this way we can save a bit of time by doing the scanning for motifs that have the same length
    t = time.time()
    unique_lengths_motifs = np.unique(
        np.array([x.shape[1] for x in known_PFM.values()])
    )
    length_to_motif_name = {len: [] for len in unique_lengths_motifs}
    for name_motif, PFM in known_PFM.items():
        length_to_motif_name[PFM.shape[1]].append(name_motif)

    padding = 0

    pos_hits, neg_hits = np.empty(
        [len(known_PFM), length_sequence + padding * 2]
    ), np.empty([len(known_PFM), length_sequence + padding * 2])
    sum_attribution_pos, sum_attribution_neg = np.empty(
        [len(known_PFM), length_sequence + padding * 2]
    ), np.empty([len(known_PFM), length_sequence + padding * 2])

    # Loop through group of length motifs --> The kernel size of conv1d has to have the same length

    curent_1st_motif = 0
    for length_PFM, name_motifs_same_length in length_to_motif_name.items():
        num_motifs = len(name_motifs_same_length)

        if length_PFM % 2 == 0:
            padding = (length_PFM // 2, length_PFM // 2 - 1)
        else:
            padding = (length_PFM // 2, length_PFM // 2)
        padding = "same"

        conv_motifs = torch.nn.Conv1d(
            4, (num_motifs + 1), length_PFM, stride=1, padding=padding, bias=False
        )  # We add an extra motif which is the one that is going to sum all the contributions in a winddow

        motifs_PFM = list(map(known_PFM.get, name_motifs_same_length))
        motifs_PFM.append(
            np.ones((4, length_PFM))
        )  # We add an extra motif which is the one that is going to sum all the contributions in a winddow

        # Fill in the conv layer with the known motifs of Hocomoco
        conv_motifs.weight = torch.nn.Parameter(torch.from_numpy(np.array(motifs_PFM)))

        if torch.cuda.is_available():
            conv_motifs = conv_motifs.cuda()

        out_pos = conv_motifs(p_mut_pos).cpu().detach().numpy()

        if sequence is not True:
            out_neg = conv_motifs(p_mut_neg).cpu().detach().numpy()

        pos_hits[curent_1st_motif : (curent_1st_motif + num_motifs)] = np.maximum(
            out_pos[:-1], 0
        )  # ReLU is the same as doing np.maximum
        sum_attribution_pos[curent_1st_motif : (curent_1st_motif + num_motifs)] = (
            np.repeat(out_pos[-1:], num_motifs, axis=0)
        )

        if sequence is not True:
            neg_hits[curent_1st_motif : (curent_1st_motif + num_motifs)] = np.maximum(
                out_neg[:-1], 0
            )
            sum_attribution_neg[curent_1st_motif : (curent_1st_motif + num_motifs)] = (
                np.repeat(out_neg[-1:], num_motifs, axis=0)
            )

        curent_1st_motif += num_motifs

    # Transform into a na pandas dataframe

    # Current shape n_motif x length_fragment

    pos_hits = pd.DataFrame(pos_hits, index=sum(length_to_motif_name.values(), []))
    pos_hits["motif"] = pos_hits.index
    pos_hits = pd.melt(
        pos_hits, id_vars="motif", var_name="position", value_name="conv_score"
    )
    pos_hits["attribution"] = "pos"

    sum_attribution_pos = pd.DataFrame(
        sum_attribution_pos, index=sum(length_to_motif_name.values(), [])
    )
    sum_attribution_pos["motif"] = sum_attribution_pos.index
    sum_attribution_pos = pd.melt(
        sum_attribution_pos,
        id_vars="motif",
        var_name="position",
        value_name="sum_attribution",
    )
    pos_hits = pos_hits.merge(
        sum_attribution_pos, how="inner", on=["motif", "position"]
    )

    if sequence:
        return pos_hits

    neg_hits = pd.DataFrame(neg_hits, index=sum(length_to_motif_name.values(), []))
    neg_hits["motif"] = neg_hits.index
    neg_hits = pd.melt(
        neg_hits, id_vars="motif", var_name="position", value_name="conv_score"
    )
    neg_hits["attribution"] = "neg"

    sum_attribution_neg = pd.DataFrame(
        sum_attribution_neg, index=sum(length_to_motif_name.values(), [])
    )
    sum_attribution_neg["motif"] = sum_attribution_neg.index
    sum_attribution_neg = pd.melt(
        sum_attribution_neg,
        id_vars="motif",
        var_name="position",
        value_name="sum_attribution",
    )
    neg_hits = neg_hits.merge(
        sum_attribution_neg, how="inner", on=["motif", "position"]
    )

    hits = pd.concat([pos_hits, neg_hits])

    return hits


# ==============================================================================
# Plotting =====================================================================
# ==============================================================================
def PARM_plot_mutagenesis(
    input : os.path,
    correlation_threshold : float,
    attribution_threshold : float,
    plot_format : str,
    output_directory : os.path = None,
    attribution_range : list = None,
):
    """
    Generate the plots of the mutagenesis data (produced by PARM_mutagenesis.py).
    Output is saved in the output directory as an image file.
    
    Parameters
    ----------
    input : os.path
        Input directory with the mutagenesis data.
    correlation_threshold : float
        Threshold for the correlation of the hits.
    attribution_threshold : float
        Threshold for the attribution of the hits.
    plot_format : str
        Format of the output plot.
    output_directory : os.path
        Output directory for the plots.
    attribution_range : list
        Range of the attribution values.
        
    Examples
    --------
    >>> PARM_plot_mutagenesis(input, 0.5, 0.5, "png", output_directory, [0, 1])
    """
    log("Reading input directory")
    input_directory = Path(input)
    # Quit if input directory does not exist
    if not input_directory.exists():
        raise NotADirectoryError(f"Input directory {input} does not exist")
    plot_extension = "." + plot_format
    total_input_files = len(
        ["_" for _ in input_directory.rglob("mutagenesis_*.txt.gz")]
    )
    # Quit if no files found
    if total_input_files == 0:
        raise FileNotFoundError("No files found in input directory")
    log("Plotting data")
    pbar = tqdm(total=total_input_files, ncols=80)
    for this_file in input_directory.rglob("mutagenesis_*.txt.gz"):
        file_name = this_file.name
        file_parent_directory = this_file.parent
        file_hits = os.path.join(
            file_parent_directory, file_name.replace("mutagenesis_", "hits_")
        )
        # Define output file
        if output_directory is None:
            output_file = os.path.join(
                file_parent_directory,
                file_name.replace("mutagenesis_", "").replace(
                    ".txt.gz", plot_extension
                ),
            )
        else:
            # create output directory
            os.makedirs(output_directory, exist_ok=True)
            output_file = os.path.join(
                output_directory,
                file_name.replace("mutagenesis_", "").replace(
                    ".txt.gz", plot_extension
                ),
            )
        # Reading input
        mutagenesis_data = pd.read_csv(this_file, compression="gzip", sep="\t")
        sequence = "".join(mutagenesis_data.Ref)
        hits_data = pd.read_csv(file_hits, compression="gzip", sep="\t")
        # Apply filters
        hits_data = hits_data.loc[abs(hits_data.rho) >= correlation_threshold]
        hits_data = hits_data.loc[abs(hits_data.att) >= attribution_threshold]
        plot_mutagenesis(
            mutagenesis_df=mutagenesis_data,
            seq=sequence,
            TSS_position=0,
            cutoff=0,
            return_fig=False,
            title=file_name.replace("mutagenesis_", "").replace(".txt.gz", ""),
            hits=hits_data,
            output_file=output_file,
            attribution_range=attribution_range,
        )
        pbar.update(1)


def plot_logo(
    matrix : np.array,
    ax_name : matplotlib.axis,
    ylabel : str,
    colors_base : int = False,
    highlight_position : int = False,
    fontsize : str = "medium",
    max_lim : int = False,
    min_lim : int = False,
):
    """
    Make plot given axis of DNA sequence importance. This plots the logo of the sequence.
    
    Parameters
    ----------
    matrix : np.array
        Importance matrix of sequence.
    ax_name : matplotlib.axis
        Axis that the logo should be plot.
    ylabel : str
        Label of the y-axis.
    colors_base : int
        If True use Vini's colors.
    highlight_position : int
        Position to highlight in the logo.
    fontsize : str
        Fontsize of the plot.
    max_lim : int
        Maximum limit of the plot.
    min_lim : int
        Minimum limit of the plot.
        
    Examples
    --------
    >>> plot_logo(matrix, ax_name, "ylabel", False, False, "medium", False, False)
    """

    tf_mut_score_plot = pd.DataFrame(np.transpose(matrix), columns=["A", "C", "G", "T"])

    if colors_base:
        color_scheme = {
            "A": colors.to_rgb("#00811C"),  # green
            "C": colors.to_rgb("#2000C7"),  # blue
            "G": colors.to_rgb("#FFB32C"),  # yeellow
            "T": colors.to_rgb("#D00001"),
        }  # red

        crp_logo = logomaker.Logo(
            tf_mut_score_plot, ax=ax_name, color_scheme=color_scheme, baseline_width=1
        )

    else:
        crp_logo = logomaker.Logo(tf_mut_score_plot, ax=ax_name, baseline_width=1)

    if highlight_position is not False:
        crp_logo.highlight_position(p=highlight_position, color="gold", alpha=0.5)

    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=["left"], visible=True)
    crp_logo.ax.set_ylabel(ylabel, labelpad=-1, fontsize=fontsize)

    if max_lim is not False and min_lim is not False:
        crp_logo.ax.set_ylim(min_lim, max_lim)

    crp_logo.ax.set_xticks([])

    return crp_logo


def find_hits_and_make_logo(
    matrix : np.array,
    ax_name,
    known_PFM,
    cutoff=0.8,
    colors_base=True,
    best_motif_in_range=10,
    known_ICM=False,
    percentage=0.15,
    multiple_one_hot : np.array = False,
    hits : pd.DataFrame = None,
    fig=None,
    split_pos_neg=False,
):
    """
    Given attribution matrix find hits (similar ), and plot the original motifs.

    Args:
        matrix (np.array) Importance matrix of sequence
        ax_name: matplotlib axis provide 2
        colors_base (bool): If True use Vini's colors
        cutoff (float): Cutoff to define similarity, default 0.8
        known_PFM: (dict) dict of known PFM with name of PFM as keys
        best_motif_in_range: (int) To avoid overlapping motifs, the best motif is found in within best_motif_in_range base pairs, default =10
        known_ICM: (dict) Same as known_PFM but IC, if False just use known_PFM.
        percentage (float): Only do the scanning of the motifs that attributions is at least 1-percentage the attribution of the max motif
        multiple_one_hot (np.array): If not False, provide one-hot encoded sequence, and the scanning will be done with the normalized sequence*one_hot
        hits: pre-computed hits df. If is None, will be computed here

        split_pos_neg: (bool) If True, split the attribution between negative and positive values when searching for hits.

    """

    len_seq = matrix.shape[1]
    # Find hits, creates dataframe

    # Get the PFM to plot and compute
    if known_ICM is not False:
        plotting_known_motifs = known_ICM
        normalize = False
    else:
        plotting_known_motifs = known_PFM
        normalize = True

    if hits is None:
        hits = run_motif_scanning(
            plotting_known_motifs,
            matrix,
            threshold=cutoff,
            attribution=True,
            multiple_one_hot=multiple_one_hot,
            normalize=normalize,
            split_pos_neg=split_pos_neg,
        )
        hits["att"] = hits["att"] / hits.sum_motif

    hits["att_abs"] = np.abs(hits.att)

    # Create empty array and fill it with hits
    relevant_hits_row1 = np.empty(matrix.shape)
    relevant_hits_row1[:, :] = 0
    relevant_hits_row2 = np.empty(matrix.shape)
    relevant_hits_row2[:, :] = 0

    # Take only hits that their mean attribution is at least 10% of the letter wiht max attribution

    hits = hits[hits.att_x_PFM.abs() > (hits.att_x_PFM.abs().max() * percentage)]

    # Avoid overlapping hits
    hits["rho_abs"] = np.abs(hits.rho)
    hits = hits.sort_values("start")
    hits["mid_point"] = ((hits.end + hits.start) / 2).astype(int)

    # Cut the promoters in bins based on the length of best_motif_in_range
    bins = np.arange(0, len_seq, best_motif_in_range)
    ind = np.digitize(hits.mid_point, bins)
    hits["bin"] = ind

    # We have to take somehow the length of the motif in the attribution sequence
    ## we prefer to have a HOCOMOCO motifs that covers all the motif in the attribution score,
    ##   instead of two that might have higher rho.

    ###### Lenght max
    hits["len"] = hits.end - hits.start
    # hits['len_max'] = hits.groupby(ind)['len'].transform(max)
    # hits = hits[hits['len_max'] == hits['len']]

    ###### Based on att*PFM
    """def secndmax(x):
        if len(x)==1:return(max(x)) 
        x = list(x)
        x.remove(max(x))
        return(max(x))"""

    hits = (
        hits.groupby(ind)
        .apply(lambda x: x.nlargest(2, ["rho_abs"], keep="first"))
        .reset_index(drop=True)
    )

    # hits['corr_max'] = hits.groupby(ind)['rho_abs'].transform(secndmax)

    # hits = hits[hits['corr_max'] <= hits['rho_abs']]

    ###### Rho max
    # hits['rho_max'] = hits.groupby(ind)['rho_abs'].transform(max)
    # hits = hits[hits['rho_max'] == hits['rho_abs']]

    ###### Based on attribution
    # hits['att_max'] = hits.groupby(ind)['att'].transform(max)
    # hits = hits[hits['att'] == hits['att_max']]

    previous_start, previous_end = -10, -10

    hits = hits.sort_values(["bin", "rho_abs"], ascending=False)

    if hits.empty:  # If no hits, delete axis
        fig.delaxes(ax_name[0])
        fig.delaxes(ax_name[1])
        return None

    ##add selected hits
    hits_to_plot_row1, hits_to_plot_row2 = [], []

    for it_rows, motif in hits.iterrows():

        start_motif, end_motif = motif[2], motif[3]

        overlap = max(
            0, min(end_motif, previous_end) - max(start_motif, previous_start)
        )

        # If the start of the motif it's a bit shifted we need to make the known motif a bit shorter s well

        if end_motif > len_seq:
            end_motif = len_seq
            rel_end = len_seq - start_motif

        else:
            rel_end = end_motif - start_motif

        if start_motif < 0:
            rel_start = -start_motif
            start_motif = 0

        else:
            rel_start = 0

        if overlap == 0 or (overlap > 0 and previous_overlap > 0):
            relevant_hits_row1[:, start_motif:end_motif] = plotting_known_motifs[
                motif[0]
            ][:, rel_start:rel_end]
            hits_to_plot_row1.append(motif)
        else:
            relevant_hits_row2[:, start_motif:end_motif] = plotting_known_motifs[
                motif[0]
            ][:, rel_start:rel_end]
            hits_to_plot_row2.append(motif)

        previous_start, previous_end, previous_overlap = start_motif, end_motif, overlap

    # Do the same for both axis
    for it, sub_ax in enumerate(ax_name):

        if it == 0:
            relevant_hits = relevant_hits_row1
            hits_to_plot = hits_to_plot_row1

        if it == 1:
            relevant_hits = relevant_hits_row2
            hits_to_plot = hits_to_plot_row2

        tf_mut_score_plot = pd.DataFrame(
            np.transpose(relevant_hits), columns=["A", "C", "G", "T"]
        )

        if colors_base:
            color_scheme = {
                "A": colors.to_rgb("#00811C"),  # green
                "C": colors.to_rgb("#2000C7"),  # blue
                "G": colors.to_rgb("#FFB32C"),  # yeellow
                "T": colors.to_rgb("#D00001"),
            }  # red

            crp_logo = logomaker.Logo(
                tf_mut_score_plot,
                ax=sub_ax,
                color_scheme=color_scheme,
                baseline_width=0,
            )

        else:
            crp_logo = logomaker.Logo(tf_mut_score_plot, ax=sub_ax, baseline_width=0)

        for selected_hit in hits_to_plot:

            motif_name = selected_hit[0].split("_")[0]
            if "-" in selected_hit[0]:
                motif_name = f"{motif_name} -"

            rho = np.abs(round(selected_hit.rho, 2))
            text_motif = f"{motif_name} R = {rho}"

            mid_point = int((selected_hit[2] + selected_hit[3]) / 2) - int(
                len(text_motif) / 2
            )

            sub_ax.text(x=mid_point, y=-0.75, s=text_motif)

        crp_logo.style_spines(visible=False)
        crp_logo.ax.set_yticks([])
        crp_logo.ax.set_xticks([])
        crp_logo.ax.set_ylim((-0.6, crp_logo.ax.get_ylim()[1]))

    return crp_logo


# function to save the effect of every possible mutations
def create_dataframe_mutation_effect(
    name,
    strand,
    model,
    output_directory,
    PFM_hocomoco_dict,
    L_max,
    reverse=False,
    seq=False,
    type_motif_scanning="PCC",
    compute_sequence_hits=False,
):
    """
    Return a dataframe with the effect of each mutation.
    Args:
            model: pytorch model
            promoters: (pd.Series) pandas series that contains chr, start, end, strand for each promoter
            output_directory: (str) Output directory to save all hits and mutagenesis dataframe.
            L_max: (int) Length max. of the model
            PFM_hocomoco_dict: (dict) Dictionary PFM or IC for motifs of interest
            compute_sequence_hits: (bool)  If True, the hits for the sequence are also computed.
            reverse: (bool) If True, compute also for reverse attribution and do the mean with the forward
            seq: (str) If seq is given, not compute sequence from genome.
            type_motif_scanning: (str) Either 'PCC' or 'conv_scanning'
    """

    if type_motif_scanning != "PCC" and type_motif_scanning != "conv_scanning":
        raise ValueError(
            f'Type of motif scanning should be either "PCC" or "conv_scanning", but you selected {type_motif_scanning}'
        )
    # Create folder if it doesn't exist

    folder_output = os.path.join(output_directory, f"{name}")

    if not os.path.exists(folder_output):
        os.makedirs(
            folder_output
        )  # Create subdirectory with the name of the promoter in the output folder

    file_hits = os.path.join(folder_output, f"hits_{name}.txt.gz")
    file_hits_seq = os.path.join(folder_output, f"hits_SEQUENCE_{name}.txt.gz")

    # Mutagenesis file
    file_mutagenesis = os.path.join(folder_output, f"mutagenesis_{name}.txt.gz")

    promoter_importance_base = pd.DataFrame()

    if not os.path.exists(file_mutagenesis):

        att_all, _ = compute_attribution(
            seq=seq, L_max=L_max, start_motif=0, end_motif=len(seq), completemodel=model
        )
        att_all = att_all

        if reverse:
            # If reverse use the same sequence but complementary and compute attributions
            reverse_DNA = "".maketrans("ACGT", "TGCA")
            rev_seq = seq.translate(reverse_DNA)[::-1]

            att_all_rev, _ = compute_attribution(
                seq=rev_seq,
                L_max=L_max,
                start_motif=0,
                end_motif=len(seq),
                completemodel=model,
            )

            att_all_rev = att_all_rev.cpu().numpy()

            att_all = np.mean(
                np.stack([att_all, np.flip(att_all_rev, (0, 1))], axis=0), axis=0
            )

        start = 1
        end = len(seq)
        start_end = list(range(start, end))

        if np.all(strand == "-"):
            start_end = list(range(end, start, -1))

        if np.all(strand == "+"):
            start_end = list(range(start, end))

        promoter_importance_base_single = pd.DataFrame(
            {
                "Ref": list(seq),
                "A": att_all[0, :],
                "C": att_all[1, :],
                "G": att_all[2, :],
                "T": att_all[3, :],
            }
        )

        promoter_importance_base = pd.concat(
            [promoter_importance_base, promoter_importance_base_single]
        )

        promoter_importance_base.to_csv(
            file_mutagenesis,
            index=False,
            sep="\t",
            compression={"method": "gzip", "compresslevel": 5},
        )

    else:
        try:
            promoter_importance_base = pd.read_csv(file_mutagenesis, sep="\t")
        except:
            promoter_importance_base = pd.read_csv(
                file_mutagenesis, sep="\t", compression="gzip"
            )

    # Now compute hits

    # now compute the average importance for each nucleotide in the reference sequnece

    onehot_seq = get_one_hot(["".join(seq)], L_max=len(seq))[0]

    plot_promoter = promoter_importance_base[["A", "C", "G", "T"]]

    vector_importance = plot_promoter.mean(axis=1)
    attribution_real = np.transpose(onehot_seq * np.expand_dims(vector_importance, 1))

    if type_motif_scanning == "PCC":

        hits = run_motif_scanning(
            PFM_hocomoco_dict,
            attribution_real,
            append=True,
            cutoff_att=0.001,
            attribution=True,
        )

        # HITS ON ONLY SEQUENCE, NOT ATTRIBUTION MATRIX
        if compute_sequence_hits:
            hits_sequence = run_motif_scanning(
                PFM_hocomoco_dict,
                np.transpose(onehot_seq),
                append=True,
                cutoff_att=0.001,
            )

    else:
        hits = conv_with_PFM_slide_through_attribution(
            PFM_hocomoco_dict, attribution_real, multiple_one_hot=False, normalize=False
        )

        # HITS ON ONLY SEQUENCE, NOT ATTRIBUTION MATRIX
        if compute_sequence_hits:
            hits_sequence = conv_with_PFM_slide_through_attribution(
                PFM_hocomoco_dict,
                np.transpose(onehot_seq),
                multiple_one_hot=False,
                normalize=False,
                sequence=True,
            )

    hits.to_csv(
        file_hits,
        index=False,
        sep="\t",
        compression={"method": "gzip", "compresslevel": 5},
    )

    if compute_sequence_hits:
        hits_sequence.to_csv(
            file_hits_seq,
            index=False,
            sep="\t",
            compression={"method": "gzip", "compresslevel": 5},
        )


def plot_mutagenesis(
    mutagenesis_df,
    seq,
    TSS_position=0,
    PFM_hocomoco_dict=False,
    ICT_hocomoco_dict=False,
    output_file=False,
    promoter_name="",
    return_fig=False,
    cutoff=0.7,
    title="",
    hits=None,
    model=None,
    attribution_range=None,
):
    """
    Args:
        mutagenesis_df (pd.DataFrame) It should be a dataframe with columns called "pos" (= position in the genome), 'A', 'C', 'G', 'T' --> These columns indicate the effect of the mutation in terms of REF-ALT

        seq: (str) String containg the sequence of the fragment, should be same length as dataframe.

        promoter_name: (str) Name of the promoter or region studied --> Used in the title
        chr: (str) Chromosome used e.g. chr2 --> Just used to add information in x-axis
        strand: (str) Either '-' or '+'
        title: (str) If necessary, add extra information to add in the title.

        PFM_hocomoco_dict: (dict) PFM dictionary to do motif scanning
        ICT_hocomoco_dict: (dict) ICT dictionary to do motif scanning

        output_file: (str) If not False, name of the figure file to be saved. Extension determines the type of figure.

        TSS_position: (int) Position of the TSS, in this way the x-axis will be relative position, and the TSS will be highlited.
        return_fig: (bool) If not False, it will return the fig object of matplotlib instead of plotting or saving plot.

        cutoff: (float) PCC cutoff to do motif scanning similarity (default 0.3).

    """

    ##IF not provided, just load  them
    if PFM_hocomoco_dict is False or ICT_hocomoco_dict is False:
        PFM_hocomoco_dict, _, ICT_hocomoco_dict = dict_jaspar(reverse=True)
    pos = pd.Series(list(range(len(seq))))
    strand = "+"
    if strand == "-":
        relative = pos - TSS_position
    elif strand == "+":
        relative = -(pos - TSS_position)
    else:
        print(
            f" Strand value neither + or -, instead value {strand} provided. Please correct this."
        )

    mutagenesis_df["rel"] = relative

    plot_promoter = mutagenesis_df[["A", "C", "G", "T"]]

    if TSS_position != 0:
        highlight_position = int(np.where(relative == 0)[0][0])
    else:
        highlight_position = False

    # The x-labels will be the relative position and the absolute
    if TSS_position != 0:
        plot_promoter.index = (
            relative.astype(str) + "\n" + mutagenesis_df.pos.astype(str)
        )
    else:
        plot_promoter.index = relative.astype(str)

    onehot_seq = get_one_hot(["".join(seq)], L_max=len(seq))[0]

    fig, ax = plt.subplots(
        figsize=(40, 7),
        nrows=6,
        ncols=2,
        gridspec_kw={"height_ratios": [1, 3, 4, 1, 1, 1], "width_ratios": [50, 1]},
    )
    plt.subplots_adjust(wspace=0.001)
    fig.delaxes(ax[3, 0])

    fig.delaxes(ax[0, 1])
    fig.delaxes(ax[1, 1])
    fig.delaxes(ax[2, 1])
    fig.delaxes(ax[3, 1])
    fig.delaxes(ax[4, 1])
    fig.delaxes(ax[5, 1])

    ############ First row plot is the DNA sequence

    ##Add arrow where TSS is
    if TSS_position != 0:
        ax[0, 0].annotate(
            "",
            xy=(highlight_position + 4, 2),
            xytext=(highlight_position, 1),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1,
                connectionstyle="angle,angleA=90,angleB=180,rad=0",
            ),
            annotation_clip=False,
        )
        # ax[0,0].annotate("", xy=(highlight_position, 1), xytext=(highlight_position, 2), arrowprops=dict(arrowstyle="-", linewidth=1), annotation_clip=False)

    plot_logo(
        np.transpose(onehot_seq),
        ax_name=ax[0, 0],
        ylabel="",
        colors_base=True,
        highlight_position=highlight_position,
    )
    # Add prediction in the tittle
    # pred = predict_fragments(seq, 600, model)
    # title = f"{title} \nPredicted wildtype activity: {pred:.4f}"
    ax[0, 0].set_title(f"{promoter_name}  {title}", fontsize="large")
    # remove y-axis info from this row
    ax[0, 0].yaxis.set_tick_params(labelleft=False)
    ax[0, 0].set_yticks([])

    ########### Third row plot is the heatmap

    # In case there are NaNs
    mask = plot_promoter.T.isnull()
    if attribution_range is not None:  # If limits are defined set them in heatmap
        _ = sns.heatmap(
            plot_promoter.T,
            ax=ax[2, 0],
            cmap="RdBu",
            center=0,
            xticklabels=10,
            cbar=False,
            vmin=attribution_range[0],
            vmax=attribution_range[1],
            mask=mask,
        )

    else:  # Otherwise take the minimal and maximal values
        _ = sns.heatmap(
            plot_promoter.T,
            ax=ax[2, 0],
            cmap="RdBu",
            center=0,
            xticklabels=10,
            cbar=False,
            vmin=plot_promoter.min(axis=None),
            vmax=plot_promoter.max(axis=None),
            mask=mask,
        )

    ax[2, 0].set_ylabel(f"")
    ax[2, 0].set_xlabel(f"Position (bp)")

    # now compute the average importance for each nucleotide in the reference sequnece
    vector_importance = plot_promoter.mean(axis=1, skipna=True)
    # If any nt is NaN, it's because all the measurements where NaN, in that case we fill it with 0
    # if vector_importance.isna().any(): vector_importance = vector_importance.fillna(0)

    attribution_real = np.transpose(onehot_seq * np.expand_dims(vector_importance, 1))

    ########### 4th and 5th row plots the hits (similar to known motifs)
    # Plot the best matches with known PFM
    find_hits_and_make_logo(
        attribution_real,
        [ax[4, 0], ax[5, 0]],
        PFM_hocomoco_dict,
        cutoff=cutoff,
        best_motif_in_range=15,
        known_ICM=ICT_hocomoco_dict,
        fig=fig,
        hits=hits,
    )

    ###########  2nd row is the variant effect
    plot_logo(
        attribution_real,
        ax_name=ax[1, 0],
        ylabel="Variant effect\n(REF-mean(ALT))",
        colors_base=True,
        highlight_position=highlight_position,
    )

    fig.colorbar(
        ax[2, 0].get_children()[0],
        ax=ax[2, 1],
        orientation="vertical",
        fraction=0.5,
        label="Variant effect\n(REF-ALT)",
    )

    # If you want return the axis and fig
    if return_fig == True:
        return (fig, ax)

    if output_file is False:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches="tight")
