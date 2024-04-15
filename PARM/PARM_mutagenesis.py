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
from Bio import motifs
import pandas as pd
import os
import io
import urllib
from PARM_utils_load_model import load_PARM
from PARM_predict import sequence_to_onehot, get_prediction
import PARM_misc
from tqdm import tqdm
from matplotlib import pyplot as plt, colors
import logomaker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def PARM_mutagenesis(
    input,
    model_weights,
    motif_database,
    parm_version,
    output_directory,
):
    """
    Reads the input fasta file and computes the mutagenesis using the PARM models.
    For each sequence in the fasta, it writes three outputs: the mutagenesis matrix,
    the motif hits, and the pdf plot of the assay.
    """
    # Initiate PARMscores dict: {K562: {seq1: score_seq1, seq2: score_seq2},
    #                            HepG2:{seq1: score_seq1, seq2: score_seq2}}
    parm_scores = dict()
    # Loading models
    complete_models = dict()
    for model in model_weights:
        model_name = os.path.basename(model).split(".parm")[0]
        PARM_misc.log(f"Loading model {model_name}", parm_version)
        complete_models[model_name] = load_PARM(model)
        parm_scores[model_name] = dict()

    # Loading motif database
    PARM_misc.log("Loading motif database", parm_version)
    motif_db_PFM, _, motif_db_ICT = load_motif_db(motif_file=motif_database)
    # ====================================================================================
    # Parsing the fasta file =============================================================
    # ====================================================================================
    PARM_misc.log("Reading input file", parm_version)
    inputs = {"sequence": [], "ID": [], "output_directory": []}
    total_interactions = len(list(SeqIO.parse(input, "fasta"))) * len(model_weights)
    for record in SeqIO.parse(input, "fasta"):
        sequence_ID = record.id
        sequence = str(record.seq).upper()
        # If sequence is longer than 600 nt, it's not inputted to PARM
        if len(sequence) > 600:
            PARM_misc.log(
                f"WARNING: sequence {sequence_ID} is longer than 600 bp. Skipping...",
                parm_version,
            )
        else:
            inputs["sequence"].append(sequence)
            inputs["ID"].append(sequence_ID)
            inputs["output_directory"].append(
                os.path.join(output_directory, f"{sequence_ID}")
            )
    # ====================================================================================
    # Running mutagenesis ================================================================
    # ====================================================================================
    pbar = tqdm(total=total_interactions, ncols=80)
    # First, create output directory
    os.makedirs(os.path.join(output_directory, sequence_ID), exist_ok=True)
    #output_pdf = os.path.join(output_directory, sequence_ID, "mutagenesis.pdf")
    #output_pdf = PdfPages(output_pdf)
    for model_name, complete_model in complete_models.items():
        PARM_misc.log(f"\nRunning in-silico mutagenesis for {model_name}", parm_version)
        parm_score, mutagenesis_df, motif_scanning_results, sequence_onehot = (
            create_dataframe_mutation_effect(
                inputs=inputs,
                complete_model=complete_model,
                output_directory=output_directory,
                PFM_hocomoco_dict=motif_db_PFM,
                pbar=pbar,
            )
        )
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


def load_motif_db(
    motif_file="https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt",
):
    """
    Load all hocomoco motifs in a dictionary format from jaspar file.
    Default is use the human hocomoco v11: "https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt"
    Also, gets the reverse of each the motifs.

    Returns:
        PFM_hocomoco_dict: (dict) PFM in np.array of all motifs. Name of motifs as keys.
        consensus_hocomoco_dict: (dict) Consensus sequence in str of all motifs. Name of motifs as keys.
        ICT_hocomoco_dict: (dict) ICT in np.array of all motifs. Name of motifs as keys.
    """

    if motif_file.startswith("http"):
        handle = urllib.request.urlopen(motif_file)
        hocomoco_jaspar = io.TextIOWrapper(handle, encoding="utf-8")

    else:
        hocomoco_jaspar = open(motif_file)

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


def create_dataframe_mutation_effect(
    inputs,
    complete_model: torch.nn.Module,
    output_directory: str,
    PFM_hocomoco_dict: dict,
    pbar,
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

    # First, create all output directories
    for dir in inputs["output_directory"]:
        os.makedirs(dir, exist_ok=True)

    (attribution_all_seqs, parm_score, reference_onehot) = (
        compute_saturation_mutagenesis(
            seq=inputs["sequence"],
            completemodel=complete_model,
            pbar=pbar,
        )
    )
    
    # Saving results -- iterate over the attribution_all_seqs matrix
    for index_sequence, attribution in enumerate(attribution_all_seqs):
        this_sequence = inputs["sequence"][index_sequence]
        this_output = inputs["output_directory"][index_sequence]
        this_output_mutagenesis = os.path.join(this_output, "mutagenesis.txt.gz")
        
        #attribution = attribution.reshape(4,len(this_sequence))
        print('')
        print(this_output_mutagenesis, flush=True)
        
        
        attribution_df = pd.DataFrame(
            {
                "Ref": list(this_sequence),
                "A": attribution[0, :],
                "C": attribution[1, :],
                "G": attribution[2, :],
                "T": attribution[3, :],
            }
        )
        # Saving as csv
        attribution_df.to_csv(
            this_output_mutagenesis ,
            index=False,
            sep="\t",
            compression={"method": "gzip", "compresslevel": 5},
        )
    
    # # now compute the average importance for each nucleotide in the reference sequnece

    # onehot_seq = reference_onehot[0]
    # plot_promoter = promoter_importance_base[["A", "C", "G", "T"]]
    # vector_importance = plot_promoter.mean(axis=1)
    # mutagenesis_attribution = np.transpose(
    #     onehot_seq * np.expand_dims(vector_importance, 1)
    # )

    # hits = run_PARM_motif_scanning(  # old slide_through_attribution_and_PFM
    #     mutagenesis_attribution=mutagenesis_attribution,
    #     known_PFM=PFM_hocomoco_dict,
    #     threshold=0.6,
    #     cutoff_att=0.001,
    #     append=True,
    #     split_pos_neg=False,
    # )

    # hits.to_csv(
    #     file_hits,
    #     index=False,
    #     sep="\t",
    #     compression={"method": "gzip", "compresslevel": 5},
    # )
    # return parm_score, promoter_importance_base, hits, reference_onehot


def compute_saturation_mutagenesis(
    seq,
    completemodel,
    ref_to_alt_attribution=True,
    index_output=0,
    alt_nt=["A", "C", "G", "T"],
    window_del=False,
    pbar='',
):
    """
    Computes saturation mutagenesis of a given region of a sequence (or the complete one) to determine
    importance (or attribution) by the model.

    Basically it mutates every base to every other possible base.

    If ref_to_alt_attribution == False: The importance of each base is determined
    by the SuRE score of that base - the mean SuRE score of the other possible 3 bases.

    If ref_to_alt_attribution == True: The importance of each base is determined
    by the REF - SuRE score of that base.


     Args:
       seq: (str or list) sequence(s) to study
       L_max: (int) input length sequence of the model. Necessary for padding.
       start_motif: (int or list) start of the motif of interest, relative to seq. If list, it should have the same length as seq. All motifs must  to have the same length.
       end_motif: (int) end of the motif of interest, relative to seq.  If list, it should have the same length as seq. All motifs in the different sequences must to have the same length.
       completemodel: (fun) model used to compute attributions
       return_reference_score: (bool) If True, returns prediction of Reference sequence as well as att
       index_output: (int) Which index of the output to take, usuallydistribution if the model only outputs one cell line and/or replicates there's only one output value and index is 0.
       alt_nt: (list of str) List of possible nucleotides to mutate to in a list. --> ['A']
       window_del: (list of int) If it is not False, provide a list of integer(s) e.g. [1], or [1,2]. It will compute the effect of a deletion of the different sizes provided in the list.

     Returns:
       att: (np.array) array of shape (idx_output, n_sequences, n_bases, length_motif) if any of this dimensions is one, it will be removed.

    """

    # If seq is a list, we have multiple sequences or index ouputs
    if type(seq) != list:
        seq = [seq]
    if type(index_output) != list:
        index_output = [index_output]

    # Defining start and end motifs/sequences
    start_motif = [0 for _ in seq]
    end_motif = [len(i) for i in seq]
    L_max = end_motif
    size_motifs = end_motif[0] - start_motif[0]

    if torch.cuda.is_available():
        completemodel = completemodel.cuda()

    # Compute SuRE score of reference sequence
    # PARM_misc.log(f"LMax in line 328: {L_max}\n\n\n\n", '2')
    reference_onehot = sequence_to_onehot(seq, L_max, for_mutagenesis=True)
    reference = torch.tensor(np.float32(reference_onehot)).permute(0, 2, 1)
    if torch.cuda.is_available():
        reference = reference.cuda()

    with torch.no_grad():
        if len(seq) == 1 and len(index_output) == 1:
            parm_score = completemodel(reference)[0, index_output].detach().cpu()
        else:
            parm_score = completemodel(reference)[:, index_output].detach().cpu()

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
#            pbar.update(1)

            # The sequences are going to be saved for each position the four nucleotides,
            # so if we compute all the possible sequences, and the model returns the predictions
            # in a flatten array we have to reconvert the array in shape base x nucleotides.

    with torch.no_grad():
        alt_reference = torch.tensor(
            np.float32(sequence_to_onehot(seqs_position_mutation, L_max, for_mutagenesis=True))
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
                        parm_score[it_seq, idx_output] - attribution[it_seq, :, :]
                    )

                att.append(att_per_seq)
            else:
                att = parm_score[idx_output] - attribution[0, :, :]

    else:  # If we are interested in the contribution of each base as REF-mean(ALT). --> Not done for deletion

        att = []
        for idx_output, attribution in enumerate(
            att_per_index
        ):  # Loop if there's more than one index output
            att_seq = []
            for it_seq in range(len(seq)):  # Loop through sequence
                mutagenesis_attribution = attribution[it_seq, :, :]

                if window_del is not False:
                    index_no_del = np.where([sub != "-" for sub in alt_nt])[
                        0
                    ]  # Check indiced where there's no deletion
                    mutagenesis_attribution = mutagenesis_attribution[index_no_del, :]

                idx = list(range(mutagenesis_attribution.shape[0]))
                att_seq.append(
                    torch.stack(
                        [
                            mutagenesis_attribution[x]
                            - mutagenesis_attribution[idx[:x] + idx[x + 1 :]].mean(
                                axis=0
                            )
                            for x in idx
                        ]
                    )
                )
            att.append(att_seq)

    att = np.squeeze(np.array(att))  # Remove one size dimensions

    ##Free space
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(parm_score) == 1:
        parm_score = parm_score[0]

    return (att, parm_score, reference_onehot)


def run_PARM_motif_scanning(
    known_PFM,
    mutagenesis_attribution,
    threshold=0.6,
    cutoff_att=0.001,
    append=True,
    split_pos_neg=False,
):
    """
    Slides through sequence, computes attribution and then check which motifs and where have a high correlation.

    Args:
        known_PFM: (dict) dict of known PFM with name of PFM as keys, optionally can also be used IC instead of PFM.
        mutagenesis_attribution: (np.array) Array showing attribution of sequence obtained from in silico mutagenesis.
        attribution: (bool) If true returns as an extra column the mean attribution * PFM of that hit.
        threshold: (float) Threshold to consider a good match between known TFBS and computed attribution.
        append: (bool) if True, then in the mutagenesis_attribution a matrix of (4,5) is appended on the end and start of the sequence.
                                In this way it allows for motifs scanning to also find hits in the edges.
        multiple_one_hot (np.array): If not False, provide one-hot encoded sequence, and the scanning will be done with the normalized sequence*one_hot.
        normalize: (bool) If normalize each base to add up to 1. If IC instead of PFM provided we recommend to not use it.
        split_pos_neg: (bool) If True, split the attribution between negative and positive values.

    Returns:
        hits: (list) Hits of list that contains [motif_name, correlation, start_hit, end_hit]

    """

    zeros_padding = np.zeros(shape=(4, 5))
    mutagenesis_attribution = np.concatenate(
        [zeros_padding, mutagenesis_attribution, zeros_padding], axis=1
    )
    length_sequence = mutagenesis_attribution.shape[-1]
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

            mutagenesis_attribution_short = mutagenesis_attribution[
                :, start_motif:end_motif
            ]

            # Avoid computing hits if the attributions are super small
            if np.abs(cutoff_att).sum() < cutoff_att:
                continue

            # Loop through motifs in the same length group

            for name_motif in name_motifs_same_length:

                PFM = known_PFM[name_motif]
                sum_motif = PFM.sum().sum()

                rho_prob = np.corrcoef(
                    [mutagenesis_attribution_short.flatten(), PFM.flatten()]
                )[0, 1]

                # if attribution is not False:

                ##If there's a hit --> Save the product of the attribution x PFM --> Useful to later take the longest motif
                if (
                    (rho_pos_prob > threshold)
                    or (rho_neg_prob > threshold)
                    or (np.abs(rho_prob) > threshold)
                ):
                    if split_pos_neg:
                        mutagenesis_attribution_short = mutagenesis_attribution[
                            :, start_motif:end_motif
                        ]
                    att_x_PFM = (
                        mutagenesis_attribution_short.flatten() * PFM.flatten()
                    ).mean()
                    att_pos = (mutagenesis_attribution_short.flatten()).sum()
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

    return hits


def plot_mutagenesis_results(
    sequence,
    mutagenesis_df,
    motif_scanning_results,
    sequence_onehot,
    sequence_score,
    model_name,
    motif_db_ICT,
):

    # Create Series of the positions
    positions = pd.Series(list(range(len(sequence))))
    mutagenesis_df["positions"] = positions

    # Start creating the plotting df
    plotting_df = mutagenesis_df[["A", "C", "G", "T"]]
    # Make positions as index
    plotting_df.index = positions.astype(str)
    onehot_seq = sequence_onehot[0]

    # Start axis
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

    # ====================================================================================
    # Row 1: DNA sequence; ax[0,0]
    plot_logo(  # old logomaker_plots_individual
        mutagenesis_df=np.transpose(onehot_seq),
        ax=ax[0, 0],
        y_label="",
    )
    ax[0, 0].set_title(f"PARM {model_name}: {sequence_score}", fontsize="large")
    # remove y-axis info from this row
    ax[0, 0].yaxis.set_tick_params(labelleft=False)
    ax[0, 0].set_yticks([])
    # ====================================================================================
    # Row 2: attribution track;  ax[1,0] =================================================
    # ====================================================================================
    # First, calculate the attribution track
    attribution = plotting_df.mean(axis=1, skipna=True)
    attribution = np.transpose(onehot_seq * np.expand_dims(attribution, 1))
    plot_logo(
        mutagenesis_df=attribution,
        ax=ax[1, 0],
        y_label="Variant effect\n(REF-mean(ALT))",
    )
    # ====================================================================================
    # Row 3: heatmap track; ax[2,0] ======================================================
    # ====================================================================================
    plotting_df_transposed = plotting_df.T
    mask = plotting_df_transposed.isnull()
    max_abs_value = plotting_df.abs().max().max()
    _ = sns.heatmap(
        plotting_df_transposed,
        ax=ax[2, 0],
        cmap="RdBu",
        center=0,
        xticklabels=10,
        cbar=False,
        mask=mask,
        vmin=(-max_abs_value),
        vmax=max_abs_value,
    )
    ax[2, 0].set_ylabel(f"")
    ax[2, 0].set_xlabel(f"Position (bp)")
    fig.colorbar(
        ax[2, 0].get_children()[0],
        ax=ax[2, 1],
        orientation="vertical",
        fraction=0.5,
        label="Variant effect\n(REF-ALT)",
    )
    # ====================================================================================
    # Row 4 (and 5): scanned motifs; ax[4,0] =============================================
    # ====================================================================================
    plot_scanned_motifs(
        attribution=attribution,
        ax=[ax[4, 0], ax[5, 0]],
        motif_scanning_results=motif_scanning_results,
        motifs_ICT=motif_db_ICT,
        fig=fig,
    )
    return (fig, ax)


def plot_logo(mutagenesis_df, ax: plt.axis, y_label: str):

    # Parse mutagenesis_df
    mutagenesis_to_plot = pd.DataFrame(
        np.transpose(mutagenesis_df), columns=["A", "C", "G", "T"]
    )
    # Define colors
    color_scheme = {
        "A": colors.to_rgb("#00811C"),  # green
        "C": colors.to_rgb("#2000C7"),  # blue
        "G": colors.to_rgb("#FFB32C"),  # yeellow
        "T": colors.to_rgb("#D00001"),
    }  # red

    plot = logomaker.Logo(
        mutagenesis_to_plot, ax=ax, color_scheme=color_scheme, baseline_width=1
    )

    plot.style_spines(visible=False)
    plot.style_spines(spines=["left"], visible=True)
    plot.ax.set_ylabel(y_label, labelpad=-1, fontsize="medium")
    plot.ax.set_xticks([])

    return plot


def plot_scanned_motifs(attribution, ax, motif_scanning_results, motifs_ICT, fig):
    attribution_shape = attribution.shape
    length_sequence = attribution_shape[1]
    percentage = 0.15  # Only do the scanning of the motifs that attributions is at least 1-percentage the attribution of the max motif

    motif_scanning_results["att_abs"] = np.abs(motif_scanning_results.att)

    # Create empty array and fill it with hits
    relevant_hits_row1 = np.empty(attribution_shape)
    # relevant_hits_row1[:,:] = 0
    relevant_hits_row2 = np.empty(attribution_shape)
    # relevant_hits_row2[:,:] = 0

    # Take only hits that their mean attribution is at least 10% of the letter wiht max attribution
    hits = motif_scanning_results[
        motif_scanning_results.att_x_PFM.abs()
        > (motif_scanning_results.att_x_PFM.abs().max() * percentage)
    ]
    # Avoid overlapping hits
    hits["rho_abs"] = np.abs(hits.rho)
    hits = hits.sort_values("start")
    hits["mid_point"] = ((hits.end + hits.start) / 2).astype(int)

    # Cut the promoters in bins based on the length of 10
    bins = np.arange(0, length_sequence, 10)
    ind = np.digitize(hits.mid_point, bins)
    hits["bin"] = ind

    # Get length of motifs
    hits["len"] = hits.end - hits.start

    hits = (
        hits.groupby(ind)
        .apply(lambda x: x.nlargest(2, ["rho_abs"], keep="first"))
        .reset_index(drop=True)
    )
    previous_start, previous_end = -10, -10
    hits = hits.sort_values(["bin", "rho_abs"], ascending=False)

    if hits.empty:  # If no hits, delete axis
        fig.delaxes(ax[0])
        fig.delaxes(ax[1])
        return None

    # Now get which motifs will be plot
    hits_to_plot_row1, hits_to_plot_row2 = [], []
    for _, motif in hits.iterrows():
        start_motif, end_motif = motif[2], motif[3]
        overlap = max(
            0, min(end_motif, previous_end) - max(start_motif, previous_start)
        )
        # If the start of the motif it's a bit shifted we need to make the known motif a bit shorter s well

        if end_motif > length_sequence:
            end_motif = length_sequence
            rel_end = length_sequence - start_motif
        else:
            rel_end = end_motif - start_motif

        if start_motif < 0:
            rel_start = -start_motif
            start_motif = 0
        else:
            rel_start = 0

        if overlap == 0 or (overlap > 0 and previous_overlap > 0):
            relevant_hits_row1[:, start_motif:end_motif] = motifs_ICT[motif[0]][
                :, rel_start:rel_end
            ]
            hits_to_plot_row1.append(motif)
        else:
            relevant_hits_row2[:, start_motif:end_motif] = motifs_ICT[motif[0]][
                :, rel_start:rel_end
            ]
            hits_to_plot_row2.append(motif)

        previous_start, previous_end, previous_overlap = start_motif, end_motif, overlap

    # Do the same for both axis
    for it, sub_ax in enumerate(ax):

        if it == 0:
            relevant_hits = relevant_hits_row1
            hits_to_plot = hits_to_plot_row1

        if it == 1:
            relevant_hits = relevant_hits_row2
            hits_to_plot = hits_to_plot_row2

        tf_mut_score_plot = pd.DataFrame(
            np.transpose(relevant_hits), columns=["A", "C", "G", "T"]
        )
        color_scheme = {
            "A": colors.to_rgb("#00811C"),  # green
            "C": colors.to_rgb("#2000C7"),  # blue
            "G": colors.to_rgb("#FFB32C"),  # yeellow
            "T": colors.to_rgb("#D00001"),
        }  # red

        crp_logo = logomaker.Logo(
            tf_mut_score_plot, ax=sub_ax, color_scheme=color_scheme, baseline_width=0
        )

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
