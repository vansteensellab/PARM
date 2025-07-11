import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import os
from .PARM_utils_load_model import load_PARM
from .PARM_misc import log, check_sequence_length
from tqdm import tqdm


def PARM_predict(
    input: str,
    output: str,
    model_directory: str,
    n_seqs_per_batch: int = 1,
    store_sequence: bool = True,
    filter_size: int = 125,
    type_loss: str = "poisson",
    test_fold: bool = False,
    L_max: int = 600,
):
    """
    Reads the input (fasta file) and predicts promoter activity scores using the PARM models.
    Writes the output as tab-separated values, where each column is a model and each row is a sequence.

    Parameters
    ----------
    input : str
        Path to the input fasta file.
    output : str
        Path to the output file.
    model_directory : str
        Path to the directory containing all the folds of the PARM models. (e.g. K562/* , with K562/K562_fold1.parm, K562_fold2.parm, etc.)
    n_seqs_per_batch : int
        Number of batches to use for prediction. If your GPUs runs out of memory, might be because of that. Default is 1.
    store_sequence : bool
        If True, the output file will contain the sequences and headers. Otherwise, only the headers and score will be saved. Default is True.
    filter_size : int
        Size of the filter to use for the PARM model. Default is 125.
    type_loss : str
        Type of loss function to use for the model. Default is 'poisson'. Other options are 'MSE' and 'heteroscedastic'.
    test_fold : bool
        If True, the function will consider that the input is a hdf5 file with the test fold data. Default is False.
    -------
    None

    Examples
    --------
    >>> PARM_predict("input.fasta", "output.tsv", ["model1.parm", "model2.parm"])
    """
    # Load models
    log("Loading models")
    list_of_models = list()
    # Iterate over the model_directory and get the folds
    path_to_all_folds = []
    for file in os.listdir(model_directory):
        if file.endswith(".parm"):
            path_to_all_folds.append(os.path.join(model_directory, file))
    if len(path_to_all_folds) == 0:
        raise ValueError(
            f"No model files (.parm) found in {model_directory}. Please check the path and ensure it contains the model files."
        )
    # Now, load the models
    log(f"Found {len(path_to_all_folds)} model files in {model_directory}")
    model_name = ""
    for fold_path in path_to_all_folds:
        if model_name == "":
            model_name = os.path.basename(fold_path).split("_fold")[0]
        elif model_name != os.path.basename(fold_path).split("_fold")[0]:
            raise ValueError(
                f"Model prefixes do not match: {model_name} and {os.path.basename(fold_path).split('_fold')[0]}. Please make sure that folds of the same model have the same prefix"
            )
        list_of_models.append(
            load_PARM(
                fold_path, filter_size=filter_size, train=False, type_loss=type_loss
            )
        )
    if test_fold:
        # If test_fold is True, we assume that the input is a hdf5 file with the test fold data
        # perform the test set prediction and create measured vs predicted plot
        log("Performing test fold predictions")
        # Use the input path as the test fold HDF5 file path
        test_fold_path = input

        # Create output directory for test results if it doesn't exist
        test_output_dir = output

        get_test_fold_predictions(
            test_fold_path=test_fold_path,
            list_of_models=list_of_models,
            cell_type=model_name,
            output_directory=test_output_dir,
        )
        return
    # Default behaviour: input is a fasta file with sequences to predict
    # Iterate over sequences and predict scores
    # Check input fasta
    check_sequence_length(input, L_max)
    log("Making predictions")
    total_sequences = sum(1 for _ in SeqIO.parse(input, "fasta"))
    log(f"Total sequences: {total_sequences}")
    total_interactions = total_sequences * len(list_of_models)
    pbar = tqdm(total=int(total_interactions / n_seqs_per_batch), ncols=80)

    i = 0
    for i_record, record in enumerate(SeqIO.parse(input, "fasta")):
        # Initiate output df
        sequence = str(record.seq).upper()
        if i == 0:
            tmp = pd.DataFrame({"sequence": [sequence], "header": [record.id]})
        else:
            tmp = pd.concat(
                [tmp, pd.DataFrame({"sequence": [sequence], "header": [record.id]})]
            )

        # Get predictions for all models
        if (i + 1) == n_seqs_per_batch or (i_record == (total_sequences - 1)):
            predictions_all_folds = []
            for model in list_of_models:
                predictions_all_folds.append(
                    get_prediction(tmp.sequence.to_list(), model)
                )
                pbar.update(1)
            # Now, take the average of the predictions and add to the tmp[model_name]
            tmp["prediction_" + model_name] = np.mean(predictions_all_folds, axis=0)

            # Store in output df
            # IF it's the first batch, save the df with headers, otherwise, save only the scores
            if i_record < n_seqs_per_batch:
                if store_sequence:
                    tmp.to_csv(output, sep="\t", index=False)
                else:
                    (tmp.drop(columns=["sequence"])).to_csv(
                        output, sep="\t", index=False
                    )
            else:
                if store_sequence:
                    tmp.to_csv(output, sep="\t", index=False, mode="a", header=False)
                else:
                    (tmp.drop(columns=["sequence"])).to_csv(
                        output, sep="\t", index=False, mode="a", header=False
                    )

            i = 0

        else:
            i += 1
    # Write output
    pbar.close()
    log("Finish output file")


def get_test_fold_predictions(
    test_fold_path, list_of_models, cell_type, output_directory
):
    """
    Perform predictions on test fold data and create measured vs predicted plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from scipy.stats import pearsonr
    import h5py

    # make output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log(f"Loading test fold data from {test_fold_path}")

    # Load HDF5 file directly
    with h5py.File(test_fold_path, "r") as f:
        # Load sequences (one-hot encoded)
        sequences = f["X"]["sequence"]["OneHotEncoding"][:]
        # Load measured values
        measured = f["Y"][f"Log2RPM_{cell_type}"][:]
        # Load the feature names of each fragment
        # temporary: create a vector that pastes the FEATstart and FEATend fields
        feature_names = np.char.add(
            np.char.add(f["FEAT"]["FEATstart"][:].astype(str), "_"),
            np.char.add(f["FEAT"]["FEATstart"][:].astype(str), "_"),
        )
        feature_names = np.char.add(feature_names, f["FEAT"]["FEATend"][:].astype(str))
        # feature_names = f['FEAT']

    log(f"Loaded {len(sequences)} test fragments")

    # Make predictions with each model
    all_predictions = []
    batch_size = 32
    for i, model in enumerate(list_of_models):
        log(f"Making predictions with model fold {i}")
        model.eval()
        predictions = []

        with torch.no_grad():
            for start_idx in range(0, len(sequences), batch_size):
                end_idx = min(start_idx + batch_size, len(sequences))
                batch_sequences = sequences[start_idx:end_idx]

                # Convert to tensor and move to GPU if available
                X = torch.tensor(batch_sequences, dtype=torch.float32).permute(0, 2, 1)
                if torch.cuda.is_available():
                    X = X.cuda()
                    model = model.cuda()

                # Make predictions
                pred = model(X).cpu().detach().numpy()
                predictions.append(pred)

        # Concatenate all batch predictions
        model_predictions = np.concatenate(predictions, axis=0)
        all_predictions.append(model_predictions)

    # Average predictions across all models
    avg_predictions = np.mean(all_predictions, axis=0).flatten()
    measured_flat = measured.flatten()

    # Now, make a dataframe with the predictions and measured values, and the feature names
    results_df = pd.DataFrame(
        {
            "measured_Log2PM": measured_flat,
            "predicted_Log2RPM": avg_predictions,
            "feature": feature_names,
        }
    )

    # Calculate correlation
    pearson_r, _ = pearsonr(measured_flat, avg_predictions)

    log(f"Pearson correlation (fragment level): {pearson_r:.3f}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 7))

    h = ax.hist2d(
        avg_predictions, measured_flat, bins=100, norm=colors.LogNorm(), cmap="viridis"
    )

    # Add correlation annotation
    ax.annotate(
        f"R = {pearson_r:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add diagonal line
    min_val = min(avg_predictions.min(), measured_flat.min())
    max_val = max(avg_predictions.max(), measured_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)

    ax.set_xlabel("Predicted Log2RPM", fontsize=12)
    ax.set_ylabel("Measured Log2RPM", fontsize=12)
    ax.set_title(f"Test Fold Results - {cell_type}", fontsize=14)

    plt.colorbar(h[3], ax=ax, label="Fragment count")

    # Save plot
    plot_path = os.path.join(
        output_directory, f"test_fold_scatter_{cell_type}_fragment_level.svg"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    results_path = os.path.join(
        output_directory, f"test_fold_predictions_{cell_type}.tsv"
    )
    results_df.to_csv(results_path, sep="\t", index=False)

    # Now, group prediction by feature and plot the measured vs. predicted values
    grouped_results = (
        results_df.groupby("feature")
        .agg({"measured_Log2PM": "mean", "predicted_Log2RPM": "mean"})
        .reset_index()
    )

    # plot the grouped results
    fig, ax = plt.subplots(figsize=(8, 7))
    h = ax.hist2d(
        grouped_results["predicted_Log2RPM"],
        grouped_results["measured_Log2PM"],
        bins=100,
        norm=colors.LogNorm(),
        cmap="viridis",
    )
    # Add correlation annotation
    pearson_r_grouped, _ = pearsonr(
        grouped_results["measured_Log2PM"], grouped_results["predicted_Log2RPM"]
    )
    
    log(f"Pearson correlation (feature level): {pearson_r_grouped:.3f}")
    
    ax.annotate(
        f"R = {pearson_r_grouped:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    # Add diagonal line
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel("Predicted Log2RPM", fontsize=12)
    ax.set_ylabel("Measured Log2RPM", fontsize=12)
    ax.set_title(f"Test Fold Results - {cell_type} (Grouped by Feature)", fontsize=14)
    plt.colorbar(h[3], ax=ax, label="Feature count")
    # Save plot
    plot_grouped_path = os.path.join(
        output_directory, f"test_fold_scatter_{cell_type}_feature_level.svg"
    )
    plt.savefig(plot_grouped_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_prediction(sequence, complete_model):
    """
    Predicts promoter activity score for input sequence
    """
    # Check if sequence is a list or not and make it a list if not
    if not isinstance(sequence, list):
        sequence = [sequence]

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
