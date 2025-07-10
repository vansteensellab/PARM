
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
                 model_directory : str,
                 n_seqs_per_batch : int = 1,
                 store_sequence : bool = True,
                 filter_size : int = 125,
                 type_loss: str = 'poisson',
                 test_fold: bool = False):
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
    complete_models = dict()
    # Iterate over the model_directory and get the folds
    model_weights = []
    for file in os.listdir(model_directory):
        if file.endswith(".parm"):
            model_weights.append(os.path.join(model_directory, file))
    if len(model_weights) == 0:
        raise ValueError(f"No model files (.parm) found in {model_directory}. Please check the path and ensure it contains the model files.")
    # Now, load the models
    model_name = ""
    for model_weight in model_weights:
        if model_name == "":
            model_name = os.path.basename(model_weight).split("_fold")[0]
        elif model_name != os.path.basename(model_weight).split("_fold")[0]:
            raise ValueError(f"Model prefixes do not match: {model_name} and {os.path.basename(model_weight).split('_fold')[0]}. Please make sure that folds of the same model have the same prefix")
        complete_models["prediction_" + model_name] = load_PARM(model_weight, filter_size = filter_size, train=False,type_loss=type_loss)
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
            list_of_models=list(complete_models.values()), 
            cell_type=model_name,
            output_directory=test_output_dir
        )
        return
    # Default behaviour: input is a fasta file with sequences to predict
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
            predictions_all_folds = []
            for _, model in complete_models.items():
                predictions_all_folds.append(get_prediction(tmp.sequence.to_list(), model))
                pbar.update(1)
            # Now, take the average of the predictions and add to the tmp[model_name] 
            tmp["prediction_" + model_name] = np.mean(predictions_all_folds, axis=0)

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

def get_test_fold_predictions(test_fold_path, list_of_models, cell_type, output_directory):
    """
    Perform predictions on test fold data and create measured vs predicted plot.
    
    Parameters
    ----------
    test_fold_path : str
        Path to the HDF5 file containing test fold data.
    list_of_models : dict
        Dictionary of loaded PARM models.
    cell_type : str
        Cell type for the data.
    output_directory : str
        Directory to save the plot.
    
    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from scipy.stats import pearsonr
    import h5py
    from .PARM_utils_data_loader import h5_dataset, shuffle_batch_sampler, pad_collate
    
    log(f"Loading test fold data from {test_fold_path}")
    
    # Create dataset for test fold
    test_dataset = h5_dataset(path=[test_fold_path], celltype=cell_type)
    
    # Create indices for all samples in test dataset
    index_dataset_test = np.arange(len(test_dataset))
    index_dataset_test = np.array([index_dataset_test, np.zeros(len(index_dataset_test), dtype=int)]).T
    
    # Create data loader for test set
    batch_size = 32  # Use reasonable batch size for prediction
    params = {
        'batch_size': None,  # Will be handled by sampler
        'shuffle': False,
        'num_workers': 1,
        'collate_fn': pad_collate(L_max=600, alternative_padding=False)
    }
    
    sampler = shuffle_batch_sampler(
        index_dataset_test, batch_size=batch_size, drop_last=False
    )
    
    test_generator = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler, **params
    )
    
    log(f"Making predictions on {len(test_dataset)} test samples")
    
    # Collect predictions from all models
    all_predictions = []
    y_test_true = np.empty((0, 1))
    
    for model_name, model in list_of_models.items():
        log(f"Predicting with model: {model_name}")
        model.eval()
        
        y_test_predicted = np.empty((0, 1))
        y_test_real = np.empty((0, 1))
        
        with torch.no_grad():
            for batch_ndx, (X, y) in enumerate(test_generator):
                X = X.permute(0, 2, 1)
                y = torch.flatten(y, 1, 2)
                
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                
                pred = model(X)
                
                y_test_predicted = np.append(
                    y_test_predicted, pred.cpu().detach().numpy(), axis=0
                )
                y_test_real = np.append(y_test_real, y.cpu().detach().numpy(), axis=0)
        
        all_predictions.append(y_test_predicted)
        if len(y_test_true) == 0:  # Only store true values once
            y_test_true = y_test_real
    
    # Average predictions across all models
    y_test_predicted_avg = np.mean(all_predictions, axis=0)
    
    # Calculate metrics
    from sklearn.metrics import r2_score
    
    true_flat = y_test_true.flatten()
    pred_flat = y_test_predicted_avg.flatten()
    
    r2 = r2_score(true_flat, pred_flat)
    pearson_r, _ = pearsonr(true_flat, pred_flat)
    mse = np.mean((true_flat - pred_flat) ** 2)
    
    log(f"Test fold results:")
    log(f"\t R2 coefficient: {r2:.4f}")
    log(f"\t Pearson correlation: {pearson_r:.4f}")
    log(f"\t Mean squared error: {mse:.4f}")
    
    # Create measured vs predicted plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create 2D histogram
    h = ax.hist2d(
        pred_flat,
        true_flat,
        bins=100,
        norm=colors.LogNorm(),
        cmap="viridis"
    )
    
    # Add correlation coefficient annotation
    ax.annotate(f'R = {pearson_r:.3f}\nR² = {r2:.3f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add diagonal line for perfect prediction
    min_val = min(pred_flat.min(), true_flat.min())
    max_val = max(pred_flat.max(), true_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    ax.set_xlabel("Predicted Log2RPM", fontsize=12)
    ax.set_ylabel("Measured Log2RPM", fontsize=12)
    ax.set_title(f"Test Fold Results - {cell_type} cell type", fontsize=14)
    
    # Add colorbar
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Save plot
    plot_path = os.path.join(output_directory, f"test_fold_scatter_{cell_type}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"Test fold plot saved to: {plot_path}")
    
    # Save predictions to file
    results_df = pd.DataFrame({
        'measured': true_flat,
        'predicted': pred_flat
    })
    results_path = os.path.join(output_directory, f"test_fold_predictions_{cell_type}.tsv")
    results_df.to_csv(results_path, sep='\t', index=False)
    
    log(f"Test fold predictions saved to: {results_path}")
    

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
