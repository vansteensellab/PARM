from pip.cmdoptions import ignore_requires_python
import numpy as np
import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import math
import torch
import torch.nn as nn
from .PARM_utils_load_model import load_PARM
from .PARM_utils_data_loader import (
    pad_collate,
    shuffle_batch_sampler,
    h5_dataset,
    gradual_warmup_scheduler,
    index_of_interest,
)
from tqdm import tqdm
from .PARM_misc import log


def PARM_train(args):
    #############
    # 1. Load arguments
    input_directory = args.input
    output_directory = args.output
    adaptor = args.adaptor
    L_max = args.L_max
    scheduler = args.cosine_scheduler
    weight_decay = args.weight_decay
    filtering_on_FEAT = args.filtering_on_FEAT
    validation_path = args.validation
    if type(validation_path) != list:
        validation_path = list(validation_path)

    # Arguments for training
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    betas = args.betas
    lr = args.lr
    if len(betas) != 2:
        sys.exit(
            f"Error: Wrong values of betas. You must provide two values, you provided {len(betas)}"
        )
    initial_weights = args.initial_weights
    #############
    # 3. Create output directory

    if not os.path.exists(output_directory):
        os.makedirs(
            output_directory
        )  # Create folder where all the output is going to be saved
    # Create subfolders for the output
    if not os.path.exists(os.path.join(output_directory, 'temp_models')):
        os.makedirs(os.path.join(output_directory, 'temp_models'))
    if not os.path.exists(os.path.join(output_directory, 'performance_stats')):
        os.makedirs(os.path.join(output_directory, 'performance_stats'))
    # All loging functions will be saved in a file
    #f = open(os.path.join(output_directory, "log.txt"), "w")
    
    log(f"GPU detected? {torch.cuda.is_available()}")

    # Check if validation data is in training data
    error = any(
        file_validation in input_directory for file_validation in validation_path
    )
    if error:
        sys.exit("Error: Your validation data is in your trainning data.")

    #############
    # 4. Run models

    param_model = {
        "output_directory": output_directory,
        "input_directory": args.input,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "betas": betas,
        "lr": lr,
        "adaptor": adaptor,
        "L_max": L_max,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "validation_path": validation_path,
        "filter_size": args.filter_size,
        "n_block": args.n_blocks,
        "cell_type": args.cell_type,
        "n_workers": args.n_workers,
        "initial_weights": initial_weights,
        "filtering_on_FEAT" : filtering_on_FEAT
    }

    objective(**param_model)


def objective(
    output_directory,
    input_directory,
    batch_size,
    n_epochs,
    betas,
    lr,
    L_max,
    weight_decay,
    validation_path,
    n_block,
    filter_size,
    cell_type,
    scheduler,
    adaptor=(False, False),
    n_workers=0,
    initial_weights=None,
    filtering_on_FEAT=False
):
    """
    Objetive function to train and validate models.

    Args:
        output_directory: (str) Directory where we want to save all output files.
        input_directory: (str) Directory where one_hot_encoding directory is with hdf5 files for training and validation.
        batch_size: (int) Batch size for training.
        n_epochs: (int) Number of total epochs.
        betas: (tuple) L1 and L2 regularization respectively.
        lr: (float) Learning rate.
        scheduler: (bool) If True, use cosine scheduler.
        adaptor: (tuple) Tuple with adaptor in 5' and adaptor in 3' in this order. If not false they are going to be used for padding.
        weight_decay: (float) Weight decay of loss
        validation_path: (str) Path to validation file hdf5.
        n_block: (int) Number of blocks in the model.
        filter_size: (int) Filter size in the model.
        cell_type: (str) Cell type to train the model.
        n_workers: (int) Number of workers to use in data loading.
        initial_weights: (str) Path to initial weights file. If None, random initialization is used.

    Returns:

    """
    log("Preparing for training")
    warmup = True
    type_optimizer = "Adam"
    padding_alternate = True
    gradient_clipping = 0.2

    ##################################
    ##Define losses

    criterion = nn.PoissonNLLLoss(log_input=False)

    def criterion(pred, y, mask=None):
        poisson = nn.PoissonNLLLoss(log_input=False)
        if mask:
            pred = pred.flatten()[mask.flatten()]
            y = y.flatten()[mask.flatten()]
        return poisson(pred, y)

    ###############################################
    ###Load model


    if initial_weights is None:
        log("Initializing model with random weights")
        model = load_PARM(
            L_max=L_max,
            n_block=n_block,
            filter_size=filter_size,
            train=True,
        )
    else:
        log(f"Initializing model using weights in {initial_weights}")
        model = load_PARM(
            weight_file=initial_weights,
            L_max=L_max,
            n_block=n_block,
            filter_size=filter_size,
        )
    dummybatch = torch.zeros(1, 4, L_max)

    if torch.cuda.is_available():
        model = model.cuda()
        dummybatch = dummybatch.cuda()

    _ = model(dummybatch)

    ################################################
    # Define optimizer

    # defining the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ################################################
    # Load dataset

    params = {
        "num_workers": n_workers,
        "pin_memory": False,
        "collate_fn": (
            pad_collate(adaptor_5=adaptor[0], adaptor_3=adaptor[1], L_max=L_max)
            if padding_alternate
            else None
        ),
    }

    ##Take dataset of interest

    #### TRAINING
    # Done inside of epoch loop

    if type(input_directory) != list:
        input_directory = [input_directory]

    # Goes through all the input files and select the indeces of the file you are interested in.
    # TODO: Clean up this section
    index_dataset_train = np.empty((2, 0), dtype=int)
    for i, directory in enumerate(input_directory):
        training_set = h5_dataset(
            path=directory, celltype=cell_type
        )

        index_train_ind = np.arange(len(training_set))

        if filtering_on_FEAT:
            index_train_ind = index_of_interest(data_length = len(training_set), 
                                                            file_path = directory, 
                                                            features = filtering_on_FEAT)

        index_dataset_train = np.append(
            index_dataset_train,
            np.array([index_train_ind, np.repeat(i, len(index_train_ind))]),
            axis=1,
        )

    index_dataset_train = np.transpose(index_dataset_train)
    training_set = h5_dataset(path=input_directory, celltype=cell_type)
    log(f"Number of fragments shorter than {L_max}: {index_dataset_train.shape[0]}")

    sampler = shuffle_batch_sampler(
        index_dataset_train, batch_size=batch_size, drop_last=False
    )
    training_generator = torch.utils.data.DataLoader(
        training_set, sampler=sampler, **params
    )

    #### VALIDATION

    index_dataset_valid = np.empty((2, 0), dtype=int)
    for i, directory in enumerate(validation_path):
        validation_set = h5_dataset(path=directory, celltype=cell_type)

        index_valid_ind = np.arange(len(validation_set))

        index_dataset_valid = np.append(
            index_dataset_valid,
            np.array([index_valid_ind, np.repeat(i, len(index_valid_ind))]),
            axis=1,
        )
    index_dataset_valid = np.transpose(index_dataset_valid)

    # This take into account different type of inputs. In case the folds are defined directly written as valid.

    validation_set = h5_dataset(path=validation_path, celltype=cell_type)

    sampler = shuffle_batch_sampler(
        index_dataset_valid, batch_size=batch_size, drop_last=False
    )

    validation_generator = torch.utils.data.DataLoader(
        validation_set, sampler=sampler, **params
    )

    ################################################
    # Load schedueler and/or warmer
    total_steps = (len(training_generator) * batch_size * n_epochs) / batch_size

    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=0, last_epoch=-1,
        )

    if warmup:
        scheduler = gradual_warmup_scheduler(
            optimizer, multiplier=1.0, total_epoch=5000, after_scheduler=scheduler
        )

    ################################################

    # empty list to store training and validation losses
    train_losses, val_losses, results = [], [], []
    for epoch in range(n_epochs):
        log(f"{'-'*20} Epoch {epoch}/{n_epochs} {'-'*20}")

        ########### TRAINING LOOP
        # Get total_iterations for the progress bar
        total_iterations_train = len(training_generator)

        ##Data  generator. Redo in every step so there;s random order
        log("  Start training")
        _, _, training_loss = train_loop(
            train_dataloader=training_generator,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            betas=betas,
            gradient_clipping=gradient_clipping,
            total_iterations=total_iterations_train,
            this_epoch=epoch,
        )

        sampler = shuffle_batch_sampler(
            index_dataset_train, batch_size=batch_size, drop_last=False
        )

        training_generator = torch.utils.data.DataLoader(
            training_set, sampler=sampler, **params
        )

        # If Nan, there's something going on in the training. We stop the training.
        if math.isnan(training_loss):
            raise ValueError(
                f"Training loss is NaN in epoch {epoch}. Something is going wrong in the training. Check your data and your model."
            )

        ########### VALIDATION LOOP
        log("  Start validation")
        with torch.no_grad():
            # Get total_iterations for the progress bar
            total_iterations_val = len(validation_generator)
            y_val_predicted, y_val_true, val_loss = validation_loop(
                valid_dataloader=validation_generator,
                model=model,
                criterion=criterion,
                betas=betas,
                total_iterations=total_iterations_val,
                output_directory=output_directory,
                this_epoch=epoch,
                cell_type=cell_type,
            )
            results.append([epoch, training_loss, val_loss])

        torch.save(
            model.state_dict(),
            os.path.join(output_directory, 'temp_models', f"model_epoch_{epoch}.pth"),
        )

        for i_cell in range(len(cell_type.split("__"))):

            cell = cell_type.split("__")[i_cell] if cell_type else "cell_line"

            true_sub = y_val_true[:, i_cell].flatten()
            predicted_sub = y_val_predicted[:, i_cell].flatten()

            MSE = (((true_sub - predicted_sub) ** 2) ** (1 / 2)).mean()
            COEFF = r2_score(true_sub, predicted_sub)
            PCC = round(pearsonr(true_sub, predicted_sub)[0], 3)

            log(f"  Summary validation for {cell} cell line")
            log(f"\t R2 coefficient: {round(COEFF,4)}")
            log(f"\t Mean sq. error: {round(MSE,4)}")
            log(f"\t Pearson's correlation: {round(PCC,4)}")

    # TRAINING is complete.

    ##We've finished all epochs
    log(f"Finished training!")
    output_basename = os.path.basename(output_directory)
    log(f"Model saved in: {os.path.join(output_directory, f'{output_basename}.parm')}")
    torch.save(model.state_dict(), os.path.join(output_directory, f"{output_basename}.parm"))

    log(f"Saving dataframe and plots with results in {output_directory}")
    column_names = ["epoch", "training_loss", "validation_loss"]
    results = pd.DataFrame(results, columns=column_names)
    results.to_csv(
        os.path.join(output_directory, 'performance_stats', f"loss_per_epoch.txt"), index=False, sep="\t"
    )

    plt.plot(results["epoch"], results["training_loss"], label="Training loss")
    plt.plot(results["epoch"], results["validation_loss"], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss: Mean Squared Error (MSE)")
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'performance_stats', f"loss_per_epoch.png"))
    plt.clf()

    return val_loss


def train_loop(
    train_dataloader,
    model,
    criterion,
    optimizer,
    scheduler,
    betas,
    gradient_clipping=False,
    total_iterations=0,
    this_epoch=0,
    cell_type=False
):
    """
    Training loop.

    Args:
        train_dataloader: Train data in torch dataloader
        Model: Pytorch model
        criterion: (fun) loss function
        optimizer:
        scheduler:
        betas: (tuple) (int, int) Beta 1 and Beta 2 respectively for regularization.
        gradient_clipping: (float) If not False, then perform gradient clipping with that max norm.
        total_iterations: (int) Total number of iterations. (for the progress bar)
        this_epoch: (int) Current epoch.

    Returns:
        y_train_predicted: (np.array) Fragment predictions
        y_train_true: (np.array) Measured SuRE score, matching fragments with the one in y_train_predicted
        training_loss: (float) Loss performance of epoch.
    """

    model.train()
    training_loss = 0.0
    if cell_type: n_cell_lines = len(cell_type.split("__"))
    else: n_cell_lines = 1

    y_train_predicted, y_train_true = np.empty((0, n_cell_lines)), np.empty((0, n_cell_lines))

    pbar = tqdm(enumerate(train_dataloader), ncols=100, total=total_iterations, file=sys.stdout)
    
    for batch_ndx, (X, y) in pbar:

        optimizer.zero_grad()

        X = X.permute(0, 2, 1)
        y = torch.flatten(y, 1, 2)

        #Check in index, if there are some cell lines for which we dont have info
        index_info_cell = (~np.isnan(y.numpy()))

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        pred = model(X)

        if batch_ndx % 13 == 0:
            y_train_predicted = np.append(
                y_train_predicted, pred.cpu().detach().numpy(), axis=0
            )
            y_train_true = np.append(y_train_true, y.cpu().detach().numpy(), axis=0)

        if betas[0] != 0 or betas[1] != 0:

            l2_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())
            l1_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())

            loss = criterion(pred, y, mask=index_info_cell) + l2_norm * betas[1] + l1_norm * betas[0]

        else:
            loss = criterion(pred, y, mask=index_info_cell)

        # Backpropagation

        loss.backward()

        if gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

        optimizer.step()

        training_loss += loss.item()
        if scheduler:
            scheduler.step()

        loss_value = training_loss / (batch_ndx + 1)

        pbar.set_postfix(
            {
                "Epoch": this_epoch,
                "Loss": f"{loss_value:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.4f}",
            }
        )

    training_loss /= batch_ndx

    for i_cell in range(n_cell_lines):
        cell = cell_type.split("__")[i_cell] if cell_type else "cell_line"

        y_train_true_cell = y_train_true[:, i_cell].flatten()
        #Position of NaN
        y_train_true_cell_nan = np.isnan(y_train_true_cell)
        y_train_true_cell = y_train_true_cell[~y_train_true_cell_nan]
        y_train_predicted_cell = y_train_predicted[:, i_cell].flatten()[~y_train_predicted_cell_nan]

        mse = (((y_train_predicted_cell - y_train_true_cell) ** 2) ** (1 / 2)).mean()
        coeff = r2_score(y_train_true_cell, y_train_predicted_cell)
        pcc = pearsonr(y_train_true_cell.flatten(), y_train_predicted_cell.flatten())[0]
        log(f"Training summary for cell line {i_cell} {cell}")
        log(f"\t Avg. loss: {training_loss:>8f}")
        log(f"\t Mean sq. error: {mse:>3f}")
        log(f"\t R2 coefficient: {coeff:>3f}")
        log(f"\t Pearson's correlation: {pcc:>3f}")



def validation_loop(
    valid_dataloader, 
    model, 
    criterion, 
    betas, 
    total_iterations, 
    output_directory, 
    this_epoch,
    cell_type,
    ):
    """
    Validation loop.
    Args:
        valid_dataloader:
        model:
        criterion:

    Returns:
        y_val_predicted: (np.array) Fragment predictions
        y_valid_true: (np.array) Measured SuRE score, matching fragments with the one in y_train_predicted
        valid_loss: (float) Loss performance of epoch.
    """

    if cell_type: n_cell_lines = len(cell_type.split("__"))
    else: n_cell_lines = 1

    y_val_predicted, y_val_real = np.empty((0, n_cell_lines)), np.empty((0, n_cell_lines))

    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for batch_ndx, (X, y) in tqdm(
            enumerate(valid_dataloader),
            total=total_iterations,
        ):

            X = X.permute(0, 2, 1)
            y = torch.flatten(y, 1, 2)

            #Check in index, if there are some cell lines for which we dont have info
            index_info_cell = (~np.isnan(y.numpy()))

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            pred = model(X)

            if batch_ndx % 5 == 0:
                y_val_predicted = np.append(
                    y_val_predicted, pred.cpu().detach().numpy(), axis=0
                )
                y_val_real = np.append(y_val_real, y.cpu().detach().numpy(), axis=0)

            if betas[0] != 0 or betas[1] != 0:

                l2_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())
                l1_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())


                loss = criterion(pred, y, mask=index_info_cell) + l2_norm * betas[1] + l1_norm * betas[0]

            else:
                loss = criterion(pred, y, mask=index_info_cell)

            # Backpropagation

            val_loss += loss.item()

    val_loss /= batch_ndx

    for i_cell in range(n_cell_lines):

        cell = cell_type.split("__")[i_cell] if cell_type else "cell_line"

        y_val_real_cell = y_val_real[:, i_cell].flatten()
        #Position of NaN
        y_val_real_cell_nan = np.isnan(y_val_real_cell)
        y_val_real_cell = y_val_real_cell[~y_val_real_cell_nan]
        y_val_predicted_cell = y_val_predicted[:, i_cell].flatten()[~y_val_predicted_cell_nan]

        # Plot the predicted vs. measurements for this validation epoch
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist2d(
            y_val_predicted_cell,
            y_val_real_cell,
            bins=(100,100),
            norm=colors.LogNorm(),
            cmap="viridis",
        )
        corrfunc(y_val_predicted_cell, y_val_real_cell, ax=ax)
        ax.set_xlabel("Predicted Log2RPM")
        ax.set_ylabel("Measured Log2RPM")
        ax.set_title(f"Validation epoch {this_epoch} - {cell} cell type")
        plt.savefig(os.path.join(output_directory, 'performance_stats', f"validation_scatter_{this_epoch}_{cell}.svg"))
        
    return (y_val_predicted, y_val_real, val_loss)

def corrfunc(x, y, ax=None, **kws):
    """
    Plot the correlation coefficient in the top left hand corner of a plot.
    (for the correlation plot)
    """
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'R = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
