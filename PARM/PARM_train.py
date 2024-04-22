import numpy as np
import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import math
from torchsummary import summary
import torch
import torch.nn as nn
from .PARM_utils_load_model import load_PARM
from .PARM_utils_data_loader import (
    pad_collate,
    shuffle_batch_sampler,
    h5_dataset,
    gradual_warmup_scheduler,
)
from .PARM_misc import log

log(f"\n Cuda working? {torch.cuda.is_available()}")


def PARM_train(args):
    #############
    # 1. Load arguments
    input_directory = args.input
    output_directory = args.output
    adaptor = args.adaptor
    L_max = args.L_max
    scheduler = args.cosine_scheduler
    weight_decay = args.weight_decay
    validation_path = args.validation
    if type(validation_path) != list:
        validation_path = list(validation_path)

    # Arguments for training
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    betas = args.betas
    lr = args.lr
    if len(betas) != 2:
        raise Exception(
            f"Wrong values of betas. You must provide two values, you provided {len(betas)}"
        )

    #############
    # 3. Create output directory

    if not os.path.exists(output_directory):
        os.makedirs(
            output_directory
        )  # Create folder where all the output is going to be saved

    # All loging functions will be saved in a file
    f = open(os.path.join(output_directory, "log.txt"), "w")

    log(f" Output directory: {output_directory}")
    sys.stdout = f

    log(f" Input Directory {input_directory}")

    # Check if validation data is in training data
    error = any(
        file_validation in input_directory for file_validation in validation_path
    )
    if error:
        raise ValueError("Error: Your validation data is in your trainning data.")

    log(f" Validation Directory {validation_path}")

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
    }

    objective(**param_model)
    f.close()


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

    Returns:

    """
    warmup = True
    type_optimizer = "Adam"
    padding_alternate = True
    gradient_clipping = 0.2

    log(f" ---- Selection of fragments ---- \n" f"      L max: {L_max} \n")

    log(
        f" ---- PARAMETERS LEARNING ---- \n"
        f"      Batch size:  {batch_size} \n"
        f"      Gradient clipping:  {gradient_clipping} \n"
        f"      Nº epoch: {n_epochs} \n"
        f"      Learning rate: {lr} \n"
        f"      Weight decay: {weight_decay} \n"
        f"      Type optimizer: {type_optimizer} \n"
        f"      Warmup: {warmup} \n"
        f"      Scheduler: {scheduler} \n"
        f"      Alternate type of paddings: {padding_alternate} \n"
        f"      Regularization: Beta1 {betas[0]} beta2 {betas[1]} \n"
    )

    log(f" ---- INPUT DATA ---- \n" f"      Adaptor: {adaptor} \n")

    ##################################
    ##Define losses

    criterion = nn.PoissonNLLLoss(log_input=False)

    ###############################################
    ###Load model

    # cell_type_strip_replicates = celltype.replace('pNK7_','').replace('_B','')
    model = load_PARM(L_max=L_max, n_block=n_block, filter_size=filter_size, train=True)
    dummybatch = torch.zeros(1, 4, L_max)

    if torch.cuda.is_available():
        model = model.cuda()
        dummybatch = dummybatch.cuda()

    # log(f' ---- MODEL ---- \n'
    #     f'{model} \n')

    _ = model(dummybatch)

    summary(model, (4, L_max))

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
        training_set = h5_dataset(path=directory, celltype=cell_type)

        index_train_ind = np.arange(len(training_set))

        index_dataset_train = np.append(
            index_dataset_train,
            np.array([index_train_ind, np.repeat(i, len(index_train_ind))]),
            axis=1,
        )

    index_dataset_train = np.transpose(index_dataset_train)
    training_set = h5_dataset(path=input_directory, celltype=cell_type)

    # index_dataset_train: shape (n_samples,info). dataset_index[:,0] --> index. dataset_index[:,1] --> n_file_dataset

    log(
        f"     Number of fragments after selection {index_dataset_train.shape} {index_dataset_train[:,0]}"
    )

    sampler = shuffle_batch_sampler(
        index_dataset_train, batch_size=batch_size, drop_last=False
    )
    training_generator = torch.utils.data.DataLoader(
        training_set, sampler=sampler, **params
    )

    #### VALIDATION

    ##feat_selection_percentage
    index_dataset_valid = np.empty((2, 0), dtype=int)
    for i, directory in enumerate(validation_path):
        validation_set = h5_dataset(path=validation_path, celltype=cell_type)

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
            optimizer, total_steps, eta_min=0, last_epoch=-1, verbose=False
        )

    if warmup:
        scheduler = gradual_warmup_scheduler(
            optimizer, multiplier=1.0, total_epoch=5000, after_scheduler=scheduler
        )

    ################################################

    # empty list to store training and validation losses
    train_losses, val_losses, results = [], [], []

    for epoch in range(n_epochs):
        log(
            f"------------------------------------------- \n --- Epoch {epoch}"
        )

        ########### TRAINING LOOP

        ##Data  generator. Redo in every step so there;s random order
        y_train_predicted, y_train_true, training_loss = train_loop(
            training_generator,
            model,
            criterion,
            optimizer,
            scheduler,
            output_directory,
            betas,
            gradient_clipping=gradient_clipping,
        )

        sampler = shuffle_batch_sampler(
            index_dataset_train, batch_size=batch_size, drop_last=False
        )

        training_generator = torch.utils.data.DataLoader(
            training_set, sampler=sampler, **params
        )

        # If Nan, there's something going on in the training. We stop the training.
        if math.isnan(training_loss):
            raise optuna.exceptions.TrialPruned()

        ########### VALIDATION LOOP

        with torch.no_grad():

            y_val_predicted, y_val_true, val_loss = validation_loop(
                validation_generator, model, criterion, output_directory, betas
            )
            results.append([epoch, training_loss, val_loss])

        torch.save(
            model.state_dict(),
            os.path.join(output_directory, f"model_epoch_{epoch}.pth"),
        )

        true_sub = y_val_true[:, 0].flatten()
        predicted_sub = y_val_predicted[:, 0].flatten()

        MSE = (((true_sub - predicted_sub) ** 2) ** (1 / 2)).mean()
        COEFF = r2_score(true_sub, predicted_sub)
        PCC = round(pearsonr(true_sub, predicted_sub)[0], 3)

        log(
            f"\n      Validation: Coeff R2 {round(COEFF,3)},  MSE: {round(MSE,2)}, PCC {round(PCC,3)}"
        )

        if (epoch) % 5 == 0 or epoch == (n_epochs - 1):
            torch.save(
                model.state_dict(),
                os.path.join(output_directory, f"tmp_model_epoch_{epoch}.parm"),
            )

    # TRAINING is complete.

    ##We've finished all epochs
    log(f"Training process has finished. Saving trained model.")

    torch.save(model.state_dict(), os.path.join(output_directory, f"model.parm"))

    column_names = ["epoch", "training_loss", "validation_loss"]

    results = pd.DataFrame(results, columns=column_names)

    results.to_csv(
        os.path.join(output_directory, f"results_model_PARM.txt"), index=False, sep="\t"
    )

    plt.plot(results["epoch"], results["training_loss"], label="Training loss")
    plt.plot(results["epoch"], results["validation_loss"], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss: Mean Squared Error (MSE)")
    plt.legend()
    plt.savefig(os.path.join(output_directory, f"loss_per_epoch.png"))
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

    Returns:
        y_train_predicted: (np.array) Fragment predictions
        y_train_true: (np.array) Measured SuRE score, matching fragments with the one in y_train_predicted
        training_loss: (float) Loss performance of epoch.
    """

    model.train()

    training_loss = 0.0
    y_train_predicted, y_train_true = np.empty((0, 1)), np.empty((0, 1))

    for batch_ndx, (X, y) in enumerate(train_dataloader):

        optimizer.zero_grad()

        X = X.permute(0, 2, 1)
        y = torch.flatten(y, 1, 2)

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        log("within y.shape", y.shape)
        pred = model(X)
        log("within pred.shape", pred.shape)

        if batch_ndx % 13 == 0:
            y_train_predicted = np.append(
                y_train_predicted, pred.cpu().detach().numpy(), axis=0
            )
            y_train_true = np.append(y_train_true, y.cpu().detach().numpy(), axis=0)

        if betas[0] != 0 or betas[1] != 0:

            l2_norm = sum(
                torch.norm(weight, p=2) for _, weight in model.named_parameters()
            )
            l1_norm = sum(
                torch.norm(weight, p=2) for _, weight in model.named_parameters()
            )

            loss = criterion(pred, y) + l2_norm * betas[1] + l1_norm * betas[0]

        else:
            loss = criterion(pred, y)

        # Backpropagation

        loss.backward()

        if gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

        optimizer.step()

        training_loss += loss.item()
        if scheduler:
            scheduler.step()

        # log results so far
        if batch_ndx % int(len(train_dataloader) / 20) == 0:
            loss, current = training_loss / (batch_ndx + 1), batch_ndx * len(X)
            perc = current / (len(train_dataloader) * X.shape[0]) * 100
            log(
                f" loss: {loss:>7f}  [{current}/{(len(train_dataloader)*X.shape[0])}]  {round(perc,3)}%"
            )

            for param_group in optimizer.param_groups:
                log(f"       Learning rate: {param_group['lr'] }")
                continue

    training_loss /= batch_ndx

    mse = (((y_train_predicted - y_train_true) ** 2) ** (1 / 2)).mean()

    log(f"Training Error: Avg loss: {training_loss:>8f}")
    log(f"                MSE {mse:>3f}")
    return (y_train_predicted, y_train_true, training_loss)


def validation_loop(valid_dataloader, model, criterion, betas):
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

    y_val_predicted, y_val_real = np.empty((0, 1)), np.empty((0, 1))

    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for batch_ndx, (X, y) in enumerate(valid_dataloader):

            X = X.permute(0, 2, 1)
            y = torch.flatten(y, 1, 2)

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

                l2_norm = sum(
                    torch.norm(weight, p=2) for _, weight in model.named_parameters()
                )

                l1_norm = sum(
                    torch.norm(weight, p=2) for _, weight in model.named_parameters()
                )

                loss = criterion(pred, y) + l2_norm * betas[1] + l1_norm * betas[0]

            else:
                loss = criterion(pred, y)

            # Backpropagation

            val_loss += loss.item()

    val_loss /= batch_ndx
    mse = (((y_val_predicted - y_val_real) ** 2) ** (1 / 2)).mean()

    log(f"Validation Error: Avg loss: {val_loss:>8f}")
    log(f"                  MSE {mse:>3f}")

    return (y_val_predicted, y_val_real, val_loss)



def validation_loop(valid_dataloader, model, output_directory, betas, cell_line:str):
    """
    Validation loop.
    Args:
        valid_dataloader:
        model:
        criterion:
        output_directory: (str) Output directory.
        cell_line: (str) Cell lines used to train and predict, if more than one separate by "_" e.g. K562_HEPG2
    
    Returns:
        y_val_predicted: (np.array) Fragment predictions
        y_valid_true: (np.array) Measured SuRE score, matching fragments with the one in y_train_predicted
        valid_loss: (float) Loss performance of epoch.
    """

    
    n_cels = len(cell_line.split('__'))
    y_val_predicted, y_val_real = np.empty((0,n_cels)), np.empty((0,n_cels))

    model.eval()

    val_loss = 0.0


    with torch.no_grad():
        for batch_ndx, (X, y) in enumerate(valid_dataloader):

            X = X.permute(0,2,1)
            y = torch.flatten(y, 1, 2)


            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            pred = model(X)

            if batch_ndx % 5 == 0:
                y_val_predicted = np.append(y_val_predicted, pred.cpu().detach().numpy(), axis=0)
                y_val_real = np.append(y_val_real, y.cpu().detach().numpy(), axis=0)

            


            if betas[0] != 0 or betas[1] != 0:

                l2_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())

                l1_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())

                loss = criterion(pred, y)  + l2_norm*betas[1] + l1_norm*betas[0]

            else:
                loss = criterion(pred, y)


            # Backpropagation

            val_loss += loss.item()


    val_loss /= (batch_ndx)
    mse = (((y_val_predicted-y_val_real)**2)**(1/2)).mean()


    log(f"Validation Error: Avg loss: {val_loss:>8f}")
    log(f"                  MSE {mse:>3f}")
    
    return(y_val_predicted, y_val_real, val_loss)

