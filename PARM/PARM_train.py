import numpy as np
import sys
import os
import pandas as pd
import time
from matplotlib import pyplot as plt, colors
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import math
import seaborn as sns
from torchsummary import summary
from optuna.trial import TrialState
import joblib
import torch
from torch.nn.functional import pad
from .PARM_utils_load_model import load_PARM
from .PARM_utils_data_loader import pad_collate, ShuffleBatchSampler, H5Dataset, index_of_interest, GradualWarmupScheduler

#from s4 import setup_optimizer, S4Model
print(f'\n Cuda working? {torch.cuda.is_available()} \n', flush=True)



def PARM_train(args=False):
    "Perform CNN for SuRE data."
    if not args: args = parse_args()


    #############
    # 1. Load arguments
    input_directory = args.dir_input
    out_dir = args.out_dir
    model_directory = args.model_dir
    cell_line = args.cell_line
    normalization = args.normalization
    criterion = args.criterion
    downsample = args.downsample
    adaptor = args.adaptor
    L_max = args.L_max
    scheduler = args.scheduler
    L_min = args.L_min
    weight_decay = args.weight_decay
    validation_path = args.validation_path
    if type(validation_path) != list: validation_path = list(validation_path)

    #Arguments for training
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    betas = args.betas
    lr = args.lr
    if len(betas) != 2: raise Exception(f'Wrong values of betas. You must provide two values, you provided {len(betas)}')

    stranded = args.stranded
    features_fragments_selection = args.features_fragments_selection

    #############
    # 2. Find which dataset we are working on from input directory

    #Define which genome we are working with
    dataset = ''.join(input_directory).split('/')

    ##possible datsets

    available_datasets = ['SuRE4n', 'SuRE23', 'focused_library', 'mouse']

    dataset_genome = []

    for single_directory in input_directory: #If there's more than one directory

        dataset = single_directory.split('/')
    
        dataset_genome_ind = [possible_dataset for possible_dataset in available_datasets if possible_dataset in single_directory]
        dataset_genome_ind = np.unique(dataset_genome_ind).tolist()
        if len(dataset_genome_ind) > 1: raise TypeError(f"Error: more than one dataset in a single path {single_directory}")

        #Try to find sub information dependign on dataset

        #if 'mouse' in single_directory and '_sample' not in single_directory: dataset_genome_ind.append([x for x in dataset if '_sample' in x][0])

        if 'thresh' in input_directory: dataset_genome_ind.append([x for x in dataset if 'thresh' in x][0])

        if 'split_' in input_directory: dataset_genome_ind.append([x for x in dataset if 'split_' in x][0]) 
        
        dataset_genome.append('_'.join(dataset_genome_ind))

        #Find which subset of feature
        if 'mouse' not in input_directory: dataset_subgroup = [x for x in dataset if ('intersection' in x or 'TSS' in x)][0]
        else: dataset_subgroup = '' 
    
    dataset_genome = np.unique(dataset_genome).tolist()
    dataset_genome = '_'.join(dataset_genome)


    #############
    # 3. Create output directory


    output_directory = os.path.join(out_dir, args.type_model, dataset_genome, dataset_subgroup, cell_line,
                                    features_fragments_selection, f'strand_{str(stranded)}',
                                    args.training_model if args.training_model else '')

    # Add possible extensions that the output directory might have

    ## If the output_directory contains "motif_" term, it means that only motifs of a specific family are going to be studied.
    if 'motif_' in output_directory:
        motif_or_family = (args.type_model).replace('motif_', '')
        print(f'Motif or family {motif_or_family} being studied', flush=True)

    ## If model_directory is diferent than False, it means that weights are going to be loaded from a previous model
    pretrained_directory = model_directory
    if model_directory:
        print(f' Pretrained directory used in: {pretrained_directory}')
        output_directory = os.path.join(output_directory, 'pretrained')


    ##If output direcotry already exists, create different trial directories
    if os.path.exists(output_directory):
        for i in range(2,100):
            trial_directory = os.path.join(output_directory, 'trial_'+str(i))
            if not os.path.exists(trial_directory):
                output_directory = trial_directory
                break

    if not os.path.exists(output_directory): os.makedirs(output_directory) #Create folder where all the output is going to be saved

    #All printing functions will be saved in a file
    f = open(os.path.join(output_directory, 'screen_messages.txt'), 'w')

    print(f' Output directory: {output_directory} \n', flush=True)
    sys.stdout = f

    print(f' Input Directory {input_directory} \n', flush=True)

    #Check if validation data is in training data
    error = any(file_validation in input_directory for file_validation in validation_path)
    if error: raise ValueError('Error: Your validation data is in your trainning data.')

    print(f' Validation Directory {validation_path} \n', flush=True)

    if model_directory:
        print(f' Pretrained directory used in: {pretrained_directory}')


    #############
    # 4. Run models

    param_model = { 'output_directory' : output_directory,
                    'input_directory' : args.dir_input,
                    'pretrained_directory'  : pretrained_directory,
                    'celltype' : cell_line, 'n_epochs' : n_epochs,
                    'batch_size' : batch_size, 'betas' : betas, 'lr':lr,
                    'features_fragments_selection': features_fragments_selection, 
                    'stranded' : stranded, 'normalization': normalization,
                    'type_criterion': criterion, 'adaptor':adaptor, 'L_max':L_max,
                    'scheduler': scheduler,
                    'weight_decay':weight_decay, 'L_min':L_min, 'validation_path':validation_path}

    #Hyperparameter optimization with optuna
    if 'tuning' in output_directory:

        objective_args = lambda trial: objective(trial, **param_model)

        print(f'Start study', flush='True')
        study = optuna.create_study(study_name='CNN_selfattention',direction='minimize') #bayesian optimization
        study.optimize(objective_args, n_trials=40, show_progress_bar=True)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print(f"Study statistics: \n"
            f"  Number of finished trials: {len(study.trials)} \n"
            f"  Number of pruned trials: {len(pruned_trials)} \n"
            f"  Number of complete trials: {len(complete_trials)} \n")

        trial = study.best_trial
        print(f"Best trial: \n"
              f"  Value: {trial.value} \n"
              f"  Params: \n")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


        joblib.dump(study, os.path.join(output_directory,"study.pkl"))
        importance_hyperparam = optuna.importance.get_param_importances(study)


        print(importance_hyperparam)

    else:

        ##########
        if not downsample:
            objective(None, **param_model)

        else:
            min_labelsnoise = 90
            step = 10
            max_labelsnoise = 100
            range_noise = list(range(min_labelsnoise,(max_labelsnoise)+step,step))

            for perc_labels in range_noise:
                print(f' ############################################# \n', flush=True)
                print(f' \n Percentage modification {perc_labels} \n', flush=True)

                perc_labels = perc_labels/100

                objective(None, perc_labels = perc_labels, **param_model)

    f.close()




def objective(trial, output_directory, input_directory, pretrained_directory, celltype, batch_size, n_epochs, betas,  lr, 
                stranded, features_fragments_selection, normalization, type_criterion, L_max, L_min, weight_decay, validation_path,
                scheduler, perc_labels = None, adaptor=(False,False)):
    """
    Objetive function to train and validate models.

    Args:
        trial: None or optuna function. Necessary if using Optuna for trainig hyperparameters.
        output_directory: (str) Directory where we want to save all output files.
        input_directory: (str) Directory where one_hot_encoding directory is with hdf5 files for training and validation.
        pretrained_directory:  (str) Directory with model_CNN.pth file to load weights.
                                        If we don't want to pretrain model just use None.
        celltype:(str) Cell type we are working with. If multiple, separated by "_
        batch_size: (int) Batch size for training.
        n_epochs: (int) Number of total epochs.
        betas: (tuple) L1 and L2 regularization respectively.
        lr: (float) Learning rate.
        scheduler: (bool) If True, use cosine scheduler.
        perc_labels: (float) Percentage used to modify a given % of the samples.
        normalization: (str) Type of normalization of measured SuRE score to choose from.
        type_criterion: (str) Criterion to use either MSE or poisson.
        adaptor: (tuple) Tuple with adaptor in 5' and adaptor in 3' in this order. If not false they are going to be used for padding.
        weight_decay: (float) Weight decay of loss 
        validation_path: (str) Path to validation file hdf5.

    Returns:

    """
    ##Necesary function for OPTUNA

    ###Define hyperparameters

    if 'tuning' in output_directory:  #If we are tuning hyperparameters
        #lr = trial.suggest_float('lr', 1e-4,6e-3)
        lr = 0.0004
        if 'selfattention' in output_directory or 'RNN' in output_directory:
            batch_size = 32

        else:
            batch_size = 256
            #batch_size = trial.suggest_categorical('batch_size', [64, 124,256,524]) #If RNN you cant use 524

        n_epochs = 5
        #type_optimizer = trial.suggest_categorical('optimizer', [ 'SGD', 'Adam']) # TRY TO USE SAM WHEN USING ATTENTION LAYERS?? OR AT LEAST ADAM WITH WEIGHT DEC.
        type_optimizer = 'Adam'
        weight_decay = trial.suggest_float('lr', high=1e-2,low=5e-5)

    else:

        warmup = True
        type_optimizer = 'Adam'
        padding_alternate = True
        gradient_clipping = 0.2
        


        if 'enformer' in output_directory:
            lr = 0.00005
            weight_decay = 0.001
            batch_size = 64

        elif 'jeremie' in output_directory:
            lr = 0.00001
            weight_decay = 0
            n_epochs = 15
            batch_size = 1024
            scheduler = False

        elif 'GLM' in output_directory:
            lr = 0.004
            weight_decay = 0

        trial = None

        if 'adaptive_sampling' in output_directory:
            n_epochs = 50

        if 'S4' in output_directory: 
            betas = (0, 0)
            weight_decay = 0.001
            gradient_clipping = False
        
        barcode = 'barcode' in output_directory

        if pretrained_directory: warmup=False
        

        
        print(f' ---- Selection of SuRE fragments ---- \n'
                f'      Stranded:  {stranded} \n'
                f'      SuRE normalization: {normalization} \n'
                f'      Barcode model: {barcode} \n'
                f'      L max: {L_max} \n'
                f'      L_min: {L_min} \n'
                f'      Features the selection was based on: {features_fragments_selection} \n')


        print(f' ---- PARAMETERS LEARNING ---- \n'
                f'      Pretrained model: {pretrained_directory} \n'
                f'      Criterion:  {type_criterion} \n'
                f'      Batch size:  {batch_size} \n'
                f'      Gradient clipping:  {gradient_clipping} \n'
                f'      Nº epoch: {n_epochs} \n'
                f'      Learning rate: {lr} \n'
                f'      Weight decay: {weight_decay} \n'
                f'      Type optimizer: {type_optimizer} \n'
                f'      Warmup: {warmup} \n'
                f'      Scheduler: {scheduler} \n'
                f'      Alternate type of paddings: {padding_alternate} \n'
                f'      Regularization: Beta1 {betas[0]} beta2 {betas[1]} \n'
                f'      Pretrained: {pretrained_directory} \n'
                f'      Percentage {perc_labels} \n')
        
        print(f' ---- INPUT DATA ---- \n'
               f'      Barcode:  {barcode} \n'
               f'      Adaptor: {adaptor} \n')


    ##################################
    ##Define losses

    def weighted_mse_loss(pred, target):

            target_bins = torch.bucketize(target.flatten(), bins)
            weights_targets = weights[target_bins].unsqueeze(1)

            loss = torch.sum(((pred - target) ** 2 )*(weights_targets))

            return loss

    def MSE_x2(pred, target, pred_track, target_track):
            MSE = nn.MSELoss()

            return(MSE(pred, target) + MSE(pred_track, target_track)*1000)

    if 'sure_track' in output_directory: criterion = MSE_x2
    elif type_criterion == 'MSE': criterion = nn.MSELoss()
    elif type_criterion == 'poisson': criterion = nn.PoissonNLLLoss(log_input=False)

    ###############################################
    ###Load model

    #cell_type_strip_replicates = celltype.replace('pNK7_','').replace('_B','')
    model = load_PARM(trial=trial, output_directory=output_directory, pretrained_directory= pretrained_directory, 
                        celltype=celltype, type_loss = type_criterion, lr= lr, L_max = L_max, L_min=L_min)
    dummybatch = torch.zeros(1, 4, L_max)

    if torch.cuda.is_available():
            model = model.cuda()
            dummybatch = dummybatch.cuda()


    #print(f' ---- MODEL ---- \n'
    #     f'{model} \n')

    _ = model(dummybatch)

    summary(model, ( 4, L_max))



    ################################################
    #Define optimizer

    # defining the optimizer
    if type_optimizer == 'Adam' and 'jeremie' not in output_directory:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)

    elif 'jeremie' in output_directory:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07, weight_decay=0, amsgrad=False)

    elif 'SGD' in type_optimizer:
            optimizer = SGD(model.parameters(), lr=lr, weight_decay = weight_decay)

    elif 'RMS' in type_optimizer :
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay = weight_decay)
        
    if 'S4' in output_directory:
         optimizer = setup_optimizer( model, lr=lr, weight_decay=weight_decay)

    ################################################
    # Load dataset

    params = {'num_workers': 12, 'pin_memory':False, 'collate_fn' : pad_collate( adaptor_5=adaptor[0], adaptor_3=adaptor[1], L_max=L_max) if padding_alternate else None}

    ##Take dataset of interest 


    #### TRAINING
    #Done inside of epoch

    selection_length =  False #True if 'focused_library' in input_directory else False

    
    if  type(input_directory) != list: input_directory = [input_directory]

    index_dataset_train = np.empty((2,0), dtype=int)
    for i, directory in enumerate(input_directory):
        training_set = H5Dataset(path = directory, celltype = celltype, Zscore_logTPM = normalization)

        order_length = False
        print(f'       Order length: {order_length}')
        
        index_train_ind = index_of_interest(data_length = len(training_set), file_path = directory, 
                                                    stranded= stranded, features = features_fragments_selection, selection_length = selection_length, 
                                                    L_max=L_max, order_length=order_length, L_min=L_min )
        
        ###TODO
        ##If we have multiple files of folds, and we want to order by length all the indeces we can't simply concat. So we have to keep track of their order.

        if perc_labels:  #If we reduce number of samples
            index_train_ind = np.random.choice(index_train_ind, size= int(len(index_train_ind) * perc_labels), replace=False) 
            index_train_ind = np.sort(index_train_ind)

        

        index_dataset_train = np.append(index_dataset_train, np.array([index_train_ind, np.repeat(i, len(index_train_ind))]), axis=1)
        
    index_dataset_train = np.transpose(index_dataset_train)
    training_set = H5Dataset(path = input_directory, celltype = celltype, Zscore_logTPM = normalization)

    #index_dataset_train: shape (n_samples,info). dataset_index[:,0] --> index. dataset_index[:,1] --> n_file_dataset 
    
        
        
    print(f'     Number of fragments after selection {index_dataset_train.shape} {index_dataset_train[:,0]}', flush=True)
    
    if order_length: sampler = torch.utils.data.sampler.BatchSampler( index_dataset_train, batch_size=batch_size, drop_last=False)
    else: sampler = ShuffleBatchSampler( index_dataset_train, batch_size=batch_size, drop_last=False)
    
    training_generator = torch.utils.data.DataLoader(training_set, sampler=sampler, **params)
    
    #### VALIDATION

    ##feat_selection_percentage
    index_dataset_valid = np.empty((2,0), dtype=int)
    for i, directory in enumerate(validation_path):
        validation_set = H5Dataset(path = validation_path, celltype = celltype, Zscore_logTPM = normalization)

        index_valid_ind = index_of_interest(data_length = len(validation_set), file_path = directory, 
                                                    stranded= stranded, features = features_fragments_selection, selection_length = selection_length, 
                                                    L_max=L_max,  feat_selection_percentage = False, L_min=L_min)

        if perc_labels:  #If we reduce number of samples
            index_valid_ind = np.random.choice(index_valid_ind, size= int(len(index_valid_ind) * perc_labels), replace=False) 
            index_valid_ind = np.sort(index_valid_ind)

        
        
        index_dataset_valid = np.append(index_dataset_valid, np.array([index_valid_ind, np.repeat(i, len(index_valid_ind))]),
                                                axis=1)
    index_dataset_valid = np.transpose(index_dataset_valid)

    #This take into account different type of inputs. In case the folds are defined directly written as valid.
    
    validation_set = H5Dataset(path = validation_path, celltype = celltype, Zscore_logTPM = normalization)


    sampler = ShuffleBatchSampler( index_dataset_valid, batch_size=batch_size, drop_last=False)


    validation_generator = torch.utils.data.DataLoader(validation_set, sampler=sampler, **params)
    


    ################################################
    # Load schedueler and/or warmer 
    total_steps =  (len(training_generator)*batch_size*n_epochs)/batch_size

    if 'S4' in output_directory:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    else:
        if scheduler:

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0, last_epoch=-1, verbose=False)
        else:
            scheduler = None

    if warmup:
        scheduler = GradualWarmupScheduler(optimizer,multiplier = 1.0, total_epoch = 5000, after_scheduler=scheduler)

    ################################################


    # empty list to store training and validation losses
    train_losses, val_losses, results = [], [], []

    for epoch in range(n_epochs):
        print(f'------------------------------------------- \n --- Epoch {epoch}', flush=True)


        ########### TRAINING LOOP

        ##Data  generator. Redo in every step so there;s random order
        y_train_predicted, y_train_true, training_loss = train_loop(training_generator, model, criterion, optimizer,
                                                                    scheduler, output_directory, betas, celltype, gradient_clipping=gradient_clipping)


        if order_length: sampler = torch.utils.data.sampler.BatchSampler( index_dataset_train, batch_size=batch_size, drop_last=False)
        else: sampler = ShuffleBatchSampler( index_dataset_train, batch_size=batch_size, drop_last=False)
    
        training_generator = torch.utils.data.DataLoader(training_set, sampler=sampler, **params)

        
        if math.isnan(training_loss):
            raise optuna.exceptions.TrialPruned()



        ########### VALIDATION LOOP

        with torch.no_grad():

            y_val_predicted, y_val_true, val_loss = validation_loop(validation_generator, model, criterion, output_directory, betas, celltype)

            

            results.append([epoch, training_loss, val_loss, perc_labels ])

        torch.save(model.state_dict(), os.path.join(output_directory,f'model_epoch_{epoch}.pth'))

        cell_lines = celltype.split('__')
        for it, cell_type in enumerate(cell_lines):
            true_sub = y_val_true[:, it].flatten()
            predicted_sub = y_val_predicted[:, it].flatten()

            MSE = (((true_sub - predicted_sub)**2)**(1/2)).mean()
            COEFF = r2_score(true_sub, predicted_sub)
            PCC = round(pearsonr(true_sub, predicted_sub)[0],3)
            
            print(f'\n      Validation {cell_type}: Coeff R2 {round(COEFF,3)},  MSE: {round(MSE,2)}, PCC {round(PCC,3)}', flush=True)




        if (epoch) % 5 == 0 or epoch==(n_epochs-1):

            plot_results(true=y_train_true, predicted=y_train_predicted, train_or_valid='train',
                                output_directory=output_directory, epoch=epoch, cell_line= celltype)
            plot_results(true=y_val_true, predicted=y_val_predicted, train_or_valid='valid',
                                output_directory=output_directory, epoch=epoch, cell_line= celltype)

            torch.save(model.state_dict(), os.path.join(output_directory,f'model_epoch_{epoch}.pth'))



        if 'tuning' in output_directory:
            trial.report(report_loss, epoch)

            #Handle pruning based on the intermediate value
            if trial.should_prune():

                plot_results(true=y_train_true, predicted=y_train_predicted, train_or_valid='train', output_directory=output_directory, epoch=epoch)
                plot_results(true=y_train_true, predicted=y_train_predicted, train_or_valid='train', output_directory=output_directory, epoch=epoch)


                raise optuna.exceptions.TrialPruned()

    # Process is complete.

    ##We've finished all epochs
    print(f'Training process has finished. Saving trained model.')
    if perc_labels: outfile_suffix = f'perc_{perc_labels}'
    else: outfile_suffix = ''

    torch.save(model.state_dict(), os.path.join(output_directory, f'model_CNN{outfile_suffix}.pth'))

    column_names = ['epoch', 'training_loss', 'validation_loss', 'percentage',
                    'Loss Motif Training', 'Loss Motif Validation', 'Loss No Motif Training', 'Loss No Motif Validation']

    column_names = column_names[:len(results[0])]

    results = pd.DataFrame(results,columns=column_names)
    
    results.to_csv(os.path.join(output_directory,f'results_model_CNN{outfile_suffix}.csv'),index=False)

    plt.plot(results['epoch'], results['training_loss'], label = 'Training loss')
    plt.plot(results['epoch'], results['validation_loss'], label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss: Mean Squared Error (MSE)')
    plt.legend()
    plt.savefig(os.path.join(output_directory, f'loss_epoch_validation_pred_training{outfile_suffix}.png'))
    plt.clf()


    if 'Motif' in column_names:

        results = pd.melt(results, id_vars=['epoch','percentage'],value_vars=[ 'training_loss', 'validation_loss',
                        'Loss Motif Training', 'Loss Motif Validation', 'Loss No Motif Training', 'Loss No Motif Validation'])
        results['size_training_motif'] = results.size_training_motif*100

        results = results.rename(columns={'variable':'Loss'})
        results['epoch'] = results['epoch'] +1
        g = sns.scatterplot(data=results, x="percentage", y="value", hue="Loss", size="epoch", alpha=0.4)
        g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('% of training samples')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(output_directory, 'size_loss.png'), bbox_inches='tight')
        plt.clf()




    return(val_loss)


def plot_results(true, predicted, train_or_valid, output_directory, epoch, cell_line=None):
    """
    Plot results after an epoch.

    Args:
        true: (np.array) Real SuRE scores.
        predicted: (np.array) Predicted SuRE scores.
        train_or_valid: (str) Either "train" or "valid".
        output_directory: (str) Output directory where figures should be saved.
        epoch: (int) Epoch where true and predicted resuts where generated.
        cell_line: (str) If multitask, mention which cell line is it. Other wise
                        use None or empty string "".

    """

    cell_lines = cell_line.split('_')

    for it, cell_type in enumerate(cell_lines):
        #Compute MSE
        true_sub = true[:, it].flatten()
        predicted_sub = predicted[:, it].flatten()
        MSE = (((true_sub - predicted_sub)**2)**(1/2)).mean()

        COEFF = r2_score(true_sub, predicted_sub)

        PCC = round(pearsonr(true_sub, predicted_sub)[0],3)

        print(f'{train_or_valid}: Coeff R2 {round(COEFF,3)},  MSE: {round(MSE,2)}, PCC {round(PCC,3)}', flush=True)

        plt.hist2d(true_sub, predicted_sub, bins=(50, 50), norm=colors.LogNorm(), cmap='Purples' )
        plt.xlabel(f'log SuRE score REAL ({train_or_valid} set)')
        plt.ylabel(f'log SuRE score PREDICTED ({train_or_valid} set)')
        plt.title(f'PCC: {PCC}, R²: {round(COEFF,3)}, MSE={round(MSE,3)}')
        plt.colorbar()
        plt.savefig(os.path.join(output_directory,f'y_{train_or_valid}_pred_vs_real_batch_epoch{epoch}_{cell_type}.png'), bbox_inches='tight')

        plt.clf()


def train_loop(train_dataloader, model, criterion, optimizer, scheduler, output_directory, betas, 
                cell_line:str, gradient_clipping=False):
    """
    Training loop.

    Args:
        train_dataloader: Train data in torch dataloader
        Model: Pytorch model
        criterion: (fun) loss function
        optimizer:
        scheduler:
        output_directory: (str) Output directory where figures should be saved.
        betas: (tuple) (int, int) Beta 1 and Beta 2 respectively for regularization.
        cell_line: (str) Cell lines used to train and predict, if more than one separate by "_" e.g. K562_HEPG2
        gradient_clipping: (float) If not False, then perform gradient clipping with that max norm.
    
    Returns:
        y_train_predicted: (np.array) Fragment predictions
        y_train_true: (np.array) Measured SuRE score, matching fragments with the one in y_train_predicted
        training_loss: (float) Loss performance of epoch.
    """

    model.train()

    training_loss = 0.0
    n_cels = len(cell_line.split('__'))
    y_train_predicted, y_train_true = np.empty((0,n_cels)), np.empty((0,n_cels))
    

    t = time.time()
    for batch_ndx, (X, y) in enumerate(train_dataloader):

        optimizer.zero_grad()

        X = X.permute(0,2,1)
        y = torch.flatten(y, 1, 2)

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        print('within y.shape', y.shape)
        pred = model(X)
        print('within pred.shape', pred.shape)

        if batch_ndx % 13 == 0:
            y_train_predicted = np.append(y_train_predicted, pred.cpu().detach().numpy(), axis=0)
            y_train_true = np.append(y_train_true, y.cpu().detach().numpy(), axis=0)


        if betas[0] != 0 or betas[1] != 0:

            l2_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())
            l1_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())

            loss = criterion(pred, y)  + l2_norm*betas[1] + l1_norm*betas[0]

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


        #Print results so far
        if batch_ndx % int(len(train_dataloader)/20) == 0:
            loss, current = training_loss/(batch_ndx+1) , batch_ndx * len(X)
            perc = current/(len(train_dataloader)*X.shape[0])*100
            print(f" loss: {loss:>7f}  [{current}/{(len(train_dataloader)*X.shape[0])}]  {round(perc,3)}%", flush=True)

            for param_group in optimizer.param_groups:
                print(f"       Learning rate: {param_group['lr'] }", flush=True)
                continue

        
    training_loss /= ((batch_ndx))

    mse = (((y_train_predicted-y_train_true)**2)**(1/2)).mean()

    print(f"Training Error: Avg loss: {training_loss:>8f}", flush=True)
    print(f"                MSE {mse:>3f} \n", flush=True)
    return(y_train_predicted, y_train_true, training_loss)


def validation_loop(valid_dataloader, model, criterion, output_directory, betas, cell_line:str):
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

                #l2_norm = sum(torch.norm(weight, p=2) if ('conv' in name and 'weight' in name) else 0
                #               for name, weight in model.named_parameters())

                l2_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())

                #l1_norm = sum(torch.norm(weight, p=1) if ('conv' in name and 'weight' in name) else 0
                #               for name, weight  in model.named_parameters())

                l1_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())

                loss = criterion(pred, y)  + l2_norm*betas[1] + l1_norm*betas[0]

            else:
                loss = criterion(pred, y)


            # Backpropagation

            val_loss += loss.item()


    val_loss /= (batch_ndx)
    mse = (((y_val_predicted-y_val_real)**2)**(1/2)).mean()


    print(f"Validation Error: Avg loss: {val_loss:>8f}", flush=True)
    print(f"                  MSE {mse:>3f} \n", flush=True)
    
    return(y_val_predicted, y_val_real, val_loss)

