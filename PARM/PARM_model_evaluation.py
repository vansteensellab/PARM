


def insert_motifs_in_random_sequences(
    models,
    L_max,
    consensus_dict_list,
    PFM_dict_list,
    model_id,
    cell_type,
    batch_size,
    num_sequences=100,
    random_sequences=False,
    output_directory=False,
):
    """
    This function will insert the motifs in random sequences and compute the correlation between known motifs and the attribution of the consensus sequence.
    Produces plots and saves dataframe.
    Args:
        models: (list of pytorch model) List of pytorch Model to use to make predictions
        L_max: (int) Max. length sequences accepted by the model
        batch_size: (int) Number of sequences to compute the attribution at the same time
        num_sequences: (int) Number of random sequences to generate
        consensus_dict: (list of dict) Dictionary with the consensus sequence of the motifs
        PFM_dict: (list of dict) Dictionary with the PFM of the motifs
        random_sequences: (list) List of random sequences to use, if False, it will be generated
        output_directory: (str) Directory where to save the plots and dataframe, if False the plots will be shown in the screen.
        cell_type: (str) Cell type working on, if several, combine them with two underscores '__' e.g. K562__HepG2

    """
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    import matplotlib.pylab as pylab


    params = {'legend.fontsize': 'large',
            'axes.titlesize':'large',
            'axes.linewidth': 0.5,
            'axes.labelsize' : 'large',
            'ytick.major.width': 0.5,
            'ytick.minor.width': 0.5,
            'pdf.fonttype': 42,
            'xtick.labelsize':'large',
            'ytick.labelsize':'large'}
    pylab.rcParams.update(params)


    from .PARM_mutagenesis import  get_one_hot, motif_attribution, run_motif_scanning
    


    # If random sequences are not given, create them
    if not random_sequences:
        print(f"                        Generating random sequences", flush=True)
        random_sequences = []
        nt = ["A", "C", "G", "T"]
        for i_seq in range(num_sequences):
            np.random.seed(i_seq * 1984)
            seq = "".join(np.random.choice(nt, size=L_max))
            # One hot encode the sequence

            random_sequences.append(seq)

    # Create dictionaries of empty lists to save the results were the keys are the cell names
    motif_correlation_across_random_sequences, database_id, motif_id = {}, {}, {}
    for cell in cell_type.split("__"):
        motif_correlation_across_random_sequences[cell] = []
        database_id[cell] = []
        motif_id[cell] = []

    n_cells = len(cell_type.split("__"))

    for id_PWM_database in consensus_dict_list.keys():  # Loop through databases
        PFM_dict = PFM_dict_list[id_PWM_database]
        consensus_dict = consensus_dict_list[id_PWM_database]

        print(f"\n     Database {id_PWM_database}", flush=True)

        pbar_PFM_keys = tqdm(PFM_dict.keys(), total=len(PFM_dict.keys()), ncols=90)
        for motif in pbar_PFM_keys:  # Loop through all motifs

            # Get consensus sequence
            cons = consensus_dict[motif]
            cons_one_hot = np.float32(get_one_hot([cons], len(cons)))[
                0
            ]  # onehot cons sequence

            # If motif is long, we wont probably have enough memory to compute the attribution for all sequences
            ## We will compute the attribution for the first 50 sequences and then the rest
            total_number_sequences = len(random_sequences) * len(cons)

            # If the total number of sequences is too big, split the random sequences in batches given by the batch_size argument

            if total_number_sequences > batch_size:
                random_sequences_split = [
                    random_sequences[(it_batch - batch_size) : it_batch]
                    for it_batch in range(
                        batch_size, len(random_sequences) + batch_size, batch_size
                    )
                ]
            else:
                random_sequences_split = [random_sequences]

            att_all = []
            for it_batch, batch_random_sequences in enumerate(random_sequences_split):
                ##Loop through all random sequences
                rand_seq_with_motif = []
                for rand_seq in batch_random_sequences:

                    seq = list(rand_seq)

                    # Place the consensus sequence in the middle of  the random/backgorund sequence
                    half_seq = int(len(seq) / 2 - len(cons) / 2)
                    seq[half_seq : (half_seq + len(cons))] = cons
                    rand_seq_with_motif.append("".join(seq))

                # Compute attribution of motif
                att_all_batch = []
                for model in (models):
                    att_all_batch.append( motif_attribution(
                        seq=rand_seq_with_motif,
                        L_max=L_max,
                        start_motif=[half_seq] * len(rand_seq_with_motif),
                        end_motif=[(half_seq + len(cons))] * len(rand_seq_with_motif),
                        ref_to_alt_attribution=True,
                        completemodel=model,
                        index_output=list(range(n_cells)),
                    ))
                
                #Average over all models
                att_all_batch = np.mean(att_all_batch, axis=0)

                # att_all_batch shape is (n_cells, n_sequences, L_max, 4)
                att_all.append(att_all_batch)

            if it_batch == 0:
                att_all = att_all[0]
            else:
                att_all = np.concatenate(att_all, axis=1)

            if len(cell_type.split("__")) == 1:
                att_all = [att_all]

            for it_cell, att_all_cell in enumerate(
                att_all
            ):  # First we loop through the cell type axes
                cell = cell_type.split("__")[it_cell]
                # Create empty array and update it for each random sequence
                corr_motif, att_motif = [], []

                for (
                    att_all_seq
                ) in att_all_cell:  # Second we loop through the random sequences axes

                    att_all_mean = att_all_seq.mean(axis=0)
                    attribution_real = np.transpose(
                        cons_one_hot * np.expand_dims(att_all_mean, 1)
                    )

                    hits = run_motif_scanning(
                        known_PFM={motif: PFM_dict[motif]},
                        attribution_seq=attribution_real,
                        append=False,
                        threshold=0.0,
                        attribution=True,
                        multiple_one_hot=False,
                        split_pos_neg=False
                    )

                    # If it's empty, there are no hits
                    if len(hits) == 0:
                        corr, att = 0, 0
                    else:
                        corr = hits.rho.iloc[0]
                        att = hits.att.iloc[0]

                    # Append the correlation
                    corr_motif.append(corr)
                    att_motif.append(att)

                # Average motif for all random_sequences
                motif_correlation_across_random_sequences[cell].append(
                    np.mean(corr_motif)
                )
                database_id[cell].append(id_PWM_database.split("/")[-1].split(".")[0])
                motif_id[cell].append(motif)

    
    return motif_correlation_across_random_sequences, database_id, motif_id

def plot_motif_correlation(motif_correlation_across_random_sequences, database_id, motif_id, cell_type, model_id, output_directory):    
        ##Also save the correlations in dataframe that contains id of the motif and  the correlation value

        for cell in cell_type.split("__"):
            motif_id_cell = motif_id[cell]
            motif_correlation_across_random_sequences_cell = (
                motif_correlation_across_random_sequences[cell]
            )
            database_id_cell = database_id[cell]

            motif_correlation = pd.DataFrame(
                {
                    "motif_id": motif_id_cell,
                    "correlation": motif_correlation_across_random_sequences_cell,
                    "database_motifs": database_id_cell,
                }
            )
            motif_correlation.to_csv(
                os.path.join(
                    output_directory, f"motif_correlations_{cell}.txt"
                ),
                index=False,
                sep="\t",
            )
            # Now plot the distribution of correlations
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            #If database is only one, dont use hue, otherwise use hue to separate the databases
            if len(set(database_id_cell)) == 1:
                sns.histplot(
                    motif_correlation,
                    x="correlation",
                    kde=True,
                    bins=50,
                    ax=ax,
                    color="black",
                    legend=True,
                    multiple="stack",
                )
            else:
                sns.histplot(
                    motif_correlation,
                    x="correlation",
                    kde=True,
                    bins=50,
                    ax=ax,
                    hue="database_motifs",
                    color="black",
                    legend=True,
                    multiple="stack",
                    palette="Set1",
                )
                # Remove legend of frame and place it outside
                sns.move_legend(
                    ax,
                    "upper left",
                    bbox_to_anchor=(1, 1),
                    frameon=False,
                    title="Motif database",
                )
            ax.set_xlabel("Correlation between known motif and model motif")
            ax.set_ylabel("Frequency")
            #Check how many motifs are higher than 0.5
            num_motifs_high_corr = (motif_correlation["correlation"] > 0.5).sum()
            #compute average correlation of the motifs with correlation higher than 0.5
            avg_corr_high_corr = motif_correlation[motif_correlation["correlation"] > 0.5]["correlation"].mean()
            percentatge = num_motifs_high_corr / len(motif_correlation) * 100
            ax.set_title(f"Model: {model_id}\n Cell type {cell}\n Motifs with corr. > 0.5: {num_motifs_high_corr} out of {len(motif_correlation)} ({percentatge:.2f}%) with avg. corr.: {avg_corr_high_corr:.2f}")

            if output_directory:
                plt.savefig(
                     os.path.join(output_directory,
                      f"3analysis_hist_motif_correlation_cell_{cell}.png"),
                     bbox_inches="tight",
                )
            #Check the correlation between the forward and reverse motif (so that is, the same motif name but one with - at the end) and plot it in a scatter plot
            motif_correlation["motif_id_no_rev"] = motif_correlation["motif_id"].apply(lambda x: x.replace("-", ""))
            motif_correlation["is_rev"] = motif_correlation["motif_id"].apply(lambda x: "-" in x)
            motif_correlation_forward = motif_correlation[motif_correlation["is_rev"] == False]
            motif_correlation_reverse = motif_correlation[motif_correlation["is_rev"] == True]
            motif_correlation_forward = motif_correlation_forward.set_index("motif_id_no_rev")
            motif_correlation_reverse = motif_correlation_reverse.set_index("motif_id_no_rev")
            motif_correlation_forward = motif_correlation_forward.loc[motif_correlation_reverse.index]
            motif_correlation_reverse = motif_correlation_reverse.loc[motif_correlation_forward.index]
            #Now plot the correlation between the forward and reverse motif
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sns.scatterplot(x=motif_correlation_forward["correlation"], y=motif_correlation_reverse["correlation"], ax=ax, color="black")
            ax.set_xlabel("Correlation forward motif")
            ax.set_ylabel("Correlation reverse motif")
            ax.set_title(f"Model: {model_id}\n Cell type {cell}\n Correlation between forward and reverse motif correlation: {motif_correlation_forward['correlation'].corr(motif_correlation_reverse['correlation']):.2f}")
            if output_directory:
                plt.savefig(
                     os.path.join(output_directory,
                      f"3analysis_scatter_correlation_forward_reverse_motif_cell_{cell}.png"),
                     bbox_inches="tight",
                )



def run_predictions_validation_mutagenesis(promoters, models, batch_size, L_max, output_directory=False, 
                                            cell_type=None
                                    ):
        """
        Return a dataframe with the effect of each mutation.
        Args:
                model: pytorch model or list of pytorch models 
                promoters: file 
                batch_size: (int) Number of sequences to compute the attribution at the same time
                output_directory: (str) Directory where to save the dataframe, if False the dataframe will be returned but not saved.    
        """
        import scipy
        import pandas as pd
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns

        import matplotlib.pylab as pylab
        params = {'legend.fontsize': 'large',
                'axes.titlesize':'large',
                'axes.linewidth': 0.5,
                'axes.labelsize' : 'large',
                'ytick.major.width': 0.5,
                'ytick.minor.width': 0.5,

                'pdf.fonttype': 42,

                'xtick.labelsize':'large',
                'ytick.labelsize':'large'}
        pylab.rcParams.update(params)

        from .PARM_predict import get_prediction


        promoters_pred = pd.read_csv(promoters, sep='\t')
        
        
        #If model is not a list, make it a list
        if type(models) != list: models = [models]

        #Split promoters in batches of batch_size to compute the predictions
        batch_promoters = [promoters_pred[i:i + batch_size] for i in range(0, len(promoters_pred), batch_size)]

        for i_cell, cell in enumerate(cell_type.split("__")):
            promoters_pred = pd.read_csv(promoters, sep='\t')
            

            for it_batch, batch in enumerate(batch_promoters):
                #print(f"           Computing mutation effect for batch {it_batch + 1} / {len(batch_promoters)}", flush=True)

                pred_mean = []
                for model in models:
                        #Make sure the sequence in batch are all in capital letters, otherwise the model will not be able to make the predictions
                        sequences = batch.sequence.str.upper()
                        model_pred = get_prediction(sequences.to_list(), model, L_max=L_max)[:, i_cell]
                        pred_mean.append(model_pred)
                
                pred = np.mean(pred_mean, axis=0)
                promoters_pred.loc[batch.index, 'pred'] = pred
            
            
            if output_directory is not False: 
                promoters_pred.to_csv(os.path.join(output_directory, f"2analysis_predictions_mutagenesis_validation_promoters_{cell}.txt"), sep='\t', index=False)  
                
            
            #Now make the correlations between the predictions and the measurements for each promoter and cell line and save them in a txt file and make heatmaps of the correlations for each cell line and promoter
            cells = ['HCT116', 'HepG2', 'K562', 'MCF7', 'LNCaP']
            #Make from wide to long format the columns of the cells now are a single column with the name of the cell and the values are in a column with the name of the mutation score
            promoters_pred = promoters_pred.melt(id_vars=['chr', 'start', 'end', 'strand', 'prom', 'mut_po', 'ref', 'alt', 'sequence', 'seq_type', 'oligo_identifyer', 'bc', 'pred'],
                                    value_vars=cells,
                                    var_name='cell',
                                    value_name='measurement')
            

            #Now group by cell and promoter and compute the correlation between the predictions and the measurements for each group
            correlations = promoters_pred.groupby(['cell', 'prom']).apply(lambda x: scipy.stats.pearsonr(x['pred'], x['measurement'])[0])
            correlations = correlations.reset_index()
            correlations.columns = ['cell', 'promoter', 'correlation']
            #Now make a heatmap of the correlations for each cell line and promoter all together, the x axis will be the cell line and the y axis will be the promoter, the color will be the correlation value
            heatmap_data = correlations.pivot(index='promoter', columns='cell', values='correlation')

            #Save the heatmap data in a txt file
            heatmap_data.to_csv(os.path.join(output_directory, f"2analysis_heatmap_correlation_predictions_measurements_mutagenesis_validation_library_{cell}.txt"), sep='\t')

            plt.figure(figsize=(10, 10))
            #add legend
            sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black', cbar_kws={'label': 'R measurement-prediction'})
            plt.xlabel("Cell line")
            plt.ylabel("Promoter")
            plt.title(f"Promoter mutagenesis validation library\n Predictions in {cell}")
            #Add legend with the correlation values
            if output_directory:
                plt.savefig(os.path.join(output_directory, f"2analysis_heatmap_correlation_predictions_measurements_mutagenesis_validation_library_{cell}.png"), bbox_inches='tight')
            plt.close()

                
        
        return promoters    



def PARM_eval_model(model_dir, 
                    output_directory,
                    L_max,
                    criterion,
                    cell_type,
                    input_h5py_file, 
                    features_fragments_selection,
                    file_input_mutagenesis_validation,
                    PWM_datasets,
                    batch_size,
                    num_sequences_rnd,
                    normalization_method="Log2RPM",
                    filter_size=125
                    ):
    """
    """
    import os
    from .PARM_utils_load_model import load_PARM
    from .PARM_mutagenesis import dict_jaspar
    from .PARM_predict import get_test_fold_predictions




    #############
    output_directory = os.path.join(output_directory, f"model_eval_{cell_type}")
    #If it doesn't exist, create it
    if not os.path.exists(output_directory): os.makedirs(output_directory)
    
    
    model_id = []
    for it_model_dir in model_dir: 
        print(f"   Loading model {it_model_dir}", flush=True)
        print(os.path.basename(it_model_dir), flush=True)
        model_id.append( os.path.splitext(os.path.basename(it_model_dir))[0])
    model_id = '_'.join(set(model_id))


    ###########
    # 1. Load the model
    models = []
    for it_model_dir in model_dir:
        models.append(
            load_PARM(
                it_model_dir,
                train=False,
                type_loss = criterion,
                filter_size=filter_size,
            ))

    print(
        f"\n --------------------------------------------------------------------------------------------------------\n\n",
        flush=True,
    )

    # 2. Make predictions on the test fold fragments and compare with the measurements of the test fold if they exist, otherwise save the predictions in a txt file to be compared with the measurements later when they are available.
    #  If argument input_h5py_file exists
    if input_h5py_file:

        for prediction_files in input_h5py_file:
            get_test_fold_predictions(
                test_fold_path=prediction_files,
                list_of_models=models,
                cell_type=cell_type,
                output_directory=output_directory,
                features_fragments_selection = features_fragments_selection,
                normalization_method=normalization_method
            )

        print(
            f"   Step 1: Predictions made\n\n\n --------------------------------------------------------------------------------------------------------\n\n",
            flush=True,
        )
        
        
        

    else:
        print(
            f"   Step 1:  No input_h5py_file provided to make predictions \n\n",
            flush=True,
        )

    

        

    ###################
    # 2. Predict on the validation library set

    if file_input_mutagenesis_validation:
        for file in file_input_mutagenesis_validation:
            print(f"   Step 2: Predict on the validation library set\n", flush=True)
            #Check that the file exists, otherwise continue
            if not os.path.exists(file):
                print(f"           File {file} does not exist, skipping\n", flush=True)
                continue

            run_predictions_validation_mutagenesis(file, models=models, 
                                                    batch_size = batch_size, L_max=L_max, 
                                                    output_directory=output_directory, 
                                                    cell_type=cell_type
                                    )

        print(
            f" Done \n --------------------------------------------------------------------------------------------------------\n\n",
            flush=True,
        )

    else:
        print(
            f"   Step 2:  No file_input_mutagenesis_validation provided to predict on the validation library set\n\n",
            flush=True,
        )


    ###########
    # 5. Insert motifs in random sequences

    # Define consensus and PFM datasets
    consensus_PWM_datasets, ICT_PWM_datasets = {}, {}

    if PWM_datasets:
        print(f"   Step 3: Insert motifs in random sequences\n", flush=True)

        if not isinstance(PWM_datasets, list):
            PWM_datasets = [PWM_datasets]

        for PWM_dataset in PWM_datasets:
            _, consensus_dict, ICT_dict = dict_jaspar(PWM_dataset, reverse=True)
            consensus_PWM_datasets[PWM_dataset] = consensus_dict
            ICT_PWM_datasets[PWM_dataset] = ICT_dict

        motif_correlation_across_random_sequences, database_id, motif_id = insert_motifs_in_random_sequences(
            models,
            L_max,
            consensus_PWM_datasets,
            ICT_PWM_datasets,
            cell_type=cell_type,
            model_id=model_id,
            num_sequences=num_sequences_rnd,
            batch_size=batch_size,
            random_sequences=False,
            output_directory=output_directory,
        )

        #Now plot the correlation between the known motifs and the attribution of the consensus sequence and save the dataframe with the correlation values
        plot_motif_correlation(motif_correlation_across_random_sequences, database_id, motif_id, cell_type, model_id, output_directory)



        print(
            f" Done \n --------------------------------------------------------------------------------------------------------\n\n",
            flush=True,
        )

    else:
        print(
            f"   Step 3:  No PWM_datasets provided to insert motifs in random sequences - It wont be computed \n\n",
            flush=True,
        )
    

    #Print where everything is saved
    print(f"   All results are saved in {output_directory} \n\n", flush=True)

