#!/usr/bin/env python3
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

import argparse
from .rrwick_help_formatter import MyParser, MyHelpFormatter
from .PARM_predict import PARM_predict
from .PARM_mutagenesis import PARM_mutagenesis, PARM_plot_mutagenesis
from .PARM_train import PARM_train
from .version import __version__
import warnings
import os
warnings.filterwarnings("ignore")


def main():
    global description
    description = (
        """
██████╗  █████╗ ██████╗ ███╗   ███╗
██╔══██╗██╔══██╗██╔══██╗████╗ ████║
██████╔╝███████║██████╔╝██╔████╔██║
██╔═══╝ ██╔══██║██╔══██╗██║╚██╔╝██║
██║     ██║  ██║██║  ██║██║ ╚═╝ ██║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
 
 Promoter Activity Regulatory Model
 Version: """
        + __version__
        + """
 """
    )

    # Main parser ========================================================================
    # ====================================================================================
    parser = MyParser(
        description="R|" + description, formatter_class=MyHelpFormatter, add_help=False
    )
    subparsers = parser.add_subparsers(dest="subparser_name", title="Tasks")

    # Train task =========================================================================
    # ====================================================================================
    train_subparser(subparsers)

    # Predict task =======================================================================
    # ====================================================================================
    predict_subparser(subparsers)

    # Run mutagenesis task ===============================================================
    # ====================================================================================
    mutagenesis_subparser(subparsers)
    
    # Plot mutagenesis task ==============================================================
    # ====================================================================================
    plot_subparser(subparsers)

    other_args = parser.add_argument_group("Other")
    other_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    other_args.add_argument(
        "--version",
        action="version",
        version="PARM v" + __version__,
        help="Show program's version number and exit",
    )
    args = parser.parse_args()

    if "func" in args:
        args.func(args)
        print(bye_message(), flush=True)
    else:
        parser.print_help()
        exit(1)


def print_arguments(left, right, total_width=80):
    left_width = len(left)
    right_width = total_width - left_width
    right_str = ', '.join(map(str, right)) if isinstance(right, list) else str(right)
    print("{0}: {1:>{2}}".format(left, right_str, right_width - 2))


def train(args):
    # Implement the logic for the train command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Predict"))
    print("-" * 80)
    print_arguments("Input", args.dir_input)
    print_arguments("Output", args.out_dir)
    print_arguments("Model", args.model_dir)
    print_arguments("Type model", args.type_model)
    print_arguments("Training model", args.training_model)
    print_arguments("Cell line", args.cell_line)
    print_arguments("Number of epochs", args.n_epochs)
    print_arguments("Batch size", args.batch_size)
    print_arguments("Betas", args.betas)
    print_arguments("Learning rate", args.lr)
    print_arguments("Scheduler", args.scheduler)
    print_arguments("Weight decay", args.weight_decay)
    print_arguments("Stranded", args.stranded)
    print_arguments("Features fragments selection", args.features_fragments_selection)
    print_arguments("Normalization", args.normalization)
    print_arguments("Criterion", args.criterion)
    print_arguments("Downsample", args.downsample)
    print_arguments("Adaptor", args.adaptor)
    print_arguments("L_max", args.L_max)
    print_arguments("L_min", args.L_min)
    print_arguments("Validation path", args.validation_path)
    print("=" * 80)
    PARM_train(args)


def predict(args):
    # Implement the logic for the predict command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Predict"))
    print("-" * 80)
    models = ",".join(args.model)
    print_arguments("Model", models)
    print_arguments("Input", args.input)
    print_arguments("Output", args.output)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_predict(
        input=args.input,
        model_weights=args.model,
        output=args.output,
        parm_version=__version__,
    )


def mutagenesis(args):
    # Check if required arguments are present
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Mutagenesis"))
    print("-" * 80)
    models = ",".join(args.model)
    print_arguments("Model:", models)
    print_arguments("Input:", args.input)
    print_arguments("Output:", args.output)
    print_arguments("Motif database:", args.motif_database)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_mutagenesis(
        input=args.input,
        model_weights=args.model,
        output_directory=args.output,
        motif_database=args.motif_database,
        parm_version=__version__,
    )


def plot(args):
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Plot"))
    print("-" * 80)
    if args.output is None:
        out = args.input
    else: 
        out = args.output
    if args.attribution_range is not None:
        r = " ".join(args.attribution_range)
    else:
        r = "None"
    print_arguments("Input:", args.input)
    print_arguments("Output:", out)
    print_arguments("Correlation threshold:", args.correlation_threshold)
    print_arguments("Attribution threshold:", args.attribution_threshold)
    print_arguments("Attribution range:", r)    
    print_arguments("Plot format:", args.plot_format)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_plot_mutagenesis(
        input=args.input,
        output_directory=args.output,
        correlation_threshold=args.correlation_threshold,
        attribution_threshold=args.attribution_threshold,
        plot_format=args.plot_format,
        parm_version=__version__,
    )


# Train task ===================================================================
def train_subparser(subparsers):
    "Parses inputs from commandline and returns them as a Namespace object."

    def str2bool(v):
        if v == 'False':
            return False
        else:
            return v

    group = subparsers.add_parser(
        "train",
        help="Train a new PARM model from pre-processed MPRA data",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description= 'R|' + description,
    )

    required_args = group.add_argument_group("Required arguments")
    # Arguments for the input files
    required_args.add_argument('--dir_input',
                        default = ['./splits/focused_library/tss_selection_m300_p100_stranded_EnhA_intersection_intersection'],
                        nargs='+',
                        help='directory with one_hot_encoding input files are. If more than one add one after the other.'
                              'default: (./splits/focused_library/tss_selection_m300_p100_stranded_EnhA_intersection_intersection)')  

    required_args.add_argument('--out_dir',
                        default = '../analysis/lbarbadillamartinez/output/models/SuRE/',
                        nargs='?',
                        help='path to a directory where output files are saved')

    required_args.add_argument('--model_dir',
                        type=str2bool,
                        nargs='?',
                        default = False,
                        help='Directory of pretrained model, if not use False. (default: False)' )

    required_args.add_argument('--type_model',
                        default = 'ResNet_Attention',
                        nargs='?',
                        choices=['ResNet', 'S4', 'ResNet_Attention', 'CNN_jeremie', 'selfattention', 'RNN', 'BPnet', 'basset', 'VAE', 'enformer_architecture', 'enformer'],
                        help=f'which model to use choose from:\n'
                                f'ResNet, S4, ResNet_Attention, CNN_jeremie,  selfattention, RNN, BPnet, basset, VAE, enformer_architecture, enformer')

    model_args = group.add_argument_group("CNN-related arguments (optional)")
    model_args.add_argument('--training_model',
                        type=str2bool,
                        default = False,
                        nargs='?',
                        help= f'Type of training you are working on. If multiple, separate by "_" \n'
                        f'  e.g. tuning, adaptive_sampling, TYY1_motif, size_sample, sure_track, multitask, TSS_vector_predict, replicates, multitask, triplet_loss ')

    model_args.add_argument('--cell_line',
                        default = 'K562',
                        nargs='?',
                        help='cell line to work with (K562, HEPG2, hNPC, HCT166, MCF7, mESC or any combination of these separated by two underscores (__)')

    model_args.add_argument('--n_epochs',
                        default = 7,
                        nargs = '?',
                        type = int,
                        help='Number of epochs to train the data to (default: 7) \n')

    model_args.add_argument('--batch_size',
                        default = 128,
                        nargs = '?',
                        type = int,
                        help='Number of samples in ech batch to train the data to (default: 128) \n')

    model_args.add_argument('--betas',
                        default = (0.0005, 0.0005),
                        nargs = '+',
                        type = float,
                        help='L1 and L2 regularization terms respectively. (default: (0.005, 0.005) ) \t run like -betas 0.1 0.2 \n')

    model_args.add_argument('--lr',
                        default = 0.001,
                        nargs = '?',
                        type = float,
                        help='Learning rate (default: 0.001) \n')
    
    model_args.add_argument('--scheduler',
                        default = True,
                        nargs = '?',
                        type = str2bool,
                        help='Cos scheduler implemented if True (default:True) \n')
    
    model_args.add_argument('--weight_decay',
                        default = 0.0,
                        nargs = '?',
                        type = float,
                        help='Weight decay (default: 0.0) \n')
    
    model_args.add_argument('--stranded',
                        default = False,
                        nargs = '?',
                        type = str2bool,
                        help=' If we are interested in stranded fragments (True, only those matching with TSS)'
                                ' or also complementary strands (False). Only valid for TSS.  (default: False) \n')
    
    model_args.add_argument('--features_fragments_selection',
                        default = 'TSS',
                        nargs = '?',
                        type = str,
                        help='Features to use to select SuRE fragments of interest (default: TSS) \t'
                                '   In humans choose from TSS, EnhA, peaks or a combination of them separated by "_" e.g. TSS_EnhA \t'
                                '   In mice choose from TSS, EnhA_many or EnhA_strong a combination of them separated by "_". \n')
    
    model_args.add_argument('--normalization',
                        default = 'LnNorm',
                        nargs = '?',
                        type = str,
                        help=' Type of normalization. Only two possibles: Log2Norm or LnNorm. '
                                '(default: LnNorm) \n')
    
    model_args.add_argument('--criterion',
                        default = 'poisson',
                        nargs = '?',
                        type = str,
                        help=' Type of criterion. Only two possibles: MSE or poisson. '
                                '(default: poisson) \n')
    

    model_args.add_argument('--downsample',
                        default = False,
                        nargs = '?',
                        help=  'Downsample dataset. What type of downsampling to do \n '  
                                '   either on number of samples (fragment_downsample) or in number of tss (TSS_downsample) \n '
                                '(default: False) \n')
    
    model_args.add_argument('--adaptor',
                        default = ('CAGTGAT', 'ACGACTG'),
                        nargs = '+',
                        help=  'If not false, give adaptor in 5 and 3 prima to use as padding. \n '  
                                '   e.g. -adaptor CAGTGAT ACGACTG \n '
                                '(default: CAGTGAT ACGACTG) \n')
    
    model_args.add_argument('--L_max',
                        default = 600,
                        nargs = '?',
                        type = int,
                        help=  'Maximum length of fragments. Necessary if we want to downsample. \n '  
                                '(default: 600) \n')
    
    model_args.add_argument('--L_min',
                        default = False,
                        nargs = '?',
                        type = int,
                        help=  'Minimum length of fragments. Necessary if we want to downsample. \n '  
                                '(default: False) \n')
    
    model_args.add_argument('--validation_path',
                        nargs='+',
                        type = str,
                        help=  'Indicate the path of the split will be used for validation ')

    other_args = group.add_argument_group("Other")
    other_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    other_args.add_argument(
        "--version",
        action="version",
        version="PARM v" + __version__,
        help="Show program's version number and exit",
    )

    group.set_defaults(func=train)

# Predict task =================================================================
def predict_subparser(subparsers):

    group = subparsers.add_parser(
        "predict",
        help="Predict promoter activity of sequences in a fasta file using a trained PARM "
        "model. The output is a tab-separated file with the sequence and the "
        "predicted score.",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description= 'R|' + description,
    )

    required_args = group.add_argument_group("Required arguments")

    required_args.add_argument(
        "--model",
        required = True,
        nargs="+",
        help="Path to the weight files for the model. If you want to perform predictions "
        "for multiple models at once, you can pass them all as a space-separated list. "
        "If you have not trained a model, you can use the pre-trained model from the "
        "pre_trained_models directory.",
    )
    required_args.add_argument(
        "--input",
        required = True,
        help="Path to the input fasta file with the sequences to be predicted.",
    )
    required_args.add_argument(
        "--output",
        required = True,
        help="Path to the output file where the predictions will be saved. Output is a "
        "tab-separated file with the sequence, header, and the predicted score.",
    )

    other_args = group.add_argument_group("Other")
    other_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    other_args.add_argument(
        "--version",
        action="version",
        version="PARM v" + __version__,
        help="Show program's version number and exit",
    )

    group.set_defaults(func=predict)

# Mutagenesis task =============================================================
def mutagenesis_subparser(subparsers):
    group = subparsers.add_parser(
        "mutagenesis",
        help="Perform mutagenesis assay of a given sequence on a trained PARM model. "
        "This produces three output files: the mutagenesis matrix witht the effect of "
        "each mutation, and the list of known motifs scanned in the sequence",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description= 'R|' + description,
    )

    required_args = group.add_argument_group("Required arguments")

    required_args.add_argument(
        "--model",
        nargs="+",
        required = True,
        help="Path to the weight files for the model. If you want to perform predictions "
        "for multiple models at once, you can pass them all as a space-separated list. "
        "If you have not trained a model, you can use the pre-trained model from the "
        "default_PARM_models directory.",
    )
    required_args.add_argument(
        "--input",        
        required = True,
        help="Path to the input fasta file with the sequences to have to mutagenesis for.",
    )
    required_args.add_argument(
        "--output",
        required = True,
        help="Path to the directory where the files will be stored. Will be created "
        "if it does not exist.",
    )

    optional_arguments = group.add_argument_group("Optional arguments")
    optional_arguments.add_argument(
        "--motif_database",
        default="https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt",
        help="Path or url to the motif databae (JASPAR format). Default is HOCOMOCOv11: https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt",
    )

    # 
    other_args = group.add_argument_group("Other")
    other_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    other_args.add_argument(
        "--version",
        action="version",
        version="PARM v" + __version__,
        help="Show program's version number and exit",
    )

    group.set_defaults(func=mutagenesis)

# Plot task ====================================================================
def plot_subparser(subparsers):
    group = subparsers.add_parser(
        "plot",
        help="Plot results of a mutagenesis assay generated by PARM. "
        "Produces a PDF file with the mutagenesis plot.",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description= 'R|' + description,
    )
    
    required_args = group.add_argument_group("Required arguments")

    required_args.add_argument(
        "--input",
        help="Path to the directory containing the `mutagenesis_[ID].txt.gz` and "
        "`hits_[ID].txt.gz` files generated by PARM mutagenesis. PARM assumes that "
        "the ID values are the same for each sequence, otherwise an error will be raised.",
    )
    optional_arguments = group.add_argument_group("Optional arguments")
    
    optional_arguments.add_argument(
        "--output",
        help="Path to the directory where the files will be stored. Default "
        "behaviour is to save the PDFs in the same directory as the input data.",
    )
    optional_arguments.add_argument(
        "--correlation_threshold",
        default=0.75,
        help="The minimum value of Pearson correlation that a scanned motif needs "
        "to present in order to be shown in the plot (Default: 0.75).",
    )
    optional_arguments.add_argument(
        "--attribution_threshold",
        default=0.001,
        help="The minimum value of attribution (i.e., the mean attribution score "
        "for the bases of a motif) that a scanned motif needs to present in order "
        "to be shown in the plot (Default: 0.001).",
    )
    optional_arguments.add_argument(
        "--attribution_range",
        default=None,
        nargs=2,
        help="Space-separated range of attribution values to be shown in the plot. "
        "(like 0.001 0.01). If not provided, the range will be calculated based on "
        "the values present in the data.",
    )
    optional_arguments.add_argument(
        "--plot_format",
        default='pdf',
        choices = ['pdf', 'svg', 'jpg', 'png'],
        help="Which format should the plots be saved? Available formats are "
        "pdf, svg, jpg, and png. (Default: pdf).",
    )

    # 
    other_args = group.add_argument_group("Other")
    other_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    other_args.add_argument(
        "--version",
        action="version",
        version="PARM v" + __version__,
        help="Show program's version number and exit",
    )
    
    group.set_defaults(func=plot)


def bye_message():
    return("\nAll done!\n"
    "If you make use of PARM in your research, please cite:\n"
    "...............\n"
    "")


def check_file(file):
    """Check if file exists, otherwise return error"""
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File {file} not found.")
    
# Main =========================================================================
if __name__ == "__main__":
    main()