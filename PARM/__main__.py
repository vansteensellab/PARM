#!/usr/bin/env python3

import argparse
from .rrwick_help_formatter import MyParser, MyHelpFormatter
from .PARM_predict import PARM_predict
from .PARM_mutagenesis import PARM_mutagenesis, PARM_plot_mutagenesis
from .PARM_train import PARM_train
from .version import __version__
from .PARM_misc import check_sequence_length, check_cuda
import warnings
import os
import sys

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
    right_str = ", ".join(map(str, right)) if isinstance(right, list) else str(right)
    print("{0}: {1:>{2}}".format(left, right_str, right_width - 2))


def train(args):
    # Implement the logic for the train command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Train"))
    print("-" * 80)
    print_arguments("Input", args.input)
    print_arguments("Validation", args.validation)
    print_arguments("Output", args.output)
    print_arguments("Cell type", args.cell_type)
    print_arguments("Measurement column name", args.measurement_column)
    print_arguments("Number of workers", args.n_workers)
    print_arguments("Number of epochs", args.n_epochs)
    print_arguments("Batch size", args.batch_size)
    print_arguments("Betas", args.betas)
    print_arguments("Learning rate", args.lr)
    print_arguments("Cosine scheduler?", args.cosine_scheduler)
    print_arguments("Weight decay", args.weight_decay)
    print_arguments("Adaptor", args.adaptor)
    print_arguments("L_max", args.L_max)
    print_arguments("Number of blocks", args.n_blocks)
    print_arguments("Filter size", args.filter_size)

    print("=" * 80)
    PARM_train(args)


def predict(args):
    # Check input fasta
    check_sequence_length(args.input, args.L_max)
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
    )


def mutagenesis(args):
    # Check input fasta
    check_sequence_length(args.input, args.L_max)
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Mutagenesis"))
    print("-" * 80)
    models = ",".join(args.model)
    print_arguments("Model", models)
    print_arguments("Input", args.input)
    print_arguments("Output", args.output)
    # check if args.motif_database is the default
    if args.motif_database == default_motif_db:
        print_arguments("Motif database", "HOCOMOCOv11 (default)")
    else:
        print_arguments("Motif database", args.motif_database)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_mutagenesis(
        input=args.input,
        model_weights=args.model,
        output_directory=args.output,
        motif_database=args.motif_database,
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
        # convert to string just for printing
        r = ", ".join(args.attribution_range)
        # Convert to float for the function
        attribution_range = [float(i) for i in args.attribution_range]
    else:
        attribution_range = None
        r = "None"
    print_arguments("Input", args.input)
    print_arguments("Output", out)
    print_arguments("Correlation threshold", args.correlation_threshold)
    print_arguments("Attribution threshold", args.attribution_threshold)
    print_arguments("Min. relative attribution", args.min_relative_attribution)
    print_arguments("Attribution range", r)
    print_arguments("Plot format", args.plot_format)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_plot_mutagenesis(
        input=args.input,
        output_directory=args.output,
        correlation_threshold=args.correlation_threshold,
        attribution_threshold=args.attribution_threshold,
        plot_format=args.plot_format,
        attribution_range=attribution_range,
    )


# Train task ===================================================================
def train_subparser(subparsers):
    "Parses inputs from commandline and returns them as a Namespace object."

    def str2bool(v):
        if v == "False":
            return False
        else:
            return v

    group = subparsers.add_parser(
        "train",
        help="Train a new PARM model from pre-processed MPRA data",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description="R|" + description,
    )

    required_args = group.add_argument_group("Required arguments")
    # Arguments for the input files
    required_args.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Path to input files. This should be a pre-processed MPRA data file. "
        "saved as a .h5 file. If you have multiple files, you can pass them as a space-separated list.",
    )

    required_args.add_argument(
        "--validation",
        nargs="+",
        required=True,
        type=str,
        help="Path to validation files. This should be a pre-processed MPRA data file. "
        "saved as a .h5 file. If you have multiple files",
    )
    
    required_args.add_argument(
        "--output", required=True,
        type=str,
        help="Path to the directory to store all the output files.",
    )
    
    required_args.add_argument(
        "--cell_type", required=True,
        type=str,
        help="The name of the cell type that you want to create a model to. "
        "This should be the same name as in the input h5 files",
    )
    
    required_args.add_argument(
        "--measurement_column", required=True,
        type=str,
        help="Which column in the input file contains the measurement data. "
        "(e.g., Log2TPM_K562)",
    )

    model_args = group.add_argument_group("Advanced arguments (for model training)")

    model_args.add_argument(
        "--n_workers",
        default=0,
        type=int,
        help="How many subprocesses to use for data loading (default: 0) \n",
    )
    
    model_args.add_argument(
        "--n_epochs",
        default=7,
        nargs="?",
        type=int,
        help="Number of epochs to train the data to (default: 7) \n",
    )

    model_args.add_argument(
        "--batch_size",
        default=128,
        nargs="?",
        type=int,
        help="Number of samples in ech batch to train the data to (default: 128) \n",
    )

    model_args.add_argument(
        "--betas",
        default=(0.0005, 0.0005),
        nargs="+",
        type=float,
        help="L1 and L2 regularization terms respectively. (default: (0.005, 0.005) ) \t run like -betas 0.1 0.2 \n",
    )

    model_args.add_argument(
        "--lr",
        default=0.001,
        nargs="?",
        type=float,
        help="Learning rate (default: 0.001) \n",
    )

    model_args.add_argument(
        "--cosine_scheduler",
        default=True,
        nargs="?",
        type=str2bool,
        help="If True, implement a cosine schedueler for learning rate. Otherwise, learning rate will be constant after warmup. (default:True)",
    )

    model_args.add_argument(
        "--weight_decay",
        default=0.0,
        nargs="?",
        type=float,
        help="Weight decay (default: 0.0) \n",
    )

    model_args.add_argument(
        "--adaptor",
        default=("CAGTGAT", "ACGACTG"),
        nargs="+",
        help="If not false, give adaptor in 5 and 3 prima to use as padding. \n "
        "   e.g. -adaptor CAGTGAT ACGACTG \n "
        "(default: CAGTGAT ACGACTG) \n",
    )

    model_args.add_argument(
        "--L_max",
        default=600,
        nargs="?",
        type=int,
        help="Maximum length of fragments. Necessary if we want to downsample. \n "
        "(default: 600) \n",
    )

    model_args.add_argument(
        "--n_blocks",
        default=5,
        type=int,
        help="Number of convolution blocks. (default: 5)",
    )

    model_args.add_argument(
        "--filter_size",
        default=125,
        type=int,
        help="Number of filters in convolution layers (default: 125)",
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
    other_args.add_argument(
        "--check_cuda",
        action=check_cuda,
        nargs=0,
        help="Check if CUDA is available and exit",
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
        description="R|" + description,
    )

    required_args = group.add_argument_group("Required arguments")

    required_args.add_argument(
        "--model",
        required=True,
        nargs="+",
        help="Path to the weight files for the model. If you want to perform predictions "
        "for multiple models at once, you can pass them all as a space-separated list. "
        "If you have not trained a model, you can use the pre-trained model from the "
        "pre_trained_models directory.",
    )
    required_args.add_argument(
        "--input",
        required=True,
        help="Path to the input fasta file with the sequences to be predicted.",
    )
    required_args.add_argument(
        "--output",
        required=True,
        help="Path to the output file where the predictions will be saved. Output is a "
        "tab-separated file with the sequence, header, and the predicted score.",
    )
    
    advanced_args = group.add_argument_group("Advanced arguments (if you trained your own model)")

    advanced_args.add_argument(
        "--L_max",
        type=int,
        default=600,
        help="The maximum length of the sequences allowed by the model. All pre-trained models "
        "have `--L_max 600`. However, if you trained your own PARM model with a different L_max value, "
        "you should specify it here. (Default: 600)"
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
        description="R|" + description,
    )

    required_args = group.add_argument_group("Required arguments")

    required_args.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="Path to the weight files for the model. If you want to perform predictions "
        "for multiple models at once, you can pass them all as a space-separated list. "
        "If you have not trained a model, you can use the pre-trained model from the "
        "default_PARM_models directory.",
    )
    required_args.add_argument(
        "--input",
        required=True,
        help="Path to the input fasta file with the sequences to have to mutagenesis for.",
    )
    required_args.add_argument(
        "--output",
        required=True,
        help="Path to the directory where the files will be stored. Will be created "
        "if it does not exist.",
    )

    optional_arguments = group.add_argument_group("Optional arguments")
    optional_arguments.add_argument(
        "--motif_database",
        default="https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt",
        help="Path or url to the motif databae (JASPAR format). Default is HOCOMOCOv11: https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt",
    )

    advanced_args = group.add_argument_group("Advanced arguments (if you trained your own model)")

    advanced_args.add_argument(
        "--L_max",
        type=int,
        default=600,
        help="The maximum length of the sequences allowed by the model. All pre-trained models "
        "have `--L_max 600`. However, if you trained your own PARM model with a different L_max value, "
        "you should specify it here. (Default: 600)"
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


    global default_motif_db
    default_motif_db = optional_arguments.get_default("motif_database")
    
    group.set_defaults(func=mutagenesis)


# Plot task ====================================================================
def plot_subparser(subparsers):
    group = subparsers.add_parser(
        "plot",
        help="Plot results of a mutagenesis assay generated by PARM. "
        "Produces a PDF file with the mutagenesis plot.",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description="R|" + description,
    )

    required_args = group.add_argument_group("Required arguments")

    required_args.add_argument(
        "--input",
        required=True,
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
        type=float,
        help="The minimum value of Pearson correlation that a scanned motif needs "
        "to present in order to be shown in the plot (Default: 0.75).",
    )
    optional_arguments.add_argument(
        "--attribution_threshold",
        default=0.001,
        type=float,
        help="The minimum value of attribution (i.e., the mean attribution score "
        "for the bases of a motif) that a scanned motif needs to present in order "
        "to be shown in the plot (Default: 0.001).",
    )
    optional_arguments.add_argument(
        "--min_relative_attribution", 
        default=0.15,
        type=float,
        help="The minimum mean attribution threshold for motif to be shown, expressed "
        "as a percentage of the maximum letter attribution within any motif. "
        "i.e. only motifs with mean attribution above this percentage of the highest attributed letter "
        "will be shown. (Default: 0.15).",
    )
    optional_arguments.add_argument(
        "--attribution_range",
        default=None,
        nargs=2,
        type=float,
        help="Space-separated range of attribution values to be shown in the plot. "
        "(like 0.001 0.01). If not provided, the range will be calculated based on "
        "the values present in the data.",
    )
    optional_arguments.add_argument(
        "--plot_format",
        default="pdf",
        type=str,
        choices=["pdf", "svg", "jpg", "png"],
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
    return (
        "\nAll done!\n"
        "If you make use of PARM in your research, please cite:\n\n"
        "  Barbadilla-Martínez L., Klaassen N.H., Franceschini-Santos V.H, et. al. (2024) \n"
        "  The regulatory grammar of human promoters uncovered by MPRA-trained deep learning. \n"
        "  bioRXiv. https://doi.org/10.1101/2024.07.09.602649\n"
        "\n"
        ""
    )



# Main =========================================================================
if __name__ == "__main__":
    main()
