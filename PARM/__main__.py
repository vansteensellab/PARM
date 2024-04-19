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
    parser_train = subparsers.add_parser(
        "train", help="Train a new PARM model from pre-processed MPRA data",
        description= 'R|' + description,
    )
    parser_train.add_argument("data_file", help="Path to the training data file")
    parser_train.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser_train.set_defaults(func=train)

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


def train(args):
    # Implement the logic for the train command here
    print(f"Training with data from {args.data_file} for {args.epochs} epochs")


def predict(args):
    # Implement the logic for the predict command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Predict"))
    print("-" * 80)
    models = ",".join(args.model)
    print("Model: {: >73}".format(models))
    print("Input:{: >74}".format(args.input))
    print("Output:{: >73}".format(args.output))
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
    print("Model: {: >73}".format(models))
    print("Input:{: >74}".format(args.input))
    print("Output:{: >73}".format(args.output))
    print("Motif database:{: >65}".format(args.motif_database))
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
    print("Input:{: >74}".format(args.input))
    print("Output:{: >73}".format(out))
    print("Correlation threshold:{: >58}".format(args.correlation_threshold))
    print("Attribution threshold:{: >58}".format(args.attribution_threshold))
    print("Plot format:{: >68}".format(args.plot_format))
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