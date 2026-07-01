#!/usr/bin/env python3

import argparse
from .rrwick_help_formatter import MyParser, MyHelpFormatter
from .version import __version__
import warnings
from .PARM_misc import check_cuda

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

    # Evaluate model task =================================================================
    # ====================================================================================
    evaluation_model_subparser(subparsers)


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
    # Lazy import to reduce initial loading time
    from .PARM_train import PARM_train
    # Implement the logic for the train command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Train"))
    print("-" * 80)
    print_arguments("Input", args.input)
    print_arguments("Validation", args.validation)
    print_arguments("Output", args.output)
    print_arguments("Cell type", args.cell_type)
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
    print_arguments("Initial weights", args.initial_weights)
    print_arguments("Dense layer after split", args.dense_layer_after_split)

    print("=" * 80)
    PARM_train(args)


def predict(args):
    # Lazy import to reduce initial loading time
    from .PARM_predict import PARM_predict
    # Implement the logic for the predict command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Predict"))
    print("-" * 80)
    print_arguments("Model", args.model)
    print_arguments("Input", args.input)
    print_arguments("Output", args.output)
    print_arguments("Number of batches", args.n_seqs_per_batch)
    print_arguments("Show only sequence's header in the output file?", args.header_only)
    print_arguments("Model filter size ", args.filter_size)
    print_arguments("Loss function type", args.type_loss)
    print_arguments("L_max", args.L_max)
    print_arguments("Predict test fold?", args.predict_test_fold)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_predict(
        input=args.input,
        model_directory=args.model,
        output=args.output,
        n_seqs_per_batch=args.n_seqs_per_batch,
        store_sequence=not args.header_only,
        filter_size= args.filter_size,
        type_loss=args.type_loss,
        L_max=args.L_max,
        test_fold = args.predict_test_fold
    )


def mutagenesis(args):
    # Lazy import to reduce initial loading time
    from .PARM_mutagenesis import PARM_mutagenesis
    from .PARM_misc import check_sequence_length
    # Check input fasta
    check_sequence_length(args.input, args.L_max)
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Mutagenesis"))
    print("-" * 80)
    # models = ",".join(args.model)
    print_arguments("Model", args.model)
    print_arguments("Input", args.input)
    print_arguments("Output", args.output)
    print_arguments("Cell type", args.cell_type)
    print_arguments("Filter size", args.filter_size)
    print_arguments("Type loss function", args.type_loss)
    # check if args.motif_database is the default
    if args.motif_database == default_motif_db:
        print_arguments("Motif database", "HOCOMOCOv11 (default)")
    else:
        print_arguments("Motif database", args.motif_database)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_mutagenesis(
        input=args.input,
        model_directory=args.model,
        output_directory=args.output,
        motif_database=args.motif_database,
        filter_size=args.filter_size,
        type_loss=args.type_loss,
        n_conv_blocks=args.n_blocks,
        cell_type=args.cell_type
    )


def plot(args):
    # Lazy import to reduce initial loading time
    from .PARM_mutagenesis import PARM_plot_mutagenesis
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
        r = ", ".join([str(i) for i in args.attribution_range])
        attribution_range = args.attribution_range
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



def evaluation_model(args):
    # Lazy import to reduce initial loading time
    from .PARM_model_evaluation import PARM_eval_model
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Evaluation model"))
    print("-" * 80)
    
    print_arguments("Model", args.model)
    print_arguments("Output directory", args.output_directory)
    print_arguments("Criterion used to train the model", args.criterion)
    print_arguments("L_max", args.L_max)
    print_arguments("Input h5py file(s) to compute predictions of MPRA fragments", args.input_h5py_file)
    print_arguments("Cell type to work with", args.cell_type)
    print_arguments("Features to select the fragments of interest", args.features_fragments_selection)
    print_arguments("File(s) of measured mutagenesis assays to validate the model", args.file_input_mutagenesis_validation)
    print_arguments("PWM datasets to use to study motifs in the model", args.PWM_datasets)
    print_arguments("Batch size to compute the attributions", args.batch_size)
    print_arguments("Number of random sequences to generate for motif insertion", args.num_sequences_rnd)
    print_arguments("Normalization method to use for the predictions and measurements", args.normalization_method)
    print_arguments("Filter size of the model", args.filter_size)
    print_arguments("Number of convolution blocks of the model", args.n_blocks)
    print_arguments("File of SNPs in SuRE format to compute the SNP effects", args.file_SNP_SuRE)
    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)
    PARM_eval_model(
        model_dir=args.model,
        output_directory=args.output_directory,
        criterion=args.criterion,
        L_max=args.L_max,
        input_h5py_file=args.input_h5py_file,
        cell_type=args.cell_type,
        features_fragments_selection=args.features_fragments_selection,
        file_input_mutagenesis_validation=args.file_input_mutagenesis_validation,
        PWM_datasets=args.PWM_datasets,
        batch_size=args.batch_size,
        num_sequences_rnd=args.num_sequences_rnd,
        normalization_method=args.normalization_method,
        filter_size=args.filter_size,
        n_conv_blocks=args.n_blocks,
        file_SNP_SuRE=args.file_SNP_SuRE
    )

def str2bool(v):
    if v == "False":
        return False
    elif v == "True":
        return True
    else:
        return v
        
# Train task ===================================================================
def train_subparser(subparsers):
    "Parses inputs from commandline and returns them as a Namespace object."

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
        default=(0.001, 0.001),
        nargs="+",
        type=float,
        help="L1 and L2 regularization terms respectively. (default: (0.001, 0.001) ) \t run like -betas 0.1 0.2 \n",
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
        default=0.2,
        nargs="?",
        type=float,
        help="Weight decay (default: 0.2) \n",
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
    
    model_args.add_argument(
        "--initial_weights",
        default=None,
        type=str,
        help="Path to initial weights file. If None, random initialization is used. (default: None)",
    )

    model_args.add_argument(
        "--filtering_on_FEAT",
        default=False,
        type=str,
        help="Filtering the h5 file based on where the fragments overlap. Only choose this option if the files have been set up for filtering in either TSS or EnhA."
    )
    
    model_args.add_argument(
        "--dense_layer_after_split",
        help="Number of dense layers after split. (default: False)",
        default=False,
        type=str2bool,
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
        help="Path to the directory of the model. If you want to perform predictions "
        "for the pre-trained K562 model, for instance, this should be "
        "pre_trained_models/K562. If you have trained your own model, "
        "you should pass the path to the directory where the .parm files are stored. ",
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

    required_args.add_argument(
        "--n_seqs_per_batch",
        type=int,
        default=1,
        help=" Number of sequences to predict simultaneously, increase only if your memory allows it. (Default: 1)"
    )

    required_args.add_argument(
        "--header_only",
        action = argparse.BooleanOptionalAction,
        default=False,
        help="If this flag is set, the output file will not contain the sequences of the\n"
                " input fasta. By default, PARM shows both the sequence and the header."
    )

    advanced_args = group.add_argument_group("Advanced arguments (if you trained your own model)")

    advanced_args.add_argument(
        "--predict_test_fold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If this flag is set, PARM will assume the input is the hdf5 file of the "
        "test fold of a trained model. This is useful if you want to evaluate the "
        "performance of a model that you trained. "
    )
        
    advanced_args.add_argument(
        "--L_max",
        type=int,
        default=600,
        help="The maximum length of the sequences allowed by the model. All pre-trained models "
        "have `--L_max 600`. However, if you trained your own PARM model with a different L_max value, "
        "you should specify it here. (Default: 600)"
    )

    advanced_args.add_argument(
        "--filter_size",
        type=int,
        default=125,
        help="The model size that torch expects (Default: 125) "
    )

    advanced_args.add_argument(
        "--type_loss",
        default = 'poisson',
        choices=['MSE', 'poisson', 'heteroscedastic'],
        type = str,
        help=' Type of loss function to use for the model. Default is "poisson". Other options are "MSE" and "heteroscedastic".'
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
        required=True,
        help="Path to the directory of the model. If you want to perform predictions "
        "for the pre-trained K562 model, for instance, this should be "
        "pre_trained_models/K562. If you have trained your own model, "
        "you should pass the path to the directory where the .parm files are stored. ",
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
    advanced_args.add_argument(
        "--filter_size",
        type=int,
        default=125,
        help="The model size that torch expects (Default: 125) "
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

#####Evaluation task of the model =================================================================

def evaluation_model_subparser(subparsers):
    "Parses inputs from commandline and returns them as a Namespace object."

    group = subparsers.add_parser(
        "evaluation_model",
        help="Evaluation of model. If provided, it can perform 4 tests: \n"
        "1) Compute predictions of MPRA fragments and compare them with the measured activity \n (if --input_h5py_file is provided) \n"
        "2) Compute mutation effects of the mutagenesis library and compare with measurements (used in the PARM paper Fig 2c-e) \n"
        "        (if --file_input_mutagenesis_validation is provided, it can be found in the repo: ./example_data/mutagenesis_library/mutagenesis_validation_promoters.txt). \n"
        "3) Check whether motifs are detected by the model. \n"
        " In a set of random sequences, we insert each motif from the database individually, compute the ISM, and measure the correlation between the attribution scores and the known motif. \n"
        " (set in --PWM_datasets HOCOMOCOv11 is used by default.) \n"
        "4) Compute the predicted effect of SNPs with significant effect in the SuRE4n data (as tested in van Arensbergen et al., 2019) (if --file_SNP_SuRE is provided, it can be found in the repo: ./example_data/SNP_SuRE/SuRE_SNPs_example.txt). \n",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description="R|" + description,
    )

    ##Required arguments
    required_args = group.add_argument_group("Required arguments")

    required_args.add_argument(
        "--model",
        required=True,
        nargs = '+',
        help="Path to the directory of the model. If you want to perform predictions "
        "for the pre-trained K562 model, for instance, this should be "
        "pre_trained_models/K562. If you have trained your own model, "
        "you should pass the path to the directory where the .parm files are stored. ",
    )

    required_args.add_argument(
        "--output_directory",
        type=str,
        required = True,
        help="Directory where to save the plots, if False the plots will be shown in the screen.\n",
    )

    #Optional arguments
    optional_arguments = group.add_argument_group("Optional arguments")

    #Optional model arguments
    optional_arguments.add_argument(
         "--criterion",
         type=str, 
         choices=['poisson', 'heteroscedastic'],
            default='poisson',
            help="General argument. \n Criterion used during training of the model, important for architecture. \n"
    )

    optional_arguments.add_argument(
        "--L_max",
        default=600,
        nargs="?",
        type=int,
        help="General argument. \n Maximum length of fragments (default: 600) \n",
    )
    
    optional_arguments.add_argument(
        "--filter_size",
        default=125,
        type=int,
        help="General argument. \n Number of filters in convolution layers (default: 125) \n",
    )
    
    optional_arguments.add_argument(
        "--n_blocks",
        default=5,
        type=int,
        help="General argument. \n Number of convolution blocks of the model (default: 5) \n",
    )

    optional_arguments.add_argument(
        "--cell_type",
        required=True,
        help="General argument. Cell line to work with (K562, HEPG2, hNPC, HCT166, MCF7, mESC or any combination of these separated by two underscores (__). Used in the input_h5py_file to select the right files to compute the predictions of the MPRA fragments. \n",
    )

    optional_arguments.add_argument(
        "--dense_layer_after_split",
        type=str2bool,
        default=False,
        help="General argument. \n Whether to use dense layers after split (default: False) \n",
    )

    # Arguments for the input files

    ###Arguments for STEP 1
    optional_arguments.add_argument(
        "--input_h5py_file",
        type=str,
        nargs="+",
        help="Step 1) \n Argument necessary for step 1 (Compute predictions of MPRA fragments). \n h5 file path, if several, separate them by a space. \n",
    )

    

    optional_arguments.add_argument(
        "--features_fragments_selection",
        default="TSS",
        nargs="?",
        type=str,
        help="Step 1) \n  Argument necessary for step 1 (Compute predictions of MPRA fragments). \n"
        "  Features to use to select SuRE fragments of interest (default: TSS) \n"
        '     In humans choose from TSS, EnhA, peaks or a combination of them separated by "_" e.g. TSS_EnhA \t'
        '     In mice choose from TSS, EnhA_many or EnhA_strong a combination of them separated by "_". \n',
    )
    
    optional_arguments.add_argument(
        "--normalization_method",
        type=str,
        default="Log2RPM",
        help="Step 1) \n Normalization method to use for the predictions and measurements. \n Default (Log2RPM) \n"
    )
    

    ###Arguments for STEP 2
    optional_arguments.add_argument(
        "--file_input_mutagenesis_validation",
        type=str,
        nargs="+",
        # default = './example_data/mutagenesis_library/mutagenesis_validation_promoters.txt',
        default=None,
        help="Step 2) \n  File of measured mutagenesis assays where the format of the file is, should contain at least the following columns: \n"
        "   chr	start	end	strand	prom	mut_po	ref	alt	sequence	seq_type	oligo_identifyer	bc	HCT116	HepG2	K562	LNCaP	MCF7\n"
        "File found in the example data: ./example_data/mutagenesis_library/mutagenesis_validation_promoters.txt \n",

    )


    ###Arguments for STEP 3
    optional_arguments.add_argument(
        "--PWM_datasets",
        type=str,
        nargs="+",
        default = False,
        help="Step 3) \n Files or paths of the PWM datasets in jaspar format to use to study motifs in the model. If several, separate them by a space. \n"
        "e.g. https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt. (default: False, if not provided, motif analysis will not be performed. If you want to use the HOCOMOCOv11 database, you can set it as https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_HUMAN_mono_jaspar_format.txt \n",
    )

    optional_arguments.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Step 3 and 4) \n Number of sequences to compute the attribution at the same time. Relevant when not enough memory. \n Default (2000) \n",
    )

    optional_arguments.add_argument(
        "--num_sequences_rnd",
        type=int,
        default=100,
        help="Step 3) \n Number of random sequences to generate for the motif insertion. \n Default (100) \n"
    )

    ##Argument for STEP 4

    optional_arguments.add_argument(
        "--file_SNP_SuRE",
        type=str,
        default=False,
        help="Step 4) \n File containing the SNPs in SuRE SNP format. \n"
        "File found in the example data in the github: example_data/eval_values/4_SNP_deltas_SuRE4n/hepg2.sign.id.LP190708_sequences.txt for HepG2 \n"
        "or example_data/eval_values/4_SNP_deltas_SuRE4n/k562.sign.id.LP190708_sequences.txt for K562 \n"
    )


    group.set_defaults(func=evaluation_model)







#####
def bye_message():
    return (
        "\nAll done!\n"
        "If you make use of PARM in your research, please cite:\n\n"
        "  Barbadilla-Martínez L., Klaassen N.H., Franceschini-Santos V.H, et. al. \n"
        "  Regulatory grammar in human promoters uncovered by MPRA-based deep learning.\n"
        "  Nature (2026). https://doi.org/10.1038/s41586-025-10093-z\n"
        "\n"
        ""
    )



# Main =========================================================================
if __name__ == "__main__":
    main()
