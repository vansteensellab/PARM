from .version import __version__
from Bio import SeqIO
import sys
import argparse

def log(message: str):
    """
    Simple function to print a log message. 
    This writes the message to the console with a version number.
    
    Parameters
    ----------
    message : str
        The message to print to the console.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> log("This is a message")
    [PARM v0.1.0] This is a message
    """
    v = 'PARM v' + __version__
    print(f"[{v}] {message}", flush=True)
    
def check_sequence_length(fasta_file, L_max = 600):
    """
    Check if any of the sequences in a fasta file are longer than a specified length.
    If any, an error message is printed.
    """
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        if len(record.seq) > L_max:
            sys.exit(f"Error: Sequence {record.id} is longer than {L_max} nucleotides.")
            

class check_cuda(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import torch
        # Show torch info
        print(f"pytorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("CUDA is available.")
        else:
            print("CUDA is not available.")
        parser.exit()