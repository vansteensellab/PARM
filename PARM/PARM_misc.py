from .version import __version__

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