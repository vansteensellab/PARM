from .version import __version__

def log(message):
    v = 'PARM v' + __version__
    print(f"[{v}] {message}", flush=True)