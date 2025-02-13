class AskNotSupportedError(Exception):
    """Raised on optimizers that don't support ask and tell interface when `ask` is called."""
