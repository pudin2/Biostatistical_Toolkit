from .load import (
    load_dataframe_from_path,
    load_dataframes,
    register_reader,
    supported_extensions,
)


def load_multiple_dataframes(*args, **kwargs):
    """Atajo con diálogo gráfico; tkinter se importa solo al llamarlo."""
    from .load_gui import load_multiple_dataframes as _impl

    return _impl(*args, **kwargs)


def load_single_dataframe(*args, **kwargs):
    from .load_gui import load_single_dataframe as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "load_dataframe_from_path",
    "load_dataframes",
    "register_reader",
    "supported_extensions",
    "load_multiple_dataframes",
    "load_single_dataframe",
]
