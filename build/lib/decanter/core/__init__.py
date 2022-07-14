"""Decanter AI Core SDK

decanter-ai-core-sdk is a python library for the easy use of
Decanter Core API.

"""
import logging
import sys

from .context import Context
from .client import CoreClient
from .plot import show_model_attr

core_logger = logging.getLogger(__name__)
core_logger.addHandler(logging.NullHandler())


def enable_default_logger():
    """Set the default logger handler for the package.

    Will set the root handles to empty list, prevent duplicate handlers added
    by other packages causing duplicate logging message.
    """
    logging.root.handlers = []

    if all(isinstance(handler, logging.NullHandler)
           for handler in core_logger.handlers):

        core_logger.setLevel(logging.INFO)
        default_handler = logging.StreamHandler(sys.stderr)
        default_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s [%(levelname)8s] '
                    '%(message)s',
                datefmt='%H:%M:%S')
        )
        core_logger.addHandler(default_handler)
