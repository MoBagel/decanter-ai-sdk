__version__ = '0.1.0'
import logging
import sys
from .client import Client

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def enable_default_logger():
    """Set the default logger handler for the package.
    Will set the root handles to empty list, prevent duplicate handlers added
    by other packages causing duplicate logging message.
    """
    logging.root.handlers = []

    if all(isinstance(handler, logging.NullHandler)
           for handler in logger.handlers):

        logger.setLevel(logging.INFO)
        default_handler = logging.StreamHandler(sys.stderr)
        default_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s [%(levelname)8s] '
                    '%(message)s',
                datefmt='%H:%M:%S')
        )
        logger.addHandler(default_handler)