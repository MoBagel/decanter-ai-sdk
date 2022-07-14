# pylint: disable=C0103,R0903
# pylint: disable=too-many-instance-attributes
# pylint: disable=super-init-not-called
"""Worker.

This module handles actions is related to worker.
"""
import logging

import decanter.core.core_api as api

logger = logging.getLogger(__name__)

class Worker:
    """Worker.
    Attributes:
        get_status: Function to get workers status.
        get_count: Function to get counts of worker.
    """
    def __init__(self):
        self.get_status = api.CoreAPI().get_worker_status
        self.get_count = api.CoreAPI().get_worker_count
