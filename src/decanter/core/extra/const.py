# pylint: disable=too-few-public-methods
"""Define variables of Decanter Core for all modules."""
from enum import Enum


class CoreStatus:
    """Status for tasks and jobs."""
    DONE = 'done'
    PENDING = 'pending'
    FAIL = 'fail'
    INVALID = 'invalid'
    RUNNING = 'running'
    DONE_STATUS = [DONE, INVALID, FAIL]
    FAIL_STATUS = [INVALID, FAIL]
    ALL_STATUS = [PENDING, RUNNING, DONE, INVALID, FAIL]


class CoreKeys(Enum):
    """Keys of Decanter Core responses."""
    id = '_id'
    progress = 'progress'
    status = 'status'
    result = 'result'
