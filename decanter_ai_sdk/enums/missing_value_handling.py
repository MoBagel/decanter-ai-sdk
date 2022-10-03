from enum import Enum


class MissingValueHandling(Enum):
    """
    Numerical missing value handling.
        - automatic
        - mean
        - median
        - zero
        - droprows
        - na
        - mode
    """

    automatic = "automatic"
    mean = "mean"
    median = "median"
    zero = "zero"
    droprows = "drop_rows"
    na = "na"
    mode = "mode"
