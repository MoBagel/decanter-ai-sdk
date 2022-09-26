from enum import Enum


class Missing_Value_Handling(Enum):
    """
    Numerical missing value handling.
        - Automatic
        - Mean
        - Median
        - Zero
        - DropRows
    """

    Automatic = "automatic"
    Mean = "mean"
    Median = "median"
    Zero = "zero"
    DropRows = "drop_rows"
    NA = "na"
    Mode = "mode"
