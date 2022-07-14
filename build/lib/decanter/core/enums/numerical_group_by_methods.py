'''available numerical group by methods supported by corex'''
from enum import Enum

class NumericalGroupByMethod(Enum):
    sum = 'sum'
    mean = 'mean'