'''available categorical group by methods supported by corex'''
from enum import Enum

class CategoricalGroupByMethod(Enum):
    count = 'count'
    mode = 'mode'