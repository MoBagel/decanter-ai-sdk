'''valid time units that are currently supported by corex'''
from enum import Enum

class TimeUnit(Enum):
    hour = 'hour'
    day = 'day'
    month = 'month'
    year = 'year'