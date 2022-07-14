"""Init jobs package"""
from .evaluators import Evaluator
from .algorithms import Algo
from .time_units import TimeUnit
from .numerical_group_by_methods import NumericalGroupByMethod
from .categorical_group_by_method import CategoricalGroupByMethod

# checks if enum is enumerated in Enum
# Throws AttributeError if not in Enum; returns enum as str otherwise
def check_is_enum(Enum, enum):
	if isinstance(enum, str):
		enum_isStr = eval('%s.%s'%(Enum.__name__, enum))
		if enum_isStr in Enum:
			return enum
		else:
			raise AttributeError(evaluator)
	elif isinstance(enum, Enum):
		return enum.value
	elif enum == None:
		return None
	else:
		raise AttributeError("[%s] Type Error"%type(enum).__name__)
