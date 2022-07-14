"""Init core_api package"""
from . import body_obj as CoreBody
from .api import CoreAPI
from .model import Model, MultiModel
from .predict_input import PredictInput, PredictTSInput
from .train_input import TrainInput, TrainTSInput, TrainClusterInput
from .setup_input import SetupInput
from .worker import Worker
