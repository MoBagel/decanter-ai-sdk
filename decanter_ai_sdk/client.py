from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any
import pandas as pd
from decanter_ai_sdk.enums.algorithms import IIDAlgorithms, TSAlgorithms
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.time_units import TimeUnit
from .enums.data_types import DataType


class AbstractClient(ABC):
    @abstractmethod
    def upload(self, data: Union[str, pd.DataFrame], name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def train_iid(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        custom_feature_types: Dict[str, DataType] = {},
        drop_features: List[str] = [],
        evaluator: Optional[Union[RegressionMetric, ClassificationMetric]] = None,
        holdout_table_id: Optional[str] = None,
        algos: Union[List[IIDAlgorithms], List[str]] = [
            IIDAlgorithms.DRF,
            IIDAlgorithms.GBM,
            IIDAlgorithms.XGBoost,
            IIDAlgorithms.GLM,
        ],
        max_model: int = 20,
        tolerance: int = 3,
        nfold: int = 5,
        stacked_ensemble: bool = True,
        validation_percentage: int = 10,
        seed: int = 1180,
        timeseries_value: List[Dict[str, Any]] = [],
        holdout_percentage: int = 10,
    ):
        raise NotImplementedError

    @abstractmethod
    def train_ts(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        datetime: str,
        time_groups: List,
        timeunit: TimeUnit,
        algos: Union[List[TSAlgorithms], List[str]] = [TSAlgorithms.GBM],
        groupby_method: Optional[str] = None,
        evaluator: RegressionMetric = RegressionMetric.WMAPE,
        exogeneous_columns_list: List = [],
        gap: int = 0,
        feature_derivation_window: int = 60,
        horizon_window: int = 1,
        validation_percentage: int = 10,
        nfold: int = 5,
        max_model: int = 20,
        tolerance: int = 3,
        seed: int = 1111,
        drop_features: List[str] = [],
        custom_feature_types: Dict[str, DataType] = {},
    ):
        raise NotImplementedError

    @abstractmethod
    def predict_iid(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict_ts(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def stop_uploading(self, id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop_training(self, id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def wait_for_response(self, url, id):
        raise NotImplementedError

    @abstractmethod
    def get_table(self, data_id: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_table_list(self) -> List[str]:
        raise NotImplementedError
