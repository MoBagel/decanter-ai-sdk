from io import StringIO
from time import sleep
from typing import Dict, List, Union, Optional, Any
import logging

import pandas as pd
from tqdm import tqdm
from decanter_ai_sdk.enums.algorithms import IIDAlgorithms, TSAlgorithms
from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.web_api.api import Api
from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.time_units import TimeUnit
from .enums.data_types import DataType

logging.basicConfig(level=logging.INFO)


class Client:
    """
    Handle client side actions.

    Support actions sunch as upload data, iid train,
    predict, time series train and predict...ect.

    Example:
    
    .. code-block:: python

    from decanter_ai_sdk.client import Client

    client = Client(auth_key="", project_id="", host="")
    
    upload_id = client.upload(data=train_file, name="train_upload")

    ...
    """
    def __init__(self, auth_key, project_id, host):
        self.auth_key = auth_key
        self.project_id = project_id
        self.host = host
        self.api = Api(
            host=host,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + auth_key,
            },
            upload_headers={
                "Authorization": "Bearer " + auth_key,
            },
            project_id=project_id,
        )

    def upload(self, data: Union[str, pd.DataFrame], name: str) -> str:
        """
        Upload csv file or pandas dataframe.

        Args:
            data (csv-file, :obj:`pandas.DataFrame`): File uploaded to gp backend server.
            name (:str:): Name for the upload action.
        Returns:
            (:str:): Uploaded table id.

        """

        if data is None:
            raise ValueError("[Upload] Uploaded None file.")

        if isinstance(data, pd.DataFrame):
            text_stream = StringIO()
            data.to_csv(text_stream, index=False)
            file = [("file", (name, text_stream.getvalue(), "text/csv"))]
            text_stream.close()

        else:
            file = [("file", (name, data, "text/csv"))]
        table_id = self.api.post_upload(file=file, name=name)
        res = self.wait_for_response("table", table_id)

        return res["_id"]

    def train_iid(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        custom_feature_types: List[Dict[str, DataType]] = [],
        drop_features: List[str] = [],
        evaluator: Optional[Union[RegressionMetric, ClassificationMetric]] = None,
        holdout_table_id: Optional[str] = None,
        algos: List[IIDAlgorithms] = [
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
        timeseries_value: List[Dict[Any, Any]] = [],
        holdout_percentage: int = 10,
    ) -> Experiment:
        """
        Train iid models.
        
        Args:
            experiment_name (:str:): Name for the training experiment action.
            experiment_table_id (:str:): Id for the table used to train.
            target (:str:): Target column.
            custom_feature_types (:list:[Dict[str, `~decanter_ai_sdk.enums.data_type.DataType`]]): Set customized feature types.
            drop_features (:list:[str]): Features that are not going to be used during experiment.
            evaluator (:class: `~decanter_ai_sdk.enums.evaluators.ClassificationMetric` or `~decanter_ai_sdk.enums.evaluators.RegressionMetric`): Evaluator used as stopping metric.
            holdout_table_id (:str:): Holdout table id.
            algos (:list:[`~decanter_ai_sdk.enums.algorithms.IIDAlgorithms`] or :list:[`~decanter_ai_sdk.enums.algorithms.TSAlgorithms`]): Algorithms used for experiment.
            max_model (:int:): Max model number for experiment.
            tolerance (:int:): Experiment tolerance. (1~10)
            nfold (:int:): Amount of folds in experiment. (2~10) for autoML. (1~10) for autoTSF.
            stacked_ensemble (:boolean:): If experiment has stack ensemble enabled.
            validation_percentage (:int:): Validation percentage of experiment. (5~20)
            seed (:int:): Random Seed of experiment. (1 ~ 65535)
            timeseries_value (:list:[Dict[Any, Any]]:): Objects containing time series values for cross validation.
            holdout_percentage (:int:): Holdout percentage for experiment.
        
        Returns:
            (:class: `~decanter_ai_sdk.web_api.experiment.Experiment`): Experiment results.
        """

        data_column_info = self.api.get_table_info(table_id=experiment_table_id)

        if validation_percentage < 5 or validation_percentage > 20:
            raise ValueError(
                "validation_percentage should be inside a range between 5 to 20."
            )

        algo_enum_values = []
        for algo in algos:
            algo_enum_values.append(algo.value)

        features = [
            feature
            for feature in data_column_info.keys()
            if feature not in drop_features + [target]
        ]

        feature_types = [
            {"id": k, "data_type": j}
            for k, j in {key: data_column_info[key] for key in features}.items()
        ]

        for cft in custom_feature_types:
            for feature in feature_types:
                if feature["id"] in cft:
                    feature["data_type"] = cft[feature["id"]].value

        if data_column_info[target] == "numerical":
            category = "regression"
            if evaluator is None:
                evaluator = RegressionMetric.MAPE
            elif evaluator.name not in RegressionMetric.__members__:
                raise ValueError("Wrong evaluator, you need to fill wmape, mse ...")

        else:
            category = "classification"
            if evaluator is None:
                evaluator = ClassificationMetric.AUC
            elif evaluator.name not in ClassificationMetric.__members__:
                raise ValueError("Wrong evaluator, you need to fill auc, logloss...")

        holdout_config: Dict[str, Any] = {}

        if holdout_percentage:
            holdout_config["percent"] = holdout_percentage

        if holdout_table_id:
            holdout_config["table"] = holdout_table_id

        training_settings = {
            "project_id": self.project_id,
            "name": experiment_name,
            "gp_table_id": experiment_table_id,
            "seed": seed,
            "target": target,
            "targetType": data_column_info[target],
            "features": features,
            "feature_types": feature_types,
            "category": category,
            "stopping_metric": evaluator.value,
            "is_binary_classification": True,
            "holdout": holdout_config,
            "tolerance": tolerance,
            "nfold": nfold,
            "max_model": max_model,
            "algos": algo_enum_values,
            "stacked_ensemble": stacked_ensemble,
            "validation_percentage": validation_percentage,
            "timeseriesValues": timeseries_value,
        }

        exp_id = self.api.post_train_iid(training_settings)

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))

        return experiment

    def train_ts(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        datetime: str,
        time_groups: List,
        timeunit: TimeUnit,
        algos: List[TSAlgorithms] = [TSAlgorithms.GBM],
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
        custom_feature_types: List[Dict[str, DataType]] = [],
    ):
        """
        Train timeseries models.
        
        Args:
            experiment_name (:str:): Name for the experiment action.
            train_table_id (:str:): Id for the table used to experiment.
            target (:str:): Target column.
            custom_feature_types (:list:[Dict[str, `~decanter_ai_sdk.enums.data_type.DataType`]]:): Set customized feature types.
            evaluator (:class: `~decanter_ai_sdk.enums.evaluators.ClassificationMetric` or `~decanter_ai_sdk.enums.evaluators.RegressionMetric`)
            algos (:list:[`~decanter_ai_sdk.enums.algorithms.IIDAlgorithms`] or :list:[`~decanter_ai_sdk.enums.algorithms.TSAlgorithms`])
            max_model (:int:): Max model number for experiment.
            tolerance (:int:): Experiment tolerance. (1~10)
            nfold (:int:): Amount of folds in experiment. (2~10) for autoML. (1~10) for autoTSF.
            validation_percentage (:int:): Validation percentage of experiment. (5~20)
            seed (:int:): Random Seed of experiment. (1 ~ 65535)
            holdout_percentage (:int:): Holdout percentage for experiment.
            horizon_window (:int:): experiment forecast horizon window value.
            gap (:int:): Forecast gap.
            feature_derivation_window (:int:): Training forecast derivation window value.
            groupby_method (:str:): Group by method used for forecast experiment.
            exogeneous_columns_list (:list:[Dict[Any, Any]]): List of exogeneous columns.
            timeunit (:class: `~decanter_ai_sdk.enums.time_units.TimeUnit`): Time unit to use for forecast experiment [`year`, `month`, `day`, `hour`].
            time_groups (:list:[Dict[Any, Any]]): List of timegroup columns.
            datetime (:str:): Date-time column for Time Series Forecast training.
        Returns:
            (:class: `~decanter_ai_sdk.web_api.experiment.Experiment`): Experiment results.
        """

        if validation_percentage < 5 or validation_percentage > 20:
            raise ValueError(
                "validation_percentage should be inside a range between 5 to 20."
            )
        algo_enum_values = []

        for algo in algos:
            algo_enum_values.append(algo.value)

        data_column_info = self.api.get_table_info(table_id=experiment_table_id)

        features = [
            feature
            for feature in data_column_info.keys()
            if feature not in drop_features + [target]
        ]

        feature_types = [
            {"id": k, "data_type": j}
            for k, j in {key: data_column_info[key] for key in features}.items()
        ]

        for cft in custom_feature_types:
            for feature in feature_types:
                if feature["id"] in cft:
                    feature["data_type"] = cft[feature["id"]].value

        training_settings = {
            "project_id": self.project_id,
            "name": experiment_name,
            "gp_table_id": experiment_table_id,
            "seed": seed,
            "target": target,
            "targetType": data_column_info[target],
            "features": features,
            "feature_types": feature_types,
            "category": "regression",
            "stopping_metric": evaluator.value,
            "is_binary_classification": True,
            "holdout": {"percent": 10},
            "tolerance": tolerance,
            "max_model": max_model,
            "algos": algo_enum_values,
            "balance_class": True,
            "is_forecast": True,
            "stacked_ensemble": False,
            "forecast_column": datetime,
            "forecast_exogeneous_columns": exogeneous_columns_list,
            "forecast_groupby_method": groupby_method,
            "forecast_gap": gap,
            "feature_derivation_start": 0,
            "feature_derivation_window": feature_derivation_window,
            "forecast_horizon_start": gap,
            "forecast_horizon_window": horizon_window,
            "forecast_time_group_columns": time_groups,
            "forecast_timeunit": timeunit,
            "validation_percentage": validation_percentage,
            "nfold": nfold,
        }

        exp_id = self.api.post_train_ts(training_settings)

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))

        return experiment

    def predict_iid(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
    ) -> Prediction:
        """
        Predict model with test iid data.

        Args:
            model (:class: `~decanter_ai_sdk.web_api.model.Model`): Model generated by train.
            keep_columns (:list:[str]): Columns to keep while predicting.
            non_negative (:bool:): Whether to convert all negative prediction to 0.
            test_table_id (:str:): Id of table used to predict.
            model_id (:str:): Model id generated by train.
            experiment_id (:str:): Experiment id generated by train.

        Returns:
            (:class: `~decanter_ai_sdk.web_api.prediction.Prediction`): Prediction results.
        """

        if model is None and (experiment_id is None or model_id is None):
            raise ValueError(
                "either model or both experiment_id and model_id should be defined"
            )

        mod_id = model.model_id if model is not None else model_id
        exp_id = model.experiment_id if model is not None else experiment_id

        prediction_settings = {
            "project_id": self.project_id,
            "experiment_id": exp_id,
            "model_id": mod_id,
            "table_id": test_table_id,
            "is_multi_model": False,
            "non_negative": non_negative,
            "keep_columns": keep_columns,
        }

        pred_id = self.api.post_predict(prediction_settings)

        prediction = Prediction(
            attributes=self.wait_for_response("prediction", pred_id)
        )
        prediction.predict_df = self.api.get_pred_data(
            prediction.attributes["_id"], data={"prediction_id": pred_id}
        )

        return prediction

    def predict_ts(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
    ) -> Prediction:
        """
        Predict model with test timeseries data.

        Args:
            model (:class: `~decanter_ai_sdk.web_api.model.Model`): Model generated by train.
            keep_columns (:list:[str]): Columns to keep while predicting.
            non_negative (:bool:): Whether to convert all negative prediction to 0.
            test_table_id (:str:): Id of table used to predict.
            model_id (:str:): Model id generated by train.
            experiment_id (:str:): Experiment id generated by train.

        Returns:
            (:class: `~decanter_ai_sdk.web_api.prediction.Prediction`): Prediction results.
        """

        if model is None and (experiment_id is None or model_id is None):
            raise ValueError(
                "either model or both experiment_id and model_id should be defined"
            )

        mod_id = model.model_id if model is not None else model_id
        exp_id = model.experiment_id if model is not None else experiment_id

        for k in self.api.get_model_type(exp_id, {"projectId": self.project_id}):
            if k["_id"] == mod_id:
                is_multi_model = k["model_type"] in [
                    "ExodusModel",
                    "MultiModel",
                    "LeviticusModel",
                ]

        prediction_settings = {
            "project_id": self.project_id,
            "experiment_id": exp_id,
            "model_id": mod_id,
            "table_id": test_table_id,
            "is_multi_model": is_multi_model,
            "non_negative": non_negative,
            "keep_columns": keep_columns,
        }

        pred_id = self.api.post_predict(prediction_settings)

        prediction = Prediction(
            attributes=self.wait_for_response("prediction", pred_id)
        )
        prediction.predict_df = self.api.get_pred_data(
            prediction.attributes["_id"], data={"prediction_id": pred_id}
        )
        return prediction

    def wait_for_response(self, url, id):
        pbar = tqdm(total=100, desc=url + " task is now pending")
        progress = 0
        while self.api.check(task=url, id=id)["status"] != "done":
            res = self.api.check(task=url, id=id)

            if res["status"] == "fail":
                raise RuntimeError(res["progress_message"])
            else:
                if res["status"] == "running":
                    pbar.set_description("[" + url + "] " + res["progress_message"])
                    pbar.update(int(float(res["progress"]) * 100) - progress)
                    progress = int(float(res["progress"]) * 100)

            sleep(3)

        pbar.update(100 - progress)
        pbar.refresh()

        pbar.close()
        logging.info("[" + url + "] Done!")

        return self.api.check(task=url, id=id)

    def get_table(self, data_id: str) -> pd.DataFrame:
        """
        Return table dataframe.

        Args:
            data_id (:str:): Uploaded table id.
        
        Returns:
            (:pandas.DataFrame:): Uploaded table dataframe.
        """
        return self.api.get_table(data_id=data_id)

    def get_table_list(self) -> List[str]:
        """
        Return list of table information.

        Returns:
            (:list:[str]): List of uploaded table information.
        """
        return self.api.get_table_list()
