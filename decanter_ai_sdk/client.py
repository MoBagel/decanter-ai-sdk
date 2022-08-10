from ctypes import Union
from io import StringIO
from time import sleep
from typing import Dict, List, Union, Any, Optional
import pandas as pd
from tqdm import tqdm
from decanter_ai_sdk.enums.algorithms import IIDAlgorithms, TSAlgorithms

from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.web_api.api import Api
import logging

from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.time_units import TimeUnit

logging.basicConfig(level=logging.INFO)


class Client:
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
            auth_headers={
                "Authorization": "Bearer " + auth_key,
            },
            project_id=project_id,
        )

    def upload(self, data: Union[str, pd.DataFrame], name: str) -> str:

        if data is None:
            raise ValueError("[Upload] Uploaded None file.")

        elif isinstance(data, pd.DataFrame):
            textStream = StringIO()
            data.to_csv(textStream, index=False)
            file = [("file", (name, textStream.getvalue(), "text/csv"))]
            textStream.close()

        else:
            file = [("file", (name, data, "text/csv"))]

        table_id = self.api.post_upload(file=file, name=name)
        res = self.wait_for_response("table", table_id)

        return res["_id"]

    def train_iid(
        self,
        experiment_name: str,
        table_id: Optional[str],
        target: Optional[str],
        drop_features: Optional[List[str]] = None,
        custom_feature_types: Optional[Dict] = [],
        evaluator: Optional[str] = None,
        holdout_table_id: Optional[str] = None,
        stopping_metric: str = "auc",
        algos: List[IIDAlgorithms] = [
            IIDAlgorithms.DRF.value,
            IIDAlgorithms.GBM.value,
            IIDAlgorithms.XGBoost.value,
            IIDAlgorithms.GLM.value,
        ],
        max_model: int = 20,
        tolerance: int = 3,
        nfold: int = 5,
        stacked_ensemble: bool = True,
        validation_percentage: int = 10,
        seed: int = 1111,
        timeseries_value: List[str] = [],
        holdout_percentage: int = 10,
    ) -> Experiment:

        for alg in algos:
            if alg not in IIDAlgorithms._value2member_map_:
                raise ValueError("Wrong alogrithm: " + alg)

        data_column_info = self.api.get_table_info(table_id=table_id)

        if validation_percentage < 5 or validation_percentage > 20:
            raise ValueError(
                "validation_percentage should be inside a range between 5 to 20."
            )

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
                if feature["id"] == list(cft.keys())[0]:
                    feature["data_type"] = list(cft.values())[0]

        if data_column_info[target] == "numerical":
            category = "regression"
            if evaluator is None:
                evaluator = "wmape"
            elif evaluator not in RegressionMetric._value2member_map_:
                raise ValueError("Wrong evaluator, you need to fill wmape, mse ...")

        else:
            category = "classification"
            if evaluator is None:
                evaluator = "auc"
            elif evaluator not in ClassificationMetric._value2member_map_:
                raise ValueError("Wrong evaluator, you need to fill auc, logloss...")

        holdout_config = dict()
        if holdout_percentage:
            holdout_config["percent"] = holdout_percentage
        if holdout_table_id:
            holdout_config["table"] = holdout_table_id

        training_settings = {
            "project_id": self.project_id,
            "name": experiment_name,
            "gp_table_id": table_id,
            "seed": seed,
            "target": target,
            "targetType": data_column_info[target],
            "features": features,
            "feature_types": feature_types,
            "evaluator": evaluator,
            "category": category,
            "stopping_metric": "auc",
            "is_binary_classification": True,
            "holdout": holdout_config,
            "tolerance": tolerance,
            "nfold": nfold,
            "max_model": max_model,
            "algos": algos,
            "stacked_ensemble": stacked_ensemble,
            "validation_percentage": validation_percentage,
            "timeseriesValues": timeseries_value,
            "stopping_metric": stopping_metric,
        }

        exp_id = self.api.post_train_iid(training_settings)

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))

        return experiment

    def train_ts(
        self,
        experiment_name: str,
        train_table_id: str,
        target: str,
        datetime: str,
        time_groups: List,
        timeunit: TimeUnit,
        algos: List[TSAlgorithms] = [TSAlgorithms.GBM.value],
        groupby_method: Optional[str] = None,
        evaluator: RegressionMetric = RegressionMetric.WMAPE.value,
        exogeneous_columns_list: List = [],
        gap: int = 0,
        feature_derivation_window: int = 60,
        horizon_window: int = 1,
        validation_percentage: int = 10,
        nfold: int = 5,
        max_model: int = 20,
        tolerance: int = 3,
        seed: int = 1111,
        drop_features=[],
        custom_feature_types: Optional[Dict] = [],
    ):

        for alg in algos:
            if alg not in TSAlgorithms._value2member_map_:
                raise ValueError("Wrong alogrithm: " + alg)

        if evaluator not in RegressionMetric._value2member_map_:
            raise ValueError("Wrong evaluator: " + evaluator)

        if validation_percentage < 5 or validation_percentage > 20:
            raise ValueError(
                "validation_percentage should be inside a range between 5 to 20."
            )

        data_column_info = self.api.get_table_info(table_id=train_table_id)

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
                if feature["id"] == list(cft.keys())[0]:
                    feature["data_type"] = list(cft.values())[0]

        training_settings = {
            "project_id": self.project_id,
            "name": experiment_name,
            "gp_table_id": train_table_id,
            "seed": seed,
            "target": target,
            "targetType": data_column_info[target],
            "features": features,
            "feature_types": feature_types,
            "category": "regression",
            "stopping_metric": evaluator,
            "is_binary_classification": True,
            "holdout": {"percent": 10},
            "tolerance": tolerance,
            "max_model": max_model,
            "algos": algos,
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

        if model is None and (experiment_id is None or model_id is None):
            raise ValueError(
                "either model or both experiment_id and model_id should be defined"
            )

        mod_id = model.model_id if model is not None else model_id
        exp_id = model.experiment_id if model is not None else experiment_id

        for k in self.api.get_model_type(exp_id, {"projectId": self.project_id}):
            if k["_id"] == mod_id:
                is_multi_model = (
                    True
                    if k["model_type"]
                    in ["ExodusModel", "MultiModel", "LeviticusModel"]
                    else False
                )

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
        """
        return self.api.get_table(data_id=data_id)

    def get_table_list(self) -> List[str]:
        """
        Return list of table information.
        """
        return self.api.get_table_list()
