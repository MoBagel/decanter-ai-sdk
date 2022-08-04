from ctypes import Union
import enum
from io import StringIO
from time import sleep
from typing import List, Union, Any
import pandas as pd

from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.web_api.api import Api
import logging

from decanter_ai_sdk.enums.evaluators import Classification_enum
from decanter_ai_sdk.enums.evaluators import Regression_enum
from decanter_ai_sdk.enums.time_units import TimeUnit


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
            upload_headers={
                "Authorization": "Bearer " + auth_key,
            },
            project_id=project_id,
        )

    def upload(self, data: Union[str, pd.DataFrame], name: str) -> str:
        if data is None:
            logging.error("[Upload] Uploaded None file.")
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
        table_id: str = None,
        target: str = None,
        # features: List[str] = None,
        drop_features: List[str] = None,
        evaluator: str = None,
        stopping_metric: str = "auc",
        default_modes: str = None,
        algos: List[str] = ["DRF", "GBM", "XGBoost"],
        max_model: int = 20,
        tolerance: int = 3,
        nfold: int = 5,
        stacked_ensemble: bool = True,
        validation_percentage: int = 10,
        seed: int = 1111,
        timeseriesValue: List[str] = [],
        holdout_percentage: int = 10,
        holdout_table_id: str = None,
    ) -> Experiment:

        data_column_info = self.api.get_table_info(table_id=table_id)

        features = [
            feature
            for feature in data_column_info.keys()
            if feature not in drop_features + [target]
        ]

        feature_types = [
            {"id": k, "data_type": j}
            for k, j in {key: data_column_info[key] for key in features}.items()
        ]

        if data_column_info[target] == "numerical":
            category = "regression"
            if evaluator is None:
                evaluator = "wmape"
            elif evaluator not in Regression_enum:
                raise ValueError("Wrong evaluator, you need to fill wmape, mse ...")

        else:
            category = "classification"
            if evaluator is None:
                evaluator = "auc"
            elif evaluator not in Classification_enum:
                raise ValueError("Wrong evaluator, you need to fill auc, logloss...")

        holdout_dict = dict()
        if holdout_percentage:
            holdout_dict["percent"] = holdout_percentage
        if holdout_table_id:
            holdout_table_id["table"] = holdout_table_id

        dict_ = {
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
            "holdout": holdout_dict,
            "tolerance": tolerance,
            "nfold": nfold,
            "max_model": max_model,
            "algos": algos,
            "stacked_ensemble": stacked_ensemble,
            "validation_percentage": validation_percentage,
            "timeseriesValues": timeseriesValue,
        }

        exp_id = self.api.post_train_iid(dict_)

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))

        return experiment

    def train_ts(
        self,
        experiment_name: str,
        train_table_id: str,
        target: str,
        datetime: str,
        evaluator: str,
        time_groups: List[Any],
        timeunit: TimeUnit,
        # enum
        exogeneous_columns_list: List[Any] = [],
        groupby_method: str = None,
        gap: int = 0,
        feature_derivation_window: int = 60,
        horizon_window: int = 1,
        validation_percentage: int = 10,
        nfold: int = 5,
        max_model: int = 20,
        tolerance: int = 3,
        algos: List[str] = ["XGBoost"],
        seed: int = 1111,
        drop_features=[],
    ):
        if evaluator is None:
            evaluator = "wmape"
        elif evaluator not in Regression_enum._value2member_map_:
            raise ValueError("Wrong evaluator, you need to fill wmape, mse ...")

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

        dist_ = {
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
            # autoTSF
            "is_forecast": True,
            "stacked_ensemble": False,
            "forecast_column": datetime,
            "forecast_exogeneous_columns": exogeneous_columns_list,
            "forecast_groupby_method": groupby_method,
            "forecast_gap": gap,
            "feature_derivation_start": 0,  # I think it outdated parameter
            "feature_derivation_window": feature_derivation_window,
            "forecast_horizon_start": gap,  # I think it outdated parameter
            "forecast_horizon_window": horizon_window,
            "forecast_time_group_columns": time_groups,
            "forecast_timeunit": timeunit,
            "validation_percentage": validation_percentage,
            "nfold": nfold,
        }

        exp_id = self.api.post_train_ts(dist_)

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))

        return experiment

    def predict_iid(
        self,
        model: Model,
        keep_columns: List[str],
        non_negative: bool,
        test_data_id: str,
    ) -> Prediction:

        data = {
            "project_id": self.project_id,
            "experiment_id": model.experiment_id,
            "model_id": model.model_id,
            "table_id": test_data_id,
            "is_multi_model": False,
            "non_negative": non_negative,
            "keep_columns": keep_columns,
        }

        pred_id = self.api.post_predict(data)

        prediction = Prediction(self.wait_for_response("prediction", pred_id))
        prediction.predict_df = self.api.get_pred_data(
            prediction.attributes.prediction_id
        )

        return prediction

    def predict_ts(
        self,
        model: Model,
        keep_columns: List[str],
        non_negative: bool,
        test_data_id: str,
    ) -> Prediction:

        data = {
            "project_id": self.project_id,
            "experiment_id": model.experiment_id,
            "model_id": model.model_id,
            "table_id": test_data_id,
            "is_multi_model": True,
            "non_negative": non_negative,
            "keep_columns": keep_columns,
        }

        print('dat:', data)

        pred_id = self.api.post_predict(data)

        prediction = Prediction(self.wait_for_response("prediction", pred_id))
        prediction.predict_df = self.api.get_pred_data(
            prediction.attributes.prediction_id
        )

        return prediction

    def wait_for_response(self, url, id):

        while self.api.check(check_url=url, id=id)["status"] != "done":
            res = self.api.check(check_url=url, id=id)

            if res["status"] == "fail":
                raise AttributeError(res["progress_message"])
            else:
                print(
                    "Progress: ",
                    int(float(res["progress"]) * 100),
                    "%\nProgress Message: ",
                    res["progress_message"],
                ) if res["status"] == "running" else print(
                    url, "task is now " + res["status"]
                )
            sleep(3)

        print(url, "Done!")

        return self.api.check(check_url=url, id=id)

    def show_table(data_id: str) -> pd.DataFrame:
        # return single data df
        pass

    def show_table_list(project_id: str) -> List[str]:
        # return list of tables
        pass
