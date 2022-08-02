from ctypes import Union
import enum
from io import StringIO
from tabnanny import check
from time import sleep
from typing import List, Union

import pandas as pd

from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.web_api.api import Api

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
            print("file is None")
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
        experiment_name: str = None,
        table_id: str = None,
        target: str = None,
        evaluator: str = None,
        features: List[str] = None,
        validation_percentage: int = None,
        default_modes: str = None,
    ) -> Experiment:

# maybe I should wrap these codes up in another python? I'm not sure if these codes look too messy.
        data_columnInfo = self.api.get_table_info(table_id=table_id)
        feature_list = [
            feature
            for feature in data_columnInfo.keys()
            if feature not in [] + [target]
        ]
        feature_dict = {key: data_columnInfo[key] for key in feature_list}
        feature_dict_list = [{"id": k, "data_type": j} for k, j in feature_dict.items()]

        if data_columnInfo[target] == 'numerical':
            category = 'regression'
            if evaluator is None:
                evaluator = 'wmape'
            elif evaluator in ['auc', 'logloss', 'mean_per_class_error', 'misclassification', 'lift_top_group']:
                raise ValueError(
                    'Wrong evaluator, you need to fill wmape, mse ...')
        else:
            category = 'classification'
            if evaluator is None:
                evaluator = 'auc'
            elif evaluator in ['wmape', 'r2', 'mse', 'rmse', 'deviance', 'mae', 'rmsle', 'mape']:
                raise ValueError(
                    'Wrong evaluator, you need to fill auc, logloss...')
#----------------------------------------------------------------------------------------------------------------

        dict_ = {
            "project_id": self.project_id,
            "name": experiment_name,
            "gp_table_id": table_id,
            "seed": 5924,
            "target": target,
            "targetType": data_columnInfo[target],
            "features": feature_list,
            "feature_types": feature_dict_list,
            "category": category,
            "stopping_metric": "auc",
            "is_binary_classification": True,
            "holdout": {"percent": 10},
            "tolerance": 7,
            "nfold": 5,
            "max_model": 5,
            "algos": ["DRF", "GLM"],
            "stacked_ensemble": False,
            "validation_percentage": 10,
            "timeseriesValues": [],
        }

        exp_id = self.api.post_train_iid(dict_)

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))

        return experiment

    def train_ts():
        pass

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

        pred_id = self.api.post_predict_iid(data)

        prediction = Prediction(self.wait_for_response("prediction", pred_id))
        prediction.predict_df = self.api.get_pred_data(prediction.pred.prediction_id)
        return prediction.predicd_df

    def predict_ts(
        self,
        model: Model,
        keep_columns: List[str],
        non_negative: bool,
        test_data_id: str,
    ) -> Prediction:

        pred_id = self.api.post_predict_iid(
            self.project_id,
            model.experiment_id,
            model.model_id,
            test_data_id,
            keep_columns,
            non_negative,
            is_multi_model=True,
        )

        prediction = Prediction(self.wait_for_response("prediction", pred_id))
        prediction.predict_df = self.api.get_pred_data(prediction.Attributes.prediction_id)

        return prediction

    def wait_for_response(self, url, id):

        while self.api.check(check_url=url, id=id)["status"] != "done":
            res = self.api.check(check_url=url, id=id)
            print(
                "Progress: ",
                int(float(res["progress"]) * 100),
                "/100\nProgress Message: ",
                res["progress_message"],
            ) if res["status"] != "pending" else print("task is now pending.")
            sleep(2)

        print(url, "Done!")
        return self.api.check(check_url=url, id=id)

    def show_table(data_id: str) -> pd.DataFrame:
        # return single data df
        pass

    def show_table_list(project_id: str) -> List[str]:
        # return list of tables
        pass
