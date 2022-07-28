import enum
from io import StringIO
from time import sleep
from typing import List

import pandas as pd

from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.web_api.api import Api

import json


class Client:
    def __init__(self, auth_key, project_id, host):
        self.auth_key = auth_key
        self.project_id = project_id
        self.host = host
        self.api = Api(
            host=host,
            headers={"Authorization": "Bearer " + auth_key},
            project_id=project_id,
        )

    def upload(self, data, name: str) -> str:
        if isinstance(data, pd.DataFrame):
            textStream = StringIO
            data.to_csv(textStream, index=False)
            file = [(textStream.getvalue(), "text/csv")]
        else:
            file = [(data, "text/csv")]

        data_id = self.api.post_upload(file=file, name=name)

        res = self.wait_for_response("table", data_id)

        return res["_id"]

    def train_iid(
        self,
        experiment_name: str = None,
        data_id: str = None,
        target: str = None,
        evaluator: str = "auc",
        features: List[str] = None,
        validation_percentage: int = 10,
        default_modes: str = "balance",
    ) -> Experiment:

        category = "category"

        data = {
            "project_id": self.project_id,
            "experiment_name": experiment_name,
            "data_id": data_id,
            "target": target,
            "category": category,
            "evaluator": evaluator,
            "features": features,
            "validation_percentage": validation_percentage,
            "default_mode": default_modes,
        }

        exp_id = self.api.post_train_iid(data)["_id"]
        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))
        print("exp_id", experiment.get_id())

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

        pred_id = self.api.post_predict_iid(data=data)["_id"]
        print('pid',pred_id)

        prediction = Prediction(self.wait_for_response("prediction", pred_id))
        return prediction

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
        return prediction

    def wait_for_response(self, url, id):
        print("url", url)

        while (
            json.loads(json.dumps(self.api.check(check_url=url, id=id)))["status"]
            != "done"
        ):
            sleep(2)
        return self.api.check(check_url=url, id=id)

    def show_table(data_id: str) -> pd.DataFrame:
        # return single data df
        pass

    def show_table_list(project_id: str) -> List[str]:
        # return list of tables
        pass

