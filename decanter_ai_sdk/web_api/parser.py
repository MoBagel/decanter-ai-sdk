from multiprocessing.connection import wait
import sys
from time import sleep
from unicodedata import category, name

from requests import head
from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from .api import Api
import json

import logging

logger = logging.getLogger(__name__)

class Parser:
    def __init__(self, host, headers, project_id):
        self.host = host
        self.headers = headers
        self.project_id = project_id
        self.api = Api(host=host, project_id=project_id, headers=headers)

    def wait_for_response(self, url, id):

        while (
            # wait for status == done
            json.loads(json.dumps(self.api.check(check_url=url, id=id)))["status"]
            != 1
        ):
            # t = json.loads(json.dumps(self.api.check(check_url=url, id=id)))
            # print(t['_id'], "hi")
            sleep(2)

        return json.loads(json.dumps(self.api.check(check_url=url, id=id)))["_id"]

    # @staticmethod
    def DataUpload(self, data, name):

        data_id = self.api.post_upload(file=data, name=name)
        print("test", data_id)
        res = self.wait_for_response("table", data_id)

        return res

    # @staticmethod
    def TrainIID(
        self,
        project_id,
        experiment_name,
        data_id,
        target,
        # category,
        evaluator,
        feature,
        default_mode,
        validation_percentage,
    ) -> Experiment:
        category = "category"
        data = {
            "project_id": project_id,
            "experiment_name": experiment_name,
            "data_id": data_id,
            "target": target,
            "category": category,
            "evaluator": evaluator,
            "feature": feature,
            "validation_percentage": validation_percentage,
            "default_mode": default_mode,
        }

        res = self.api.post_train_iid(data)
        # print(res['is_favorited'])
        # exp = Experiment(name = "name")
        # print(exp)
        for v in res['attributes']:
            # print(res['attributes'][v], "\n")
            logger.info(res["attributes"][v])
        experiment = Experiment(res)
        # print(experiment.get_best_model)
        # print("attr", experiment.attributes)
        # need another model list?
        # pass
        return experiment

    # @staticmethod
    def PredictIID(
        self, project_id, experiment_id, model_id, data_id, keep_columns, non_negative
    ) -> Prediction:

        data = {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "model_id": model_id,
            "data_id": data_id,
            "non_negative": non_negative,
            "keep_columns": keep_columns,
            "is_multi_model": False,
        }

        res = self.api.post_predict_iid(data)
        # while loop
        prediction = Prediction(
            # constuctor input
        )

        return prediction
