from typing import Dict
from io import StringIO
import json
import requests
import pandas as pd
from decanter_ai_sdk.web_api.api import Api
import sys
import os

sys.path.append("..")

current_path = os.path.dirname(os.path.abspath(__file__))


class TestingApi(Api):
    def __init__(self):
        self.url = None
        self.headers = None
        self.project_id = None
        self.auth_headers = None

    def post_upload(self, file: Dict, name: str):
        return "62ff59883a6eef99be6e8e86"

    def post_train_iid(self, data):
        return "6302f089818547e247f5aa26"

    def post_train_ts(self, data):
        return "6302f089818547e247f5aa26"

    def post_predict(self, data):
        return "6302f53cf52233e377e53a37"

    def get_table_info(self, table_id):
        f = open(current_path + "/data/table_info.json")
        table_info = json.load(f)
        return table_info

    def check(self, task, id):
        if task == "table":
            f = open(current_path + "/data/table.json")
            table_data = json.load(f)
            return table_data

        if task == "experiment":
            f = open(current_path + "/data/experiment.json")
            experiment_data = json.load(f)
            return experiment_data

        if task == "prediction":
            f = open(current_path + "/data/predict.json")
            pred_data = json.load(f)
            return pred_data

    def get_pred_data(self, pred_id, data):
        data = {"Name": ["Tom", "nick", "krish", "jack"], "Age": [20, 21, 19, 18]}
        return pd.DataFrame(data)

    def get_table_list(self):
        f = open(current_path + "/data/table_list.json")
        data_list = json.load(f)
        return data_list

    def get_table(self, data_id):
        table_data = {"Name": ["Tom", "nick", "krish", "jack"], "Age": [20, 21, 19, 18]}
        return pd.DataFrame(table_data)

    def get_model_list(self, experiment_id, query):
        f = open(current_path + "/data/model_list.json")
        model_list_data = json.load(f)
        return model_list_data
