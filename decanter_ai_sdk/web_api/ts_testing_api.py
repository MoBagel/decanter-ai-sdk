from typing import Dict
import json
import pandas as pd
from decanter_ai_sdk.web_api.api import ApiClient
import sys
import os

sys.path.append("..")

current_path = os.path.dirname(os.path.abspath(__file__))

# The following code is generated for local unit testing simulation.
class TestingTsApiClient(ApiClient):
    def __init__(self):
        self.url = None
        self.headers = None
        self.project_id = None
        self.auth_headers = None

    def post_upload(self, file: Dict, name: str):
        if name == "ts_train_file":
            return "63044594818547e247f5aa44"
        else:
            return "6304459bf52233e377e53a41"

    def post_train_iid(self, data):
        return "6304459bf52233e377e53a41"

    def post_train_ts(self, data):
        return "63044594818547e247f5aa44"

    def post_predict(self, data):
        return "6302f53cf52233e377e53a37"

    def get_table_info(self, table_id):
        f = open(current_path + "/data/table_info.json")
        table_info = json.load(f)
        return table_info

    def check(self, task, id):
        if task == "table":
            # Decide which json file (train or test) should be returned by checking id.
            if id == "63044594818547e247f5aa44":
                f = open(current_path + "/data/ts_train_table.json")
                table_data = json.load(f)
            elif id == "6304459bf52233e377e53a41":
                f = open(current_path + "/data/ts_test_table.json")
                table_data = json.load(f)

            return table_data

        if task == "experiment":
            f = open(current_path + "/data/ts_exp.json")
            return json.load(f)

        if task == "prediction":
            f = open(current_path + "/data/ts_predict.json")
            return json.load(f)

    def get_pred_data(self, pred_id, data):
        data = {"Name": ["Tom", "nick", "krish", "jack"], "Age": [20, 21, 19, 18]}
        return pd.DataFrame(data)

    def get_table_list(self, page=1):
        f = open(current_path + "/data/table_list.json")
        data_list = json.load(f)
        return data_list

    def get_table(self, data_id):
        table_data = {"name": ["Tom", "nick", "krish", "jack"], "Age": [20, 21, 19, 18]}
        return pd.DataFrame(table_data)

    def get_model_list(self, experiment_id):
        f = open(current_path + "/data/model_list.json")
        model_list_data = json.load(f)
        return model_list_data

    def stop_uploading(self, id):
        if id == "":
            return False
        return True

    def stop_training(self, id):
        if id == "":
            return False
        return True
