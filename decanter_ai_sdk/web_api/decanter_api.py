from typing import Dict
from io import StringIO
import json
import requests
import pandas as pd

from decanter_ai_sdk.web_api.api import ApiClient


class DecanterApiClient(ApiClient):
    def __init__(self, host, headers, auth_headers, project_id):  # pragma: no cover
        self.url = host + "/v1/"
        self.headers = headers
        self.project_id = project_id
        self.auth_headers = auth_headers

    def post_upload(self, file: Dict, name: str):  # pragma: no cover

        res = requests.post(
            f"{self.url}table/upload",
            files=file,
            data={"name": name, "project_id": self.project_id},
            headers=self.auth_headers,
            verify=False,
        )

        if not res.ok:
            raise RuntimeError(res.json()["message"])
        return res.json()["table"]["_id"]

    def post_train_iid(self, data) -> str:  # pragma: no cover

        res = requests.post(
            f"{self.url}experiment/create",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        if not res.ok:
            raise RuntimeError(res.json()["message"])
        return res.json()["experiment"]["_id"]

    def post_train_ts(self, data) -> str:  # pragma: no cover

        res = requests.post(
            f"{self.url}experiment/create",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        if not res.ok:
            raise RuntimeError(res.json()["message"])
        return res.json()["experiment"]["_id"]

    def post_predict(self, data) -> str:  # pragma: no cover

        res = requests.post(
            f"{self.url}prediction/predict",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        if not res.ok:
            raise RuntimeError(res.json()["message"])

        return res.json()["prediction"]["_id"]

    def get_table_info(self, table_id):  # pragma: no cover

        table_response = requests.get(
            f"{self.url}table/{table_id}/columns", headers=self.headers, verify=False
        )
        table_info = {}

        if not table_response.ok:
            raise RuntimeError(table_response.json()["message"])

        for column in table_response.json()["columns"]:
            table_info[column["id"]] = column["data_type"]
        return table_info

    def check(self, task, id):  # pragma: no cover
        if task == "table":

            res = requests.get(
                f"{self.url}table/{id}",
                verify=False,
                headers=self.headers,
            ).json()
            return res["table"]

        if task == "experiment":

            res = requests.get(
                f"{self.url}experiment/{id}", verify=False, headers=self.headers
            ).json()
            return res["experiment"]

        if task == "prediction":

            res = requests.get(
                f"{self.url}prediction/{id}", verify=False, headers=self.headers
            ).json()
            return res["data"]

    def get_pred_data(self, pred_id, data):  # pragma: no cover

        prediction_get_response = requests.get(
            f"{self.url}/prediction/{pred_id}/download",
            headers=self.headers,
            data=data,
            verify=False,
        )

        read_file = StringIO(prediction_get_response.text)
        prediction_df = pd.read_csv(read_file)

        return prediction_df

    def get_table_list(self):  # pragma: no cover

        table_list_res = requests.get(
            f"{self.url}/table/getlist/{self.project_id}",
            headers=self.headers,
            verify=False,
        )
        return table_list_res.json()["tables"]

    def get_table(self, data_id):  # pragma: no cover

        table_res = requests.get(
            f"{self.url}table/{data_id}/csv",
            headers=self.headers,
            verify=False,
        )

        read_file = StringIO(table_res.text)
        table_df = pd.read_csv(read_file)

        return table_df

    def get_model_list(self, experiment_id, query):  # pragma: no cover

        res = requests.get(
            f"{self.url}experiment/{experiment_id}/model/getlist?projectId={self.project_id}",
            headers=self.auth_headers,
            verify=False,
        )
        return res.json()["model_list"]

    def stop_uploading(self, id) -> bool:  # pragma: no cover
        res = requests.post(
            f"{self.url}table/stop",
            headers=self.auth_headers,
            verify=False,
            data={"table_id": id, "project_id": self.project_id},
        )
        return res.ok

    def stop_training(self, id) -> bool:  # pragma: no cover
        res = requests.post(
            f"{self.url}experiment/stop",
            headers=self.auth_headers,
            verify=False,
            data={"experiment_id": id, "project_id": self.project_id},
        )
        return res.ok
