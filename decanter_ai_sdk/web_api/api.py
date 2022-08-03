from io import StringIO
import json
import requests
import pandas as pd


class Api:
    def __init__(self, host, headers, upload_headers, project_id):
        self.url = host + "/v1/"
        self.headers = headers
        self.project_id = project_id
        self.upload_headers = upload_headers

    def post_upload(self, file: str, name: str):

        res = requests.post(
            f"{self.url}table/upload",
            files=file,
            data={"name": name, "project_id": self.project_id},
            headers=self.upload_headers,
            verify=False,
        )

        return res.json()["table"]["_id"]

    def post_train_iid(self, data):

        res = requests.post(
            f"{self.url}experiment/create",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        return res.json()["experiment"]["_id"]

    def post_train_ts(self, data):

        res = requests.post(
            f"{self.url}experiment/create",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )

        return res.json()["experiment"]["_id"]

    def post_predict_iid(self, data):

        res = requests.post(
            f"{self.url}prediction/predict",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        return res.json()["prediction"]["_id"]

    def get_table_info(self, table_id):
        table_response = requests.get(
            f"{self.url}table/{table_id}/columns", headers=self.headers, verify=False
        )
        table_info = {}
        for column in table_response.json()["columns"]:
            table_info[column["id"]] = column["data_type"]
        return table_info

    def check(self, check_url, id):
        if check_url == "table":

            res = requests.get(
                f"{self.url}table/{id}",
                verify=False,
                headers=self.headers,
            ).json()

            return res["table"]

        if check_url == "experiment":

            res = requests.get(
                f"{self.url}experiment/{id}", verify=False, headers=self.headers
            ).json()

            return res["experiment"]

        if check_url == "prediction":

            res = requests.get(
                f"{self.url}prediction/{id}", verify=False, headers=self.headers
            ).json()

            return res["data"]

    def get_pred_data(self, pred_id):
        data = {"prediction_id": pred_id}

        prediction_get_response = requests.get(
            f"{self.url}/prediction/{pred_id}/download",
            headers=self.headers,
            data=data,
            verify=False,
        )

        read_file = StringIO(prediction_get_response.text)
        prediction_df = pd.read_csv(read_file)
        return prediction_df
