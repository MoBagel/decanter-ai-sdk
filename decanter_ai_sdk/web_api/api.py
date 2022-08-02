from io import StringIO
import logging
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from time import sleep
import pandas as pd

logger = logging.getLogger(__name__)
requests.packages.urllib3.disable_warnings()

requests_session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])

requests_session.mount("http://", HTTPAdapter(max_retries=retries))
requests_session.mount("https://", HTTPAdapter(max_retries=retries))


class Api:
    def __init__(self, host, headers, upload_headers, project_id):
        self.url = host + "/v1/"
        self.headers = headers
        self.project_id = project_id
        self.upload_headers = upload_headers

    @staticmethod
    def requests_(http, url, json=None, data=None, files=None, headers=None):
        try:
            if http == "GET":
                return requests_session.get(
                    url=url,
                    verify=False,
                )
            if http == "POST":
                return requests_session.post(
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    verify=False,
                    headers=headers,
                )
            if http == "PUT":
                return requests_session.put(
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    verify=False,
                    headers=headers,
                )
            if http == "DELETE":
                return requests_session.delete(
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    verify=False,
                )

            raise Exception("No such HTTP Method.")

        except requests.exceptions.RequestException as err:
            logger.error("Request Failed :(")
            raise Exception(err)

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
        url = self.url + "prediction/" + pred_id + "/download"
        data = {"prediction_id": pred_id}
        status_code = None

        while status_code != 200:
            prediction_get_response = requests.get(
                f"{self.url}/prediction/{pred_id}/download",
                headers=self.headers,
                data=data,
                verify=False,
            )
            status_code = prediction_get_response.status_code
            sleep(2)

        read_file = StringIO(prediction_get_response.text)
        prediction_df = pd.read_csv(read_file)
        return prediction_df
