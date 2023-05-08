import json
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from requests_toolbelt import MultipartEncoder

from decanter_ai_sdk.web_api.api import ApiClient


class DecanterApiClient(ApiClient):
    def __init__(self, host, headers, auth_headers, project_id):  # pragma: no cover
        self.url = host + "/v1/"
        self.headers = headers
        self.project_id = project_id
        self.auth_headers = auth_headers
        self.session = requests.Session()

        # Retry when having temporary connection issue
        # ref: https://stackoverflow.com/a/35504626
        retries = urllib3.Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS"],
        )

        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def post_upload(self, file: tuple, name: str):  # pragma: no cover
        m = MultipartEncoder(
            fields={"project_id": self.project_id, "name": name, "file": file}
        )
        headers = self.auth_headers
        headers["Content-Type"] = m.content_type

        res = self.session.post(
            f"{self.url}table/upload",
            data=m,
            headers=headers,
            verify=False,
        )

        if not res.ok:
            raise RuntimeError(res.json()["message"])
        return res.json()["table"]["_id"]

    def post_train_iid(self, data) -> str:  # pragma: no cover
        res = self.session.post(
            f"{self.url}experiment/create",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )

        if not res.ok:
            raise RuntimeError(res.json()["message"])
        return res.json()["experiment"]["_id"]

    def post_train_ts(self, data) -> str:  # pragma: no cover
        res = self.session.post(
            f"{self.url}experiment/create",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        if not res.ok:
            raise RuntimeError(res.json()["message"])
        return res.json()["experiment"]["_id"]

    def post_predict(self, data) -> str:  # pragma: no cover
        res = self.session.post(
            f"{self.url}prediction/predict",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        if not res.ok:
            raise RuntimeError(res.json()["message"])

        return res.json()["prediction"]["_id"]

    def batch_predict(
        self,
        pred_df: pd.DataFrame,
        experiment_id: str,
        model_id: str,
        timestamp_format: str,
    ) -> pd.Series:
        data = {
            "project_id": self.project_id,
            "featuresList": pred_df.to_dict(orient="records"),
            "timestamp_format": timestamp_format,
        }

        res = self.session.post(
            f"{self.url}experiment/{experiment_id}/model/{model_id}/batch_predict",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )

        if not res.ok:
            raise RuntimeError(res.json()["message"])

        return pd.DataFrame(res.json()["data"])["prediction"]

    def get_table_info(self, table_id):  # pragma: no cover
        table_response = self.session.get(
            f"{self.url}table/{table_id}/columns", headers=self.headers, verify=False
        )
        table_info = {}

        if not table_response.ok:
            raise RuntimeError(table_response.json()["message"])

        for column in table_response.json()["columns"]:
            table_info[column["id"]] = column["data_type"]
        return table_info

    def update_table(self, table_id, updated_column, updated_type) -> None:
        data = {
            "project_id": self.project_id,
            "columns": [{"data_type": updated_type, "id": updated_column}],
            "table_id": table_id,
        }

        update_response = self.session.put(
            f"{self.url}table/update",
            headers=self.headers,
            data=json.dumps(data),
            verify=False,
        )
        if update_response.status_code == 200:
            print("Update Successfully!!")
        else:
            print(f"Update Fail - {update_response.status_code}")
            print(
                "typo? Only categorical, numerical, datetime, and id can be filled in"
            )

    def check(self, task, id):  # pragma: no cover
        if task == "table":
            res = self.session.get(
                f"{self.url}table/{id}",
                verify=False,
                headers=self.headers,
            ).json()
            return res["table"]

        if task == "experiment":
            res = self.session.get(
                f"{self.url}experiment/{id}", verify=False, headers=self.headers
            ).json()
            return res["experiment"]

        if task == "prediction":
            res = self.session.get(
                f"{self.url}prediction/{id}", verify=False, headers=self.headers
            ).json()
            return res["data"]

    def get_experiment_list(self, page):
        experiment = {}
        experiment_response = self.session.get(
            f"{self.url}experiment/getlist/{self.project_id}",
            headers=self.headers,
            params={"page": page},
            verify=False,
        )
        for experiment_ in experiment_response.json()["experiments"]:
            experiment_name = experiment_["name"]
            create_time = pd.to_datetime(experiment_["started_at"][:19]) + pd.Timedelta(
                8, unit="h"
            )
            experiment[experiment_["_id"]] = {
                "ExperimentName": experiment_name,
                "StartTime": create_time.strftime("%Y-%m-%d %H:%M"),
            }
        return experiment

    def get_prediction_list(self, model_id):
        res = self.session.get(
            f"{self.url}prediction/getlist/{model_id}",
            verify=False,
            params={"project_id": self.project_id},
            headers=self.headers,
        )

        return res.json()["predictions"]

    def get_pred_data(self, pred_id, download):  # pragma: no cover
        prediction_get_response = self.session.get(
            f"{self.url}prediction/{pred_id}/download",
            headers=self.headers,
            params={"download": download},
            verify=False,
        )

        read_file = StringIO(prediction_get_response.text)
        prediction_df = pd.read_csv(read_file).reset_index()

        return prediction_df

    def get_table_list(self, page):  # pragma: no cover
        table_list_res = self.session.get(
            f"{self.url}table/getlist/{self.project_id}",
            headers=self.headers,
            verify=False,
            params={"page": page},
        )
        return table_list_res.json()["tables"]

    def get_table(self, data_id):  # pragma: no cover
        table_res = self.session.get(
            f"{self.url}table/{data_id}/csv",
            headers=self.headers,
            verify=False,
        )

        read_file = StringIO(table_res.text)
        table_df = pd.read_csv(read_file)

        return table_df

    def get_model_list(self, experiment_id):  # pragma: no cover
        res = self.session.get(
            f"{self.url}experiment/{experiment_id}/model/getlist?projectId={self.project_id}",
            headers=self.auth_headers,
            verify=False,
        )
        return res.json()["model_list"]

    def get_model_threshold(self, experiment_id, model_id) -> float:
        res = self.session.get(
            f"{self.url}experiment/{experiment_id}/model/{model_id}/predict_threshold",
            headers=self.headers,
            params={"project_id": self.project_id},
            verify=False,
        )
        return res.json()["threshold"]

    def get_performance_metrics(self, model_id, table_id) -> List:
        res = self.session.get(
            f"{self.url}prediction/getlist/{model_id}",
            verify=False,
            params={"project_id": self.project_id},
            headers=self.headers,
        )
        res_pred = res.json()["predictions"]
        perf_list = [
            {
                "metrics": res_pred[x]["performance"]["metrics"],
                "threshold": res_pred[x]["threshold"]
                if "threshold" in list(res_pred[0].keys())
                else np.nan,
            }
            for x in range(len(res_pred))
            if res_pred[x]["table_id"] == table_id
        ]
        return perf_list

    def stop_uploading(self, id) -> bool:  # pragma: no cover
        res = self.session.post(
            f"{self.url}table/stop",
            headers=self.auth_headers,
            verify=False,
            data={"table_id": id, "project_id": self.project_id},
        )
        return res.ok

    def stop_training(self, id) -> bool:  # pragma: no cover
        res = self.session.post(
            f"{self.url}experiment/stop",
            headers=self.auth_headers,
            verify=False,
            data={"experiment_id": id, "project_id": self.project_id},
        )
        return res.ok

    def delete_experiment(self, experiment_id) -> str:
        res = self.session.post(
            f"{self.url}experiment/delete",
            headers=self.headers,
            data=json.dumps(
                {"project_id": self.project_id, "experiment_id": experiment_id}
            ),
            verify=False,
        )

        return res.json()["message"]
