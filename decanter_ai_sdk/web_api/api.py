import logging
from random import random
import json
from tabnanny import check
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# from typing import Dict, Any


logger = logging.getLogger(__name__)
requests.packages.urllib3.disable_warnings()

requests_session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])

requests_session.mount("http://", HTTPAdapter(max_retries=retries))
requests_session.mount("https://", HTTPAdapter(max_retries=retries))


class Api:
    def __init__(self, host, headers, project_id):
        # self.host = host
        self.url = "http://" + host + "/v1/"
        self.headers = headers
        self.project_id = project_id

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

    def post_upload(self, file, name):

        res_data = {"url": self.url, "_id": "upload_id"}
        data = {"project_id": self.project_id, "name": name}
        # print(self.url + "upload", data, self.headers, file)
        # return requests.post(self.url+"upload", files=file, data=data, headers=self.headers)
        pass
        # return self.requests_(http="POST", url= self.url + "upload", files = file, headers=self.headers, data = data)

    def post_train_iid(self, data):
        return { "_id":"train_id" }
        pass

    def post_predict_iid(self, data):
        return { "_id":"prediction_id" }
        pass

    # @staticmethod
    def check(self, check_url, id):
        # check with server and print process using fake data right now
        if(check_url=="table"):
            return {"_id": "returned_data_id", "status": "done"}
        if(check_url=="experiment"):
            return {
            "_id": "62d7bc5e30833d5b1bd36ee7",
            "algos": ["DRF", "GLM", "StackedEnsemble"],
            "attributes": {
                "Generalized Linear Model 1": {
                    "cv_averages": {
                        "auc": 0.9866558000000001,
                        "lift_top_group": 2.8763037999999996,
                        "logloss": 0.12999312,
                        "mean_per_class_error": 0.05677256,
                        "misclassification": 0.0440251585,
                    },
                    "cv_deviations": {
                        "auc": 0.00588660000000002,
                        "lift_top_group": 0.18138849999999995,
                        "logloss": 0.02253636,
                        "mean_per_class_error": 0.02482341,
                        "misclassification": 0.012578615500000001,
                    },
                    "feature_explanations": [],
                    "holdout_scores": {
                        "auc": 1.0,
                        "lift_top_group": 2.3529411764705883,
                        "logloss": 0.08566723308675815,
                        "mean_per_class_error": 0.0,
                        "misclassification": 0.0,
                    },
                    "model_id": "62d7bc63817e9251bbe70fe6",
                    "name": "Generalized Linear Model 1",
                    "validation_scores": {
                        "auc": 0.9974999999999999,
                        "lift_top_group": 2.0,
                        "logloss": 0.0732032089302445,
                        "mean_per_class_error": 0.025,
                        "misclassification": 0.025000000000000022,
                    },
                },
                "Random Forest 1": {
                    "cv_averages": {
                        "auc": 0.9807285,
                        "lift_top_group": 2.8763037999999996,
                        "logloss": 0.25653913500000003,
                        "mean_per_class_error": 0.051124877,
                        "misclassification": 0.047169812000000005,
                    },
                    "cv_deviations": {
                        "auc": 0.010203700000000038,
                        "lift_top_group": 0.18138849999999995,
                        "logloss": 0.110729635,
                        "mean_per_class_error": 0.010701147000000001,
                        "misclassification": 0.009433962,
                    },
                    "feature_explanations": [],
                    "holdout_scores": {
                        "auc": 0.9923273657289002,
                        "lift_top_group": 2.3529411764705883,
                        "logloss": 0.13205309928357106,
                        "mean_per_class_error": 0.043478260869565216,
                        "misclassification": 0.050000000000000044,
                    },
                    "model_id": "62d7bc71817e9251bbe70fe8",
                    "name": "Random Forest 1",
                    "validation_scores": {
                        "auc": 1.0,
                        "lift_top_group": 2.0,
                        "logloss": 0.10012234957144561,
                        "mean_per_class_error": 0.0,
                        "misclassification": 0.0,
                    },
                },
                "Random Forest 2": {
                    "cv_averages": {
                        "auc": 0.9817619200000001,
                        "lift_top_group": 2.8763037999999996,
                        "logloss": 0.253587275,
                        "mean_per_class_error": 0.055362165000000005,
                        "misclassification": 0.0503144655,
                    },
                    "cv_deviations": {
                        "auc": 0.00917028000000003,
                        "lift_top_group": 0.18138849999999995,
                        "logloss": 0.107155655,
                        "mean_per_class_error": 0.006463858999999999,
                        "misclassification": 0.0062893085,
                    },
                    "feature_explanations": [],
                    "holdout_scores": {
                        "auc": 0.9910485933503835,
                        "lift_top_group": 2.3529411764705883,
                        "logloss": 0.13470767844604475,
                        "mean_per_class_error": 0.05115089514066496,
                        "misclassification": 0.050000000000000044,
                    },
                    "model_id": "62d7bc70817e9251bbe70fe7",
                    "name": "Random Forest 2",
                    "validation_scores": {
                        "auc": 1.0,
                        "lift_top_group": 2.0,
                        "logloss": 0.10556964288643853,
                        "mean_per_class_error": 0.0,
                        "misclassification": 0.0,
                    },
                },
                "Stacked Ensemble 1": {
                    "cv_averages": {
                        "auc": 0.9856530349999999,
                        "lift_top_group": 2.8878688,
                        "logloss": 0.12722805,
                        "mean_per_class_error": 0.057015905,
                        "misclassification": 0.043803419999999996,
                    },
                    "cv_deviations": {
                        "auc": 0.0028228650000000077,
                        "lift_top_group": 0.23213119999999998,
                        "logloss": 0.016054870000000006,
                        "mean_per_class_error": 0.007015904999999996,
                        "misclassification": 0.011752136,
                    },
                    "feature_explanations": [],
                    "holdout_scores": {
                        "auc": 0.9948849104859335,
                        "lift_top_group": 2.3529411764705883,
                        "logloss": 0.10315118048404706,
                        "mean_per_class_error": 0.021739130434782608,
                        "misclassification": 0.025000000000000022,
                    },
                    "model_id": "62d7bc73817e9251bbe70fe9",
                    "name": "Stacked Ensemble 1",
                    "validation_scores": {
                        "auc": 1.0,
                        "lift_top_group": 2.0,
                        "logloss": 0.055118882016197766,
                        "mean_per_class_error": 0.0,
                        "misclassification": 0.0,
                    },
                },
                "Stacked Ensemble 2": {
                    "cv_averages": {
                        "auc": 0.986166315,
                        "lift_top_group": 2.8878688,
                        "logloss": 0.12748642500000001,
                        "mean_per_class_error": 0.054374395,
                        "misclassification": 0.043803419999999996,
                    },
                    "cv_deviations": {
                        "auc": 0.0029588150000000035,
                        "lift_top_group": 0.23213119999999998,
                        "logloss": 0.016494524999999996,
                        "mean_per_class_error": 0.009657414999999996,
                        "misclassification": 0.011752136,
                    },
                    "feature_explanations": [],
                    "holdout_scores": {
                        "auc": 0.9948849104859335,
                        "lift_top_group": 2.3529411764705883,
                        "logloss": 0.10534238333252433,
                        "mean_per_class_error": 0.021739130434782608,
                        "misclassification": 0.025000000000000022,
                    },
                    "model_id": "62d7bc76817e9251bbe70fea",
                    "name": "Stacked Ensemble 2",
                    "validation_scores": {
                        "auc": 0.9974999999999999,
                        "lift_top_group": 2.0,
                        "logloss": 0.0630490827022057,
                        "mean_per_class_error": 0.025,
                        "misclassification": 0.025000000000000022,
                    },
                },
            },
            "bagel_id": "M005",
            "balance_class": "False",
            "best_model": "Generalized Linear Model 1",
            "best_model_id": "62d7bc63363ac1ad75cc34d1",
            "best_score": 0.9866558000000001,
            "category": "classification",
            "company_id": "62d6e045e0c2efffdc84c39b",
            "completed_at": "2022-07-20T08:27:38.000000Z",
            "corex_models": [
                "62d7bc70817e9251bbe70fe7",
                "62d7bc71817e9251bbe70fe8",
                "62d7bc76817e9251bbe70fea",
                "62d7bc73817e9251bbe70fe9",
                "62d7bc63817e9251bbe70fe6",
            ],
            "created_at": "2022-07-20T08:27:10.589000Z",
            "created_by": {
                "_id": "62d7a20f03e0c11ee280b3e8",
                "active_project_id": "62d7a23003e0c11ee280b3ea",
                "company_id": "62d6e045e0c2efffdc84c39b",
                "created_at": "2022-07-20T06:34:55.898000Z",
                "is_active": "true",
                "is_removed": "False",
                "last_login": "2022-07-20T06:35:15.717000Z",
                "login_attempts": 0,
                "mfa_code": "",
                "mfa_email": "",
                "mfa_email_verify": "False",
                "need_new_pass": "False",
                "need_onboarding": "true",
                "need_pass_reset": "False",
                "role": 0,
                "unlocked_at": "None",
                "updated_at": "2022-07-20T06:35:15.723000Z",
                "username": "stest",
            },
            "data_id": "62d7bc60817e9251bbe70fe5",
            "error": {},
            "feature_types": [
                {"data_type": "numerical", "id": "ID"},
                {"data_type": "numerical", "id": "mean radius"},
                {"data_type": "numerical", "id": "mean texture"},
                {"data_type": "numerical", "id": "mean perimeter"},
                {"data_type": "numerical", "id": "mean area"},
                {"data_type": "numerical", "id": "mean smoothness"},
                {"data_type": "numerical", "id": "mean compactness"},
                {"data_type": "numerical", "id": "mean concavity"},
                {"data_type": "numerical", "id": "mean concave points"},
                {"data_type": "numerical", "id": "mean symmetry"},
                {"data_type": "numerical", "id": "mean fractal dimension"},
            ],
            "features": [
                "ID",
                "mean radius",
                "mean texture",
                "mean perimeter",
                "mean area",
                "mean smoothness",
                "mean compactness",
                "mean concavity",
                "mean concave points",
                "mean symmetry",
                "mean fractal dimension",
            ],
            "forecast_column": "None",
            "forecast_exogeneous_columns": [],
            "forecast_time_group_columns": [],
            "gp_table_id": "62d7bc2f30833d5b1bd36ee6",
            "holdout": {"percent": 10},
            "holdout_percentage": 0.1,
            "hyperparameters": {
                "algos": "StackedEnsemble,GLM,DRF",
                "apu_name": "None",
                "balance_class": "False",
                "data_rows": 398,
                "endogenous_features": [],
                "exogenous_features": [],
                "feature_types": [
                    {"data_type": "numerical", "id": "ID"},
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYradius",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYtexture",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYperimeter",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYarea",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYsmoothness",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYcompactness",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYconcavity",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYconcaveuFjLKCYXwlW2WAhXI6E0cz5CcYpoints",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYsymmetry",
                    },
                    {
                        "data_type": "numerical",
                        "id": "meanuFjLKCYXwlW2WAhXI6E0cz5CcYfractaluFjLKCYXwlW2WAhXI6E0cz5CcYdimension",
                    },
                    {"data_type": "categorical", "id": "Diagnosis"},
                ],
                "holdout_percentage": 0.1,
                "max_model": 3,
                "max_runtime": 259200,
                "model_type": "binary classification",
                "nfold": 2,
                "preprocessing": "None",
                "seed": 5924,
                "stopping_metric": "auc",
                "target": "Diagnosis",
                "time_group": [],
                "time_series_split": "None",
                "tolerance": 0.005,
                "transformed_features": [
                    "ID",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYradius",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYtexture",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYperimeter",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYarea",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYsmoothness",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYcompactness",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYconcavity",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYconcaveuFjLKCYXwlW2WAhXI6E0cz5CcYpoints",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYsymmetry",
                    "meanuFjLKCYXwlW2WAhXI6E0cz5CcYfractaluFjLKCYXwlW2WAhXI6E0cz5CcYdimension",
                ],
                "validation_percentage": 0.1,
            },
            "is_binary_classification": "true",
            "is_favorited": "False",
            "is_forecast": "False",
            "is_starred": "False",
            "max_model": 3,
            "name": "API_breast_cancer_example",
            "nfold": 2,
            "preprocessing": {},
            "progress": 1.0,
            "progress_message": "task completed",
            "project_id": "62d7a23003e0c11ee280b3ea",
            "recommendations": [
                {
                    "cv_average": 0.9866558000000001,
                    "cv_deviation": 0.00588660000000002,
                    "evaluator": "auc",
                    "model_id": "62d7bc63817e9251bbe70fe6",
                    "name": "Generalized Linear Model 1",
                },
                {
                    "cv_average": 0.051124877,
                    "cv_deviation": 0.010701147000000001,
                    "evaluator": "mean_per_class_error",
                    "model_id": "62d7bc71817e9251bbe70fe8",
                    "name": "Random Forest 1",
                },
                {
                    "cv_average": 0.12722805,
                    "cv_deviation": 0.016054870000000006,
                    "evaluator": "logloss",
                    "model_id": "62d7bc73817e9251bbe70fe9",
                    "name": "Stacked Ensemble 1",
                },
                {
                    "cv_average": 0.043803419999999996,
                    "cv_deviation": 0.011752136,
                    "evaluator": "misclassification",
                    "model_id": "62d7bc73817e9251bbe70fe9",
                    "name": "Stacked Ensemble 1",
                },
                {
                    "cv_average": 2.8878688,
                    "cv_deviation": 0.23213119999999998,
                    "evaluator": "lift_top_group",
                    "model_id": "62d7bc73817e9251bbe70fe9",
                    "name": "Stacked Ensemble 1",
                },
            ],
            "seed": 5924,
            "stacked_ensemble": "true",
            "started_at": "2022-07-20T08:27:10.000000Z",
            "status": "done",
            "stopping_metric": "auc",
            "target": "Diagnosis",
            "target_type": "categorical",
            "task_id": "62d7bc5ee708141485442fd5",
            "timeseriesValues": {},
            "tolerance": 0.005,
            "train_table": {
                "_id": "62d7bc2f30833d5b1bd36ee6",
                "celery_task_id": "1834f3b9-3d71-4039-b3b0-c8cd4ec49a3f",
                "columns": [
                    "62d7bc3e1e5975996af6c15c",
                    "62d7bc3e1e5975996af6c15d",
                    "62d7bc3e1e5975996af6c15e",
                    "62d7bc3e1e5975996af6c15f",
                    "62d7bc3e1e5975996af6c160",
                    "62d7bc3e1e5975996af6c161",
                    "62d7bc3e1e5975996af6c162",
                    "62d7bc3e1e5975996af6c163",
                    "62d7bc3e1e5975996af6c164",
                    "62d7bc3e1e5975996af6c165",
                    "62d7bc3e1e5975996af6c166",
                    "62d7bc3e1e5975996af6c167",
                ],
                "company_id": "62d6e045e0c2efffdc84c39b",
                "completed_at": "2022-07-20T08:26:38.000000Z",
                "created_at": "2022-07-20T08:26:23.343000Z",
                "data_id": "62d7bc2fa9960d32427c2796",
                "eda_id": "62d7bc2fa9960d32427c2796",
                "eda_status": "done",
                "error": "None",
                "file_size": 32305,
                "is_active": "true",
                "name": "breast_cancer_train_API",
                "progress": 1.0,
                "progress_message": "task completed",
                "project_id": "62d7a23003e0c11ee280b3ea",
                "rows": 398,
                "started_at": "2022-07-20T08:26:23.000000Z",
                "status": "done",
                "tag": "all",
                "task_id": "62d7bc2fe708141485442fd2",
                "updated_at": "2022-07-20T08:26:23.348000Z",
                "upload_uid": "yMEe2Qu01ZYu",
                "uploaded_by": "62d7a20f03e0c11ee280b3e8",
            },
            "updated_at": "2022-07-20T08:27:38.803000Z",
            "validation_percentage": 0.1,
        }
        if(check_url=="prediction"):
            return {
      "_id":"62df711d3c02dec8a049e0f2",
      "apu_mock_model":False,
      "company_id":"62dec9aed92d6a1c31cd04e7",
      "compared_to_what":"cv",
      "completed_at":"2022-07-26T04:44:45.000000Z",
      "created_at":"2022-07-26T04:44:13.768000Z",
      "data_id":"62df711fcdcc1337e8d7a825",
      "download_count":0,
      "error":{
         
      },
      "experiment_id":"62df70543465481f32deec2b",
      "is_auto_predict":False,
      "is_multi_model":False,
      "keep_columns":[
         "ID"
      ],
      "model_id":"62df706b0d2ebed0bc921bb3",
      "performance":{
         "cross_threshold":False,
         "drift":{
            "auc":2.07,
            "lift_top_group":-3.14,
            "logloss":-90.02,
            "mean_per_class_error":-269.0,
            "misclassification":-248.56
         },
         "drift_diff":{
            "auc":0.020588457824142647,
            "lift_top_group":-0.08542668852459068,
            "logloss":-0.09543853665021831,
            "mean_per_class_error":-0.07366007007600595,
            "misclassification":-0.06255310624561403
         },
         "metrics":{
            "auc":0.9749627421758573,
            "lift_top_group":2.8032786885245904,
            "logloss":0.2014578441502183,
            "mean_per_class_error":0.10104321907600596,
            "misclassification":0.08771929824561403
         }
      },
      "plot_key":"62df713dcdcc1337e8d7a82d",
      "progress":1.0,
      "progress_message":"task completed",
      "project_id":"62df5f26c64f9b760444d911",
      "shapley_only":False,
      "started_at":"2022-07-26T04:44:13.000000Z",
      "status":"done",
      "table_id":"62df7019d92d6a1c31cd0516",
      "task_id":"62df711da42f883c147314de",
      "updated_at":"2022-07-26T04:44:13.768000Z"
   }
