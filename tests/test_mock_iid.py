from cmath import exp
import sys

sys.path.append("..")
from decanter_ai_sdk.client import Client
import os
from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.data_types import DataType
import pandas as pd
from typing import List


def test_iid():
    print("---From test iid---")

    client = Client(
        auth_key="auth_API_key",
        project_id="project_id",
        host="host_url",
        dry_run_type="iid",
    )

    current_path = os.path.dirname(os.path.abspath(__file__))

    train_file_path = os.path.join(current_path, "train.csv")
    train_file = open(train_file_path, "rb")
    train_id = client.upload(train_file, "test_file")

    test_file_path = os.path.join(current_path, "test.csv")
    test_file = open(test_file_path, "rb")
    test_id = client.upload(test_file, "test_file")

    exp_name = "exp_name"
    experiment = client.train_iid(
        experiment_name=exp_name,
        experiment_table_id=train_id,
        target="Survived",
        evaluator=ClassificationMetric.AUC,
        custom_feature_types={
            "Pclass": DataType.categorical,
            "Parch": DataType.categorical,
        },
    )

    best_model = experiment.get_best_model()

    predict = client.predict_iid(
        keep_columns=[], non_negative=False, test_table_id=test_id, model=best_model
    )
