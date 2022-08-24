import sys
from typing import Dict, List

sys.path.append("..")
from decanter_ai_sdk.enums.time_units import TimeUnit
from decanter_ai_sdk.client import Client
import os
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.data_types import DataType
import pandas as pd


def test_ts():
    print("---From test ts---")

    client = Client(
        auth_key="auth_API_key",
        project_id="project_id",
        host="host_url",
        dry_run_type="ts",
    )

    current_path = os.path.dirname(os.path.abspath(__file__))

    train_file_path = os.path.join(current_path, "ts_train.csv")
    train_file = open(train_file_path, "rb")
    train_id = client.upload(train_file, "ts_train_file")

    test_file_path = os.path.join(current_path, "ts_test.csv")
    test_file = open(test_file_path, "rb")
    test_id = client.upload(test_file, "ts_test_file")

    assert isinstance(client.get_table_list(), List)

    exp_name = "exp_name"
    experiment = client.train_ts(
        experiment_name=exp_name,
        experiment_table_id=train_id,
        target="Passengers",
        datetime="Month",
        time_groups=[],
        timeunit=TimeUnit.month,
        groupby_method="sum",
        max_model=5,
        evaluator=RegressionMetric.MAPE,
        custom_feature_types={"Pclass": DataType.numerical},
    )

    best_model = experiment.get_best_model()
    assert isinstance(experiment.get_model_list(), List)
    assert experiment.experiment_info()["name"] == exp_name

    for metric in RegressionMetric:
        assert (
            experiment.get_best_model_by_metric(metric).model_id
            == "63044b72ed266c3d7b2f895f"
        )

    predict = client.predict_ts(
        keep_columns=[], non_negative=False, test_table_id=test_id, model=best_model
    )

    assert isinstance(predict.get_predict_df(), pd.DataFrame)