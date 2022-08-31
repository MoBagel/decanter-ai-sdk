import os
import pandas as pd
from decanter_ai_sdk.enums.time_units import TimeUnit
from decanter_ai_sdk.client import Client
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.data_types import DataType
from typing import List


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
    train_file_df = pd.read_csv(open(train_file_path, "rb"))
    train_id = client.upload(train_file_df, "train_file")

    test_file_path = os.path.join(current_path, "ts_test.csv")
    test_file = open(test_file_path, "rb")
    test_id = client.upload(test_file, "test_file")

    assert client.get_table(train_id)["name"][0] == "Tom"
    assert isinstance(client.get_table_list(), List)
    assert client.get_table_list().__len__() == 2
    assert client.get_table_list()[0]["name"] == "train_file"
    assert client.get_table_list()[1]["name"] == "test_file"

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

    for metric in RegressionMetric:
        assert (
            experiment.get_best_model_by_metric(metric).model_id
            == "63044b72ed266c3d7b2f895f"
        )
    assert isinstance(experiment.get_model_list(), List)
    assert experiment.get_model_list().__len__() == 4
    for model in experiment.get_model_list():
        assert model.experiment_id == "63044b583a6eef99be6e8e9b"
    assert experiment.experiment_info()["name"] == exp_name

    predict = client.predict_ts(
        keep_columns=[], non_negative=False, test_table_id=test_id, model=best_model
    )

    assert isinstance(predict.get_predict_df(), pd.DataFrame)
    assert predict.attributes["model_id"] == best_model.model_id
    assert predict.attributes["table_id"] == test_id
