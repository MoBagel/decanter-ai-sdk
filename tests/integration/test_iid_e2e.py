import os
import pandas as pd
from decanter_ai_sdk.enums.algorithms import IIDAlgorithms
from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.data_types import DataType
from typing import List
from setup import *


def test_train_and_predict_titanic(client):
    print("---From test iid---")
    current_path = os.path.dirname(os.path.abspath(__file__))

    train_file_path = "data/train.csv"
    train_file = open(train_file_path, "rb")
    train_id = client.upload(train_file, "train_file")
    assert train_id is not None

    test_file_path = "data/test.csv"
    test_file = open(test_file_path, "rb")
    test_id = client.upload(test_file, "test_file")
    assert test_id is not None
    assert isinstance(client.get_table_list(), List)

    exp_name = "exp_test_iid"
    experiment = client.train_iid(
        experiment_name=exp_name,
        experiment_table_id=train_id,
        target="Survived",
        evaluator=ClassificationMetric.AUC,
        custom_column_types={
            "Pclass": DataType.categorical,
            "Parch": DataType.categorical,
        },
        algos=["GLM", "GBM", IIDAlgorithms.DRF, IIDAlgorithms.XGBoost],
        max_model=2,
    )

    assert experiment.status == "done"
    assert isinstance(experiment.get_model_list(), List)

    best_model = experiment.get_best_model()

    predict = client.predict_iid(
        keep_columns=[],
        non_negative=False,
        test_table_id=test_id,
        model=best_model,
        threshold=0.5,
    )

    assert predict.attributes["status"] == "done"
    assert isinstance(predict.get_predict_df(), pd.DataFrame)
    assert predict.attributes["model_id"] == best_model.model_id
    assert predict.attributes["table_id"] == test_id
