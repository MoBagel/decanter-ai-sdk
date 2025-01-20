import pandas as pd
from setup import *

from decanter_ai_sdk.enums.algorithms import TSAlgorithms
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.time_units import TimeUnit


def test_ts(client):
    print("---From test ts---")
    train_file_path = "data/ts_train.csv"
    train_df = pd.read_csv(open(train_file_path, "rb"))
    train_id = client.upload(train_df, "train_file")
    assert train_id is not None

    test_file_path = "data/ts_test.csv"
    test_file = open(test_file_path, "rb")
    test_id = client.upload(test_file, "test_file")
    assert test_id is not None

    exp_name = "exp_name_ts"
    experiment = client.train_ts(
        experiment_name=exp_name,
        experiment_table_id=train_id,
        target="Passengers",
        datetime="Month",
        time_groups=[],
        timeunit=TimeUnit.month,
        groupby_method="sum",
        evaluator=RegressionMetric.WMAPE,
        algos=["ets", TSAlgorithms.theta],
    )
    assert experiment.status == "done"
    assert len(experiment.get_model_list()) == 2

    best_model = experiment.get_best_model()

    # predict on best model
    predict = client.predict_ts(
        keep_columns=[], non_negative=False, test_table_id=test_id, model=best_model
    )
    
    first_model_id = client.get_model_list(experiment.id)[0]['_id']
    predictions = client.batch_predict(
        pred_df=train_df,
        experiment_id=experiment.id,
        model_id=first_model_id)

    assert predict.attributes["status"] == "done"
    assert isinstance(predict.get_predict_df(), pd.DataFrame)
    assert isinstance(predictions, pd.Series)
    assert predict.attributes["model_id"] == best_model.model_id
    assert predict.attributes["table_id"] == test_id