import sys
sys.path.append("..")
from decanter_ai_sdk.enums.time_units import TimeUnit
from decanter_ai_sdk.client import Client
import os
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.data_types import DataType


def test_iid():
    print("---From test iid---")

    client = Client(
        auth_key="auth_API_key", 
        project_id="project_id", 
        host="host_url",
        test=True)

    current_path = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(current_path, "ts_train.csv")
    train_file = open(train_file_path, "rb")
    train_id = client.upload(train_file, "train_file")

    test_file_path = os.path.join(current_path, "ts_test.csv")
    test_file = open(test_file_path, "rb")
    test_id = client.upload(test_file, "test_file")

    client.get_table_list()

    experiment = client.train_ts(
    experiment_name="exp_name", 
    experiment_table_id=train_id, 
    target="Passengers",
    datetime="Month",
    time_groups=[],
    timeunit=TimeUnit.month,
    groupby_method="sum",
    max_model=5,
    evaluator=RegressionMetric.MAPE,
    custom_feature_types={"Pclass" : DataType.numerical}
    )

    best_model = experiment.get_best_model()
    
    predict = client.predict_ts(
        keep_columns=[],
        non_negative= False,
        test_table_id= test_id,
        model_id = "6303208ec4f10cea106ed838",
        experiment_id = "630317b7818547e247f5aa2e"
    )

    predict.get_predict_df()
    
