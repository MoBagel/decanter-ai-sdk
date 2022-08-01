# from locale import currency
from re import S
from decanter_ai_sdk.client import Client
import os
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd

def test_demo1():
    print("---From test demo1---")

    client = Client(
        auth_key="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNjJlNzJjNmQ5YzAyMWQ3MzRiN2RlMThlIiwiaXNzIjoiZ3AtYmUiLCJzdWIiOiJhcGlrZXkiLCJhdWQiOiJjbGllbnQiLCJpYXQiOjE2NTkzMTczNTcuODU4NDk2fQ.TokV7SC4PtRI8RLpYE6Z1Py_oFtutR8NA_76mBBluUw",
        project_id="62e72cf19c021d734b7de190",
        host="https://192.168.2.18",
    )

    current_path = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(current_path, "train.csv")
    train_file = open(train_file_path, "rb")
    # print("1", train_file)
    # t = train_file
    # keys = pd.read_csv(t).keys()
    # print(keys)
    # print("2", train_file)

    # table_id = client.upload(data=train_file, name="test_upload")
    # print("table", table_id)
    table_id = "62e76fdd75aba53da8c47042"

    # df = pd.read_csv(train_file)
    # print("keys", df.keys()[0])

    experiment = client.train_iid(
        experiment_name="Training Data",
        table_id=table_id,
        target="Survived",
        # evaluator="eva",
        features=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
        validation_percentage = 10,
        default_modes="dm",
    )

    # model = experiment.get_best_model_by_metric("auc")

    # prediction = client.predict_iid(
    #     model=model, keep_columns=["ID"], non_negative=False, test_data_id=data_id
    # )

    # print("pred_id", prediction.pred.prediction_id)
