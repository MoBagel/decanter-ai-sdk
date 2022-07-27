# from locale import currency
from decanter_ai_sdk.client import Client
import os
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd

# class a(BaseModel):
#     a : int
#     b : Dict[str, int]

# class Model(BaseModel):
#     model_id: str = None
#     model_name: str = None
#     model_data: Any = None
#     experiment_id: str = None
#     experiment_name: str = None
#     attributes: Dict[str, Any] = None

def test_demo1():
   print("---From test demo1---")

   client = Client(
      auth_key="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNjJkZjVmMGRkOTJkNmExYzMxY2QwNTE0IiwiaXNzIjoiZ3AtYmUiLCJzdWIiOiJhcGlrZXkiLCJhdWQiOiJjbGllbnQiLCJpYXQiOjE2NTg4MDYwMjkuMjEyMDg2fQ.C-fo1_69ubOBtDpuZ3izBX9v7jgUWjersCOKmmS9bSM", 
      project_id="62df5f26c64f9b760444d911", 
      host="192.168.2.18")

   current_path = os.path.dirname(os.path.abspath(__file__))
   train_file_path = os.path.join(current_path, "train.csv")
   train_file = open(train_file_path, "rb")

   data_id = client.upload(data=train_file, name="test_upload")
   print(data_id)
   # print("data_id:", data_id)

   experiment = client.train_iid(experiment_name="name" , data_id=data_id, target="target", evaluator="eva", features={"features":"ok"}, validation_percentage="0.1", default_modes="dm")
   # print(experiment)

   # print(experiment.get_best_model())
