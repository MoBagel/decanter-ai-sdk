# from locale import currency
from decanter_ai_sdk.client import Client
import os
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd

def test_demo1():
   print("---From test demo1---")

   client = Client(
      auth_key="auth_key", 
      project_id="project_id", 
      host="host_ip")

   current_path = os.path.dirname(os.path.abspath(__file__))
   train_file_path = os.path.join(current_path, "train.csv")
   train_file = open(train_file_path, "rb")

   data_id = client.upload(data=train_file, name="test_upload")
   print(data_id)
   # print("data_id:", data_id)

   experiment = client.train_iid(experiment_name="name" , data_id=data_id, target="target", evaluator="eva", features={"features":"ok"}, validation_percentage="0.1", default_modes="dm")
   # print(experiment)

   # print(experiment.get_best_model())
