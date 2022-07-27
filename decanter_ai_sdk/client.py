import enum
from io import StringIO
from typing import List, Optional, Union

import pandas

from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from decanter_ai_sdk.model import Model

from decanter_ai_sdk.web_api.parser import Api
from decanter_ai_sdk.web_api.parser import Parser

import pandas as pd

inputType = Union[str, pd.DataFrame]

class Client:
    def __init__(self, auth_key, project_id, host):
        self.auth_key = auth_key
        self.project_id = project_id
        self.host = host
        self.parser = Parser(host, headers = {"Authorization": "Bearer " + auth_key}, project_id=project_id)
        
        # self.api = Api(host, {"Authorization": "Bearer " + host})

    def upload(self, data: Optional[inputType], name: str) -> str:
        if(isinstance(data, pd.DataFrame)):
            textStream = StringIO
            data.to_csv(textStream, index=False)
            file = [(textStream.getvalue(), 'text/csv')]
        else:
            file = [(data, 'text/csv')]

        data_id = self.parser.DataUpload(
            data=file,
            name = name
        )

        return data_id

    def train_iid(
        self,
        experiment_name: str,
        data_id: str,
        target: str,
        evaluator: str,
        features: List[str],
        validation_percentage: int,
        default_modes: str,
    ) -> Experiment:

        experiment = self.parser.TrainIID(
            project_id= self.project_id,
            experiment_name= experiment_name,
            data_id= data_id,
            target= target,
            evaluator= evaluator,
            feature= features,
            validation_percentage= validation_percentage,
            default_mode= default_modes,
        )

        return experiment

    def predict_iid(
        self,
        model: Model,
        keep_columns: List[str],
        non_negative: bool,
        test_data_id: str,
    ) -> Prediction:

        prediction = self.parser.PredictIID(
            self.project_id,
            model.experiment_id,
            model.model_id,
            test_data_id,
            keep_columns,
            non_negative,
        )

        return prediction

    def show_table(data_id: str) -> pandas.DataFrame:
        # return single data df
        pass

    def show_table_list(project_id: str) -> List[str]:
        # return list of tables
        pass

    
    # def predict_batch():

    #     pass
