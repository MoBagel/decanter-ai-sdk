from typing import Dict
from abc import ABC, abstractmethod


class Api(ABC):
    @abstractmethod
    def post_upload(self, file: Dict, name: str):
        pass

    def post_train_iid(self, data):
        pass

    def post_train_ts(self, data):
        pass

    def post_predict(self, data):
        pass

    def get_table_info(self, table_id):
        pass

    def check(self, task, id):
        pass

    def get_pred_data(self, pred_id, data):
        pass

    def get_table_list(self):
        pass

    def get_table(self, data_id):
        pass

    def get_model_list(self, experiment_id, query):
        pass
