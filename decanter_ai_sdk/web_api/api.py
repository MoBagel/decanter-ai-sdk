from typing import Dict
from abc import ABC, abstractmethod


class ApiClient(ABC):
    @abstractmethod
    def post_upload(self, file: Dict, name: str):
        pass

    @abstractmethod
    def post_train_iid(self, data):
        pass

    @abstractmethod
    def post_train_ts(self, data):
        pass

    @abstractmethod
    def post_predict(self, data):
        pass

    @abstractmethod
    def get_table_info(self, table_id):
        pass

    @abstractmethod
    def check(self, task, id):
        pass

    @abstractmethod
    def get_pred_data(self, pred_id, data):
        pass

    @abstractmethod
    def get_table_list(self):
        pass

    @abstractmethod
    def get_table(self, data_id):
        pass

    @abstractmethod
    def get_model_list(self, experiment_id, query):
        pass
