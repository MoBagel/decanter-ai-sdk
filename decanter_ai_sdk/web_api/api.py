from typing import Dict
from abc import ABC, abstractmethod


class ApiClient(ABC):
    @abstractmethod
    def post_upload(self, file: tuple, name: str):
        raise NotImplementedError

    @abstractmethod
    def post_train_iid(self, data):
        raise NotImplementedError

    @abstractmethod
    def post_train_ts(self, data):
        raise NotImplementedError

    @abstractmethod
    def post_predict(self, data):
        raise NotImplementedError

    @abstractmethod
    def get_table_info(self, table_id):
        raise NotImplementedError

    @abstractmethod
    def check(self, task, id):
        raise NotImplementedError

    @abstractmethod
    def get_pred_data(self, pred_id, data):
        raise NotImplementedError

    @abstractmethod
    def get_table_list(self):
        raise NotImplementedError

    @abstractmethod
    def get_table(self, data_id):
        raise NotImplementedError

    @abstractmethod
    def get_model_list(self, experiment_id):
        raise NotImplementedError

    @abstractmethod
    def stop_uploading(self, id):
        raise NotImplementedError

    @abstractmethod
    def stop_training(self, id):
        raise NotImplementedError

    @abstractmethod
    def delete_tables(self, table_ids):
        raise NotImplementedError
