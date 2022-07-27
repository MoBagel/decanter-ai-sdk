from typing import Any, Dict
from pydantic import BaseModel


class Model(BaseModel):
    model_id: str = None
    model_name: str = None
    model_data: Any = None
    experiment_id: str = None
    experiment_name: str = None
    attributes: Dict[str, Any] = None

    # def __init__(self, model_id, model_name, model_data, experiment_id, experiment_name):
    #     self.model_id = model_id
    #     self.model_name = model_name
    #     self.model_data = model_data
    #     self.experiment_id = experiment_id
    #     self.experiment_name = experiment_name
