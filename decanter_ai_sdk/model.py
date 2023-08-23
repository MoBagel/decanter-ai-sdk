from typing import Dict, Union

from pydantic import BaseModel


class Model(BaseModel):
    model_id: str
    model_name: str
    metrics_score: Dict[str, Union[float, None]]
    experiment_id: str
    experiment_name: str
    attributes: Dict
    """
    Model class returned by training action.
    """
