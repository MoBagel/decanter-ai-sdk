from typing import Dict
from pydantic import BaseModel


class Model(BaseModel):
    model_id: str
    model_name: str
    metrics_score: Dict[str, float]
    experiment_id: str
    experiment_name: str
    attributes: Dict
    """
    Model class returned by training action.
    """
