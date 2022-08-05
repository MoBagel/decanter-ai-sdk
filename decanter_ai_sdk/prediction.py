from typing import Any, List
from pydantic import BaseModel, Field
import pandas as pd


class Prediction:
    def __init__(self, data) -> None:
        self.attributes = Attributes.parse_obj(data)
        self.predict_df = None

    def get_predict_df(self) -> pd.DataFrame:
        return self.predict_df


class Attributes(BaseModel):
    prediction_id: str = Field(..., alias="_id")
    apu_mock_model: str
    compare_to_what: str = None
    completed_at: str
    created_at: str
    data_id: str = Field(..., alias="data_id")
    download_count: int
    error: Dict[str, str]
    is_auto_predict: bool
    is_multi_model: bool
    keep_columns: List[str]
    model_id: str
    performance: Any
    plot_key: str
    progress: float
    progress_message: str
    project_id: str
    shapley_only: bool
    started_at: str
    status: str
    table_id: str
    task_id: str
    updated_at: str

    def get_metrics(self):
        return self.performance["metrics"]
