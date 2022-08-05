from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field
import pandas as pd


class Prediction(BaseModel):
    attributes: Dict[str, Any]
    predict_df: Optional[pd.DataFrame]

    class Config:
        arbitrary_types_allowed = True

    def get_predict_df(self) -> pd.DataFrame:
        return self.predict_df