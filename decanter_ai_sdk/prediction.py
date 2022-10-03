from typing import Any, Dict, Optional
from pydantic import BaseModel
from decanter_ai_sdk.enums.status import Status
import pandas as pd


class Prediction(BaseModel):
    attributes: Dict[str, Any]
    predict_df: Optional[pd.DataFrame]
    """
    Prediction class returned by prediction action.
    """

    class Config:
        arbitrary_types_allowed = True

    def get_predict_df(self) -> pd.DataFrame:
        """

        Returns:
        ----------
        (pandas.DataFrame)
            Predict result in pandas.DataFrame.

        """
        return self.predict_df


class PredictionResult(BaseModel):
    result: Optional[Prediction]
    status: Status
