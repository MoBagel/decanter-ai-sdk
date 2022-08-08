from unicodedata import category
from attr import attributes
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from decanter_ai_sdk.enums.evaluators import ClassificationMetric, RegressionMetric
from decanter_ai_sdk.model import Model
import sys


class Experiment(BaseModel):
    algos: List[str]
    attributes: Dict
    bagel_id: str
    best_model: str
    best_model_id: str
    best_score: float
    category: str
    company_id: str
    completed_at: str
    corex_models: List[str]
    created_at: str
    created_by: Any
    data_id: str
    error: Dict[str, str]
    feature_types: List[Dict[str, str]]
    features: List[str]
    forecast_column: Optional[str]
    forecast_exogeneous_columns: List[str] = []
    forecast_time_group_columns: List[str] = []
    gp_table_id: str
    holdout: Dict[str, str]
    holdout_percentage: float
    is_binary_classification: bool
    is_favorited: bool
    is_forecast: bool
    is_starred: bool
    max_model: int
    name: str
    nfold: int
    preprocessing: Dict[str, str]
    progress: float
    progress_message: str
    project_id: str
    recommendations: List[Dict]
    seed: int
    stacked_ensemble: bool
    started_at: str
    status: str
    stopping_metric: str
    target: str
    target_type: str
    task_id: str
    timeseriesValues: Dict
    tolerance: float
    train_table: Dict
    updated_at: str
    validation_percentage: float
    id: str = Field(..., alias="_id")

    def get_best_model(self) -> Model:

        return Model(
            model_id=self.best_model_id,
            model_name=self.best_model,
            metrics_score=self.attributes[self.best_model]["cv_averages"],
            experiment_id=self.id,
            experiment_name=self.name,
            attributes=self.attributes[self.best_model],
        )

    def get_best_model_by_metric(self, metric: ClassificationMetric) -> Model:
        result = None
        if (
            metric == ClassificationMetric.AUC
            or metric == RegressionMetric.R2
            or metric == ClassificationMetric.LIFT_TOP_GROUP
        ):
            score = 0
            for attr in self.attributes:
                if float(self.attributes[attr]["cv_averages"][metric]) > score:
                    score = float(self.attributes[attr]["cv_averages"][metric])
                    result = Model(
                        model_id=self.attributes[attr]["model_id"],
                        model_name=self.attributes[attr]["name"],
                        metrics_score=self.attributes[attr]["cv_averages"],
                        experiment_id=self.id,
                        experiment_name=self.name,
                        attributes=self.attributes[attr],
                    )
        else:
            score = sys.maxsize
            for attr in self.attributes:
                if float(self.attributes[attr]["cv_averages"][metric]) < score:
                    score = float(self.attributes[attr]["cv_averages"][metric])
                    result = Model(
                        model_id=self.attributes[attr]["model_id"],
                        model_name=self.attributes[attr]["name"],
                        metrics_score=self.attributes[attr]["cv_averages"],
                        experiment_id=self.id,
                        experiment_name=self.name,
                        attributes=self.attributes[attr],
                    )

        return result

    def get_model_list(self) -> List[Model]:
        list = []

        for attr in self.attributes:

            list.append(
                Model(
                    model_id=self.attributes[attr]["model_id"],
                    model_name=self.attributes[attr]["name"],
                    metrics_score=self.attributes[attr]["cv_averages"],
                    experiment_id=self.id,
                    experiment_name=self.name,
                    attributes=self.attributes[attr],
                )
            )

        return list

    def experiment_info(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
