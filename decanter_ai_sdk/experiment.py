from unicodedata import category
from attr import attributes
from pydantic import BaseModel, Field
from typing import Any, List, Dict
from decanter_ai_sdk.model import Model
import sys

class Experiment(BaseModel):
    algos: List[str] = None
    attributes: Dict[str, Any] = None
    bagel_id: str = None
    best_model: str = None
    best_model_id: str = None
    best_score: float = None
    category: str = None
    company_id: str = None
    completed_at: str = None
    corex_models: List[str] = None
    created_at: str = None
    created_by: Dict[str, str] = None
    data_id: str = None
    error: Dict[str, str] = None
    feature_types: List[Dict[str, str]] = None
    features: List[str] = None
    forecast_column: str = None
    forecast_exogeneous_columns: List[str] = None
    forecast_time_group_columns: List[str] = None
    gp_table_id: str = None
    holdout: Dict[str, str] = None
    holdout_percentage: float = None
    is_binary_classification: bool = None
    is_favorited: bool = None
    is_forecast: bool = None
    is_starred: bool = None
    max_model: int = None
    name: str = None
    nfold: int = None
    preprocessing: Dict[str, str] = None
    progress: float = None
    progress_message: str = None
    project_id: str = None
    recommendations: List[Dict[str, Any]] = None
    seed: int = None
    stacked_ensemble: bool = None
    started_at: str = None
    status: str = None
    stopping_metric: str = None
    target: str = None
    target_type: str = None
    task_id: str = None
    timeseriesValues: Dict[str, Any] = None
    tolerance: float = None
    train_table: Dict[str, Any] = None
    updated_at: str = None
    validation_percentage: float = None
    id: str = Field(..., alias='_id')

    def get_best_model(self) -> Model:
        return Model(
                model_id = self.best_model_id,
                model_name = self.best_model,
                metrics_score = self.attributes[self.best_model]['validation_scores'],
                experiment_id = self.id,
                experiment_name = self.name,
                attributes = self.attributes[self.best_model]
            )

    def get_id(self) -> str:
        return self.id

    def get_best_model_by_metric(self, metric: str) -> Model:
        result = None
        if(metric=="auc" or metric =="r2" or metric=="left_top_group" or metric=="custom_increasing"):
            t = 0
            for attr in self.attributes:
                print(self.attributes[attr]['cv_averages'][metric])
                if float(self.attributes[attr]['cv_averages'][metric]) > t:
                    t = float(self.attributes[attr]['cv_averages'][metric])
                    result = Model(
                        model_id = self.attributes[attr]['model_id'],
                        model_name = self.attributes[attr]['name'],
                        metrics_score = self.attributes[attr]['validation_scores'],
                        experiment_id = self.id,
                        experiment_name = self.name,
                        attributes= self.attributes[attr])
        else:
            t = sys.maxint
            for attr in self.attributes:
                if float(self.attributes[attr]['cv_averages'][metric]) < t:
                    t = float(self.attributes[attr]['cv_averages'][metric])
                    result = Model(
                        model_id = self.attributes[attr]['model_id'],
                        model_name = self.attributes[attr]['name'],
                        metrics_score = self.attributes[attr]['validation_scores'],
                        experiment_id = self.id,
                        experiment_name = self.name,
                        attributes= self.attributes[attr])
        
        return result

    def get_model_list(self) -> List[Model]:
        list = []

        for attr in self.attributes:

            list.append(Model(
                model_id = self.attributes[attr]['model_id'],
                model_name = self.attributes[attr]['name'],
                metrics_score = self.attributes[attr]['validation_scores'],
                experiment_id = self.id,
                experiment_name = self.name,
                attributes= self.attributes[attr]))

        return list

    def experiment_info(self) -> Dict:
        return {
            "id" : self.id,
            "name" : self.name,
            "created_at" : self.created_at,
            "completed_at" : self.completed_at
        }