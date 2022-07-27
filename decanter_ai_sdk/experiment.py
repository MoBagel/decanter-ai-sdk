from unicodedata import category
from pydantic import BaseModel
from typing import Any, List, Dict
from decanter_ai_sdk.model import Model
class Experiment():
    def __init__(self, data) -> None:
        self.Attr = pyExp.parse_obj(data)
    
    def get_best_model() -> Model:
        return "best_model"

class pyExp(BaseModel):
    id: int = None
    name: str = None
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

    # def __init__(self) -> None:
    #     # super().__init__(**data)
    #     pass

    def get_best_model() -> Model:
        
        pass

    # def get_best_model_by_metric(metric: str) -> Model:
    #     pass

    # def get_model_list() -> List[Model]:
    #     pass

    # def get_best_model_by_matrics(matric) -> Model:
    #     pass

    # def experiment_info() -> Dict:
    #     pass


# class Experiment():
#     def __init__(self, experiment_id, experiment_name, model_list, response_data):
#         self.experiment_id = experiment_id
#         self.experiment_name = experiment_name
#         self.model_list = model_list
#         self.attributes = response_data
