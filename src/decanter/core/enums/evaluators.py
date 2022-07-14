"""
Function for user access the Evaluators Metrics
of the Decanter AI Core SDK.
"""
from enum import Enum


class Evaluator(Enum):
    """
    The Evaluator enumeration is the metrics currently
    supported by the Decanter AI Core SDK

    - Regression
        - auto (deviance)
        - deviance
        - mse
        - mae
        - rmsle
        - r2
        - mape (Supported from Decanter AI 4.9~)
        - wmape (Supported from Decanter AI 4.9~)
    - Binary Classification
        - auto (logloss)
        - logloss
        - lift_top_group
        - auc
        - misclassification
    - Multinomial Classification
        - auto (logloss)
        - logloss
        - misclassification
    - Clustering
        - auto (tot_withinss)
        - tot_withinss
    """
    auto = 'auto'
    mse = 'mse'
    rmse = 'rmse'
    mae = 'mae'
    rmsle = 'rmsle'
    auc = 'auc'
    logloss = 'logloss'
    deviance = 'deviance'
    r2 = 'r2'
    mape = 'mape'
    wmape = 'wmape'
    lift_top_group = 'lift_top_group'
    misclassification = 'misclassification'
    mean_per_class_error = 'mean_per_class_error'
    tot_withinss = 'tot_withinss'

    @staticmethod
    def resolve_select_model_by(select_model_by, model_type):
        if select_model_by == Evaluator.auto.value:
            if model_type == "regression":
                return Evaluator.deviance.value
            elif model_type == "multinomial classification" or model_type == "binary classification":
                return Evaluator.logloss.value
            elif model_type == "clustering":
                return Evaluator.tot_withinss.value
            else:
                raise Exception(f"Illegal model type: {model_type}")
        else:
            return select_model_by
