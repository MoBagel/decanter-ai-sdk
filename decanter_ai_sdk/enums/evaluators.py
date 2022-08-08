from enum import Enum


class ClassificationMetric(Enum):
    """
    The Evaluator enumeration is the metrics currently
    supported by the Decanter AI Core SDK
    - Classification
        - auto (logloss)
        - logloss
        - lift_top_group
        - auc
        - misclassification
        - mean_per_class_error
    """

    LOGLOSS = "logloss"
    LIFT_TOP_GROUP = "lift_top_group"
    AUC = "auc"
    MISCLASSIFICATION = "misclassification"
    MEAN_PER_CLASS_ERROR = "mean_per_class_error"


class RegressionMetric(Enum):
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
    """

    WMAPE = "wmape"
    R2 = "r2"
    MSE = "mse"
    RMSE = "rmse"
    DEVIANCE = "deviance"
    MAE = "mae"
    RMSLE = "rmsle"
    MAPE = "mape"
