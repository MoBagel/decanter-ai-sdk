from enum import Enum


class IIDAlgorithms(Enum):
    """
    IID Algorithms.
    """

    DRF = "DRF"
    GBM = "GBM"
    GLM = "GLM"
    XGBoost = "XGBoost"
    StackedEnsemble = "StackedEnsemble"
    DeepLearning = "DeepLearning"
    autotpot = "autotpot"
    rfmulticlassifier = "rfmulticlassifier"
    autolgbm = "autolgbm"
    autoxgboost = "autoxgboost"


class TSAlgorithms(str, Enum):
    """
    Time series algorithms.
    """

    XGBoost = "XGBoost"
    GBM = "GBM"
    GLM = "GLM"
    theta = "theta"
    ets = "ets"
    metaprophet = "metaprophet"
    arima = "arima"
    lgbmspeed = "lgbmspeed"
    lgbmaccuracy = "lgbmaccuracy"
    lstm = "lstm"
    fewshotlearning = "fewshotlearning"

    class Config:
        use_enum_values = True
