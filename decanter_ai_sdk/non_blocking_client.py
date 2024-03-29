from io import StringIO
from typing import Dict, List, Union, Optional, Any
import pandas as pd
from decanter_ai_sdk.enums.algorithms import IIDAlgorithms, TSAlgorithms
from decanter_ai_sdk.experiment import ExperimentResult
from decanter_ai_sdk.prediction import Prediction, PredictionResult
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.web_api.iid_testing_api import TestingIidApiClient as IidMockApi
from decanter_ai_sdk.web_api.decanter_api import DecanterApiClient as Api
from decanter_ai_sdk.web_api.ts_testing_api import TestingTsApiClient as TsMockApi
from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.time_units import TimeUnit
from decanter_ai_sdk.enums.status import Status
from .enums.data_types import DataType
from .enums.missing_value_handling import MissingValueHandling

import logging

logging.basicConfig(level=logging.INFO)


class NonBlockingClient:
    """
    Handle client side actions without blocking.
    The functions in this client will return the results immediately, so users can do mulitiple works at the same time and check the status.

    Support actions sunch as upload data, iid train,
    predict, time series train and predict...etc.

    Example:

    .. code-block:: python

    from decanter_ai_sdk.client import NonBlockingClient

    ...

    train_file_path = os.path.join("path_to_file", "train.csv")

    client = Client(auth_key="API key get from decanter", project_id="project id from decanter", host="decanter host")

    upload_id = client.upload(data=train_file, name="train_upload")

    ...
    """

    def __init__(self, auth_key, project_id, host, dry_run_type=None):
        self.auth_key: str = auth_key
        self.project_id: str = project_id
        self.host: str = host
        if dry_run_type == "ts":
            self.api = TsMockApi()
        elif dry_run_type == "iid":
            self.api = IidMockApi()
        else:  # pragma: no cover
            self.api = Api(
                host=host,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + auth_key,
                },
                auth_headers={
                    "Authorization": "Bearer " + auth_key,
                },
                project_id=project_id,
            )

    def upload(self, data: Union[str, pd.DataFrame], name: str) -> str:
        """
        Upload csv file or pandas dataframe to gp.

        Parameters:
        ----------
            data Union(str, pandas.DataFrame)
                Can be the path to a csv file or a pandas dataframe.
            name (str)
                Name for the upload action.

        Returns:
        ----------
            (str)
                Upload table id.

        """

        if data is None:
            raise ValueError("[Upload] Uploaded None file.")  # pragma: no cover

        if isinstance(data, pd.DataFrame):
            text_stream = StringIO()
            data.to_csv(text_stream, index=False)
            file = (name, text_stream.getvalue(), "text/csv")
            text_stream.close()

        else:
            file = (name, data, "text/csv")
        table_id = self.api.post_upload(file=file, name=name)

        return table_id

    def train_iid(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        custom_column_types: Dict[str, DataType] = {},
        drop_features: List[str] = [],
        evaluator: Optional[Union[RegressionMetric, ClassificationMetric]] = None,
        holdout_table_id: Optional[str] = None,
        algos: Union[List[IIDAlgorithms], List[str]] = [
            IIDAlgorithms.DRF,
            IIDAlgorithms.GBM,
            IIDAlgorithms.XGBoost,
            IIDAlgorithms.GLM,
        ],
        max_model: int = 20,
        tolerance: int = 3,
        nfold: int = 5,
        stacked_ensemble: bool = True,
        validation_percentage: int = 10,
        seed: int = 1180,
        timeseries_value: List[Dict[str, Any]] = [],
        holdout_percentage: int = 10,
        missing_value_settings: Dict[str, MissingValueHandling] = {},
    ) -> str:
        """
        Train iid models.

        Parameters:
        ----------
            experiment_name (str)
                Name of the experiment.
            experiment_table_id (str)
                Id for the table used in experiment.
            target (str)
                Name of the target column.
            custom_column_types (Dict[str: `~decanter_ai_sdk.enums.data_type.DataType`])
                Set customized feature types by inputting {feature_name_1: feature_type_1, feature_name_2: feature_type_2}.
            drop_features (List[str])
                Feature names that are not going to be used during experiment.
            evaluator (Union[`~decanter_ai_sdk.enums.evaluators.ClassificationMetric`, `~decanter_ai_sdk.enums.evaluators.RegressionMetric`])
                Evaluator used as stopping metric.
            holdout_table_id (str)
                Id of the table used to perform holdout.
            algos (Union[List[`~decanter_ai_sdk.enums.algorithms.IIDAlgorithms`], List[`~decanter_ai_sdk.enums.algorithms.TSAlgorithms`]])
                Algorithms used for experiment.
            max_model (int)
                Limit for the number of models to train for this experiment.
            tolerance (int)
                Larger error tolerance will let the training stop earlier. Smaller error tolerance usually generates more accurate models but takes more time. (1~10)
            nfold (int)
                Amount of folds in experiment. (2~10) for autoML. (1~10) for autoTSF.
            stacked_ensemble (boolean)
                If stacked ensemble models will be trained.
            validation_percentage (int)
                Validation percentage of experiment. (5~20)
            seed (int)
                Random Seed of experiment. (1 ~ 65535)
            timeseries_value (List[Dict[str, Any]])
                Folds for time series cross validation (train, window, test, holdout_timeseries, cv, holdout_Percentage, split_By, lag).
                \tExample:
                    {
                        'train': {
                            'start': '2011-03-18T00:00:00',
                            'end': '2012-04-21T00:00:00'
                        },
                        'window': 40,
                        'test': 20,
                        'holdout_timeseries': {
                            'start': '2012-04-21T00:00:00',
                            'end': '2020-03-28T00:00:00'
                        },
                        'cv': [
                            {
                                'train': {
                                    'start': '2009-10-25T00:00:00',
                                    'end': '2010-11-29T00:00:00'
                                },
                                'test': {
                                    'start': '2010-11-29T00:00:00',
                                    'end': '2011-06-17T00:00:00'
                                }
                            }
                        ],
                        'holdoutPercentage': 10,
                        'splitBy': 'MonitorDateTime',
                        'lag': 0
                    }
            holdout_percentage (int)
                Holdout percentage for experiment.

        Returns:
        ----------
            (`~decanter_ai_sdk.web_api.experiment.Experiment`)
                Experiment id.
        """
        data_column_info = self.api.get_table_info(table_id=experiment_table_id)

        # cast target column
        if target in custom_column_types.keys():
            data_column_info[target] = custom_column_types[target].value

        if validation_percentage < 5 or validation_percentage > 20:
            raise ValueError(
                "validation_percentage should be inside a range between 5 to 20."
            )  # pragma: no cover

        algo_values = []
        for algo in algos:
            if type(algo) == str:
                algo_values.append(algo)
            elif type(algo) == IIDAlgorithms:
                algo_values.append(algo.value)

        features = [
            feature
            for feature in data_column_info.keys()
            if feature not in drop_features + [target]
        ]

        feature_types = [
            {"id": k, "data_type": j}
            for k, j in {key: data_column_info[key] for key in features}.items()
        ]

        for feature in feature_types:
            if feature["id"] in custom_column_types.keys():
                feature["data_type"] = custom_column_types[feature["id"]].value

        column_list = []
        for column in missing_value_settings.keys():
            column_list.append(
                {
                    "columnName": column,
                    "method": {"imputation": missing_value_settings[column].value},
                }
            )

        if data_column_info[target] == "numerical":
            category = "regression"
            is_binary_classification: bool = False
            if evaluator is None:
                evaluator = RegressionMetric.MAPE
            elif evaluator.name not in RegressionMetric.__members__:
                raise ValueError(
                    "Wrong evaluator, you need to fill wmape, mse ..."
                )  # pragma: no cover
        else:
            category = "classification"

            if self.get_table(experiment_table_id)[target].nunique() > 2:
                is_binary_classification = False
            else:
                is_binary_classification = True

            if evaluator is None:
                evaluator = (
                    ClassificationMetric.AUC
                    if is_binary_classification
                    else ClassificationMetric.LOGLOSS
                )
            elif evaluator.name not in ClassificationMetric.__members__:
                raise ValueError(
                    "Wrong evaluator, you need to fill auc, logloss..."
                )  # pragma: no cover

        holdout_config: Dict[str, Any] = {}

        if holdout_percentage:
            holdout_config["percent"] = holdout_percentage

        if holdout_table_id:
            holdout_config["table"] = holdout_table_id

        training_settings = {
            "project_id": self.project_id,
            "name": experiment_name,
            "gp_table_id": experiment_table_id,
            "seed": seed,
            "target": target,
            "targetType": data_column_info[target],
            "features": features,
            "feature_types": feature_types,
            "category": category,
            "stopping_metric": evaluator.value,
            "is_binary_classification": is_binary_classification,
            "holdout": holdout_config,
            "tolerance": tolerance,
            "nfold": nfold,
            "max_model": max_model,
            "algos": algo_values,
            "stacked_ensemble": stacked_ensemble,
            "validation_percentage": validation_percentage,
            "timeseriesValues": timeseries_value,
            "preprocessing": {"columnWise": column_list},
        }

        exp_id = self.api.post_train_iid(training_settings)

        return exp_id

    def train_ts(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        datetime: str,
        time_groups: List,
        timeunit: TimeUnit,
        nfold: int,
        validation_percentage: int,
        holdout_percentage: int,
        algos: Union[List[TSAlgorithms], List[str]] = [TSAlgorithms.GBM],
        groupby_method: Optional[str] = None,
        evaluator: RegressionMetric = RegressionMetric.WMAPE,
        exogeneous_columns_list: List = [],
        gap: int = 0,
        feature_derivation_window: int = 60,
        horizon_window: int = 1,
        max_model: int = 20,
        tolerance: int = 3,
        seed: int = 1111,
        drop_features: List[str] = [],
        custom_column_types: Dict[str, DataType] = {},
        missing_value_settings: Dict[str, MissingValueHandling] = {},
        train_fusion_model: bool = False,
    ) -> str:
        """
        Train timeseries models.

        Parameters:
        ----------
            experiment_name (str)
                Name of the experiment.
            experiment_table_id (str)
                Id for the table used in experiment.
            target (str)
                Name of the target column.
            datetime (str)
                Date-time column for Time Series Forecast training.
            custom_column_types (Dict[str: `~decanter_ai_sdk.enums.data_type.DataType`])
                Set customized feature types by inputting {feature_name_1: feature_type_1, feature_name_2: feature_type_2}.
            evaluator (`~decanter_ai_sdk.enums.evaluators.ClassificationMetric`, `~decanter_ai_sdk.enums.evaluators.RegressionMetric`)
                Evaluator used as stopping metric.
            algos (List[`~decanter_ai_sdk.enums.algorithms.IIDAlgorithms`],  List[`~decanter_ai_sdk.enums.algorithms.TSAlgorithms`])
                Algorithms used for experiment.
            max_model (int)
                Limit for the number of models to train for this experiment.
            tolerance (int)
                Larger error tolerance will let the training stop earlier. Smaller error tolerance usually generates more accurate models but takes more time. (1~10)
            nfold (int)
                Amount of folds in experiment. (2~10) for autoML. (1~10) for autoTSF.
            validation_percentage (int)
                Validation percentage of experiment. (5~20)
            seed (int)
                Random Seed of experiment. (1 ~ 65535)
            holdout_percentage (int)
                Holdout percentage for experiment.
            horizon_window (int)
                experiment forecast horizon window value.
            gap (int)
                Forecast gap.
            feature_derivation_window (int)
                Training forecast derivation window value.
            groupby_method (str)
                Group by method used for forecast experiment.
            #TODO Discuss with Ken about this.
            exogeneous_columns_list (List[Dict[Any, Any]])
                List of exogeneous columns.
            timeunit (`~decanter_ai_sdk.enums.time_units.TimeUnit`)
                Time unit to use for forecast experiment [`year`, `month`, `day`, `hour`].
            #TODO Discuss with Ken about this.
            time_groups (List[Dict[Any, Any]])
                List of timegroup columns.
            train_fusion_model (bool)
                Whether to train fusion model or not.
                The fusion model feature can be accessed exclusively in versions later than v4.12.28.

        Returns:
        ----------
            (`~decanter_ai_sdk.web_api.experiment.Experiment`)
                Experiment id.
        """

        if validation_percentage < 5 or validation_percentage > 20:
            raise ValueError(
                "validation_percentage should be inside a range between 5 to 20."
            )  # pragma: no cover

        algo_values = []
        for algo in algos:
            if type(algo) == str:
                algo_values.append(algo)
            elif type(algo) == TSAlgorithms:
                algo_values.append(algo.value)

        data_column_info = self.api.get_table_info(table_id=experiment_table_id)

        features = [
            feature
            for feature in data_column_info.keys()
            if feature not in drop_features + [target]
        ]

        feature_types = [
            {"id": k, "data_type": j}
            for k, j in {key: data_column_info[key] for key in features}.items()
        ]

        for feature in feature_types:
            if feature["id"] in custom_column_types.keys():
                feature["data_type"] = custom_column_types[feature["id"]].value

        column_list = []
        for column in missing_value_settings.keys():
            column_list.append(
                {
                    "columnName": column,
                    "method": {"imputation": missing_value_settings[column]},
                }
            )

        training_settings = {
            "project_id": self.project_id,
            "name": experiment_name,
            "gp_table_id": experiment_table_id,
            "seed": seed,
            "target": target,
            "targetType": data_column_info[target],
            "features": features,
            "feature_types": feature_types,
            "category": "regression",
            "stopping_metric": evaluator.value,
            "is_binary_classification": False,
            "holdout": {"percent": holdout_percentage},
            "tolerance": tolerance,
            "max_model": max_model,
            "algos": algo_values,
            "balance_class": False,
            "is_forecast": True,
            "stacked_ensemble": False,
            "forecast_column": datetime,
            "forecast_exogeneous_columns": exogeneous_columns_list,
            "forecast_groupby_method": groupby_method,
            "forecast_gap": gap,
            "feature_derivation_start": 0,
            "feature_derivation_window": feature_derivation_window,
            "forecast_horizon_start": gap,
            "forecast_horizon_window": horizon_window,
            "forecast_time_group_columns": time_groups,
            "forecast_timeunit": timeunit.value,
            "validation_percentage": validation_percentage,
            "nfold": nfold,
            "preprocessing": {"columnWise": column_list},
            "train_fusion_model": train_fusion_model,
        }

        exp_id = self.api.post_train_ts(training_settings)

        return exp_id

    def predict_iid(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Predict model with test iid data.

        Parameters:
        ----------
            model (`~decanter_ai_sdk.web_api.model.Model`)
                Model generated by train.
            keep_columns (List[str])
                Columns to include in the prediction result.
            non_negative (bool)
                Whether to convert all negative predictions to 0.
            test_table_id (str)
                Id of table used to predict.
            model_id (str)
                Id of model used to predict.
            experiment_id (str)
                Id of experiment used to predict.

        Returns:
        ----------
            (`~decanter_ai_sdk.web_api.prediction.Prediction`)
                Prediction id.
        """

        if model is None and (experiment_id is None or model_id is None):
            raise ValueError(
                "either model or both experiment_id and model_id should be defined"
            )  # pragma: no cover

        mod_id = model.model_id if model is not None else model_id
        exp_id = model.experiment_id if model is not None else experiment_id

        prediction_settings = {
            "project_id": self.project_id,
            "experiment_id": exp_id,
            "model_id": mod_id,
            "table_id": test_table_id,
            "is_multi_model": False,
            "non_negative": non_negative,
            "keep_columns": keep_columns,
            "threshold": threshold,
        }

        pred_id = self.api.post_predict(prediction_settings)

        return pred_id

    def predict_ts(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
    ) -> str:
        """
        Predict model with test timeseries data.

        Parameters:
        ----------
            model (`~decanter_ai_sdk.web_api.model.Model`)
                Model generated by train.
            keep_columns (List[str])
                Columns to include in the prediction result.
            non_negative (bool)
                Whether to convert all negative predictions to 0.
            test_table_id (str)
                Id of table used to predict.
            model_id (str)
                Id of model used to predict.
            experiment_id (str)
                Id of experiment used to predict.

        Returns:
        ----------
            (`~decanter_ai_sdk.web_api.prediction.Prediction`)
                Prediction id.
        """

        if model is None and (experiment_id is None or model_id is None):
            raise ValueError(
                "either model or both experiment_id and model_id should be defined"
            )  # pragma: no cover

        mod_id = model.model_id if model is not None else model_id
        exp_id = model.experiment_id if model is not None else experiment_id
        is_multi_model = False
        for k in self.api.get_model_list(exp_id):
            if k["_id"] == mod_id:
                is_multi_model = k["model_type"] in [
                    "ExodusModel",
                    "MultiModel",
                    "LeviticusModel",
                ]

        prediction_settings = {
            "project_id": self.project_id,
            "experiment_id": exp_id,
            "model_id": mod_id,
            "table_id": test_table_id,
            "is_multi_model": is_multi_model,
            "non_negative": non_negative,
            "keep_columns": keep_columns,
        }

        pred_id = self.api.post_predict(prediction_settings)

        return pred_id

    def stop_uploading(self, id: str) -> None:
        if self.api.stop_uploading(id):
            logging.info("Uploading task: " + id + " stopped successfully.")
        else:
            logging.info("This task has already stopped or doesn't exist.")
        return None

    def stop_training(self, id: str) -> None:
        if self.api.stop_training(id):
            logging.info("Experiment: " + id + " stopped successfully.")
        else:
            logging.info("This task has already stopped or doesn't exist.")
        return None

    def get_table(self, data_id: str) -> pd.DataFrame:
        """
        Return table dataframe.

        Params:
        ----------
            data_id (str)
                Uploaded table id.

        Returns:
        ----------
            (pandas.DataFrame)
                Uploaded table dataframe.
        """
        return self.api.get_table(data_id=data_id)

    def get_table_list(self) -> List[str]:
        """
        Return list of table information.

        Returns:
        ----------
            (List[str])
                List of uploaded table information.
        """
        return self.api.get_table_list()

    def check_upload_status(self, id) -> Status:
        """
        You can check the status of the uploading task by inputting upload id using this function.

        Params:
        ----------
            id (str)
                Uploaded table id.

        Returns:
        ----------
            status (str)
        """
        return Status(self.api.check(task="table", id=id)["status"])

    def check_exp_status(self, id) -> Status:
        """
        You can check the status of the experiment by inputting experiment id using this function.

        Params:
        ----------
            id (str)
                Experiment table id.

        Returns:
        ----------
            status (str)
        """
        return Status(self.api.check(task="experiment", id=id)["status"])

    def check_pred_status(self, id) -> Status:
        """
        You can check the status of the prediction by inputting prediction id using this function.

        Params:
        ----------
            id (str)
                Prediction table id.

        Returns:
        ----------
            status (str)
        """
        return Status(self.api.check(task="prediction", id=id)["status"])

    def get_exp_result(self, exp_id) -> ExperimentResult:
        """
        You can get experiment results in ExperimentResult type using this function.
        """
        response = self.api.check(task="experiment", id=exp_id)
        if response["status"] != "done":
            logging.info("Experiment is not done yet.")

        experiment = ExperimentResult(
            result=response, status=Status(response["status"])
        )

        return experiment

    def get_pred_result(self, pred_id) -> PredictionResult:
        """
        You can get prediction results in PredictionResult type using this function.
        """
        response = self.api.check(task="prediction", id=pred_id)
        if response["status"] != "done":
            logging.info("Predict task is not done yet.")

        prediction = PredictionResult(
            result=Prediction(
                attributes=response,
                predict_df=self.api.get_pred_data(
                    pred_id=pred_id, data={"prediction_id": pred_id}
                ),
            ),
            status=Status(response["status"]),
        )

        return prediction
