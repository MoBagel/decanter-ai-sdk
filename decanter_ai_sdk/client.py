from time import sleep
from typing import Dict, List, Union, Optional, Any
from tqdm import tqdm
import pandas as pd
from decanter_ai_sdk.enums.algorithms import IIDAlgorithms, TSAlgorithms
from decanter_ai_sdk.experiment import Experiment
from decanter_ai_sdk.prediction import Prediction
from decanter_ai_sdk.model import Model
from decanter_ai_sdk.web_api.iid_testing_api import TestingIidApiClient as IidMockApi
from decanter_ai_sdk.web_api.decanter_api import DecanterApiClient as Api
from decanter_ai_sdk.web_api.ts_testing_api import TestingTsApiClient as TsMockApi
from decanter_ai_sdk.enums.evaluators import ClassificationMetric
from decanter_ai_sdk.enums.evaluators import RegressionMetric
from decanter_ai_sdk.enums.time_units import TimeUnit
from decanter_ai_sdk.non_blocking_client import NonBlockingClient
from .enums.data_types import DataType
from decanter_ai_sdk.enums.missing_value_handling import MissingValueHandling

import logging

logging.basicConfig(level=logging.INFO)


class Client:
    """
    Handle client side actions.

    Support actions sunch as upload data, iid train,
    predict, time series train and predict...etc.

    Example:

    .. code-block:: python

    from decanter_ai_sdk.client import Client

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
        self.non_blocking_client = NonBlockingClient(
            auth_key=auth_key,
            project_id=project_id,
            host=host,
            dry_run_type=dry_run_type,
        )
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
                Uploaded table id.

        """
        table_id = self.non_blocking_client.upload(data=data, name=name)

        res = self.wait_for_response("table", table_id)

        return res["_id"]

    def train_iid(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        custom_feature_types: Dict[str, DataType] = {},
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
    ) -> Experiment:
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
            custom_feature_types (Dict[str: `~decanter_ai_sdk.enums.data_type.DataType`])
                Set customized feature types by inputting {feature_name_1: feature_type_1, feature_name_2: feature_type_2}.
            missing_value_settings (Dict[str: `~decanter_ai_sdk.enums.data_type.DataType`])
                Set missing value handling method by inputting {feature_name_1: feature_type_1, feature_name_2: feature_type_2}.
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
                Experiment results.
        """

        exp_id = self.non_blocking_client.train_iid(
            experiment_name=experiment_name,
            experiment_table_id=experiment_table_id,
            target=target,
            custom_feature_types=custom_feature_types,
            drop_features=drop_features,
            evaluator=evaluator,
            holdout_table_id=holdout_table_id,
            algos=algos,
            max_model=max_model,
            tolerance=tolerance,
            nfold=nfold,
            stacked_ensemble=stacked_ensemble,
            validation_percentage=validation_percentage,
            seed=seed,
            timeseries_value=timeseries_value,
            holdout_percentage=holdout_percentage,
            missing_value_settings=missing_value_settings,
        )

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))

        return experiment

    def train_ts(
        self,
        experiment_name: str,
        experiment_table_id: str,
        target: str,
        datetime: str,
        time_groups: List,
        timeunit: TimeUnit,
        algos: Union[List[TSAlgorithms], List[str]] = [TSAlgorithms.GBM],
        groupby_method: Optional[str] = None,
        evaluator: RegressionMetric = RegressionMetric.WMAPE,
        exogeneous_columns_list: List = [],
        gap: int = 0,
        feature_derivation_window: int = 60,
        horizon_window: int = 1,
        validation_percentage: int = 10,
        nfold: int = 5,
        max_model: int = 20,
        tolerance: int = 3,
        seed: int = 1111,
        drop_features: List[str] = [],
        custom_feature_types: Dict[str, DataType] = {},
        holdout_percentage: int = 10,
        missing_value_settings: Dict[str, MissingValueHandling] = {},
    ) -> Experiment:
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
            custom_feature_types (Dict[str: `~decanter_ai_sdk.enums.data_type.DataType`])
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
            missing_value_settings (Dict[str: `~decanter_ai_sdk.enums.data_type.DataType`])
                Set missing value handling method by inputting {feature_name_1: feature_type_1, feature_name_2: feature_type_2}.
            #TODO Discuss with Ken about this.
            exogeneous_columns_list (List[Dict[Any, Any]])
                List of exogeneous columns.
            timeunit (`~decanter_ai_sdk.enums.time_units.TimeUnit`)
                Time unit to use for forecast experiment [`year`, `month`, `day`, `hour`].
            #TODO Discuss with Ken about this.
            time_groups (List[Dict[Any, Any]])
                List of timegroup columns.


        Returns:
        ----------
            (`~decanter_ai_sdk.web_api.experiment.Experiment`)
                Experiment results.
        """

        exp_id = self.non_blocking_client.train_ts(
            experiment_name=experiment_name,
            experiment_table_id=experiment_table_id,
            target=target,
            datetime=datetime,
            time_groups=time_groups,
            timeunit=timeunit,
            algos=algos,
            groupby_method=groupby_method,
            evaluator=evaluator,
            exogeneous_columns_list=exogeneous_columns_list,
            gap=gap,
            feature_derivation_window=feature_derivation_window,
            horizon_window=horizon_window,
            validation_percentage=validation_percentage,
            nfold=nfold,
            drop_features=drop_features,
            custom_feature_types=custom_feature_types,
            max_model=max_model,
            tolerance=tolerance,
            holdout_percentage=holdout_percentage,
            seed=seed,
        )

        experiment = Experiment.parse_obj(self.wait_for_response("experiment", exp_id))
        return experiment

    def predict_iid(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
    ) -> Prediction:
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
                Prediction results.
        """
        pred_id = self.non_blocking_client.predict_iid(
            keep_columns=keep_columns,
            non_negative=non_negative,
            test_table_id=test_table_id,
            model_id=model_id,
            experiment_id=experiment_id,
            model=model,
        )

        prediction = Prediction(
            attributes=self.wait_for_response("prediction", pred_id)
        )
        prediction.predict_df = self.api.get_pred_data(
            prediction.attributes["_id"], data={"prediction_id": pred_id}
        )

        return prediction

    def predict_ts(
        self,
        keep_columns: List[str],
        non_negative: bool,
        test_table_id: str,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[Model] = None,
    ) -> Prediction:
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
                Prediction results.
        """

        pred_id = self.non_blocking_client.predict_ts(
            keep_columns=keep_columns,
            non_negative=non_negative,
            test_table_id=test_table_id,
            model_id=model_id,
            experiment_id=experiment_id,
            model=model,
        )

        prediction = Prediction(
            attributes=self.wait_for_response("prediction", pred_id)
        )
        prediction.predict_df = self.api.get_pred_data(
            prediction.attributes["_id"], data={"prediction_id": pred_id}
        )
        return prediction

    def stop_uploading(self, id: str) -> None:
        self.non_blocking_client.stop_uploading(id)

    def stop_training(self, id: str) -> None:
        self.non_blocking_client.stop_training(id)

    def wait_for_response(self, url, id):
        pbar = tqdm(total=100, desc=url + " task is now pending")
        progress = 0
        while self.api.check(task=url, id=id)["status"] != "done":  # pragma: no cover
            res = self.api.check(task=url, id=id)

            if res["status"] == "fail":
                raise RuntimeError(res["progress_message"])

            if res["status"] == "running":
                pbar.set_description(
                    "[" + url + "] " + "id: " + id + " " + res["progress_message"]
                )
                pbar.update(int(float(res["progress"]) * 100) - progress)
                progress = int(float(res["progress"]) * 100)

            sleep(3)

        pbar.update(100 - progress)
        pbar.refresh()

        pbar.close()
        logging.info("[" + url + "] Done!")

        return self.api.check(task=url, id=id)

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
