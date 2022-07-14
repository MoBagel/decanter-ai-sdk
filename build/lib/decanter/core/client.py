# pylint: disable=too-many-arguments
"""Function for user handle the use of Decanter Core API."""
import io
import logging

import pandas as pd

from decanter.core import Context
from decanter.core.jobs import DataUpload, DataSetup,\
    Experiment, ExperimentTS, ExperimentCluster,\
    PredictResult, PredictTSResult
from decanter.core.enums.evaluators import Evaluator
from decanter.core.enums import check_is_enum

logger = logging.getLogger(__name__)


class CoreClient(Context):
    """Handle client side actions.

    Support actions sunch as setup data, upload data, train,
    predict, time series train and predict...ect.

    Example:
        .. code-block:: python

            from decanter import core
            client = core.CoreClient()
            client.upload(data={csv-file-type/dataframe})

    """

    def __init__(self, username, password, host):
        super().__init__()
        """Create context instance and init neccessary variable and objects.

            Setting the user, password, and host for the funture connection when
            calling APIs, and create an event loop if it isn't exist. Check if the
            connection is healthy after args be set.

        Args:
            username (str): User name for login Decanter Core server
            password (str): Password name for login Decanter Core server
            host (str): Decanter Core server URL.
        """
        Context.create(username=username, password=password, host=host)
        self.api = Context.api

    @staticmethod
    def setup(setup_input, name=None):
        """Setup data reference.

        Create a DataSetup Job and scheduled the execution in CORO_TASKS list.
        Record the Job in JOBS list.

        Args:
            setup_input
                (:class:`~decanter.core.core_api.setup_input.SetupInput`):
                stores the settings for training.
            name (:obj:`str`, optional): name for setup action.

        Returns:
            :class:`~decanter.core.jobs.data_setup.DataSetup` object

        Raises:
            AttributeError: If the function is called without
                :class:`~decanter.core.context.Context` created.

        """

        data = DataSetup(setup_input=setup_input, name=name)

        try:
            if Context.LOOP is None:
                raise AttributeError('[Core] event loop is \'NoneType\'')
            task = Context.LOOP.create_task(
                data.wait())
            Context.CORO_TASKS.append(task)
            Context.JOBS.append(data)
        except AttributeError:
            logger.error('[Core] Context not created')
            raise
        return data

    @staticmethod
    def upload(file, name=None, eda=True):
        """Upload csv file or pandas dataframe.

        Create a DataUpload Job and scheduled the execution in CORO_TASKS list.
        Record the Job in JOBS list.

        Args:
            file (csv-file, :obj:`pandas.DataFrame`): File uploaded to
                core server.
            name (str, optional): Name for upload action.
            eda (bool, optional): Whether to perform eda on data upload

        Returns:
            :class:`~decanter.core.jobs.data_upload.DataUpload` object

        Raises:
            AttributeError: If the function is called without
                :class:`~decanter.core.context.Context` created.

        """
        logger.debug('[Core] Create DataUpload Job')

        # check file validation
        if file is None:
            logger.error('[Core] upload file is \'NoneType\'')
            raise Exception
        if isinstance(file, pd.DataFrame):
            file = file.to_csv(index=False)
            file = io.StringIO(file)
            file.name = 'no_name'

        data = DataUpload(file=file, name=name, eda=eda)
        # check context validation
        try:
            if Context.LOOP is None:
                raise AttributeError('[Core] event loop is \'NoneType\'')
            task = Context.LOOP.create_task(data.wait())
            Context.CORO_TASKS.append(task)
            Context.JOBS.append(data)
        except AttributeError:
            logger.error('[Core] Context not created')
            raise
        return data

    @staticmethod
    def train(train_input, select_model_by=Evaluator.auto, name=None):
        """Train model with data.

        Create a Experiment Job and scheduled the execution in CORO_TASKS list.
        Record the Job in JOBS list.

        Args:
            train_input
                (:class:`~decanter.core.core_api.train_input.TrainInput`):
                stores the settings for training.
            select_model_by
                (:class:`~decanter.core.enums.evaluators.Evaluator`):
                if predict by trained experiment, how should we select best model
            name (:obj:`str`, optional): name for train action.

        Returns:
            :class:`~decanter.core.jobs.experiment.Experiment` object

        Raises:
            AttributeError: If the function is called without
                :class:`~decanter.core.context.Context` created.
        """
        select_model_by = check_is_enum(Evaluator, select_model_by)
        logger.debug('[Core] Create Train Job')
        exp = Experiment(
            train_input=train_input,
            select_model_by=select_model_by, name=name)
        try:
            if Context.LOOP is None:
                raise AttributeError('[Core] event loop is \'NoneType\'')
            task = Context.LOOP.create_task(exp.wait())
            Context.CORO_TASKS.append(task)
            Context.JOBS.append(exp)
        except AttributeError:
            logger.error('[Core] Context not created')
            raise
        return exp

    @staticmethod
    def train_ts(train_input, select_model_by=Evaluator.auto, name=None):
        """Train time series model with data.

        Create a Time Series Experiment Job and scheduled the execution
        in CORO_TASKS list.  Record the Job in JOBS list.

        Args:
            train_input
                (:class:`~decanter.core.core_api.train_input.TrainTSInput`):
                Settings for training.
            select_model_by
                (:class:`~decanter.core.enums.evaluators.Evaluator`):
                if predict by trained experiment, how should we select best model
            name (:obj:`str`, optional): name for train time series action.

        Returns:
            :class:`~decanter.core.jobs.experiment.ExperimentTS` object

        Raises:
            AttributeError: If the function is called without
                :class:`~decanter.core.context.Context` created.
        """
        select_model_by = check_is_enum(Evaluator, select_model_by)
        logger.debug('[Core] Create Train Job')
        exp_ts = ExperimentTS(
            train_input=train_input,
            select_model_by=select_model_by,
            name=name)
        try:
            if Context.LOOP is None:
                raise AttributeError('[Core] event loop is \'NoneType\'')
            task = Context.LOOP.create_task(exp_ts.wait())
            Context.CORO_TASKS.append(task)
            Context.JOBS.append(exp_ts)
        except AttributeError:
            logger.error('[Core] Context not created')
            raise
        return exp_ts

    @staticmethod
    def train_cluster(train_input, name=None):
        """Train cluster model with data.

        Create a Cluster Experiment Job and scheduled the execution
        in CORO_TASKS list.  Record the Job in JOBS list.

        Args:
            train_input
                (:class:`~decanter.core.core_api.train_input.TrainClusterInput`):
                Settings for training.
            name (:obj:`str`, optional): name for train time series action.

        Returns:
            :class:`~decanter.core.jobs.experiment.ExperimentTS` object

        Raises:
            AttributeError: If the function is called without
                :class:`~decanter.core.context.Context` created.
        """
        logger.debug('[Core] Create Train Cluster Job')
        exp = ExperimentCluster(
            train_input=train_input,
            select_model_by=Evaluator.tot_withinss, name=name)
        try:
            if Context.LOOP is None:
                raise AttributeError('[Core] event loop is \'NoneType\'')
            task = Context.LOOP.create_task(exp.wait())
            Context.CORO_TASKS.append(task)
            Context.JOBS.append(exp)
        except AttributeError:
            logger.error('[Core] Context not created')
            raise
        return exp

    @staticmethod
    def predict(predict_input, name=None):
        """Predict model with test data.

        Create a PredictResult Job and scheduled the execution
        in CORO_TASKS list. Record the Job in JOBS list.

        Args:
            predict_input
                (:class:`~decanter.core.core_api.predict_input.PredictInput`):
                stores the settings for prediction.
            name (:obj:`str`, optional): string, name for predict action.

        Returns:
            :class:`~decanter.core.jobs.predict_result.PredictResult` object

        Raises:
            AttributeError: If the function is called without
                :class:`~decanter.core.context.Context` created

        """
        logger.debug('[Core] Create Predict Job')
        predict_res = PredictResult(predict_input=predict_input, name=name)
        try:
            if Context.LOOP is None:
                raise AttributeError('[Core] event loop is \'NoneType\'')
            task = Context.LOOP.create_task(predict_res.wait())
            Context.CORO_TASKS.append(task)
            Context.JOBS.append(predict_res)
        except AttributeError:
            logger.error('[Core] Context not created')
            raise
        return predict_res

    @staticmethod
    def predict_ts(predict_input, name=None):
        """Predict time series model with test data.

        Create a Time Series PredictResult Job and scheduled the execution
        in CORO_TASKS list.  Record the Job in JOBS list.

        Args:
            predict_input
                (:class:`~decanter.core.core_api.predict_input.PredictTSInput`):
                stores the settings for prediction.
            name (:obj:`str`, optional): name for predict time series action.

        Returns:
            :class:`~decanter.core.jobs.predict_result.PredictTSResult`
            object.

        Raises:
            AttributeError: If the function is called without
                :class:`~decanter.core.context.Context` created
        """
        logger.debug('[Core] Create Predict Job')
        predict_ts_res = PredictTSResult(
            predict_input=predict_input, name=name)
        try:
            if Context.LOOP is None:
                raise AttributeError('[Core] event loop is \'NoneType\'')
            task = Context.LOOP.create_task(predict_ts_res.wait())
            Context.CORO_TASKS.append(task)
            Context.JOBS.append(predict_ts_res)
        except AttributeError:
            logger.error('[Core] Context not created')
            raise

        return predict_ts_res
