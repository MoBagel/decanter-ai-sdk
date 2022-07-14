# pylint: disable=C0103
"""This module defines tasks for different :class:`~decanter.core.jobs.job.Job`.

Handle the completion of tasks such as upload data, train, prediction...ect.
Return the result to Job.
"""
import abc
import logging

from functools import partial

from decanter.core import Context
from decanter.core.core_api import CoreAPI
from decanter.core.extra import CoreStatus, CoreKeys
from decanter.core.extra.utils import check_response, gen_id, isnotebook, \
    exception_handler, exception_handler_for_class_method

try:
    if isnotebook():
        raise ImportError
except ImportError:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger(__name__)


class Task:
    """Handle Action's result.

    Handle the execution of the actions (ex. upload data), the
    update of the results, and the tracking of status to determine
    the end of execution.

    Attributes:
        status (str): Status of task.
        result (value of the task result): The result of executing the task.
        name (str): Name of task for tracking process.
    """

    def __init__(self, name=None):
        self.status = CoreStatus.PENDING
        self.result = None
        self.name = name

    def is_done(self):
        """
        Return:
            bool. True for task in `DONE_STATUS`, False otherwise.
        """
        return self.status in CoreStatus.DONE_STATUS

    def not_done(self):
        """
        Return:
            bool. True for task not in `DONE_STATUS`, False otherwise.
        """
        return not self.is_done()

    def is_success(self):
        """
        Return:
            bool. True for success, False otherwise.
        """
        return self.status == CoreStatus.DONE and self.result is not None

    def is_fail(self):
        """
        Return:
            bool. True for failed, False otherwise.
        """
        return self.status in CoreStatus.FAIL_STATUS or \
            (self.status == CoreStatus.DONE and self.result is None)

    @abc.abstractmethod
    def run(self):
        """Execute task.

        Raises:
            NotImplementedError: If child class do not implement this function.
        """
        raise NotImplementedError('Please Implement run method')

    @abc.abstractmethod
    async def update(self):
        """Update attribute by response or result.

        Raises:
            NotImplementedError: If child class do not implement this function.
        """
        raise NotImplementedError('Please Implement update method')


class CoreTask(Task):
    """Handle Decanter Core Action's result

    Handle the task relate to Decanter Core server, such as upload data,
    training, prediction.

    Attributes:
        core_service (str): Status of task.
        id (str): Task identifier created by creating a task.
        response (dict): Responses from the request api.
        progress (float): Progress of the task process.
        name (str): Name of task for tracking process.
    """
    BAR_CNT = 0
    'int: The position of progress bar to avoid overlapping.'

    def __init__(self, name=None):
        super().__init__(name=name)
        self.core_service = CoreAPI()
        self.id = None
        self.response = None
        self.progress = 0
        self.pbar = None

    async def update(self):
        """Update the response from Decanter server.

        Get the task from sending api request and update the result
        of response.
        """
        func = partial(self.core_service.get_tasks_by_id, task_id=self.id)
        self.response = await Context.LOOP.run_in_executor(None, func)
        if self.status in CoreStatus.DONE_STATUS:
            return
        self.response = check_response(self.response).json()
        self.update_task_response()
        logger.debug(
            '[Task]\'%s\' done update. status: %s', self.name, self.status)

    @exception_handler_for_class_method
    def update_task_response(self):
        """Update the result from response

        Update progress, and status. Update progress bar due to the
        progress key value of response.
        """
        logger.debug(
            '[Task] \'%s\' start update task resp. status: %s',
            self.name, self.status)

        @exception_handler
        def update_pbar(resp_progress):
            diff = int((resp_progress - self.progress)*100)
            self.pbar.update(diff)
            self.progress = resp_progress

        for key_ in [CoreKeys.id, CoreKeys.progress, CoreKeys.result, CoreKeys.status]:
            attr, key = key_.name, key_.value
            try:
                if key_ == CoreKeys.progress:
                    update_pbar(self.response[key])
                setattr(self, attr, self.response[key])
            except KeyError as err:
                logger.debug(err)

    @abc.abstractmethod
    def run(self):
        """Execute Decanter Core task.
        Raises:
            NotImplementedError: If child class do not implement this function.
        """
        raise NotImplementedError('Please Implement run method in CoreTask')

    @exception_handler_for_class_method
    def run_core_task(self, api_func, **kwargs):
        """Start running Decanter Core task by calling api_func.

        Args:
            api_func (func): CoreAPI function.
            kwargs: Parameters for api_func.
        """
        logger.debug('[%s] \'%s\' start.', self.__class__.__name__, self.name)
        self.response = check_response(
            api_func(**kwargs), key=CoreKeys.id.value)
        self.response = self.response.json()
        self.id = self.response[CoreKeys.id.value]
        self.pbar = tqdm(
            total=100, position=CoreTask.BAR_CNT, leave=True,
            bar_format='{l_bar}{bar}', desc='Progress %s' % self.name)
        CoreTask.BAR_CNT += 1

        self.status = CoreStatus.RUNNING

    def stop(self):
        """Stop undone task in Decanter Core server

        Send the stop task api to stop the running or pending task.
        """
        if self.id is not None:
            check_response(self.core_service.put_tasks_stop_by_id(self.id))
        logger.info(
            '[CoreTask] Stop Task %s id:%s while %s',
            self.name, self.id, self.status)
        self.status = CoreStatus.FAIL


class UploadTask(CoreTask):
    """Upload data to Decanter Core.

    Attributes:
        file (csv-file-object): The csv file to be uploaded.
    """

    def __init__(self, file, name=None, eda=True):
        super().__init__(name=gen_id('UploadTask', name))
        self.file = file
        self.eda = eda

    def run(self):
        """Execute upload data by sending the upload api."""
        super().run_core_task(
            api_func=self.core_service.post_upload,
            filename=self.file.name,
            file=self.file,
            eda=self.eda,
            encoding='text/plain(UTF-8)')


class TrainTask(CoreTask):
    """Train model on Decanter Core.

    Attributes:
        train_input (:class:`~decanter.core.core_api.train_input.TrainInput`):
            Settings for training.
    """

    def __init__(self, train_input, name=None):
        super().__init__(name=gen_id('TrainTask', name))
        self.train_input = train_input

    def run(self):
        """Execute model training by sending the triain api."""
        train_params = self.train_input.get_train_params()
        super().run_core_task(api_func=self.core_service.post_tasks_train, **train_params)


class TrainTSTask(CoreTask):
    """Train time series forecast models on Decanter Core.

    Attributes:
        train_input (:class:`~decanter.core.core_api.train_input.TrainTSInput`):
            Settings for training time series forecast models.
    """

    def __init__(self, train_input, name=None):
        super().__init__(name=gen_id('TrainTSTask', name))
        self.train_input = train_input

    def run(self):
        """Execute time seires forecast model training by sending the auto
        time series forecast train api."""
        train_params = self.train_input.get_train_params()
        super().run_core_task(
            api_func=self.core_service.post_tasks_auto_ts_train, **train_params)


class TrainClusterTask(CoreTask):
    """Train time series forecast models on Decanter Core.

    Attributes:
        train_input (:class:`~decanter.core.core_api.train_input.TrainTSInput`):
            Settings for training time series forecast models.
    """

    def __init__(self, train_input, name=None):
        super().__init__(name=gen_id('TrainClusterTask', name))
        self.train_input = train_input

    def run(self):
        """Execute clustering training by sending the cluster triain api."""
        train_params = self.train_input.get_train_params()
        super().run_core_task(
            api_func=self.core_service.post_tasks_cluster_train, **train_params)


class PredictTask(CoreTask):
    """Predict model on Decanter Core.

    Attributes:
        predict_input (:class:`~decanter.core.core_api.predict_input.PredictInput`):
            Settings for prediction.
    """

    def __init__(self, predict_input, name=None):
        super().__init__(name=gen_id('PredictTask', name))
        self.predict_input = predict_input

    def run(self):
        """Execute predict model training by sending the predict api."""
        pred_params = self.predict_input.getPredictParams()
        super().run_core_task(
            api_func=self.core_service.post_tasks_predict,
            **pred_params)


class PredictTSTask(CoreTask):
    """Predict time series model on Decanter Core.

    Attributes:
        predict_input \
            (:class:`~decanter.core.core_api.predict_input.PredictTSInput`):
            Settings for time series prediction.
    """

    def __init__(self, predict_input, name=None):
        super().__init__(name=gen_id('PredictTSTask', name))
        self.predict_input = predict_input

    def run(self):
        """Execute time series models prediction by sending the time
        series predict api."""
        pred_params = self.predict_input.getPredictParams()
        super().run_core_task(
            api_func=self.core_service.post_tasks_auto_ts_predict,
            **pred_params)


class SetupTask(CoreTask):
    """V0 version is for normal use, changing columns type. V2 version is for
    V2 eda use, setting v2 eda result for the data, preparing for custom eda.

    Attributes:
        setup_input (:class:`~decanter.core.core_api.setup_input.SetupInput`):
            Settings for set up data.
    """

    def __init__(self, setup_input, name='Setup'):
        super().__init__(name=name)
        self.setup_input = setup_input

    def run(self):
        """
        Execute setup data by sending the setup api.
        """
        setup_params = self.setup_input.get_setup_params()
        super().run_core_task(
            api_func=self.core_service.post_tasks_setup,
            **setup_params)
