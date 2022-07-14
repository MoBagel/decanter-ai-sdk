# pylint: disable=too-many-instance-attributes
# pylint: disable=super-init-not-called
"""
Experiment and ExperimentTS handles the training of models on Decanter Core server,
and stores Experiment results in its attributes.
"""
import logging
import numpy as np
from decanter.core.core_api import CoreAPI, Model, MultiModel
from decanter.core.extra import CoreStatus
from decanter.core.extra.decorators import update
from decanter.core.extra.utils import check_response, gen_id
from decanter.core.jobs.job import Job
from decanter.core.jobs.task import TrainTask, TrainTSTask, TrainClusterTask
from decanter.core.enums.evaluators import Evaluator
from decanter.core.enums import check_is_enum

logger = logging.getLogger(__name__)


class Experiment(Job):
    """Experiment manage to get the results from model training.

    Handle the execution of training task in order to train model on
    Decanter Core server Stores the training results in Experiment's attributes.

    Attributes:
        jobs (list(:class:`~decanter.core.jobs.job.Job`)): [DataUpload]. List of jobs
            that Experiment needs to wait till completed.
        task(:class:`~decanter.core.jobs.task.TrainTask`): Train task runned by
            Experiment Job.
        train_input(:class:`~decanter.core.core_api.train_input.TrainInput`):
            Settings for training models.
        best_model(:class:`~decanter.core.core_api.model.Model`): Model with the best score in
            `select_model_by` argument.
        select_model_by (str): The score to select best model.
        features (list(str)): The features used for training.
        train_data_id (str): The ID of the train data.
        target (str): The target of the experiment.
        test_base_id (str): The ID for the test base data.
        models (list(str)): The models' id of the experiment.
        hyperparameters (dict): The hyperparameters of the experiment.
        attributes (dict): The experiment attributes.
        recommendations (dict): Recommended model for each evaluator.
        created_at (str): The date the data was created.
        options (dict): Extra information for experiment.
        updated_at (str): The time the data was last updated.
        completed_at (str): The time the data was completed at.
        name (str): Name to track Job progress.
    """

    def __init__(self, train_input, select_model_by=Evaluator.auto, name=None):
        super().__init__(
            jobs=[train_input.data],
            task=TrainTask(train_input, name=name),
            name=gen_id(self.__class__.__name__, name))

        select_model_by = check_is_enum(Evaluator, select_model_by)
        self.train_input = train_input
        self.best_model = Model()
        self.select_model_by = select_model_by
        self.features = None
        self.train_data_id = None
        self.target = None
        self.test_base_id = None
        self.models = None
        self.hyperparameters = None
        self.attributes = None
        self.recommendations = None
        self.options = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None

    @classmethod
    def create(cls, exp_id, name=None):
        """Create Experiment by exp_id.

        Args:
            exp_id (str): ObjectId in 24 hex digits.
            name (:obj:`str`, optional): Name to track Job progress.

        Returns:
            :class:`~decanter.core.jobs.experiment.Experiment`: Experiment object
                with the specific id.
        """
        core_service = CoreAPI()
        exp_resp = check_response(
            core_service.get_experiments_by_id(exp_id)).json()
        exp = cls(train_input=None)
        exp.update_result(exp_resp)
        exp.status = CoreStatus.DONE
        exp.name = name
        return exp

    @update
    def update_result(self, task_result):
        """Update Job's attribute from Task's result."""
        self.get_best_model()

    def get_best_model(self):
        """Get the best model in experiment by `select_model_by` and stores
        in best model attribute."""
        if not self.task.is_success():
            return
        class_ = self.__class__.__name__
        logger.debug('[%s] \'%s\' get best model', class_, self.name)
        select_by_evaluator = Evaluator.resolve_select_model_by(
            self.select_model_by, self.hyperparameters['model_type'])
        minlevel = {Evaluator.mse.value, Evaluator.mae.value, Evaluator.mean_per_class_error.value,
                    Evaluator.deviance.value, Evaluator.logloss.value, Evaluator.rmse.value,
                    Evaluator.rmsle.value, Evaluator.misclassification.value,
                    Evaluator.mape.value, Evaluator.wmape.value}

        # Get the best model among models with valid score
        model_list = list(filter(lambda x: not np.isnan(
            x['cv_averages'][select_by_evaluator]), self.attributes.values()))
        best_model_id = None
        try:
            if select_by_evaluator in minlevel:
                best_model_id = min(
                    model_list,
                    key=lambda x: x['cv_averages'][select_by_evaluator])['model_id']
            else:
                best_model_id = max(
                    model_list,
                    key=lambda x: x['cv_averages'][select_by_evaluator])['model_id']
        except AttributeError:
            logger.error('[%s] no models in %s result', class_, self.name)
        except KeyError as err:
            logger.error(err)

        if best_model_id is not None:
            self.best_model.update(self.id, best_model_id)
            self.best_model.task_status = self.task.status
            logger.debug(
                '[%s] \'%s\' best model id: %s',
                class_, self.name, best_model_id)
        else:
            logger.error(
                '[%s] fail to get best model', class_)


class ExperimentTS(Experiment, Job):
    """ExperimentTS manage to get the result from time series model training.

    Handle the execution of time series training task in order train
    time series model on Decanter Core server Stores the training results in
    ExperimentTS's attributes.

    Attributes:
        jobs (list(:class:`~decanter.core.jobs.job.Job`)): [DataUpload]. List of jobs
            that ExperimentTS needs to wait till completed.
        task (:class:`~decanter.core.jobs.task.TrainTSTask`):
            Time series training task run by ExperimentTS Job.
        train_input (:class:`~decanter.core.core_api.train_input.TrainTSInput`):
            Settings for time series training models.
        best_model (:class:`~decanter.core.core_api.model.MultiModel`): MultiModel with the
            best score in `select_model_by` argument
        select_model_by (str): The score to select best model
        features (list(str)): The features used for training
        train_data_id (str): The ID of the train data
        target (str): The target of the experiment
        test_base_id (str): The ID for the test base data
        models (list(str)): The models' id of the experiment
        hyperparameters (dict): The hyperparameters of the experiment.
        attributes (dict): The experiment attributes.
        recommendations (dict): Recommended model for each evaluator.
        created_at (str): The date the data was created.
        options (dict): Extra information for experiment.
        updated_at (str): The time the data was last updated.
        completed_at (str): The time the data was completed at.
        name (str): Name to track Job progress.
    """

    def __init__(self, train_input, select_model_by=Evaluator.auto, name=None):
        Job.__init__(
            self,
            jobs=[train_input.data],
            task=TrainTSTask(train_input, name=name),
            name=gen_id(self.__class__.__name__, name))

        select_model_by = check_is_enum(Evaluator, select_model_by)
        self.train_input = train_input
        self.best_model = MultiModel()
        self.select_model_by = select_model_by
        self.features = None
        self.train_data_id = None
        self.target = None
        self.test_base_id = None
        self.models = None
        self.hyperparameters = None
        self.attributes = None
        self.recommendations = None
        self.options = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None

    @classmethod
    def create(cls, exp_id, name=None):
        """Create Time series Experiment by exp_id. Inherit from
        :func:`~Experiment.create`

        Args:
            exp_id (str): ObjectId in 24 hex digits
            name (:obj:`str`, optional): (opt) Name to track Job progress

        Returns:
            :class:`~decanter.core.jobs.experiment.ExperimentTS`: Experiment object\
                with the specific id.
        """
        return super(ExperimentTS, cls).create(exp_id=exp_id, name=name)


class ExperimentCluster(Experiment, Job):
    """ExperimentTS manage to get the result from clustering model training.

    Handle the execution of clustering training task in order train
    clustering model on Decanter Core server Stores the training results in
    ExperimentCluster's attributes.

    Attributes:
        jobs (list(:class:`~decanter.core.jobs.job.Job`)): [DataUpload]. List of jobs
            that ExperimentTS needs to wait till completed.
        task (:class:`~decanter.core.jobs.task.TrainTSTask`):
            Time series training task run by ExperimentTS Job.
        train_input (:class:`~decanter.core.core_api.train_input.TrainTSInput`):
            Settings for time series training models.
        best_model (:class:`~decanter.core.core_api.model.MultiModel`): MultiModel with the
            best score in `select_model_by` argument
        select_model_by (str): The score to select best model
        features (list(str)): The features used for training
        train_data_id (str): The ID of the train data
        target (str): The target of the experiment
        test_base_id (str): The ID for the test base data
        models (list(str)): The models' id of the experiment
        hyperparameters (dict): The hyperparameters of the experiment.
        attributes (dict): The experiment attributes.
        recommendations (dict): Recommended model for each evaluator.
        created_at (str): The date the data was created.
        options (dict): Extra information for experiment.
        updated_at (str): The time the data was last updated.
        completed_at (str): The time the data was completed at.
        name (str): Name to track Job progress.
    """

    def __init__(self, train_input, select_model_by=Evaluator.auto, name=None):
        Job.__init__(
            self,
            jobs=[train_input.data],
            task=TrainClusterTask(train_input, name=name),
            name=gen_id(self.__class__.__name__, name))

        select_model_by = check_is_enum(Evaluator, select_model_by)
        self.train_input = train_input
        self.best_model = Model()
        self.select_model_by = select_model_by
        self.features = None
        self.train_data_id = None
        self.target = None
        self.test_base_id = None
        self.models = None
        self.hyperparameters = None
        self.attributes = None
        self.recommendations = None
        self.options = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None

    @classmethod
    def create(cls, exp_id, name=None):
        """Create Clustering Experiment by exp_id. Inherit from
        :func:`~Experiment.create`

        Args:
            exp_id (str): ObjectId in 24 hex digits
            name (:obj:`str`, optional): (opt) Name to track Job progress

        Returns:
            :class:`~decanter.core.jobs.experiment.ExperimentCluster`: Experiment object\
                with the specific id.
        """
        return super(ExperimentCluster, cls).create(exp_id=exp_id, name=name)
