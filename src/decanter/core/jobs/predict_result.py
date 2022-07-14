# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=super-init-not-called
"""
PredictResult and PredictTSResult handle the prediction of the model training
on Decanter Core server, and stores the predict results in its attributes.
"""
import io
import logging

import pandas as pd

from decanter.core.extra.decorators import update
from decanter.core.extra.utils import check_response, gen_id
from decanter.core.jobs.job import Job
from decanter.core.jobs.task import PredictTask, PredictTSTask


logger = logging.getLogger(__name__)


class PredictResult(Job):
    """PredictResult manage to get the results from predictions.

    Handle the execution of predict task in order to predict model on
    Decanter Core server Stores the predict results in PredictResult's attributes.

    Attributes:
        jobs (list(:class:`~decanter.core.jobs.job.Job`)): List of jobs that
            PredictResult needs to wait for, [TestData, Experiment].
        task (:class:`~decanter.core.jobs.task.PredictTask`): Predict task runned by
            PredictResult Job.
        accessor (dict): Accessor for files in hdfs.
        schema (dict): The original data schema.
        originSchema (dict): The original data schema.
        annotationsMeta (dict): information: Extra information for data.
        options (dict): Extra information for data.
        created_at (str): The date the data was created.
        updated_at (str): The time the data was last updated.
        completed_at (str): The time the data was completed at.
    """
    def __init__(self, predict_input, name=None):
        super().__init__(
            jobs=[predict_input.data, predict_input.experiment],
            task=PredictTask(predict_input, name=name),
            name=gen_id(self.__class__.__name__, name))
        self.accessor = None
        self.schema = None
        self.originSchema = None
        self.annotations = None
        self.options = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None

    @update
    def update_result(self, task_result):
        """Update Job's attributes from Task's result."""
        return

    def show(self):
        """Show content of predict result.

        Returns:
            str: Content of PredictResult.
        """
        pred_txt = ''
        if self.is_success():
            pred_txt = check_response(
                self.core_service.get_data_file_by_id(self.id)).text
        else:
            logger.error('[%s] fail', self.__class__.__name__)
        return pred_txt

    def show_df(self):
        """Show predict result in pandas dataframe.

        Returns:
            :class:`pandas.DataFrame`: Content of predict result.
        """
        pred_df = None
        if self.is_success():
            pred_csv = check_response(
                self.core_service.get_data_file_by_id(self.id))
            pred_csv = pred_csv.content.decode('utf-8')
            pred_df = pd.read_csv(io.StringIO(pred_csv))
        else:
            logger.error('[%s] fail', self.__class__.__name__)
        return pred_df

    def download_csv(self, path):
        """DownLoad csv format of the predict result.

        Args:
            path (str): The path to download csv file.
        """
        if self.is_success():
            data_csv = check_response(
                self.core_service.get_data_file_by_id(self.id)).text
            save_csv = open(path, 'w+')
            save_csv.write(data_csv)
            save_csv.close()
        else:
            logger.error('[%s] Fail to Download', self.__class__.__name__)


class PredictTSResult(PredictResult, Job):
    """Predict time series's model result.

    Handle time series's model Prediction on Decanter Core server  Stores
    predict Result to attribute.

    Attributes:
        jobs (list(:class:`~decanter.core.jobs.job.Job`)): List of jobs that
            PredictTSResult needs to wait for, [TestData, Experiment].
        task (:class:`~decanter.core.jobs.task.PredictTSTask`): Predict task runned by
            PredictResult Job.
        accessor (dict): Accessor for files in hdfs.
        schema (dict): The original data schema.
        originSchema (dict): The original data schema.
        annotationsMeta (dict): information: Extra information for data.
        options (dict): Extra information for data.
        created_at (str): The date the data was created.
        updated_at (str): The time the data was last updated.
        completed_at (str): The time the data was completed at.
    """
    def __init__(self, predict_input, name=None):
        Job.__init__(
            self,
            jobs=[predict_input.data, predict_input.experiment],
            task=PredictTSTask(predict_input, name=name),
            name=gen_id(self.__class__.__name__, name))
        self.accessor = None
        self.schema = None
        self.originSchema = None
        self.annotations = None
        self.options = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None
