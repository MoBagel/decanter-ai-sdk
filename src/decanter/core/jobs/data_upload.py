# pylint: disable=C0103
# pylint: disable=too-many-instance-attributes
"""DataUpload

Handle data upload to Decanter Core server Stores Data
attribute.

"""
import io
import logging

import pandas as pd

from decanter.core.extra import CoreStatus
from decanter.core.extra.decorators import update
from decanter.core.extra.utils import check_response, gen_id
from decanter.core.jobs.job import Job
from decanter.core.jobs.task import UploadTask


logger = logging.getLogger(__name__)


class DataUpload(Job):
    """DataUpload manage to get the result from data upload.

    Handle the execution of upload task in order to upload data to
    Decanter Core server Stores the upload results in DataUpload attributes.

    Attributes:
        jobs (list): None, list up jobs that DataUpload needs to wait for.
        task (:class:`~decanter.core.jobs.task.UploadTask`): Upload task run by
            DataUpload.
        accessor (dict): Accessor for files in hdfs.
        schema (dict): The original data schema.
        originSchema (dict): The original data schema.
        annotationsMeta (dict): information: Extra information for data.
        options (dict): Extra information for data.
        created_at (str): The date the data was created.
        updated_at (str): The time the data was last updated.
        completed_at (str): The time the data was completed at.
        name (str): Name to track Job progress, will give default name if None.
    """
    def __init__(self, file=None, name=None, eda=True):
        """DataUpload Init.

        Args:
            file (file-object): DataUpload file to upload.
            name (:obj:`str`, optional): Name to track Job progress
        """
        super().__init__(jobs=None,
                         task=UploadTask(file, name, eda),
                         name=gen_id(self.__class__.__name__, name))

        self.accessor = None
        self.schema = None
        self.originSchema = None
        self.annotations = None
        self.options = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None

    @classmethod
    def create(cls, data_id, name=None):
        """Create data by data_id.

        Args:
            data_id (str): ObjectId in 24 hex digits
            name (str): (opt) Name to track Job progress

        Returns:
            :class:`~decanter.core.jobs.data_upload.DataUpload` object
        """
        data = cls()
        data_resp = check_response(
            data.core_service.get_data_by_id(data_id)).json()
        data.update_result(data_resp)
        data.status = CoreStatus.DONE
        data.name = name
        return data

    @update
    def update_result(self, task_result):
        """Update from 'result' in Task response."""
        return

    def show(self):
        """Show data content.

        Returns:
            str: Content of uploaded data.
        """
        data_txt = None
        if self.is_success():
            data_txt = check_response(
                self.core_service.get_data_file_by_id(self.id)).text
        else:
            logger.error(
                '[%s] \'%s\' show data failed',
                self.__class__.__name__, self.name)
        return data_txt

    def show_df(self):
        """Show data in pandas dataframe.

        Returns:
            :class:`pandas.DataFrame`: Content of uploaded data.
        """
        data_df = None
        if self.is_success():
            data_csv = check_response(
                self.core_service.get_data_file_by_id(self.id))
            data_csv = data_csv.content.decode('utf-8')
            data_df = pd.read_csv(io.StringIO(data_csv))
        else:
            logger.error('[%s get result] fail', self.__class__.__name__)
        return data_df

    def download_csv(self, path):
        """DownLoad csv format of the uploaded data.

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
            logger.error('[%s get result] fail', self.__class__.__name__)
