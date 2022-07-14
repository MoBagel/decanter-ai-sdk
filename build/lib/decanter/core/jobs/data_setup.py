# pylint: disable=C0103
# pylint: disable=too-many-instance-attributes
"""DataSetup

Handle data setup to Decanter Core server Stores Data
attribute.

"""
import io
import logging

import pandas as pd

from decanter.core.extra import CoreStatus
from decanter.core.extra.decorators import update
from decanter.core.extra.utils import check_response, gen_id
from decanter.core.jobs.job import Job
from decanter.core.jobs.task import SetupTask


logger = logging.getLogger(__name__)


class DataSetup(Job):
    """DataSetup manage to get the result from data setup.

    Handle the execution of setup task in order to setup data to
    Decanter Core server Stores the setup results in DataSetup attributes.

    Attributes:
        jobs (list): None, list up jobs that DataSetup needs to wait for.
        task (:class:`~decanter.core.jobs.task.SetupTask`): Setup task run by
            DataSetup.
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
    def __init__(self, setup_input=None, name=None):
        """DataSetup Init.

        Args:
            setup_input (:obj:`~decanter.core.jobs.data_setup.DataSetup)
            name (:obj:`str`, optional): Name to track Job progress
        """
        super().__init__(
            jobs=[setup_input.data],
            task=SetupTask(setup_input, name),
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
            :class:`~decanter.core.jobs.data_setup.DataSetup` object
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
            str: Content of setup data.
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
            :class:`pandas.DataFrame`: Content of setup data.
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
        """DownLoad csv format of the setup data.

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
