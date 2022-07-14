import json
from typing import NamedTuple

from decanter.core.core_api import CoreBody


class SetupInput:
    """Setup Input for Experiment Job.

    Settings for model training.

    Args:
        data (:class:`~decanter.core.jobs.data_upload.DataUpload`):
            Train data uploaded on Decanter Core server
        data_columns (list of :class:`ColumnSpec`):
            Request body for sending setup api.
        eda (bool): Whether to perform eda on data setup

    Example:
        .. code-block:: python

            setup_input = SetupInput(
                data = upload_data,
                data_source=upload_data.accessor,
                data_columns=[{ 'id': 'Pclass', 'data_type': 'categorical'}]
            )
    """
    def __init__(
            self, data, data_columns, callback=None,
            eda=None, preprocessing=None, version=None):

        self.data = data

        tmp_accessor = CoreBody.Accessor.create(uri='tmp_uri', format='csv')
        dataColumns = CoreBody.column_array(data_columns)
        self.setup_body = CoreBody.SetupBody.create(
            data_source=tmp_accessor,
            data_id='tmp_data_id',
            callback=callback,
            eda=eda,
            data_columns=dataColumns,
            preprocessing=preprocessing,
            version=version)

    def get_setup_params(self):
        """Using setup_body to create the JSON request body for setting data.

        Returns:
            :obj:`dict`
        """
        setattr(self.setup_body, 'data_id', self.data.id)
        setattr(self.setup_body, 'data_source', self.data.accessor)
        params = json.dumps(
            self.setup_body.jsonable(), cls=CoreBody.ComplexEncoder)
        params = json.loads(params)
        return params


class ColumnSpec(NamedTuple):
    id: str
    data_type: str


