# pylint: disable=C0103
# pylint: disable=too-many-instance-attributes
# pylint: disable=super-init-not-called
"""Model and Multimodel.

This module handles actions relate to models. Storing Attributes of
model metadata from decanter.core server, also support model downloading from
server or uploading form local zip fils.
"""
import logging

from decanter.core.core_api import CoreAPI
from decanter.core.extra import CoreStatus, CoreKeys
from decanter.core.extra.decorators import block_method
from decanter.core.extra.utils import check_response


logger = logging.getLogger(__name__)


class Model:
    """Model from training.

    Attributes:
        get_model: Function to get models metadata.
        download_model: Function to get model mojo file.
        task_status (str): Status of training task.
        id (str): ObjectId in 24 hex digits.
        key (str): The unique key for the model.
        name (str): The name of the model (may not be unique across projects).
        exp_id (str): The experiment ID of the model.
        importances (dict): The feature importance.
        attributes (dict): Model related scores and feature importance.
        hyperparameters (dict): The model's hyperparameters.
        created_at (str): The time the model was created at.
        updated_at (str): The time the model was last updated.
        completed_at (str): The time the model was completed_at.
    """
    def __init__(self):
        self.get_model = CoreAPI().get_models_by_id
        self.download_model = CoreAPI().get_models_download_by_id
        self.task_status = CoreStatus.PENDING
        self.id = None
        self.key = None
        self.name = None
        self.exp_id = None
        self.importances = None
        self.attributes = None
        self.hyperparameters = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None

    @classmethod
    def create(cls, exp_id, model_id):
        """Create :class:`Model` or :class:`MultiModel` depends on which
        instance type has called.

        Args:
            exp_id (str): The experiment ID of the model.
            model_id (str): ObjectId in 24 hex digits.

        Returns:
            :class:`~decanter.core.core_api.model.Model` or :class:`~decanter.\
            core.core_api.mode.MultiModel` object.

        Raises:
            AttributeError: Occurred when getting no results from decanter
                server when getting model's metadata.
        """
        model = cls()
        get_models_resp = check_response(
            model.get_model(exp_id, model_id)).json()
        try:
            for attr, val in get_models_resp.items():
                if attr == CoreKeys.id.value:
                    attr = CoreKeys.id.name
                model.__dict__.update({attr: val})
        except AttributeError:
            logger.error('[Model] No result from decanter.core.')

        model.task_status = CoreStatus.DONE
        return model

    def update(self, exp_id, model_id):
        """Update model attributes.

        Get and Set attributes from the response attribute from
        decanter server.
        """
        get_models_resp = check_response(
            self.get_model(exp_id, model_id)).json()
        try:
            for attr, val in get_models_resp.items():
                if attr == CoreKeys.id.value:
                    attr = CoreKeys.id.name
                self.__dict__.update({attr: val})
        except AttributeError:
            logger.error('[%s] No result from decanter.core.', self.__class__.__name__)

        self.task_status = CoreStatus.DONE

    def is_done(self):
        """Check if training task is done.

        Returns:
            bool: True if :class:`Model`'s task is done, else False.
        """
        return self.task_status in CoreStatus.DONE_STATUS

    @block_method
    def get(self, attr):
        """Get attribute of model.

        Args:
            attr (str): model's attribute.

        Returns:
            Value of the attribute of the given object.
        """
        return getattr(self, attr)

    @classmethod
    def download_by_id(cls, model_id, model_path):
        """Download model file to local.

        Getting Mojo model zip file from decanter.core server and
        download to local.

        Args:
            model_id (str): ObjectId in 24 hex digits.
            model_path (str): Path to store zip mojo file.
        """
        resp = check_response(
            cls().download_model(model_id))
        zfile = open(model_path, 'wb')
        zfile.write(resp.content)
        zfile.close()

    def download(self, model_path):
        """Download model file to local.

        Download the trained mojo model from Model instance to
        local in the format of zip file.

        Args:
            model_path (str): Path to store zip mojo file.
        """
        if self.id is None:
            logger.info('[Model] %s model id is NoneType', self.name)
        resp = check_response(
            self.download_model(model_id=self.id))
        zfile = open(model_path, 'wb')
        zfile.write(resp.content)
        zfile.close()


class MultiModel(Model):
    """MultiModel from time series training.

    Attributes:
        get_model: Get multi models metadata.
        download_model: Get model mojo file.
        task_status (str): Status of training task.
        id (str): ObjectId in 24 hex digits.
        name (str): The name of the model (may not be unique across projects).
        exp_id (str): The experiment ID of the model.
        attributes (dict): Model related scores and feature importance.
        predictionPipeline (dict): TODO
        hyperparameters (dict): TODO
        created_at (str): The time the model was created at.
        updated_at (str): The time the model was last updated.
        completed_at (str): The time the model was completed_at.
    """
    def __init__(self):
        self.get_model = CoreAPI().get_multimodels_by_id
        self.download_model = CoreAPI().get_models_download_by_id
        self.task_status = CoreStatus.PENDING
        self.id = None
        self.name = None
        self.exp_id = None
        self.attributes = None
        self.predictionPipeline = None
        self.hyperparameters = None
        self.created_at = None
        self.updated_at = None
        self.completed_at = None

    @classmethod
    def create(cls, exp_id, model_id):
        """Create Multimodel. Inherit from :func:`Model.create`."""
        return super(MultiModel, cls).create(exp_id=exp_id, model_id=model_id)

    @classmethod
    def download_by_id(cls, model_id, model_path):
        """MultiModel has no download method"""
        logger.info('MultiModel doesn\'t support download in local.')

    def download(self, model_path):
        """MultiModel has no download method"""
        logger.info('MultiModel doesn\'t support download in local.')
