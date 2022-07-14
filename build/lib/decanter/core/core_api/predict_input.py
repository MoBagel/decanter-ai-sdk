# pylint: disable=C0103
# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods
"""Settings for the PredictResult and PredictTSResult"""
import json

import logging
from decanter.core.core_api import CoreBody
from decanter.core.enums.evaluators import Evaluator
from decanter.core.enums import check_is_enum

logger = logging.getLogger(__name__)


class PredictInput:
    """Predict Input for PredictResult Job.

    Setting test data, best model, and the request body for prediction.

    Args:
        data (:class:`~decanter.core.jobs.data_upload.DataUpload`): Test data.
        experiment (:class:`~decanter.core.jobs.experiment.Experiment`):
            Experiment from training.
        select_model (str, optional): Methods of screening models

            - `best` (default): predict with the best model scored on cv average
            - `model_id`: predict with the model designated by model_id
            - `recommendation`: predict with the recommended model

        select_opt (str, optional): Based on the options required by the select_model.
            value with select_model case:

            - `best`: None
            - `model_id`: given the model ID (model ID will be in the format of ObjectID)
            - `recommendation`: metric, ex: auc ...

        keep_columns (:obj:`list`, optional): The names of the columns
            that will be appended to the prediction data.
        threshold (:obj:`double`, optional): Prediction threshold for
            binary classification models. Max = 1, Min = 0

    Examples:
        .. code-block:: python

            predict_input = PredictInput(
                data=test_data,
                experiment=exp,
                threshold=0.9,
                keep_columns=['col_1', 'col_3']
            )

    """
    def __init__(
            self, data, experiment, select_model='best', select_opt=None, callback=None,
            keep_columns=None, threshold=None, version=None):
        self.data = data
        self.experiment = experiment
        self.pred_body = CoreBody.PredictBody.create(
                data_id='tmp_data_id', model_id='tmp_model_id', callback=callback,
                keep_columns=keep_columns, threshold=threshold, version=version)
        self.select_model = select_model
        self.select_opt = select_opt

    def getPredictParams(self):
        """Using pred_body to create the JSON request body for prediction.

        Returns:
            :obj:`dict`
        """
        if self.select_model == 'best':
            select_model_id = self.experiment.best_model.id
        elif self.select_model == 'model_id':
            if self.select_opt in self.experiment.models:
                select_model_id = self.select_opt
            else:
                logger.error('[%s] Invalid input model ID: %s',
                                self.__class__.__name__, self.select_opt)
                raise ValueError('Invalid input model ID: %s' %self.select_opt)
        elif self.select_model == 'recommendation':
            self.select_opt = check_is_enum(Evaluator, self.select_opt)
            for rec in self.experiment.recommendations:
                if self.select_opt == rec['evaluator']:
                   select_model_id = rec['model_id']
            if 'select_model_id' not in locals().keys():
                logger.error('[%s] Invalid input metric: %s',
                                self.__class__.__name__, self.select_opt)
                raise ValueError('Invalid input metric: %s' %self.select_opt)
        setattr(self.pred_body, 'data_id', self.data.id)
        setattr(self.pred_body, 'model_id', select_model_id)

        params = json.dumps(
            self.pred_body.jsonable(), cls=CoreBody.ComplexEncoder)
        params = json.loads(params)
        return params

class PredictTSInput(PredictInput):
    """Time series predict input for  PredictTSResult Job.

    Setting test data, best model, and the request body for prediction.

    Args:
        data (:class:`~decanter.core.jobs.data_upload.DataUpload`): Test data.
        experiment (:class:`~decanter.core.jobs.experiment.Experiment`):
            Experiment from training.
        threshold_max_by (float, optional): Prediction threshold for
            binary classification only, can choose a threshold that
            optimizes the given metric.Available options are:precision, f1,
            accuracy, evaluator, recall, specificity, f2, f0point5, mean_per_class_error
    """
    def __init__(self, data, experiment, callback=None, threshold_max_by=None, version=None):
        super().__init__(data=data, experiment=experiment)

        self.pred_body = CoreBody.PredictBodyTSModel.create(
            data_id='tmp_data_id', model_id='tmp_model_id', callback=callback,
            threshold_max_by=threshold_max_by, version=version)
