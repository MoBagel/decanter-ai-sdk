# pylint: disable=arguments-differ
# pylint: disable=too-few-public-methods
# pylint: disable=invalid-name
"""CoreX Objects

The unit of the request body when sending corex apis. Support turning
each CoreX Object to json format.

"""
import json

from decanter.core.extra.decorators import corex_obj


class ComplexEncoder(json.JSONEncoder):
    """Extending JSONEncoder

    Define own JSON Encoder for the object having the function `jsonable`. It will
    return Dictionary object is having `jsonable`.

    Returns:
        function: `jsonable`
    """

    def default(self, obj):
        if hasattr(obj, 'jsonable'):
            return obj.jsonable()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class CoreBodyObj:
    """Base class of all CoreX Object.

    Has the jsonable function to return the Dictionary of object.

    """

    def __init__(self, **kwargs):
        """Add the argument pass in as attributes."""
        self.__dict__.update(
            (k, v) for k, v in kwargs.items() if v is not None)

    def jsonable(self):
        """Reutrn the Dictionary of Object"""
        return self.__dict__


class CVTrain(CoreBodyObj):
    """Specification for the train fold"""
    @classmethod
    @corex_obj(required={'start', 'end'})
    def create(cls, **kwargs):
        """Return CVTrain object with passed kwargs as attributes"""
        return cls(**kwargs)


class CVObject(CoreBodyObj):
    """The specifications for the cross validation folds"""
    @classmethod
    @corex_obj(required={'cvTrain', 'cvTest'})
    def create(cls, **kwargs):
        """Return CVObject object with passed kwargs as attributes"""
        return cls(**kwargs)


class Column(CoreBodyObj):
    """The column names of the data"""
    @classmethod
    @corex_obj(required={'id', 'data_type'})
    def create(cls, **kwargs):
        """Return Column object with passed kwargs as attributes"""
        return cls(**kwargs)


class TimeSeriesSplit(CoreBodyObj):
    """The specifications for the folds in time series cross validation.
    Cannot be used if nfold is specified"""
    @classmethod
    @corex_obj(required={'split_by', 'cv', 'train'})
    def create(cls, **kwargs):
        """Return TimeSeriesSplit object with passed kwargs as attributes"""
        return cls(**kwargs)


class TrainBody(CoreBodyObj):
    """The specifications for the folds in time series cross validation.
    Cannot be used if nfold is specified"""
    @classmethod
    @corex_obj(required={'target', 'train_data_id', 'algos'})
    def create(cls, **kwargs):
        """Return TrainBody object with passed kwargs as attributes"""
        return cls(**kwargs)


class ClusterTrainBody(CoreBodyObj):
    """ClusterTrainBody"""
    @classmethod
    @corex_obj(required={'train_data_id'})
    def create(cls, **kwargs):
        """Return ClusterTrainBody object with passed kwargs as attributes"""
        return cls(**kwargs)


class ModelBuildControl(CoreBodyObj):
    """Model build control"""
    @classmethod
    @corex_obj(required=None)
    def create(cls, **kwargs):
        """Return ModelBuildControl object with passed kwargs as attributes"""
        return cls(**kwargs)


class GeneticAlgorithmParams(CoreBodyObj):
    """The Genetic Algorithm parameters to be used in Auto Time Series"""
    @classmethod
    @corex_obj(required=None)
    def create(cls, **kwargs):
        """Return GeneticAlgorithmParams object with passed kwargs as
        attributes.

        Dictionary at_least specifies that the arguments passed in must
        includes at least one of the feature in the Dictionary.

        """
        at_least = {
            'max_iteration',
            'generation_size',
            'mutation_rate',
            'crossover_rate'}
        if all(kwargs[x] is None for x in at_least):
            return None
        return cls(**kwargs)


class BuildControl(CoreBodyObj):
    """Model build control."""
    @classmethod
    @corex_obj(required={'genetic_algorithm'})
    def create(cls, **kwargs):
        """Return BuildControl object with passed kwargs as attributes"""
        return cls(**kwargs)


class ModelSpec(CoreBodyObj):
    """The model specification"""
    @classmethod
    @corex_obj(required={'endogenous_features'})
    def create(cls, **kwargs):
        """Return ModelSpec object with passed kwargs as attributes"""
        return cls(**kwargs)


class TSGroupBy(CoreBodyObj):
    """Time series group by."""
    @classmethod
    @corex_obj(required=None)
    def create(cls, **kwargs):
        """Return TSGroupBy object with passed kwargs as attributes"""
        return cls(**kwargs)


class InputSpec(CoreBodyObj):
    """	The input specification"""
    @classmethod
    @corex_obj(required={'train_data_id', 'target', 'datetime_column', 'forecast_horizon', 'gap'})
    def create(cls, **kwargs):
        """Return InputSpec object with passed kwargs as attributes"""
        return cls(**kwargs)


class BuildSpec(CoreBodyObj):
    """Attribute for TrainAutoTSBody."""
    @classmethod
    @corex_obj(required={'genetic_algorithm'})
    def create(cls, **kwargs):
        """Return BuildSpec object with passed kwargs as attributes"""
        return cls(**kwargs)


class TrainAutoTSBody(CoreBodyObj):
    """Body for auto time series train api."""
    @classmethod
    @corex_obj(required={'build_spec', 'input_spec'})
    def create(cls, **kwargs):
        """Return TrainAutoTSBody object with passed kwargs as attributes"""
        return cls(**kwargs)


class PredictBody(CoreBodyObj):
    """Body for predict api."""
    @classmethod
    @corex_obj(required={'data_id', 'model_id'})
    def create(cls, **kwargs):
        """Return PredictBody object with passed kwargs as attributes"""
        return cls(**kwargs)


class PredictBodyTSModel(CoreBodyObj):
    """Body for predict time series forecast model api."""
    @classmethod
    @corex_obj(required={'data_id', 'model_id'})
    def create(cls, **kwargs):
        """Return PredictBodyTSModel object with passed kwargs as attributes"""
        return cls(**kwargs)


class SetupBody(CoreBodyObj):
    """Body for setup api."""
    @classmethod
    @corex_obj(required={'data_source', 'data_columns'})
    def create(cls, **kwargs):
        """Return SetupBody object with passed kwargs as attributes"""
        return cls(**kwargs)


class Accessor(CoreBodyObj):
    """Data accessor"""
    @classmethod
    @corex_obj(required={'uri', 'format'})
    def create(cls, **kwargs):
        """Return SetupBody object with passed kwargs as attributes"""
        return cls(**kwargs)


def column_array(cols):
    """Turn Josn columns to list of Column objects."""
    if cols is None:
        return None
    result = []
    for col in cols:
        nullable = col.get('nullable', None)
        try:
            result.append(
                Column(
                    id=col['id'], data_type=col['data_type'],
                    nullable=nullable))
        except KeyError as err:
            raise ValueError('missing required value in Column %s' % err)
    return result


def cv_obj_array(cv):
    """Turn JSON cv to list of CVObjects."""
    def build_cv_obj(cv):
        cv_objs = []
        for cv_obj in cv:
            try:
                cv_objs.append(Column(train=cv_obj['train'], test=cv_obj['test']))
            except KeyError as err:
                raise ValueError('missing required value in cv types %s' % err)
        return cv_objs

    return None if cv is None else build_cv_obj(cv)
