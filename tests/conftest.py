# pylint: disable=redefined-builtin
"""Config for pytest

Define pytest fixtures and variables.
"""
import pandas as pd
import pytest
import responses

from decanter.core import Context, CoreClient
from decanter.core.core_api import Model, MultiModel,\
    TrainInput, TrainTSInput,\
    PredictInput, PredictTSInput
from decanter.core.extra import CoreStatus
from decanter.core.jobs import DataUpload, Experiment, ExperimentTS


USR = 'test-usr'
PWD = 'test-pwd'
HOST = 'http://mobagel.test'
TEST_HEALTHY_URL = 'http://mobagel.test/data/test'
UPLOAD_ID = '4uploadid'
TRAIN_ID = '4traintaskid'
PREDICT_ID = '4predtaskid'
DATA_ID = '4dataid'
EXP_ID = '4expid'
PREDRES_ID = '4resid'
MODEL_ID = '4modelid'

csv_file = open('./tests/data/test.csv', 'rb')
txt_file = open('./tests/data/test.txt', 'rb')
df_file = pd.read_csv('./tests/data/test.csv')

core_results = {
    'upload': {'_id': DATA_ID},
    'train': {'_id': EXP_ID},
    'train_ts': {'_id': EXP_ID},
    'predict': {'_id': PREDRES_ID},
    'predict_ts': {'_id': PREDRES_ID}
}

data = DataUpload()
data.id, data.status, data.result = DATA_ID, CoreStatus.DONE, core_results['upload']

fail_data = DataUpload()
fail_data.status = CoreStatus.FAIL

best_model = Model()
best_multi_model = MultiModel()
best_multi_model.id = best_model.id = MODEL_ID

train_input = TrainInput(data=data, target='test-target', algos=['test-algo'])
train_ts_input = TrainTSInput(
    data=data, target='test', datetime_column='test',
    endogenous_features=['test'], forecast_horizon='test', gap='test',
    time_unit='test', regression_method='test', classification_method='test',
    max_window_for_feature_derivation='test')


exp = Experiment(train_input)
exp.id, exp.status, exp.result = EXP_ID, CoreStatus.DONE, core_results['train']
exp.best_model = best_model

fail_exp = Experiment(train_input)
fail_exp.status = CoreStatus.FAIL

exp_ts = ExperimentTS(train_ts_input)
exp_ts.id, exp_ts.status, exp_ts.result = EXP_ID, CoreStatus.DONE, core_results['train_ts']
exp_ts.best_model = best_multi_model

fail_exp_ts = ExperimentTS(train_ts_input)
fail_exp_ts.status = CoreStatus.FAIL

predict_input = PredictInput(data=data, experiment=exp)
predict_ts_input = PredictTSInput(data=data, experiment=exp_ts)

global_data = {
    'upload': UPLOAD_ID,
    'train': TRAIN_ID,
    'train_ts': TRAIN_ID,
    'predict': PREDICT_ID,
    'predict_ts': PREDICT_ID,
    'data': DATA_ID,
    'exp': EXP_ID,
    'predict_res': PREDRES_ID,
    'model': MODEL_ID,
    'test_csv_file': csv_file,
    'test_txt_file': txt_file,
    'test_df': df_file,
    'fine_data': data,
    'fail_data': fail_data,
    'fine_exp': exp,
    'fail_exp': fail_exp,
    'fine_exp_ts': exp_ts,
    'fail_exp_ts': fail_exp_ts,
    'train_input': train_input,
    'train_ts_input': train_ts_input,
    'predict_input': predict_input,
    'predict_ts_input': predict_ts_input,
    'results': core_results
}


def get_urls(action, task=None):
    """Return complete urls."""
    endpoints = {
        'upload': lambda task: '/v2/upload',
        'train': lambda task: '/v2/tasks/train',
        'predict': lambda task: '/v2/tasks/predict',
        'train_ts': lambda task: '/v2/tasks/train_time_series',
        'predict_ts': lambda task: '/v2/tasks/predict/tsmodel',
        'stop': lambda task: '/tasks/' + global_data[task] + '/stop',
        'task': lambda task: '/tasks/' + global_data[task]
    }
    return HOST + endpoints[action](task)


@pytest.fixture(scope='session')
def client():
    """Return class:`CoreClient <CoreClient>` for unittests."""
    return CoreClient()


@pytest.fixture(scope='session')
def globals():
    """Return dictionary for unittest in use of global data."""
    return global_data


@pytest.fixture(scope='session')
def urls():
    """Return urls for unittest to mock apis responses."""
    def url(action, task=None):
        return get_urls(action, task)
    return url


@pytest.fixture(autouse=True)
def mock_test_responses():
    """Mock the responses for api responses of running and getting tasks."""
    def mocked_responses(task, status=None, task_result='default'):
        if task_result == 'default':
            task_result = core_results[task]

        with responses.RequestsMock() as rsps:
            responses.add(
                responses.POST, get_urls(task),
                json={
                    '_id': global_data[task]
                },
                status=200,
                content_type='application/json')
            responses.add(
                responses.GET, get_urls('task', task),
                json={
                    '_id': global_data[task],
                    'status': status,
                    'result': task_result
                },
                status=200,
                content_type='application/json')
            return rsps
    yield mocked_responses


@pytest.fixture
def context_fixture():
    """Create and close context before and after unittest.

    Mock different conditions when creating context.
    """
    context_ = []

    def mock_request_body(cond):
        return {
            'Healthy':
                {'status': 400},
            'AuthFailed':
                {'json': {}, 'status': 405},
            'RequestException':
                {'json': {}, 'status': 405}
        }.get(cond, 'error')

    def mocked_context(cond):
        params = mock_request_body(cond)
        responses.add(
            responses.DELETE, TEST_HEALTHY_URL, **params)
        context = Context.create(username=USR, password=PWD, host=HOST)
        context_.append(context)
        return context

    yield mocked_context

    if len(context_) > 0:
        context_[0].close()
