# pylint: disable=redefined-builtin
# pylint: disable=too-many-arguments
"""Test related method and functionality of Experiemt."""
import asyncio

import responses
import pytest

from decanter.core import Context
from decanter.core.core_api import TrainInput, TrainTSInput,\
                                    PredictInput, PredictTSInput
from decanter.core.extra import CoreStatus

fail_conds = [(stat, res) for stat in CoreStatus.FAIL_STATUS for res in [None, 'result']]
fail_conds.append((CoreStatus.DONE, None))


def get_train_ts_input(data):
    """Return different train_input with different data"""
    return TrainTSInput(
                data=data, target='test',
                datetime_column='test',
                endogenous_features=['test'],
                forecast_horizon='test',
                gap='test', time_unit='test',
                regression_method='test',
                classification_method='test',
                max_window_for_feature_derivation='test')


@responses.activate
def test_exp_success(
        globals, client, mock_test_responses, context_fixture):
    """Experiment gets the result and id from CoreClient.train()"""
    context = context_fixture('Healthy')
    mock_test_responses(task='train', status=CoreStatus.DONE)

    exp = client.train(globals['train_input'])
    context.run()

    assert exp.task.id == globals['train']
    assert exp.id == globals['exp']
    assert exp.status == CoreStatus.DONE
    assert exp.result == globals['results']['train']


@responses.activate
@pytest.mark.parametrize('status, result', fail_conds)
def test_exp_fail(
        globals, client, status, result, mock_test_responses, context_fixture):
    """Experiment fails when status and result create fail conditions."""
    context = context_fixture('Healthy')
    mock_test_responses(task='train', status=status, task_result=result)

    exp = client.train(globals['train_input'])
    context.run()

    assert exp.task.id == globals['train']
    assert exp.id is None
    assert exp.status == status
    assert exp.result == result


@responses.activate
def test_exp_fail_by_data(
        globals, client, context_fixture):
    """Experiment fails if required data fails."""
    context = context_fixture('Healthy')
    exp = client.train(
        train_input=TrainInput(
            data=globals['fail_data'], target='test-target',
            algos=['test-algo']))

    context.run()

    assert exp.task.id is None
    assert exp.id is None
    assert exp.status == CoreStatus.FAIL
    assert exp.result is None


@responses.activate
@pytest.mark.parametrize('status', [CoreStatus.PENDING, CoreStatus.RUNNING, CoreStatus.FAIL])
def test_exp_stop(
        globals, urls, client, status, mock_test_responses, context_fixture):
    """Experiment status is fail if stopped during pending, running, and fail
    status, remains if in done status. The prediction following will failed
    if Experiment failed.
    """
    async def cancel(exp):
        await asyncio.sleep(1)
        exp.stop()
        return

    context = context_fixture('Healthy')
    mock_test_responses(task='train', status=status)
    responses.add(
        responses.PUT, urls('stop', 'train'),
        json={
            'message': 'task removed'
        },
        status=200,
        content_type='application/json')

    exp = client.train(globals['train_input'])

    pred_res = client.predict(
        predict_input=PredictInput(data=globals['fine_data'], experiment=exp))
    cancel_task = Context.LOOP.create_task(cancel(exp))
    Context.CORO_TASKS.append(cancel_task)
    context.run()

    assert exp.status == CoreStatus.FAIL
    assert pred_res.status == CoreStatus.FAIL


@responses.activate
def test_exp_ts_success(
        globals, client, mock_test_responses, context_fixture):
    """Time series Experiment getting the result and id
    from CoreClient.train_ts()"""
    context = context_fixture('Healthy')
    mock_test_responses(task='train_ts', status=CoreStatus.DONE)

    exp = client.train_ts(globals['train_ts_input'])
    context.run()

    assert exp.task.id == globals['train']
    assert exp.id == globals['exp']
    assert exp.status == CoreStatus.DONE
    assert exp.result == globals['results']['train']


@responses.activate
@pytest.mark.parametrize('status, result', fail_conds)
def test_exp_ts_fail(
        globals, client, status, result, mock_test_responses, context_fixture):
    """Time series Experiment fails when status and result create fail
    conditions."""
    context = context_fixture('Healthy')
    mock_test_responses(task='train_ts', status=status, task_result=result)

    exp = client.train_ts(train_input=globals['train_ts_input'])
    context.run()

    assert exp.task.id == globals['train']
    assert exp.id is None
    assert exp.status == status
    assert exp.result == result


@responses.activate
def test_exp_ts_fail_by_data(
        globals, client, context_fixture):
    """Time series Experiment fails if required data fails."""
    context = context_fixture('Healthy')

    train_ts_input = get_train_ts_input(globals['fail_data'])
    exp = client.train_ts(train_ts_input)
    context.run()
    assert exp.task.id is None
    assert exp.id is None
    assert exp.status == CoreStatus.FAIL
    assert exp.result is None


@responses.activate
@pytest.mark.parametrize('status', [CoreStatus.PENDING, CoreStatus.RUNNING, CoreStatus.FAIL])
def test_exp_ts_stop(
        globals, urls, client, status, mock_test_responses, context_fixture):
    """Time series Experiment status fails if stopped during pending, running,
    and fail status, remains if in done status. The prediction following will
    fail if Time series Experiment fails.
    """
    async def cancel(exp):
        await asyncio.sleep(1)
        exp.stop()
        return

    context = context_fixture('Healthy')
    mock_test_responses(task='train_ts', status=status)
    responses.add(
        responses.PUT, urls('stop', 'train'),
        json={
            'message': 'task removed'
        },
        status=200,
        content_type='application/json')

    exp = client.train_ts(globals['train_ts_input'])

    pred_res = client.predict_ts(
        predict_input=PredictTSInput(
            data=globals['fine_data'], experiment=exp))
    cancel_task = Context.LOOP.create_task(cancel(exp))
    Context.CORO_TASKS.append(cancel_task)
    context.run()

    assert exp.status == CoreStatus.FAIL
    assert pred_res.status == CoreStatus.FAIL
