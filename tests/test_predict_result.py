# pylint: disable=redefined-builtin
# pylint: disable=too-many-arguments
"""Test related method and functionality of PredictResult."""
import asyncio

import pytest
import responses

from decanter.core import Context
from decanter.core.core_api import PredictInput, PredictTSInput
from decanter.core.extra import CoreStatus

fail_conds = [(stat, res) for stat in CoreStatus.FAIL_STATUS for res in [None, 'result']]
fail_conds.append((CoreStatus.DONE, None))


@responses.activate
def test_pred_res_success(globals, client, mock_test_responses, context_fixture):
    """PredictResult gets the result and id from CoreClient.predict()"""
    context = context_fixture('Healthy')
    mock_test_responses(task='predict', status=CoreStatus.DONE)

    pred_res = client.predict(globals['predict_input'])
    context.run()

    assert pred_res.task.id == globals['predict']
    assert pred_res.id == globals['predict_res']
    assert pred_res.result == globals['results']['predict']
    assert pred_res.status == CoreStatus.DONE


@responses.activate
@pytest.mark.parametrize('status, result', fail_conds)
def test_pred_res_fail(
        globals, client, status, result, mock_test_responses, context_fixture):
    """Predict Result fails when status and result create fail conditions."""
    context = context_fixture('Healthy')
    mock_test_responses(task='predict', status=status, task_result=result)

    pred_res = client.predict(globals['predict_input'])

    context.run()

    assert pred_res.task.id == globals['predict']
    assert pred_res.id is None
    assert pred_res.status == status
    assert pred_res.result == result


@responses.activate
@pytest.mark.parametrize(
    'data, exp', [
        ('fail_data', 'fine_exp'), ('fine_data', 'fail_exp'),
        ('fail_data', 'fail_exp')])
def test_pred_res_fail_by_jobs(globals, client, data, exp, context_fixture):
    """Predict Result fails if required jobs fails."""
    context = context_fixture('Healthy')

    pred_res = client.predict(
        predict_input=PredictInput(
            data=globals[data], experiment=globals[exp]))

    context.run()

    assert pred_res.task.id is None
    assert pred_res.id is None
    assert pred_res.status is CoreStatus.FAIL
    assert pred_res.result is None


@responses.activate
@pytest.mark.parametrize('status', [CoreStatus.PENDING, CoreStatus.RUNNING, CoreStatus.FAIL])
def test_pred_stop(
        globals, urls, client, status, mock_test_responses, context_fixture):
    """Predict Result status fails if stopped during pending, running, and fail
    status, remains if in done status.
    """
    async def cancel(pred):
        await asyncio.sleep(1)
        pred.stop()
        return

    context = context_fixture('Healthy')
    mock_test_responses(task='predict', status=status)
    responses.add(
        responses.PUT, urls('stop', 'predict'),
        json={
            'message': 'task removed'
        },
        status=200,
        content_type='application/json')

    pred_res = client.predict(globals['predict_input'])

    cancel_task = Context.LOOP.create_task(cancel(pred_res))
    Context.CORO_TASKS.append(cancel_task)
    context.run()

    assert pred_res.status == CoreStatus.FAIL


@responses.activate
def test_pred_ts_res_success(
        globals, client, mock_test_responses, context_fixture):
    """Time series Prediction gets the result and id
    from CoreClient.predict_ts()"""
    context = context_fixture('Healthy')
    mock_test_responses(task='predict_ts', status=CoreStatus.DONE)

    pred_res = client.predict_ts(globals['predict_ts_input'])
    context.run()

    assert pred_res.task.id == globals['predict']
    assert pred_res.id == globals['predict_res']
    assert pred_res.result == globals['results']['predict_ts']
    assert pred_res.status == CoreStatus.DONE


@responses.activate
@pytest.mark.parametrize('status, result', fail_conds)
def test_pred_ts_res_fail(
        globals, client, status, result, mock_test_responses, context_fixture):
    """Time series Predict Result fails when status and result
    create fail conditions."""
    context = context_fixture('Healthy')
    mock_test_responses(task='predict_ts', status=status, task_result=result)

    pred_res = client.predict_ts(globals['predict_ts_input'])

    context.run()

    assert pred_res.task.id == globals['predict']
    assert pred_res.id is None
    assert pred_res.status == status
    assert pred_res.result == result


@responses.activate
@pytest.mark.parametrize(
    'data, exp', [
        ('fail_data', 'fine_exp_ts'), ('fine_data', 'fail_exp_ts'),
        ('fail_data', 'fail_exp_ts')])
def test_pred_ts_res_fail_by_jobs(
        globals, client, data, exp, context_fixture):
    """Time series Predict Result fails if required jobs fails."""
    context = context_fixture('Healthy')

    pred_res = client.predict_ts(
        predict_input=PredictTSInput(
            data=globals[data], experiment=globals[exp]))

    context.run()

    assert pred_res.task.id is None
    assert pred_res.id is None
    assert pred_res.status == CoreStatus.FAIL
    assert pred_res.result is None


@responses.activate
@pytest.mark.parametrize('status', [CoreStatus.PENDING, CoreStatus.RUNNING, CoreStatus.FAIL])
def test_pred_ts_stop(
        globals, urls, client, status, mock_test_responses, context_fixture):
    """Time series Predict Result status fails if stopped during pending,
    running, and fail status, remains if in done status.
    """
    async def cancel(pred):
        await asyncio.sleep(1)
        pred.stop()
        return

    context = context_fixture('Healthy')
    mock_test_responses(task='predict_ts', status=status)
    responses.add(
        responses.PUT, urls('stop', 'predict_ts'),
        json={
            'message': 'task removed'
        },
        status=200,
        content_type='application/json')

    pred_res = client.predict_ts(globals['predict_ts_input'])

    cancel_task = Context.LOOP.create_task(cancel(pred_res))
    Context.CORO_TASKS.append(cancel_task)
    context.run()

    assert pred_res.status == CoreStatus.FAIL
