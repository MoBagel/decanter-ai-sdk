# pylint: disable=redefined-builtin
# pylint: disable=too-many-arguments
"""Test related method and functionality of Context."""
import pytest
import responses

from decanter.core import Context
from decanter.core.core_api import TrainInput
from decanter.core.extra import CoreStatus
from decanter.core.jobs import DataUpload

fail_conds = [(stat, res) for stat in CoreStatus.FAIL_STATUS for res in [None, 'result']]
fail_conds.append((CoreStatus.DONE, None))


@responses.activate
def test_data_success(
        globals, client, mock_test_responses, context_fixture):
    """DataUpload gets the id and result when upload csv file or datafram."""
    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=CoreStatus.DONE)
    data = client.upload(file=globals['test_csv_file'])
    data_ = client.upload(file=globals['test_df'])
    context.run()
    assert data.task.id == data_.task.id == globals['upload']
    assert data.id == data_.id == globals['data']
    assert data.status == data_.status == CoreStatus.DONE
    assert data.result == data_.result == globals['results']['upload']


@responses.activate
@pytest.mark.parametrize('status, result', fail_conds)
def test_data_fail(
        globals, client, status, result, mock_test_responses, context_fixture):
    """DataUpload fails when status and result create fail conditions."""
    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=status, task_result=result)
    data = client.upload(file=globals['test_csv_file'])
    context.run()
    assert data.task.id == globals['upload']
    assert data.id is None
    assert data.status == status
    assert data.result == result


@responses.activate
def test_no_file(client, mock_test_responses, context_fixture):
    """Raise exceptions when upload empty files."""
    context_fixture('Healthy')
    mock_test_responses(task='upload')
    with pytest.raises(Exception):
        client.upload(file=None)


@responses.activate
@pytest.mark.parametrize('status', [CoreStatus.PENDING, CoreStatus.RUNNING, CoreStatus.FAIL])
def test_data_stop(
        globals, urls, client, status, mock_test_responses, context_fixture):
    """DataUpload status is fail if stopped during pending, running, and fail status,
    remains if in done status. The experiment following will failed if data
    failed.
    """
    async def cancel(data):
        data.stop()
        return

    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=status)
    mock_test_responses(task='train', status=CoreStatus.DONE)
    responses.add(
        responses.PUT, urls('stop', 'upload'),
        json={
            'message': 'task removed'
        },
        status=200,
        content_type='application/json')
    if status == CoreStatus.DONE:
        data = DataUpload()
        data.status = CoreStatus.DONE
    else:
        data = client.upload(file=globals['test_csv_file'])

    exp = client.train(TrainInput(
        data=data, target='test-target', algos=['test-algo']))
    cancel_task = Context.LOOP.create_task(cancel(data))
    Context.CORO_TASKS.append(cancel_task)
    context.run()

    if status == CoreStatus.DONE:
        assert data.status == CoreStatus.DONE
        assert exp.status == CoreStatus.DONE
    else:
        assert data.status == CoreStatus.FAIL
        assert exp.status == CoreStatus.FAIL
