# pylint: disable=redefined-builtin
"""Test related method and functionality of Context."""
import pytest
import responses

from decanter.core import Context
from decanter.core.extra import CoreStatus


def test_no_context(globals, client):
    """Test calling coerx_api when no context created.

    CoreClient will call context.LOOP, check if every api has raises
    AttributeError with message "event loop is \'NoneType\'".
    """
    with pytest.raises(AttributeError, match=r'event loop is \'NoneType\''):
        client.upload(file=globals['test_csv_file'])

    with pytest.raises(AttributeError, match=r'event loop is \'NoneType\''):
        client.train(train_input=globals['train_input'])

    with pytest.raises(AttributeError, match=r'event loop is \'NoneType\''):
        client.predict(
            predict_input=globals['predict_input'])


@responses.activate
def test_connection_fail(context_fixture):
    """Context exits from Python when meeting any RequestException."""
    with pytest.raises(SystemExit):
        context_fixture('RequestException')


@responses.activate
def test_auth_fail(context_fixture):
    """Context exits from Python when authorization failed."""
    with pytest.raises(SystemExit):
        context_fixture('AuthFailed')


@responses.activate
def test_stop_jobs(globals, urls, client, mock_test_responses, context_fixture):
    """Context stops the jobs in the list passed by `Context.stop.jobs()`"""
    async def cancel():
        context.stop_jobs([datas[0], datas[2]])
        responses.add(
            responses.GET, urls('task', 'upload'),
            json={
                '_id': globals['upload'],
                'status': CoreStatus.DONE,
                'result': {
                    '_id': globals['data']}
                },
            status=200,
            content_type='application/json')

    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=CoreStatus.RUNNING)
    responses.add(
        responses.PUT, urls('stop', 'upload'),
        json={
            'message': 'task removed'
        },
        status=200,
        content_type='application/json')
    datas = []
    for i in range(3):
        data = client.upload(file=globals['test_csv_file'], name=str(i))
        datas.append(data)

    cancel_task = Context.LOOP.create_task(cancel())
    Context.CORO_TASKS.append(cancel_task)
    context.run()

    assert datas[0].status == CoreStatus.FAIL
    assert datas[2].status == CoreStatus.FAIL
    assert datas[1].status == CoreStatus.DONE


@responses.activate
def test_stop_all_jobs(
        globals, urls, client, mock_test_responses, context_fixture):
    """Context stops all jobs in running or pending status."""
    async def cancel():
        context.stop_all_jobs()
        return

    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=CoreStatus.RUNNING)
    responses.add(
        responses.PUT, urls('stop', 'upload'),
        json={
            'message': 'task removed'
        },
        status=200,
        content_type='application/json')

    datas = []
    for i in range(3):
        data = client.upload(file=globals['test_csv_file'], name=str(i))
        datas.append(data)

    cancel_task = Context.LOOP.create_task(cancel())
    Context.CORO_TASKS.append(cancel_task)
    context.run()

    assert all(data.status == CoreStatus.FAIL for data in datas)

@responses.activate
def test_get_jobs_by_name(
        globals, client, mock_test_responses, context_fixture):
    """Context gets jobs with name in name list."""

    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=CoreStatus.DONE)
    data_list = []
    for i in range(3):
        data = client.upload(file=globals['test_csv_file'], name=str(i))
        data_list.append(data)

    context.run()
    jobs = context.get_jobs_by_name(names=['0', '2'])

    assert jobs[0] is data_list[0]
    assert jobs[1] is data_list[2]


@responses.activate
def test_get_all_jobs(
        globals, client, mock_test_responses, context_fixture):
    """Context gets all jobs."""

    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=CoreStatus.DONE)
    data_list = []
    for i in range(2):
        data = client.upload(file=globals['test_csv_file'], name=str(i))
        data_list.append(data)

    context.run()
    jobs = context.get_all_jobs()

    assert jobs[0] is data_list[0]
    assert jobs[1] is data_list[1]


@responses.activate
def test_get_jobs_status(
        globals, urls, client, mock_test_responses, context_fixture):
    """Context shows jobs status in dataframe with specific status."""

    context = context_fixture('Healthy')
    mock_test_responses(task='upload', status=CoreStatus.DONE)
    responses.add(
        responses.GET, urls('task', 'upload'),
        json={
            '_id': globals['upload'],
            'status': CoreStatus.FAIL
        },
        status=200,
        content_type='application/json')

    for i in range(2):
        client.upload(file=globals['test_csv_file'], name=str(i))

    context.run()
    job_fail = context.get_jobs_status(status=['fail'])

    assert job_fail.iloc[0]['status'] == 'fail'
    assert job_fail.iloc[0]['Job'] == '1'
