import os
from http.client import HTTPMessage
from typing import List
from unittest.mock import ANY, Mock, call, patch

from decanter_ai_sdk.client import Client


@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_retry_request_on_http(getconn_mock):
    getconn_mock.return_value.getresponse.side_effect = [
        Mock(status=500, msg=HTTPMessage()),
        Mock(status=502, msg=HTTPMessage()),
        Mock(status=200, msg=HTTPMessage()),
    ]

    client = Client(
        auth_key="auth_API_key",
        project_id="project_id",
        host="http://test.decanter.ai",
    )
    try:
        client.get_table_list()
    except:
        pass

    assert getconn_mock.return_value.request.mock_calls == [
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
    ]


@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_retry_request_on_https(getconn_mock):
    getconn_mock.return_value.getresponse.side_effect = [
        Mock(status=500, msg=HTTPMessage()),
        Mock(status=502, msg=HTTPMessage()),
        Mock(status=200, msg=HTTPMessage()),
    ]

    client = Client(
        auth_key="auth_API_key",
        project_id="project_id",
        host="https://test.decanter.ai",
    )
    try:
        client.get_table_list()
    except:
        pass

    assert getconn_mock.return_value.request.mock_calls == [
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
    ]


@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_retry_request_with_max_3_times(getconn_mock):
    getconn_mock.return_value.getresponse.side_effect = [
        Mock(status=500, msg=HTTPMessage()),
        Mock(status=502, msg=HTTPMessage()),
        Mock(status=500, msg=HTTPMessage()),
        Mock(status=500, msg=HTTPMessage()),
    ]

    client = Client(
        auth_key="auth_API_key",
        project_id="project_id",
        host="http://test.decanter.ai",
    )
    try:
        client.get_table_list()
    except:
        pass

    assert getconn_mock.return_value.request.mock_calls == [
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
        call("GET", "/v1/table/getlist/project_id?page=1", body=None, headers=ANY),
    ]
