import pytest
from decanter_ai_sdk.client import Client
from dotenv import dotenv_values


@pytest.fixture(scope="session")
def client():
    config = dotenv_values(".env")

    return Client(
        auth_key=config["API_KEY"],
        project_id=config["PROJECT_ID"],
        host=config["HOST"],
    )
