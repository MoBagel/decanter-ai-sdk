import sys
import pytest
from tests import test_mock_iid
from tests import test_mock_ts

sys.path.append("..")


class Test_demo:
    test_mock_iid.test_iid
    test_mock_ts.test_ts


if __name__ == "__name__":
    pytest.main()
