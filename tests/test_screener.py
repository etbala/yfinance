import pytest
from unittest.mock import patch, MagicMock

from yfinance.screener.screener import screen
from yfinance.screener.query import EquityQuery
from tests.mocks import MockResponse


@pytest.fixture
def equity_query():
    return EquityQuery("gt", ["eodprice", 3])


def test_set_large_size_raises(equity_query):
    with pytest.raises(ValueError):
        screen(equity_query, size=251)


def test_fetch_query(equity_query):
    with patch("yfinance.data.YfData.post") as mock_post:
        mock_post.return_value = MockResponse({"finance": {"result": [{"key": "value"}]}})
        result = screen(equity_query)
    assert result == {"key": "value"}


def test_fetch_predefined():
    with patch("yfinance.data.YfData.get") as mock_get:
        mock_get.return_value = MockResponse({"finance": {"result": [{"key": "value"}]}})
        result = screen("aggressive_small_caps")
    assert result == {"key": "value"}
