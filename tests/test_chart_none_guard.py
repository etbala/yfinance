"""
Regression test for issue #2670:
When Yahoo returns {"chart": null}, accessing data["chart"]["result"] raised
a TypeError. The fix adds a None guard before key access.
"""
import unittest
from unittest.mock import MagicMock, patch

from yfinance.scrapers.history import PriceHistory
from yfinance.exceptions import YFPricesMissingError


def _make_price_history(json_payload):
    """Build a PriceHistory instance whose HTTP call returns json_payload."""
    mock_response = MagicMock()
    mock_response.text = ""
    mock_response.json.return_value = json_payload

    mock_data = MagicMock()
    mock_data.get.return_value = mock_response
    mock_data.cache_get.return_value = mock_response

    return PriceHistory(data=mock_data, ticker="FAKE", tz="America/New_York")


class TestChartNoneGuard(unittest.TestCase):

    def test_chart_none_raises_prices_missing_not_type_error(self):
        """data["chart"] = None must not crash with TypeError."""
        ph = _make_price_history({"chart": None})
        with self.assertRaises(YFPricesMissingError):
            ph.history(period="1mo", raise_errors=True)

    def test_chart_result_none_raises_prices_missing(self):
        """data["chart"]["result"] = None must raise YFPricesMissingError."""
        ph = _make_price_history({"chart": {"result": None, "error": None}})
        with self.assertRaises(YFPricesMissingError):
            ph.history(period="1mo", raise_errors=True)

    def test_chart_missing_raises_prices_missing(self):
        """Missing 'chart' key entirely must raise YFPricesMissingError."""
        ph = _make_price_history({})
        with self.assertRaises(YFPricesMissingError):
            ph.history(period="1mo", raise_errors=True)

    def test_chart_with_error_description_raises_prices_missing(self):
        """data["chart"]["error"] present must raise YFPricesMissingError (not crash)."""
        ph = _make_price_history({
            "chart": {
                "result": None,
                "error": {"code": "Not Found", "description": "No data for symbol"},
            }
        })
        with self.assertRaises(YFPricesMissingError):
            ph.history(period="1mo", raise_errors=True)


if __name__ == "__main__":
    unittest.main()
