import pandas as pd
import pytest
import yfinance as yf


@pytest.fixture
def calendars():
    return yf.Calendars()


def test_get_earnings_calendar(calendars):
    result = calendars.get_earnings_calendar(limit=1)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    tickers = result.index.tolist()
    assert isinstance(tickers, list)
    assert len(tickers) == 1
    assert tickers == calendars.earnings_calendar.index.tolist()


def test_get_earnings_calendar_future_dates(calendars):
    result = calendars.get_earnings_calendar(limit=5)
    assert len(result) == 5
    now = pd.Timestamp.now(tz="UTC")
    assert result["Event Start Date"].iloc[0] >= now


def test_get_ipo_info_calendar(calendars):
    result = calendars.get_ipo_info_calendar(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5


def test_get_economic_events_calendar(calendars):
    result = calendars.get_economic_events_calendar(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5


def test_get_splits_calendar(calendars):
    result = calendars.get_splits_calendar(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5
