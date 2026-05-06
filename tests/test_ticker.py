import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Any, get_args, _GenericAlias

from tests.context import yfinance as yf
from yfinance.exceptions import (
    YFPricesMissingError, YFInvalidPeriodError, YFNotImplementedError,
    YFTickerMissingError, YFTzMissingError, YFDataException,
)
from yfinance.config import YfConfig

from unittest.mock import patch, MagicMock


ticker_attributes = (
    ("major_holders", pd.DataFrame),
    ("institutional_holders", pd.DataFrame),
    ("mutualfund_holders", pd.DataFrame),
    ("insider_transactions", pd.DataFrame),
    ("insider_purchases", pd.DataFrame),
    ("insider_roster_holders", pd.DataFrame),
    ("splits", pd.Series),
    ("actions", pd.DataFrame),
    ("shares", pd.DataFrame),
    ("info", dict),
    ("calendar", dict),
    ("recommendations", Union[pd.DataFrame, dict]),
    ("recommendations_summary", Union[pd.DataFrame, dict]),
    ("upgrades_downgrades", Union[pd.DataFrame, dict]),
    ("ttm_cashflow", pd.DataFrame),
    ("quarterly_cashflow", pd.DataFrame),
    ("cashflow", pd.DataFrame),
    ("quarterly_balance_sheet", pd.DataFrame),
    ("balance_sheet", pd.DataFrame),
    ("ttm_income_stmt", pd.DataFrame),
    ("quarterly_income_stmt", pd.DataFrame),
    ("income_stmt", pd.DataFrame),
    ("analyst_price_targets", dict),
    ("earnings_estimate", pd.DataFrame),
    ("revenue_estimate", pd.DataFrame),
    ("earnings_history", pd.DataFrame),
    ("eps_trend", pd.DataFrame),
    ("eps_revisions", pd.DataFrame),
    ("growth_estimates", pd.DataFrame),
    ("sustainability", pd.DataFrame),
    ("options", tuple),
    ("news", Any),
    ("earnings_dates", pd.DataFrame),
)


def assert_attribute_type(instance, attribute_name, expected_type):
    try:
        attribute = getattr(instance, attribute_name)
    except (YFNotImplementedError, YFDataException):
        return
    if attribute is not None and expected_type is not Any:
        err_msg = f"{attribute_name} type is {type(attribute)} not {expected_type}"
        if isinstance(expected_type, _GenericAlias) and expected_type.__origin__ is Union:
            allowed_types = get_args(expected_type)
            assert isinstance(attribute, allowed_types), err_msg
        else:
            assert type(attribute) == expected_type, err_msg


class TestTicker:
    def setup_method(self):
        YfConfig.debug.hide_exceptions = True

    def test_getTz(self):
        tkrs = ["IMP.JO", "BHG.JO", "SSW.JO", "BP.L", "INTC"]
        for tkr in tkrs:
            yf.cache.get_tz_cache().store(tkr, None)
            dat = yf.Ticker(tkr)
            tz = dat._get_ticker_tz(timeout=5)
            assert tz is not None

    def test_badTicker(self):
        tkr = "DJI"
        dat = yf.Ticker(tkr)

        dat.history(period="5d")
        dat.history(start="2022-01-01")
        dat.history(start="2022-01-01", end="2022-03-01")
        yf.download([tkr], period="5d", threads=False, ignore_tz=False)
        yf.download([tkr], period="5d", threads=True, ignore_tz=False)
        yf.download([tkr], period="5d", threads=False, ignore_tz=True)
        yf.download([tkr], period="5d", threads=True, ignore_tz=True)

        for k in dat.fast_info:
            dat.fast_info[k]

        for attribute_name, attribute_type in ticker_attributes:
            assert_attribute_type(dat, attribute_name, attribute_type)

        assert isinstance(dat.dividends, pd.Series)
        assert dat.dividends.empty
        assert isinstance(dat.splits, pd.Series)
        assert dat.splits.empty
        assert isinstance(dat.capital_gains, pd.Series)
        assert dat.capital_gains.empty
        with pytest.raises(YFNotImplementedError):
            _ = dat.shares
        assert isinstance(dat.actions, pd.DataFrame)
        assert dat.actions.empty

    def test_invalid_period(self):
        tkr = "VALE"
        dat = yf.Ticker(tkr)
        YfConfig.debug.hide_exceptions = False
        with pytest.raises(YFInvalidPeriodError):
            dat.history(period="2wks", interval="1d")
        with pytest.raises(YFInvalidPeriodError):
            dat.history(period="2mos", interval="1d")

    def test_valid_custom_periods(self):
        valid_periods = [
            ("1d", "1m"), ("5d", "15m"), ("1mo", "1d"), ("3mo", "1wk"),
            ("6mo", "1d"), ("1y", "1mo"), ("5y", "1wk"), ("max", "1mo"),
            ("2d", "30m"), ("10mo", "1d"), ("1y", "1d"), ("3y", "1d"),
            ("2wk", "15m"), ("6mo", "5d"), ("10y", "1wk"),
        ]
        dat = yf.Ticker("AAPL")
        YfConfig.debug.hide_exceptions = False
        for period, interval in valid_periods:
            df = dat.history(period=period, interval=interval)
            assert isinstance(df, pd.DataFrame), f"period={period}, interval={interval}"
            assert not df.empty, f"No data for period={period}, interval={interval}"
            assert "Close" in df.columns

            now = datetime.now()
            if period == "max":
                continue
            if period.endswith("d"):
                expected_start = now - timedelta(days=int(period[:-1]))
            elif period.endswith("mo"):
                expected_start = now - timedelta(days=30 * int(period[:-2]))
            elif period.endswith("y"):
                expected_start = now - timedelta(days=365 * int(period[:-1]))
            elif period.endswith("wk"):
                expected_start = now - timedelta(weeks=int(period[:-2]))
            else:
                continue

            actual_start = df.index[0].to_pydatetime().replace(tzinfo=None)
            expected_start = expected_start.replace(hour=0, minute=0, second=0, microsecond=0)
            assert actual_start >= expected_start - timedelta(days=10), \
                f"Start {actual_start} out of range for period={period}"
            assert df.index[-1].to_pydatetime().replace(tzinfo=None) <= now, \
                f"End {df.index[-1]} out of range for period={period}"

    def test_ticker_missing(self):
        tkr = "ATVI"
        dat = yf.Ticker(tkr)
        with pytest.raises((YFTickerMissingError, YFTzMissingError, YFPricesMissingError)):
            YfConfig.debug.hide_exceptions = False
            dat.history(period="3mo", interval="1d")

    def test_goodTicker(self):
        tkrs = ["IBM", "QCSTIX"]
        for tkr in tkrs:
            dat = yf.Ticker(tkr)
            dat.history(period="5d")
            dat.history(start="2022-01-01")
            dat.history(start="2022-01-01", end="2022-03-01")
            yf.download([tkr], period="5d", threads=False, ignore_tz=False)
            yf.download([tkr], period="5d", threads=True, ignore_tz=False)
            yf.download([tkr], period="5d", threads=False, ignore_tz=True)
            yf.download([tkr], period="5d", threads=True, ignore_tz=True)

            for k in dat.fast_info:
                dat.fast_info[k]

            for attribute_name, attribute_type in ticker_attributes:
                assert_attribute_type(dat, attribute_name, attribute_type)

    def test_goodTicker_withProxy(self):
        tkr = "IBM"
        dat = yf.Ticker(tkr)
        dat._fetch_ticker_tz(timeout=5)
        dat._get_ticker_tz(timeout=5)
        dat.history(period="5d")
        for attribute_name, attribute_type in ticker_attributes:
            assert_attribute_type(dat, attribute_name, attribute_type)

    def test_ticker_with_symbol_mic(self):
        equities = [
            ("OR", "XPAR"),
            ("AAPL", "XNYS"),
            ("GOOGL", "XNAS"),
            ("BMW", "XETR"),
        ]
        for eq in equities:
            yf.Ticker(eq)
            yf.Ticker((eq[0], eq[1].lower()))

    def test_ticker_with_symbol_mic_invalid(self):
        with pytest.raises(ValueError, match="Unknown MIC code: 'XXXX'"):
            yf.Ticker(("ABC", "XXXX"))


class TestTickerHistory:
    def setup_method(self):
        self.symbol = "IBM"
        self.ticker = yf.Ticker(self.symbol)
        self.symbols = ["AMZN", "MSFT", "NVDA"]

    def test_history(self):
        md = self.ticker.history_metadata
        assert "IBM" in md.values(), "metadata missing"
        data = self.ticker.history("1y")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_history_metadata(self):
        self.ticker.history("1mo", repair=True)
        md = self.ticker.history_metadata
        assert md["YF repair?"]

    def test_download(self):
        tomorrow = pd.Timestamp.now().date() + pd.Timedelta(days=1)
        for t in [False, True]:
            for i in [False, True]:
                for m in [False, True]:
                    for n in [1, "all"]:
                        symbols = self.symbols[0] if n == 1 else self.symbols
                        data = yf.download(symbols, end=tomorrow, threads=t, ignore_tz=i, multi_level_index=m)
                        assert isinstance(data, pd.DataFrame)
                        assert not data.empty
                        if i:
                            assert data.index.tz is None
                        else:
                            assert data.index.tz is not None
                        if (not m) and n == 1:
                            assert not isinstance(data.columns, pd.MultiIndex)
                        else:
                            assert isinstance(data.columns, pd.MultiIndex)

    def test_dividends(self):
        data = self.ticker.dividends
        assert isinstance(data, pd.Series)
        assert not data.empty

    def test_splits(self):
        data = self.ticker.splits
        assert isinstance(data, pd.Series)

    def test_actions(self):
        data = self.ticker.actions
        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_chained_history_calls(self):
        _ = self.ticker.history(period="2d")
        data = self.ticker.dividends
        assert isinstance(data, pd.Series)
        assert not data.empty


class TestTickerEarnings:
    def setup_method(self):
        self.ticker = yf.Ticker("GOOGL")

    def test_earnings_dates(self):
        data = self.ticker.earnings_dates
        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_earnings_dates_with_limit(self):
        ticker = yf.Ticker("IBM")
        limit = 100
        data = ticker.get_earnings_dates(limit=limit)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert len(data) == limit

        data_cached = ticker.get_earnings_dates(limit=limit)
        assert data is data_cached


class TestTickerHolders:
    def setup_method(self):
        self.ticker = yf.Ticker("GOOGL")

    def test_major_holders(self):
        data = self.ticker.major_holders
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.major_holders

    def test_institutional_holders(self):
        data = self.ticker.institutional_holders
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.institutional_holders

    def test_mutualfund_holders(self):
        data = self.ticker.mutualfund_holders
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.mutualfund_holders

    def test_insider_transactions(self):
        data = self.ticker.insider_transactions
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.insider_transactions

    def test_insider_purchases(self):
        data = self.ticker.insider_purchases
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.insider_purchases

    def test_insider_roster_holders(self):
        data = self.ticker.insider_roster_holders
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.insider_roster_holders


class TestTickerMiscFinancials:
    def setup_method(self):
        self.ticker = yf.Ticker("GOOGL")
        self.ticker_old_fmt = yf.Ticker("BSE.AX")

    @pytest.mark.skip(reason="Hits businessinsider api manually which is more complex to mock in this testing system. Need to investigate further.")
    def test_isin(self):
        data = self.ticker.isin
        assert isinstance(data, str)
        assert data == "CA02080M1005"
        assert data is self.ticker.isin

    def test_options(self):
        data = self.ticker.options
        assert isinstance(data, tuple)
        assert len(data) > 1

    def test_shares_full(self):
        data = self.ticker.get_shares_full()
        assert isinstance(data, pd.Series)
        assert not data.empty

    def test_income_statement(self):
        expected_keys = ["Total Revenue", "Basic EPS"]
        expected_periods_days = 365

        data = self.ticker.get_income_stmt(pretty=True)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        period = abs((data.columns[0] - data.columns[1]).days)
        assert abs(period - expected_periods_days) < 20

        data2 = self.ticker.income_stmt
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_income_stmt(pretty=False)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_income_stmt(as_dict=True)
        assert isinstance(data, dict)

    def test_quarterly_income_statement(self):
        expected_keys = ["Total Revenue", "Basic EPS"]
        expected_periods_days = 365 // 4

        data = self.ticker.get_income_stmt(pretty=True, freq="quarterly")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        period = abs((data.columns[0] - data.columns[1]).days)
        assert abs(period - expected_periods_days) < 20

        data2 = self.ticker.quarterly_income_stmt
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_income_stmt(pretty=False, freq="quarterly")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_income_stmt(as_dict=True)
        assert isinstance(data, dict)

    def test_ttm_income_statement(self):
        expected_keys = ["Total Revenue", "Pretax Income", "Normalized EBITDA"]

        data = self.ticker.get_income_stmt(pretty=True, freq="trailing")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        assert len(data.columns) == 1

        data2 = self.ticker.ttm_income_stmt
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_income_stmt(pretty=False, freq="trailing")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_income_stmt(as_dict=True, freq="trailing")
        assert isinstance(data, dict)

    def test_balance_sheet(self):
        expected_keys = ["Total Assets", "Net PPE"]
        expected_periods_days = 365

        data = self.ticker.get_balance_sheet(pretty=True)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        period = abs((data.columns[0] - data.columns[1]).days)
        assert abs(period - expected_periods_days) < 20

        data2 = self.ticker.balance_sheet
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_balance_sheet(pretty=False)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_balance_sheet(as_dict=True)
        assert isinstance(data, dict)

    def test_quarterly_balance_sheet(self):
        expected_keys = ["Total Assets", "Net PPE"]
        expected_periods_days = 365 // 4

        data = self.ticker.get_balance_sheet(pretty=True, freq="quarterly")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        period = abs((data.columns[0] - data.columns[1]).days)
        assert abs(period - expected_periods_days) < 20

        data2 = self.ticker.quarterly_balance_sheet
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_balance_sheet(pretty=False, freq="quarterly")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_balance_sheet(as_dict=True, freq="quarterly")
        assert isinstance(data, dict)

    def test_cash_flow(self):
        expected_keys = ["Operating Cash Flow", "Net PPE Purchase And Sale"]
        expected_periods_days = 365

        data = self.ticker.get_cashflow(pretty=True)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        period = abs((data.columns[0] - data.columns[1]).days)
        assert abs(period - expected_periods_days) < 20

        data2 = self.ticker.cashflow
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_cashflow(pretty=False)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_cashflow(as_dict=True)
        assert isinstance(data, dict)

    def test_quarterly_cash_flow(self):
        expected_keys = ["Operating Cash Flow", "Net PPE Purchase And Sale"]
        expected_periods_days = 365 // 4

        data = self.ticker.get_cashflow(pretty=True, freq="quarterly")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        period = abs((data.columns[0] - data.columns[1]).days)
        assert abs(period - expected_periods_days) < 20

        data2 = self.ticker.quarterly_cashflow
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_cashflow(pretty=False, freq="quarterly")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_cashflow(as_dict=True)
        assert isinstance(data, dict)

    def test_ttm_cash_flow(self):
        expected_keys = ["Operating Cash Flow", "Net PPE Purchase And Sale"]

        data = self.ticker.get_cashflow(pretty=True, freq="trailing")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys:
            assert k in data.index
        assert len(data.columns) == 1

        data2 = self.ticker.ttm_cashflow
        assert data.equals(data2)

        expected_keys_raw = [k.replace(" ", "") for k in expected_keys]
        data = self.ticker.get_cashflow(pretty=False, freq="trailing")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        for k in expected_keys_raw:
            assert k in data.index

        data = self.ticker.get_cashflow(as_dict=True, freq="trailing")
        assert isinstance(data, dict)

    def test_income_alt_names(self):
        i1 = self.ticker.income_stmt
        assert i1.equals(self.ticker.incomestmt)
        assert i1.equals(self.ticker.financials)

        i1 = self.ticker.get_income_stmt()
        assert i1.equals(self.ticker.get_incomestmt())
        assert i1.equals(self.ticker.get_financials())

        i1 = self.ticker.quarterly_income_stmt
        assert i1.equals(self.ticker.quarterly_incomestmt)
        assert i1.equals(self.ticker.quarterly_financials)

        i1 = self.ticker.get_income_stmt(freq="quarterly")
        assert i1.equals(self.ticker.get_incomestmt(freq="quarterly"))
        assert i1.equals(self.ticker.get_financials(freq="quarterly"))

        i1 = self.ticker.ttm_income_stmt
        assert i1.equals(self.ticker.ttm_incomestmt)
        assert i1.equals(self.ticker.ttm_financials)

        i1 = self.ticker.get_income_stmt(freq="trailing")
        assert i1.equals(self.ticker.get_incomestmt(freq="trailing"))
        assert i1.equals(self.ticker.get_financials(freq="trailing"))

    def test_balance_sheet_alt_names(self):
        assert self.ticker.balance_sheet.equals(self.ticker.balancesheet)
        assert self.ticker.get_balance_sheet().equals(self.ticker.get_balancesheet())
        assert self.ticker.quarterly_balance_sheet.equals(self.ticker.quarterly_balancesheet)
        assert self.ticker.get_balance_sheet(freq="quarterly").equals(
            self.ticker.get_balancesheet(freq="quarterly")
        )

    def test_cash_flow_alt_names(self):
        assert self.ticker.cash_flow.equals(self.ticker.cashflow)
        assert self.ticker.get_cash_flow().equals(self.ticker.get_cashflow())
        assert self.ticker.quarterly_cash_flow.equals(self.ticker.quarterly_cashflow)
        assert self.ticker.get_cash_flow(freq="quarterly").equals(
            self.ticker.get_cashflow(freq="quarterly")
        )
        assert self.ticker.ttm_cash_flow.equals(self.ticker.ttm_cashflow)
        assert self.ticker.get_cash_flow(freq="trailing").equals(
            self.ticker.get_cashflow(freq="trailing")
        )

    def test_bad_freq_value_raises_exception(self):
        with pytest.raises(ValueError):
            self.ticker.get_cashflow(freq="badarg")

    def test_calendar(self):
        data = self.ticker.calendar
        assert isinstance(data, dict)
        assert len(data) > 0
        assert "Earnings Date" in data
        assert "Earnings Average" in data
        assert "Earnings Low" in data
        assert "Earnings High" in data
        assert "Revenue Average" in data
        assert "Revenue Low" in data
        assert "Revenue High" in data
        assert data is self.ticker.calendar


class TestTickerAnalysts:
    def setup_method(self):
        self.ticker = yf.Ticker("GOOGL")
        self.ticker_no_analysts = yf.Ticker("^GSPC")

    def test_recommendations(self):
        data = self.ticker.recommendations
        data_summary = self.ticker.recommendations_summary
        assert data.equals(data_summary)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.recommendations

    def test_recommendations_summary(self):
        data = self.ticker.recommendations_summary
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.recommendations_summary

    def test_upgrades_downgrades(self):
        data = self.ticker.upgrades_downgrades
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data is self.ticker.upgrades_downgrades

    def test_analyst_price_targets(self):
        data = self.ticker.analyst_price_targets
        assert isinstance(data, dict)
        assert data is self.ticker.analyst_price_targets

    def test_earnings_estimate(self):
        data = self.ticker.earnings_estimate
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.earnings_estimate

    def test_revenue_estimate(self):
        data = self.ticker.revenue_estimate
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.revenue_estimate

    def test_earnings_history(self):
        data = self.ticker.earnings_history
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data is self.ticker.earnings_history

    def test_eps_trend(self):
        data = self.ticker.eps_trend
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.eps_trend

    def test_growth_estimates(self):
        data = self.ticker.growth_estimates
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data is self.ticker.growth_estimates

    def test_no_analysts(self):
        attributes = [
            "recommendations",
            "upgrades_downgrades",
            "earnings_estimate",
            "revenue_estimate",
            "earnings_history",
            "eps_trend",
            "growth_estimates",
        ]
        for attribute in attributes:
            try:
                data = getattr(self.ticker_no_analysts, attribute)
                assert isinstance(data, pd.DataFrame), f"{attribute}: wrong type"
                assert data.empty, f"{attribute}: not empty"
            except Exception as e:
                pytest.fail(f"Exception raised for attribute '{attribute}': {e}")


class TestTickerInfo:
    def setup_method(self):
        self.symbols = []
        self.symbols += ["ESLT.TA", "BP.L", "GOOGL"]
        self.symbols.append("QCSTIX")
        self.symbols += ["BTC-USD", "IWO", "VFINX", "^GSPC"]
        self.symbols += ["SOKE.IS", "ADS.DE"]
        self.symbols += ["EXTO"]
        self.tickers = [yf.Ticker(s) for s in self.symbols]

    def test_fast_info(self):
        f = yf.Ticker("AAPL").fast_info
        for k in f:
            assert f[k] is not None

    def test_info(self):
        data = self.tickers[0].info  # ESLT.TA
        assert isinstance(data, dict)
        assert "symbol" in data
        assert data["symbol"] == self.symbols[0]

    def test_complementary_info(self):
        data1 = self.tickers[0].info  # ESLT.TA — no PEG ratio
        assert data1["trailingPegRatio"] is None

        data2 = self.tickers[2].info  # GOOGL — has PEG ratio
        assert isinstance(data2["trailingPegRatio"], float)

    def test_isin_info(self):
        isin_list = {
            "ES0137650018": True,
            "does_not_exist": True,
            "INF209K01EN2": True,
            "INX846K01K35": False,
            "INF846K01K35": True,
        }
        for isin, should_succeed in isin_list.items():
            if not should_succeed:
                with pytest.raises(ValueError) as cm:
                    yf.Ticker(isin)
                assert str(cm.value) in [f"Invalid ISIN number: {isin}", "Empty tickername"]
            else:
                ticker = yf.Ticker(isin)
                try:
                    ticker.info
                except Exception:
                    pass

    def test_empty_info(self):
        data = self.tickers[10].info  # EXTO
        assert isinstance(data, dict)
        assert "quoteType" in data
        assert "trailingPegRatio" in data


class TestTickerFundsData:
    def setup_method(self):
        self.test_tickers = [
            yf.Ticker("SPY"),
            yf.Ticker("JNK"),
            yf.Ticker("VTSAX"),
        ]

    def test_fetch_and_parse(self):
        for ticker in self.test_tickers:
            try:
                ticker.funds_data._fetch_and_parse()
            except Exception as e:
                pytest.fail(f"_fetch_and_parse raised for {ticker.ticker}: {e}")

        with pytest.raises(YFDataException):
            yf.Ticker("AAPL").funds_data._fetch_and_parse()

    def test_description(self):
        for ticker in self.test_tickers:
            description = ticker.funds_data.description
            assert isinstance(description, str)
            assert len(description) > 0

    def test_fund_overview(self):
        for ticker in self.test_tickers:
            fund_overview = ticker.funds_data.fund_overview
            assert isinstance(fund_overview, dict)

    def test_fund_operations(self):
        for ticker in self.test_tickers:
            fund_operations = ticker.funds_data.fund_operations
            assert isinstance(fund_operations, pd.DataFrame)

    def test_asset_classes(self):
        for ticker in self.test_tickers:
            asset_classes = ticker.funds_data.asset_classes
            assert isinstance(asset_classes, dict)

    def test_top_holdings(self):
        for ticker in self.test_tickers:
            top_holdings = ticker.funds_data.top_holdings
            assert isinstance(top_holdings, pd.DataFrame)

    def test_equity_holdings(self):
        for ticker in self.test_tickers:
            equity_holdings = ticker.funds_data.equity_holdings
            assert isinstance(equity_holdings, pd.DataFrame)

    def test_bond_holdings(self):
        for ticker in self.test_tickers:
            bond_holdings = ticker.funds_data.bond_holdings
            assert isinstance(bond_holdings, pd.DataFrame)

    def test_bond_ratings(self):
        for ticker in self.test_tickers:
            bond_ratings = ticker.funds_data.bond_ratings
            assert isinstance(bond_ratings, dict)

    def test_sector_weightings(self):
        for ticker in self.test_tickers:
            sector_weightings = ticker.funds_data.sector_weightings
            assert isinstance(sector_weightings, dict)


class TestTickerValuationMeasures:
    def test_valuation_measures(self):
        data = yf.Ticker("AAPL").valuation
        assert data.shape == (4, 3)
        assert list(data.columns) == ["Current", "12/2023", "12/2022"]
        assert "Market Cap (intraday)" in data.index
        assert "Trailing P/E" in data.index
        assert "Forward P/E" in data.index
        assert data.index.name is None
        assert data.loc["Market Cap (intraday)", "Current"] == "2.50T"
        assert data.loc["Forward P/E", "12/2023"] == "26.1"

    def test_valuation_measures_no_table(self):
        no_table = MagicMock()
        no_table.text = "<html><body><p>No tables here</p></body></html>"
        with patch("yfinance.data.YfData.get", return_value=no_table):
            data = yf.Ticker("AAPL").valuation
        assert isinstance(data, pd.DataFrame)
        assert data.empty

    def test_valuation_measures_fetch_error(self):
        with patch("yfinance.data.YfData.cache_get", side_effect=Exception("network error")):
            data = yf.Ticker("AAPL").valuation
        assert isinstance(data, pd.DataFrame)
        assert data.empty
