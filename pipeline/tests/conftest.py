import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, PREP  # noqa: E402


@pytest.fixture(scope="session")
def samples():
    return pd.read_parquet(PREP / "samples.parquet")


@pytest.fixture(scope="session")
def feat():
    return lambda s: pd.read_parquet(FEAT / f"set_{s}.parquet").set_index("sid")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Skip, rather than fail, a test whose input is not in this checkout.

    The repository ships only the labels and the result tables. Checks on the extracted
    time series, the feature tables and the maps need the full pipeline run first.
    """
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(FileNotFoundError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0,
                           f"Skipped: input missing, run the pipeline first ({call.excinfo.value})")
