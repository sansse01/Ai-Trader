from datetime import datetime, timezone

import pytest

from ai_trader_strats.config_schemas import RiskConfig
from ai_trader_strats.risk_hooks import RiskEngineAdapter


class BrokerStub:
    def __init__(self, value: float = 100.0):
        self._value = value

    def getvalue(self) -> float:
        return self._value

    def setvalue(self, value: float) -> None:
        self._value = value


@pytest.fixture
def daily_lock_adapter() -> tuple[RiskEngineAdapter, BrokerStub]:
    broker = BrokerStub()
    cfg = RiskConfig(dd_lock_pct=-0.095, daily_loss_lock_pct=-0.02, single_asset_cap_pct=5.0, gross_cap_pct=5.0)
    adapter = RiskEngineAdapter(broker, cfg)
    adapter.update_clock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    return adapter, broker


def test_drawdown_lock():
    broker = BrokerStub()
    cfg = RiskConfig(dd_lock_pct=-0.095, daily_loss_lock_pct=-0.5, single_asset_cap_pct=5.0, gross_cap_pct=5.0)
    adapter = RiskEngineAdapter(broker, cfg)
    adapter.update_clock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert not adapter.locked()
    broker.setvalue(89.0)
    assert adapter.locked()
    broker.setvalue(95.0)
    assert adapter.locked()
    broker.setvalue(105.0)
    assert not adapter.locked()


def test_daily_lock_reset(daily_lock_adapter: tuple[RiskEngineAdapter, BrokerStub]):
    adapter, broker = daily_lock_adapter
    adapter.locked()  # prime state
    broker.setvalue(97.0)
    adapter.update_clock(datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
    assert adapter.locked()
    adapter.update_clock(datetime(2024, 1, 2, tzinfo=timezone.utc))
    broker.setvalue(101.0)
    assert not adapter.locked()
