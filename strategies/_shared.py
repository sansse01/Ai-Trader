from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import pandas as pd


def make_signal_backtest_builder(
    default_params: Mapping[str, Any],
    signal_map: Mapping[str, str],
):
    """Create a simple backtesting.py Strategy builder based on signal columns."""

    long_col = signal_map.get("long")
    short_col = signal_map.get("short")
    exit_long_col = signal_map.get("exit_long")
    exit_short_col = signal_map.get("exit_short")
    long_stop_col = signal_map.get("long_stop")
    short_stop_col = signal_map.get("short_stop")
    long_tp_col = signal_map.get("long_take_profit")
    short_tp_col = signal_map.get("short_take_profit")

    def builder(
        df: pd.DataFrame,
        df_sig: pd.DataFrame,
        params: Mapping[str, Any],
    ):
        from backtesting import Strategy

        merged_params: MutableMapping[str, Any] = dict(default_params)
        merged_params.update(dict(params or {}))

        allow_shorts = bool(merged_params.get("allow_shorts", True))
        stop_loss_percent = float(merged_params.get("stop_loss_percent", 0.0))
        take_profit_percent = float(merged_params.get("take_profit_percent", 0.0))
        risk_fraction = float(merged_params.get("risk_fraction", 0.25))
        sizing_mode = str(merged_params.get("sizing_mode", "Whole units (int)"))
        contract_size = float(merged_params.get("contract_size", 0.001))

        class SignalStrategy(Strategy):
            def _position_size(self, price: float) -> float:
                if sizing_mode.startswith("Whole"):
                    cash = float(self.equity)
                    notional = cash * risk_fraction
                    units = max(1, int(notional / max(price, 1e-9)))
                    return units
                return max(0.001, min(0.9999, risk_fraction))

            def _get_bool(self, row: pd.Series, column: str | None) -> bool:
                if column and column in row:
                    return bool(row[column])
                return False

            def _get_price(self, row: pd.Series, column: str | None) -> float | None:
                if column and column in row:
                    value = row[column]
                    if value is not None and not pd.isna(value):
                        return float(value) * contract_size
                return None

            def next(self):
                ts = self.data.index[-1]
                if ts not in df_sig.index:
                    return
                row = df_sig.loc[ts]
                base_row = df.loc[ts] if ts in df.index else None
                raw_close = (
                    float(base_row["Close"])
                    if base_row is not None and not pd.isna(base_row["Close"])
                    else float(self.data.Close[-1]) / max(contract_size, 1e-9)
                )

                long_signal = self._get_bool(row, long_col)
                short_signal = self._get_bool(row, short_col)

                if self.position:
                    if self.position.is_long:
                        if self._get_bool(row, exit_long_col) or (
                            short_signal and allow_shorts
                        ):
                            self.position.close()
                            return
                        stop_price = self._get_price(row, long_stop_col)
                        if stop_price is None and stop_loss_percent > 0:
                            stop_price = raw_close * (1 - stop_loss_percent / 100.0) * contract_size
                        if stop_price is not None and self.data.Low[-1] <= stop_price:
                            self.position.close()
                            return
                        take_profit_price = self._get_price(row, long_tp_col)
                        if take_profit_price is None and take_profit_percent > 0:
                            take_profit_price = raw_close * (1 + take_profit_percent / 100.0) * contract_size
                        if (
                            take_profit_price is not None
                            and self.data.High[-1] >= take_profit_price
                        ):
                            self.position.close()
                            return
                    else:
                        if self._get_bool(row, exit_short_col) or long_signal:
                            self.position.close()
                            return
                        stop_price = self._get_price(row, short_stop_col)
                        if stop_price is None and stop_loss_percent > 0:
                            stop_price = raw_close * (1 + stop_loss_percent / 100.0) * contract_size
                        if stop_price is not None and self.data.High[-1] >= stop_price:
                            self.position.close()
                            return
                        take_profit_price = self._get_price(row, short_tp_col)
                        if take_profit_price is None and take_profit_percent > 0:
                            take_profit_price = raw_close * (1 - take_profit_percent / 100.0) * contract_size
                        if (
                            take_profit_price is not None
                            and self.data.Low[-1] <= take_profit_price
                        ):
                            self.position.close()
                            return

                if not self.position:
                    size = self._position_size(float(self.data.Close[-1]))
                    if long_signal:
                        self.buy(size=size)
                    elif short_signal and allow_shorts:
                        self.sell(size=size)

        return SignalStrategy

    return builder


__all__ = ["make_signal_backtest_builder"]
