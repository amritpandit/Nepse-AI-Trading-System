from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkType(StrEnum):
    BUY_AND_HOLD = "buy_and_hold"
    NEPSE_INDEX = "nepse_index"
    SECTOR_INDEX = "sector_index"


class StrategyBase(BaseModel):
    name: str
    version: str
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class StrategyCreate(StrategyBase):
    pass


class StrategyResponse(BaseModel):
    id: int
    name: str
    version: str
    created_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class StrategyDetailResponse(StrategyResponse):
    description: str | None = None
    config: dict[str, Any]


class BacktestConfig(BaseModel):
    start_date: date
    end_date: date
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    commission_rate: float = Field(default=0.005, ge=0, le=0.1)
    slippage_bps: float = Field(default=5.0, ge=0, le=5_000)
    execution_delay_bars: int = Field(default=1, ge=0, le=20)

    @model_validator(mode="after")
    def validate_date_range(self) -> "BacktestConfig":
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        return self


class BacktestCreate(BaseModel):
    strategy_id: int
    config: BacktestConfig


class BacktestResultResponse(BaseModel):
    backtest_id: int
    strategy_id: int
    metrics: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class BenchmarkCompareRequest(BaseModel):
    benchmark_type: BenchmarkType
    start_date: str
    end_date: str
    symbol: str | None = None
    sector: str | None = None
    initial_capital: float = 1000000.0


class BenchmarkResultResponse(BaseModel):
    period_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float | None = None
    equity_curve: list[dict[str, Any]]
