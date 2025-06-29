# stat_arb_trader_dhan/api_models/__init__.py

from pydantic import BaseModel
from typing import List, Optional

class SignalRequest(BaseModel):
    """
    Defines the structure for a request from the frontend.
    """
    strategy_id: int
    symbol: str
    timeframe: str
    num_candles: int

class Candle(BaseModel):
    """
    Defines the structure for a single candlestick, compatible with Lightweight Charts.
    """
    time: int  # UNIX timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

class IndicatorData(BaseModel):
    """
    Defines the structure for a single time-series indicator line.
    """
    time: int  # UNIX timestamp
    value: float

class SignalMarker(BaseModel):
    """
    Defines the structure for a signal marker to be placed on the chart.
    """
    time: int
    position: str  # 'aboveBar' or 'belowBar'
    color: str
    shape: str  # 'arrowUp' or 'arrowDown'
    text: str

class SignalResponse(BaseModel):
    """
    Defines the complete data structure sent back to the frontend.
    """
    status: str
    symbol: str
    historical_candles: List[Candle]
    ema9: List[IndicatorData]
    ema15: List[IndicatorData]
    ema200: List[IndicatorData]
    rsi14: List[IndicatorData]
    volume_data: List[IndicatorData]
    signal_markers: List[SignalMarker]
    latest_signal_text: str