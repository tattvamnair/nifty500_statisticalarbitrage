# stat_arb_trader_dhan/strategy_logic/ema_crossover.py

import pandas as pd
import numpy as np
from core.logger_setup import logger

def generate_signals(df: pd.DataFrame, use_trend_filter: bool = True):
    """
    Generates trading signals using an ADAPTIVE strategy.
    - For Intraday, it uses a VWAP filter.
    - For Daily/Positional, it uses an EMA_200 filter.
    The EMA crossover pair (9/15) is constant for all timeframes.
    """
    if df.empty:
        return pd.DataFrame()
    
    signals_df = df.copy()

    is_intraday_run = signals_df.index.to_series().dt.date.duplicated().any()

    if is_intraday_run and 'VWAP' in signals_df.columns:
        trend_filter_col = 'VWAP'
        adx_threshold = 22
        logger.info(f"  > Running in INTRADAY mode (9/15 EMA with VWAP filter).")
    else:
        trend_filter_col = 'EMA_200'
        adx_threshold = 25
        logger.info(f"  > Running in DAILY/POSITIONAL mode (9/15 EMA with EMA_200 filter).")

    fast_ema, slow_ema = 'EMA_9', 'EMA_15'

    required_cols = ['close', trend_filter_col, fast_ema, slow_ema, 'ADX_14', 'SMA_50']
    if not all(col in signals_df.columns for col in required_cols):
        logger.warning(f"  > Strategy prerequisite columns missing. Skipping.")
        return pd.DataFrame()

    # ========================== FIX 1: CORRECT NAN CHECK ==========================
    # We only check the FINAL row for NaN. This correctly skips new stocks without
    # incorrectly flagging all of them.
    if pd.isna(signals_df[trend_filter_col].iloc[-1]):
        logger.warning(f"  > Trend filter '{trend_filter_col}' is invalid for the latest candle. Insufficient history. Skipping.")
        return pd.DataFrame()
    # ==============================================================================

    signals_df['is_uptrend'] = signals_df['close'] > signals_df[trend_filter_col]
    signals_df['is_trending'] = signals_df['ADX_14'].fillna(0) > adx_threshold
    slope_period = 5
    slope_threshold = 0.0005 
    sma50_slope = (signals_df['SMA_50'] - signals_df['SMA_50'].shift(slope_period)) / signals_df['SMA_50'].shift(slope_period)
    signals_df['is_ma_angled_up'] = sma50_slope > slope_threshold
    signals_df['is_ma_angled_down'] = sma50_slope < -slope_threshold
    signals_df['can_go_long'] = signals_df['is_uptrend'] & signals_df['is_trending'] & signals_df['is_ma_angled_up']
    signals_df['can_go_short'] = ~signals_df['is_uptrend'] & signals_df['is_trending'] & signals_df['is_ma_angled_down']
    ema_fast_above_slow = signals_df[fast_ema] > signals_df[slow_ema]
    ema_crossed_up = ema_fast_above_slow & ~ema_fast_above_slow.shift(1, fill_value=False)
    ema_crossed_down = ~ema_fast_above_slow & ema_fast_above_slow.shift(1, fill_value=False)
    signals_df['long_entry_trigger'] = ema_crossed_up & signals_df['can_go_long']
    signals_df['short_entry_trigger'] = ema_crossed_down & signals_df['can_go_short']
    close_above_slow_ema = signals_df['close'] > signals_df[slow_ema]
    signals_df['long_exit_trigger'] = ~close_above_slow_ema & close_above_slow_ema.shift(1, fill_value=False)
    signals_df['short_exit_trigger'] = close_above_slow_ema & ~close_above_slow_ema.shift(1, fill_value=False)

    # ================= FIX 2: ROBUST FOR-LOOP (IMMUNE TO BAD INDEX) =================
    # This loop now uses integer positions (iloc/iat) which are always unique.
    # This prevents the "ambiguous truth value" error.
    signals_df['position'] = 0
    signals_df['signal'] = 'HOLD' 
    position_col_loc = signals_df.columns.get_loc('position')
    signal_col_loc = signals_df.columns.get_loc('signal')
    position_state = 0
    for i in range(1, len(signals_df)):
        long_entry = signals_df['long_entry_trigger'].iloc[i]
        short_entry = signals_df['short_entry_trigger'].iloc[i]
        long_exit = signals_df['long_exit_trigger'].iloc[i]
        short_exit = signals_df['short_exit_trigger'].iloc[i]
        
        if position_state == 0:
            if long_entry:
                position_state = 1
                signals_df.iat[i, signal_col_loc] = 'BUY'
            elif short_entry:
                position_state = -1
                signals_df.iat[i, signal_col_loc] = 'SELL'
        elif position_state == 1:
            if long_exit:
                position_state = 0
                signals_df.iat[i, signal_col_loc] = 'EXIT_LONG'
            else:
                signals_df.iat[i, signal_col_loc] = 'HOLD_LONG'
        elif position_state == -1:
            if short_exit:
                position_state = 0
                signals_df.iat[i, signal_col_loc] = 'EXIT_SHORT'
            else:
                signals_df.iat[i, signal_col_loc] = 'HOLD_SHORT'
        signals_df.iat[i, position_col_loc] = position_state
    # ================================================================================

    columns_to_drop = [
        'is_uptrend', 'is_trending', 'is_ma_angled_up', 'is_ma_angled_down',
        'can_go_long', 'can_go_short', 'long_entry_trigger', 'short_entry_trigger',
        'long_exit_trigger', 'short_exit_trigger'
    ]
    signals_df.drop(columns=[col for col in columns_to_drop if col in signals_df.columns], inplace=True)
    
    return signals_df