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

    # --- ADAPTIVE STRATEGY LOGIC ---
    
    # --- FINAL BUG FIX: Use a foolproof method to detect intraday runs ---
    # We check if any date in the index is duplicated. This is only true for intraday data.
    is_intraday_run = signals_df.index.to_series().dt.date.duplicated().any()

    # Set parameters based on the correctly detected mode
    if is_intraday_run and 'VWAP' in signals_df.columns:
        # --- INTRADAY-OPTIMIZED PARAMETERS ---
        trend_filter_col = 'VWAP'
        adx_threshold = 22
        logger.info(f"  > Running in INTRADAY mode (9/15 EMA with VWAP filter).")
    else:
        # --- DAILY/POSITIONAL PARAMETERS ---
        trend_filter_col = 'EMA_200'
        adx_threshold = 25
        logger.info(f"  > Running in DAILY/POSITIONAL mode (9/15 EMA with EMA_200 filter).")

    fast_ema, slow_ema = 'EMA_9', 'EMA_15'

    # Defensive check to ensure all needed columns exist before proceeding
    required_cols = [trend_filter_col, fast_ema, slow_ema, 'ADX_14', 'SMA_50']
    if not all(col in signals_df.columns for col in required_cols):
        logger.warning(f"  > Strategy prerequisite columns missing for the selected mode. Cannot generate signals.")
        return pd.DataFrame()

    # Layer 1: The Trend Filter (Adaptive)
    signals_df['is_uptrend'] = signals_df['close'] > signals_df[trend_filter_col]
    
    # Layer 2: The Sideways Market Detector
    signals_df['is_trending'] = signals_df['ADX_14'].fillna(0) > adx_threshold
    slope_period = 5
    slope_threshold = 0.0005 
    sma50_slope = (signals_df['SMA_50'] - signals_df['SMA_50'].shift(slope_period)) / signals_df['SMA_50'].shift(slope_period)
    signals_df['is_ma_angled_up'] = sma50_slope > slope_threshold
    signals_df['is_ma_angled_down'] = sma50_slope < -slope_threshold

    # Combine filters into master conditions
    signals_df['can_go_long'] = signals_df['is_uptrend'] & signals_df['is_trending'] & signals_df['is_ma_angled_up']
    signals_df['can_go_short'] = ~signals_df['is_uptrend'] & signals_df['is_trending'] & signals_df['is_ma_angled_down']

    # Layer 3: The Entry Trigger
    ema_fast_above_slow = signals_df[fast_ema] > signals_df[slow_ema]
    ema_crossed_up = ema_fast_above_slow & ~ema_fast_above_slow.shift(1, fill_value=False)
    ema_crossed_down = ~ema_fast_above_slow & ema_fast_above_slow.shift(1, fill_value=False)
    
    signals_df['long_entry_trigger'] = ema_crossed_up & signals_df['can_go_long']
    signals_df['short_entry_trigger'] = ema_crossed_down & signals_df['can_go_short']
    
    # Exit Triggers (using the slower EMA of the pair)
    close_above_slow_ema = signals_df['close'] > signals_df[slow_ema]
    signals_df['long_exit_trigger'] = ~close_above_slow_ema & close_above_slow_ema.shift(1, fill_value=False)
    signals_df['short_exit_trigger'] = close_above_slow_ema & ~close_above_slow_ema.shift(1, fill_value=False)

    # --- Determine Position State & Generate Clearer Signals ---
    signals_df['position'] = 0
    signals_df['signal'] = 'HOLD' 

    position_state = 0
    for i in range(1, len(signals_df)):
        idx = signals_df.index[i]
        
        if position_state == 0:
            if signals_df.loc[idx, 'long_entry_trigger']:
                position_state = 1
                signals_df.loc[idx, 'signal'] = 'BUY'
            elif signals_df.loc[idx, 'short_entry_trigger']:
                position_state = -1
                signals_df.loc[idx, 'signal'] = 'SELL'
        
        elif position_state == 1:
            if signals_df.loc[idx, 'long_exit_trigger']:
                position_state = 0
                signals_df.loc[idx, 'signal'] = 'EXIT_LONG'
            else:
                signals_df.loc[idx, 'signal'] = 'HOLD_LONG'
        
        elif position_state == -1:
            if signals_df.loc[idx, 'short_exit_trigger']:
                position_state = 0
                signals_df.loc[idx, 'signal'] = 'EXIT_SHORT'
            else:
                signals_df.loc[idx, 'signal'] = 'HOLD_SHORT'

        signals_df.loc[idx, 'position'] = position_state

    # Clean up all intermediate helper columns
    columns_to_drop = [
        'is_uptrend', 'is_trending', 'is_ma_angled_up', 'is_ma_angled_down',
        'can_go_long', 'can_go_short', 'long_entry_trigger', 'short_entry_trigger',
        'long_exit_trigger', 'short_exit_trigger'
    ]
    signals_df.drop(columns=[col for col in columns_to_drop if col in signals_df.columns], inplace=True)
    
    return signals_df