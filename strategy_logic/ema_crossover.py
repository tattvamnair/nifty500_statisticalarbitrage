# stat_arb_trader_dhan/strategy_logic/ema_crossover.py

import pandas as pd
import numpy as np
from core.logger_setup import logger

def generate_signals(df: pd.DataFrame, use_trend_filter: bool = True):
    """
    Generates trading signals based on the 9/15 EMA Crossover strategy,
    enhanced with ADX and Moving Average Slope filters to avoid whipsaws.
    """
    if df.empty or 'EMA_9' not in df.columns:
        return pd.DataFrame() # Return empty if essential columns are missing
    
    signals_df = df.copy()

    # --- ADVANCED STRATEGY LOGIC ---

    # Layer 1: The Long-Term Trend Filter (The "Ocean Current")
    signals_df['is_uptrend'] = signals_df['close'] > signals_df['EMA_200']
    
    # Layer 2: The Sideways Market Detector (The "Doldrums")
    adx_threshold = 25
    signals_df['is_trending'] = signals_df.get('ADX_14', pd.Series(0, index=df.index)) > adx_threshold

    slope_period = 5
    slope_threshold = 0.0005 # 0.05% change over the slope period
    sma50_slope = (signals_df['SMA_50'] - signals_df['SMA_50'].shift(slope_period)) / signals_df['SMA_50'].shift(slope_period)
    signals_df['is_ma_angled_up'] = sma50_slope > slope_threshold
    signals_df['is_ma_angled_down'] = sma50_slope < -slope_threshold

    # Combine filters into master conditions
    signals_df['can_go_long'] = signals_df['is_uptrend'] & signals_df['is_trending'] & signals_df['is_ma_angled_up']
    signals_df['can_go_short'] = ~signals_df['is_uptrend'] & signals_df['is_trending'] & signals_df['is_ma_angled_down']

    # Layer 3: The Entry Trigger (The "Go" Signal)
    ema9_above_ema15 = signals_df['EMA_9'] > signals_df['EMA_15']
    
    # --- FIX for FutureWarning & BUG in short logic ---
    # Using modern `fill_value` syntax to silence warnings and correctly define crossovers.
    # A long entry is when the current state is True and the prior state was False.
    ema_crossed_up = ema9_above_ema15 & ~ema9_above_ema15.shift(1, fill_value=False)
    
    # A short entry is when the current state is False and the prior state was True.
    ema_crossed_down = ~ema9_above_ema15 & ema9_above_ema15.shift(1, fill_value=False)
    
    signals_df['long_entry_trigger'] = ema_crossed_up & signals_df['can_go_long']
    signals_df['short_entry_trigger'] = ema_crossed_down & signals_df['can_go_short']
    
    # Exit Triggers using modern syntax
    close_above_ema15 = signals_df['close'] > signals_df['EMA_15']
    signals_df['long_exit_trigger'] = ~close_above_ema15 & close_above_ema15.shift(1, fill_value=False)
    signals_df['short_exit_trigger'] = close_above_ema15 & ~close_above_ema15.shift(1, fill_value=False)
    # --- END OF FIX ---

    # --- Determine Position State & Generate Clearer Signals ---
    signals_df['position'] = 0
    signals_df['signal'] = 'HOLD' # Default signal

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