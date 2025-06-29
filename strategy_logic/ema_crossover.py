# stat_arb_trader_dhan/strategy_logic/ema_crossover.py

import pandas as pd
from core.logger_setup import logger

def generate_signals(df: pd.DataFrame, use_trend_filter: bool = True):
    """
    Generates trading signals based on the 9/15 EMA Crossover strategy.
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original DataFrame
    signals_df = df.copy()

    # --- Signal Logic ---
    # 1. Calculate the primary boolean conditions
    signals_df['ema9_above_ema15'] = signals_df['EMA_9'] > signals_df['EMA_15']
    signals_df['close_above_ema15'] = signals_df['close'] > signals_df['EMA_15']
    
    # 2. Calculate the trend filter condition (optional)
    if use_trend_filter:
        signals_df['price_above_ema200'] = signals_df['close'] > signals_df['EMA_200']

    # 3. Generate Entry Signals based on crossovers
    # A long entry happens when the fast EMA crosses ABOVE the slow EMA
    signals_df['long_entry_trigger'] = (signals_df['ema9_above_ema15']) & (~signals_df['ema9_above_ema15'].shift(1).fillna(False))
    
    # A short entry happens when the fast EMA crosses BELOW the slow EMA
    signals_df['short_entry_trigger'] = (~signals_df['ema9_above_ema15']) & (signals_df['ema9_above_ema15'].shift(1).fillna(False))

    # Apply trend filter if enabled
    if use_trend_filter:
        signals_df.loc[~signals_df['price_above_ema200'], 'long_entry_trigger'] = False
        signals_df.loc[signals_df['price_above_ema200'], 'short_entry_trigger'] = False

    # 4. Generate Exit Signals based on price vs. the slow EMA
    # A long exit happens when the close price crosses BELOW the slow EMA
    signals_df['long_exit_trigger'] = (~signals_df['close_above_ema15']) & (signals_df['close_above_ema15'].shift(1).fillna(False))

    # A short exit happens when the close price crosses ABOVE the slow EMA
    signals_df['short_exit_trigger'] = (signals_df['close_above_ema15']) & (~signals_df['close_above_ema15'].shift(1).fillna(False))

    # 5. Determine Position State
    signals_df['position'] = 0
    signals_df['signal'] = 'HOLD'

    position_state = 0
    for i in range(1, len(signals_df)):
        idx = signals_df.index[i]
        
        if position_state == 0:
            if signals_df.loc[idx, 'long_entry_trigger']:
                position_state = 1
                signals_df.loc[idx, 'signal'] = 'LONG_ENTRY'
            elif signals_df.loc[idx, 'short_entry_trigger']:
                position_state = -1
                signals_df.loc[idx, 'signal'] = 'SHORT_ENTRY'
        
        elif position_state == 1 and signals_df.loc[idx, 'long_exit_trigger']:
            position_state = 0
            signals_df.loc[idx, 'signal'] = 'LONG_EXIT'
        
        elif position_state == -1 and signals_df.loc[idx, 'short_exit_trigger']:
            position_state = 0
            signals_df.loc[idx, 'signal'] = 'SHORT_EXIT'

        signals_df.loc[idx, 'position'] = position_state

    # Clean up intermediate helper columns
    columns_to_drop = [
        'ema9_above_ema15', 'close_above_ema15', 'long_entry_trigger',
        'short_entry_trigger', 'long_exit_trigger', 'short_exit_trigger'
    ]
    if use_trend_filter:
        columns_to_drop.append('price_above_ema200')
        
    signals_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    return signals_df