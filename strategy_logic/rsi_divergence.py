# stat_arb_trader_dhan/strategy_logic/rsi_divergence.py

import pandas as pd
import numpy as np
import pandas_ta as ta # Import the pandas_ta library to calculate indicators
from scipy.signal import find_peaks
from core.logger_setup import logger

def generate_signals(df: pd.DataFrame, use_trend_filter: bool = True):
    """
    Generates trading signals based on the RSI Divergence strategy.
    Now includes a 'divergence_status' column for better logging.
    """
    if df.empty:
        return pd.DataFrame() # Return empty if no data

    # --- 1. CALCULATE REQUIRED INDICATORS ---
    signals_df = df.copy()
    signals_df['RSI_14'] = ta.rsi(signals_df['close'], length=14)
    signals_df['VMA_20'] = ta.sma(signals_df['volume'], length=20)
    signals_df['SMA_200'] = ta.sma(signals_df['close'], length=200)

    # --- 2. INITIALIZE COLUMNS & ADD DIVERGENCE TAG ---
    signals_df['signal'] = 'HOLD'
    signals_df['position'] = 0
    # ** NEW: Add a column to track the divergence status for the dashboard **
    signals_df['divergence_status'] = 'NONE'

    # --- 3. ROBUSTNESS FIX: Check if indicators are valid before proceeding ---
    required_cols = ['RSI_14', 'VMA_20', 'SMA_200']
    if signals_df[required_cols].isnull().iloc[-1].any():
        logger.warning("RSI Divergence strategy cannot run; required indicators have NaN values (insufficient history). Skipping.")
        return pd.DataFrame()

    # --- 4. Find Swing Highs and Lows ---
    swing_high_indices, _ = find_peaks(signals_df['high'], distance=10)
    swing_low_indices, _ = find_peaks(-signals_df['low'], distance=10)

    # --- 5. Detect Divergences and Generate Entry Signals ---
    # Bullish Divergence
    for i in range(len(swing_low_indices) - 1):
        idx1, idx2 = swing_low_indices[i], swing_low_indices[i+1]
        
        price_low1, price_low2 = signals_df['low'].iloc[idx1], signals_df['low'].iloc[idx2]
        rsi_low1, rsi_low2 = signals_df['RSI_14'].iloc[idx1], signals_df['RSI_14'].iloc[idx2]

        if price_low2 < price_low1 and rsi_low2 > rsi_low1 and rsi_low1 <= 35:
            # ** NEW: Tag the divergence status from this point forward **
            signals_df.iloc[idx2:, signals_df.columns.get_loc('divergence_status')] = 'BULLISH'
            for j in range(idx2 + 1, min(idx2 + 6, len(signals_df))):
                confirmation_candle = signals_df.iloc[j]
                
                is_bullish_candle = confirmation_candle['close'] > confirmation_candle['open']
                is_breakout = confirmation_candle['close'] > signals_df['high'].iloc[idx2]
                volume_confirmed = confirmation_candle['volume'] > confirmation_candle['VMA_20']
                trend_confirmed = not use_trend_filter or (use_trend_filter and confirmation_candle['close'] > confirmation_candle['SMA_200'])

                if is_bullish_candle and is_breakout and volume_confirmed and trend_confirmed:
                    signals_df.iloc[j, signals_df.columns.get_loc('signal')] = 'BULLISH_DIVERGENCE_ENTRY'
                    break

    # Bearish Divergence
    for i in range(len(swing_high_indices) - 1):
        idx1, idx2 = swing_high_indices[i], swing_high_indices[i+1]

        price_high1, price_high2 = signals_df['high'].iloc[idx1], signals_df['high'].iloc[idx2]
        rsi_high1, rsi_high2 = signals_df['RSI_14'].iloc[idx1], signals_df['RSI_14'].iloc[idx2]

        if price_high2 > price_high1 and rsi_high2 < rsi_high1 and rsi_high1 >= 65:
            # ** NEW: Tag the divergence status from this point forward **
            signals_df.iloc[idx2:, signals_df.columns.get_loc('divergence_status')] = 'BEARISH'
            for j in range(idx2 + 1, min(idx2 + 6, len(signals_df))):
                confirmation_candle = signals_df.iloc[j]

                is_bearish_candle = confirmation_candle['close'] < confirmation_candle['open']
                is_breakdown = confirmation_candle['close'] < signals_df['low'].iloc[idx2]
                volume_confirmed = confirmation_candle['volume'] > confirmation_candle['VMA_20']
                trend_confirmed = not use_trend_filter or (use_trend_filter and confirmation_candle['close'] < confirmation_candle['SMA_200'])

                if is_bearish_candle and is_breakdown and volume_confirmed and trend_confirmed:
                    signals_df.iloc[j, signals_df.columns.get_loc('signal')] = 'BEARISH_DIVERGENCE_ENTRY'
                    break

    # --- 6. Generate Exit Signals and Position State (Robust Loop) ---
    position_state = 0
    signal_col_loc = signals_df.columns.get_loc('signal')
    position_col_loc = signals_df.columns.get_loc('position')
    
    for i in range(1, len(signals_df)):
        current_signal = signals_df.iat[i, signal_col_loc]
        
        if position_state == 0:
            if current_signal == 'BULLISH_DIVERGENCE_ENTRY': position_state = 1
            elif current_signal == 'BEARISH_DIVERGENCE_ENTRY': position_state = -1
        
        elif position_state == 1:
            signals_df.iat[i, signal_col_loc] = 'HOLD_LONG'
            if signals_df['RSI_14'].iloc[i] > 65:
                signals_df.iat[i, signal_col_loc] = 'LONG_EXIT_RSI'
                position_state = 0
        
        elif position_state == -1:
            signals_df.iat[i, signal_col_loc] = 'HOLD_SHORT'
            if signals_df['RSI_14'].iloc[i] < 35:
                signals_df.iat[i, signal_col_loc] = 'SHORT_EXIT_RSI'
                position_state = 0

        signals_df.iat[i, position_col_loc] = position_state
        
    return signals_df