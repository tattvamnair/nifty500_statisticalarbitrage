# stat_arb_trader_dhan/strategy_logic/rsi_divergence.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from core.logger_setup import logger

def generate_signals(df: pd.DataFrame, use_trend_filter: bool = True):
    """
    Generates trading signals based on the RSI Divergence strategy.
    """
    if df.empty or 'RSI_14' not in df.columns:
        return df

    signals_df = df.copy()
    
    # --- Find Swing Highs and Lows ---
    # The 'distance' parameter is crucial to avoid detecting minor fluctuations.
    swing_high_indices, _ = find_peaks(signals_df['high'], distance=10)
    swing_low_indices, _ = find_peaks(-signals_df['low'], distance=10)

    signals_df['signal'] = 'HOLD'
    signals_df['position'] = 0

    # --- Detect Divergences and Generate Entry Signals ---
    # Bullish Divergence: Lower Lows in Price, Higher Lows in RSI
    for i in range(len(swing_low_indices) - 1):
        idx1, idx2 = swing_low_indices[i], swing_low_indices[i+1]
        
        price_low1, price_low2 = signals_df['low'].iloc[idx1], signals_df['low'].iloc[idx2]
        rsi_low1, rsi_low2 = signals_df['RSI_14'].iloc[idx1], signals_df['RSI_14'].iloc[idx2]

        if price_low2 < price_low1 and rsi_low2 > rsi_low1 and rsi_low1 <= 35:
            # Found a divergence. Now look for the confirmation candle.
            for j in range(idx2 + 1, min(idx2 + 6, len(signals_df))):
                confirmation_candle = signals_df.iloc[j]
                
                # Confirmation Criteria
                is_bullish_candle = confirmation_candle['close'] > confirmation_candle['open']
                is_breakout = confirmation_candle['close'] > signals_df['high'].iloc[idx2]
                volume_confirmed = confirmation_candle['volume'] > confirmation_candle['VMA_20']
                trend_confirmed = not use_trend_filter or (use_trend_filter and confirmation_candle['close'] > confirmation_candle['SMA_200'])

                if is_bullish_candle and is_breakout and volume_confirmed and trend_confirmed:
                    signals_df.loc[signals_df.index[j], 'signal'] = 'BULLISH_DIVERGENCE_ENTRY'
                    break # Stop looking for confirmation once found

    # Bearish Divergence: Higher Highs in Price, Lower Highs in RSI
    for i in range(len(swing_high_indices) - 1):
        idx1, idx2 = swing_high_indices[i], swing_high_indices[i+1]

        price_high1, price_high2 = signals_df['high'].iloc[idx1], signals_df['high'].iloc[idx2]
        rsi_high1, rsi_high2 = signals_df['RSI_14'].iloc[idx1], signals_df['RSI_14'].iloc[idx2]

        if price_high2 > price_high1 and rsi_high2 < rsi_high1 and rsi_high1 >= 65:
            # Found a divergence. Now look for the confirmation candle.
            for j in range(idx2 + 1, min(idx2 + 6, len(signals_df))):
                confirmation_candle = signals_df.iloc[j]

                is_bearish_candle = confirmation_candle['close'] < confirmation_candle['open']
                is_breakdown = confirmation_candle['close'] < signals_df['low'].iloc[idx2]
                volume_confirmed = confirmation_candle['volume'] > confirmation_candle['VMA_20']
                trend_confirmed = not use_trend_filter or (use_trend_filter and confirmation_candle['close'] < confirmation_candle['SMA_200'])

                if is_bearish_candle and is_breakdown and volume_confirmed and trend_confirmed:
                    signals_df.loc[signals_df.index[j], 'signal'] = 'BEARISH_DIVERGENCE_ENTRY'
                    break

    # --- Generate Exit Signals and Position State ---
    position_state = 0
    for i in range(1, len(signals_df)):
        current_signal = signals_df.loc[signals_df.index[i], 'signal']
        
        if position_state == 0:
            if current_signal == 'BULLISH_DIVERGENCE_ENTRY': position_state = 1
            elif current_signal == 'BEARISH_DIVERGENCE_ENTRY': position_state = -1
        
        elif position_state == 1:
            if signals_df['RSI_14'].iloc[i] > 65: # Exit long if RSI becomes overbought
                signals_df.loc[signals_df.index[i], 'signal'] = 'LONG_EXIT_RSI'
                position_state = 0
        
        elif position_state == -1:
            if signals_df['RSI_14'].iloc[i] < 35: # Exit short if RSI becomes oversold
                signals_df.loc[signals_df.index[i], 'signal'] = 'SHORT_EXIT_RSI'
                position_state = 0

        signals_df.loc[signals_df.index[i], 'position'] = position_state
        
    return signals_df