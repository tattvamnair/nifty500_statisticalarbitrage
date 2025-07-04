# stat_arb_trader_dhan/strategy_logic/stat_arb.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from collections import namedtuple
import time

from core.logger_setup import logger

# --- Configuration Parameters for the Strategy ---
ROLLING_WINDOW = 60
CORRELATION_THRESHOLD = 0.90
ADF_P_VALUE_THRESHOLD = 0.01
MIN_HALF_LIFE = 5
MAX_HALF_LIFE = 100
Z_SCORE_ENTRY = 2.5
Z_SCORE_STOP_LOSS = 3
Z_SCORE_EXIT = 0.5

# --- Data Structures ---
PairSignal = namedtuple('PairSignal', [
    'pair', 'signal_type', 'reason', 'z_score', 'hedge_ratio', 'half_life'
])

strategy_cache = { "cointegrated_pairs": None }

# --- Helper Functions (Unchanged) ---
def _calculate_adf_test(series):
    try:
        result = adfuller(series.dropna())
        return result[1]
    except Exception:
        return 1.0

def _calculate_half_life(spread):
    try:
        spread_lag = spread.shift(1)
        spread_delta = spread - spread_lag
        df = pd.DataFrame({'delta': spread_delta, 'lag': spread_lag}).dropna()
        if len(df) < 10: return -1
        reg = sm.OLS(df['delta'], sm.add_constant(df['lag'])).fit()
        lambda_val = reg.params.iloc[1]
        if abs(lambda_val) < 1e-6 or lambda_val >= 0: return -1
        return -np.log(2) / lambda_val
    except Exception:
        return -1

# --- Core Logic Functions ---
def find_cointegrated_pairs(all_data_dict, formation_candles):
    """Analyzes historical data to find high-quality cointegrated pairs."""
    logger.info(f"--- Running Cointegration Analysis on {formation_candles} candles ---")
    log_prices = {
        symbol: np.log(df['close'].dropna())
        for symbol, df in all_data_dict.items() if len(df) >= formation_candles
    }
    valid_symbols = list(log_prices.keys())
    if len(valid_symbols) < 2:
        logger.warning("Not enough symbols with sufficient data to form pairs.")
        return []

    price_df = pd.DataFrame(log_prices)
    corr_matrix = price_df.corr()
    potential_pairs = [
        (s1, s2) for s1, s2 in combinations(valid_symbols, 2)
        if s1 in corr_matrix and s2 in corr_matrix.columns and corr_matrix.loc[s1, s2] > CORRELATION_THRESHOLD
    ]

    cointegrated_pairs = []
    for s1, s2 in potential_pairs:
        try:
            series1, series2 = log_prices[s1], log_prices[s2]
            temp_df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
            if len(temp_df) < formation_candles * 0.9: continue
            if _calculate_adf_test(temp_df['s1']) < ADF_P_VALUE_THRESHOLD or \
               _calculate_adf_test(temp_df['s2']) < ADF_P_VALUE_THRESHOLD:
                continue

            # (Nit #6b) Add try/except for robustness against singular matrix errors
            try:
                model = sm.OLS(temp_df['s1'], sm.add_constant(temp_df['s2'])).fit()
            except np.linalg.LinAlgError:
                logger.debug(f"OLS regression failed for pair ({s1}, {s2}) due to singular matrix. Skipping.")
                continue
                
            alpha, beta = model.params
            spread = temp_df['s1'] - (alpha + beta * temp_df['s2'])
            p_value_spread = _calculate_adf_test(spread)

            if p_value_spread < ADF_P_VALUE_THRESHOLD:
                half_life = _calculate_half_life(spread)
                if MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE:
                    cointegrated_pairs.append({
                        "pair": (s1, s2), "half_life": half_life, "hedge_ratio": (alpha, beta)
                    })
                    logger.info(f"  > SUCCESS: Pair ({s1}, {s2}) cointegrated. P-val: {p_value_spread:.4f}, Half-life: {half_life:.2f}")
        except Exception as e:
            logger.debug(f"Could not process pair ({s1}, {s2}) during cointegration search: {e}")
            continue
    logger.info(f"--- Cointegration Analysis Complete. Found {len(cointegrated_pairs)} pairs. ---")
    return cointegrated_pairs

def generate_pair_signals(pair_data_slice, pair_info, open_position_state):
    """Generates a trading signal for a SINGLE pair based on the latest data slice."""
    try:
        s1_symbol, s2_symbol = pair_info['pair']
        s1_df, s2_df = pair_data_slice[s1_symbol], pair_data_slice[s2_symbol]
        
        # We need ROLLING_WINDOW + 1 bars to have a held-out bar for the z-score
        if len(s1_df) < ROLLING_WINDOW + 1 or len(s2_df) < ROLLING_WINDOW + 1:
            return None

        s1_log_prices = np.log(s1_df['close'])
        s2_log_prices = np.log(s2_df['close'])

        # (Fix #3) DYNAMIC HEDGE RATIO: Fit OLS only on the rolling window for adaptiveness.
        # We use the full slice passed to us for this calculation.
        
        # Fit on data[:-1] (the lookback window)
        model = sm.OLS(s1_log_prices.iloc[:-1], sm.add_constant(s2_log_prices.iloc[:-1])).fit()
        alpha, beta = model.params
        hedge_ratio = (alpha, beta)

        # Calculate spread and z-score based on this dynamic hedge ratio
        spread = s1_log_prices - (hedge_ratio[0] + hedge_ratio[1] * s2_log_prices)
        mean_spread = spread.iloc[:-1].mean()
        std_spread = spread.iloc[:-1].std()

        if std_spread < 1e-6: return None

        current_z_score = (spread.iloc[-1] - mean_spread) / std_spread
        time_stop_candles = int(3 * pair_info['half_life'])

        if open_position_state:
            is_closing, exit_reason = False, ""
            if abs(current_z_score) <= Z_SCORE_EXIT:
                is_closing, exit_reason = True, f"PROFIT TARGET (Z-Score crossed {Z_SCORE_EXIT:.2f})"
            elif (open_position_state['direction'] == 'LONG' and current_z_score <= -Z_SCORE_STOP_LOSS) or \
                 (open_position_state['direction'] == 'SHORT' and current_z_score >= Z_SCORE_STOP_LOSS):
                is_closing, exit_reason = True, f"STATISTICAL STOP (Z-Score hit {Z_SCORE_STOP_LOSS:.1f})"
            elif 'bars_held' in open_position_state and open_position_state['bars_held'] > time_stop_candles:
                 is_closing, exit_reason = True, f"TIME STOP ({open_position_state['bars_held']} > {time_stop_candles} bars)"
            
            if is_closing:
                return PairSignal(pair_info['pair'], f"EXIT_{open_position_state['direction']}", exit_reason, current_z_score, hedge_ratio, pair_info['half_life'])
            else:
                return PairSignal(pair_info['pair'], f"HOLD_{open_position_state['direction']}", "Position Open", current_z_score, hedge_ratio, pair_info['half_life'])
        else:
            if current_z_score > Z_SCORE_ENTRY:
                return PairSignal(pair_info['pair'], "ENTER_SHORT", f"Z-Score > {Z_SCORE_ENTRY:.1f}", current_z_score, hedge_ratio, pair_info['half_life'])
            elif current_z_score < -Z_SCORE_ENTRY:
                return PairSignal(pair_info['pair'], "ENTER_LONG", f"Z-Score < -{Z_SCORE_ENTRY:.1f}", current_z_score, hedge_ratio, pair_info['half_life'])
                
    except Exception as e:
        s1_symbol, s2_symbol = pair_info['pair']
        logger.debug(f"Signal generation failed for pair ({s1_symbol}, {s2_symbol}): {e}")
        return None
    return None