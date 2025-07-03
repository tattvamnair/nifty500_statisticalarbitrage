# stat_arb_trader_dhan/strategy_logic/stat_arb.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from collections import namedtuple
import time

from core.logger_setup import logger

# --- Configuration Parameters for the Strategy (Tuned for Higher Quality Signals) ---
FORMATION_PERIOD_CANDLES = 252
ROLLING_WINDOW = 60
CORRELATION_THRESHOLD = 0.90
ADF_P_VALUE_THRESHOLD = 0.01  # Stricter P-value
MIN_HALF_LIFE = 5
MAX_HALF_LIFE = 50            # Tighter Half-Life Window
Z_SCORE_ENTRY = 2.5           # Stricter Entry Threshold
Z_SCORE_STOP_LOSS = 3.0
Z_SCORE_EXIT = 0.0

# --- Data Structures ---
PairSignal = namedtuple('PairSignal', [
    'pair', 'signal_type', 'reason', 'z_score', 'hedge_ratio', 'half_life', 'trade_plan', 'trade_details'
])

strategy_cache = {
    "cointegrated_pairs": None,
    # Will store entry_index for correct bar-based time stops
    "open_positions": {}
}

# --- Helper Functions ---
def _calculate_adf_test(series):
    """Helper function to run the Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return result[1]

def _calculate_half_life(spread):
    """Calculates the half-life of mean reversion for a spread."""
    try:
        spread_lag = spread.shift(1)
        spread_delta = spread - spread_lag
        df = pd.DataFrame({'delta': spread_delta, 'lag': spread_lag}).dropna()
        if len(df) < 10: return -1
        reg = sm.OLS(df['delta'], sm.add_constant(df['lag'])).fit()
        lambda_val = reg.params.iloc[1]
        
        # FIX: Handle tiny lambda values to prevent extreme half-life calculations
        if abs(lambda_val) < 1e-6: return -1
        if lambda_val >= 0: return -1

        return -np.log(2) / lambda_val
    except Exception:
        return -1
        
# --- Core Logic Functions ---
def find_cointegrated_pairs(all_data_dict, num_candles):
    """Analyzes historical data to find high-quality cointegrated pairs."""
    logger.info("--- Starting Cointegration Analysis ---")
    log_prices = {
        symbol: np.log(df['close'].dropna())
        for symbol, df in all_data_dict.items() if len(df) >= num_candles
    }
    valid_symbols = list(log_prices.keys())
    if len(valid_symbols) < 2:
        logger.warning("Not enough symbols with sufficient data to form pairs.")
        return []
    logger.info(f"Phase 1: Screening {len(valid_symbols)} stocks for correlation > {CORRELATION_THRESHOLD}.")
    price_df = pd.DataFrame(log_prices)
    corr_matrix = price_df.corr()
    potential_pairs = [
        (s1, s2) for s1, s2 in combinations(valid_symbols, 2)
        if s1 in corr_matrix and s2 in corr_matrix.columns and corr_matrix.loc[s1, s2] > CORRELATION_THRESHOLD
    ]
    if not potential_pairs:
        logger.info("No potential pairs found after correlation filter.")
        return []
    logger.info(f"Found {len(potential_pairs)} potential pairs. Proceeding to cointegration testing.")
    cointegrated_pairs = []
    for s1, s2 in potential_pairs:
        try:
            series1, series2 = log_prices[s1], log_prices[s2]
            temp_df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
            if len(temp_df) < num_candles * 0.8: continue
            aligned_s1, aligned_s2 = temp_df['s1'], temp_df['s2']
            if _calculate_adf_test(aligned_s1) < ADF_P_VALUE_THRESHOLD or _calculate_adf_test(aligned_s2) < ADF_P_VALUE_THRESHOLD: continue
            model = sm.OLS(aligned_s1, sm.add_constant(aligned_s2)).fit()
            spread = aligned_s1 - model.predict(sm.add_constant(aligned_s2))
            p_value_spread = _calculate_adf_test(spread)
            if p_value_spread < ADF_P_VALUE_THRESHOLD:
                half_life = _calculate_half_life(spread)
                if MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE:
                    cointegrated_pairs.append({"pair": (s1, s2), "half_life": half_life})
                    logger.info(f"  > \033[92mSUCCESS:\033[0m Pair ({s1}, {s2}) is cointegrated. P-value: {p_value_spread:.4f}, Half-life: {half_life:.2f} candles.")
        except Exception as e:
            logger.warning(f"Could not process pair ({s1}, {s2}) during cointegration search due to an error: {e}. Skipping pair.")
            continue
    logger.info(f"--- Cointegration Analysis Complete. Found {len(cointegrated_pairs)} high-quality pairs. ---")
    return cointegrated_pairs

def generate_pair_signals(all_data_dict, pairs_to_check):
    """Generates trading signals with full per-leg trade plans."""
    if not pairs_to_check: return []
    
    signals = []
    open_positions = strategy_cache["open_positions"]

    for pair_info in pairs_to_check:
        try:
            pair = pair_info['pair']
            s1_symbol, s2_symbol = pair
            
            if s1_symbol not in all_data_dict or s2_symbol not in all_data_dict: continue

            s1_df, s2_df = all_data_dict[s1_symbol], all_data_dict[s2_symbol]
            s1_log_prices, s2_log_prices = np.log(s1_df['close']), np.log(s2_df['close'])

            rolling_df = pd.DataFrame({'s1': s1_log_prices, 's2': s2_log_prices}).dropna()
            s1_rolling, s2_rolling = rolling_df['s1'].tail(ROLLING_WINDOW), rolling_df['s2'].tail(ROLLING_WINDOW)

            if len(s1_rolling) < ROLLING_WINDOW: continue

            model = sm.OLS(s1_rolling, sm.add_constant(s2_rolling)).fit()
            hedge_ratio = model.params.iloc[1]
            spread = s1_rolling - (hedge_ratio * s2_rolling)
            mean_spread, std_spread = spread.mean(), spread.std()
            
            if std_spread == 0: continue

            current_z_score = (spread.iloc[-1] - mean_spread) / std_spread

            def get_trade_details(z_stop_level):
                details = {}
                s1_current_price, s2_current_price = s1_df['close'].iloc[-1], s2_df['close'].iloc[-1]
                s1_log_target = (Z_SCORE_EXIT * std_spread) + mean_spread + (hedge_ratio * np.log(s2_current_price))
                details['s1_target'] = np.exp(s1_log_target)
                s2_log_target = (np.log(s1_current_price) - (Z_SCORE_EXIT * std_spread) - mean_spread) / hedge_ratio
                details['s2_target'] = np.exp(s2_log_target)
                s1_log_stop = (z_stop_level * std_spread) + mean_spread + (hedge_ratio * np.log(s2_current_price))
                details['s1_stop'] = np.exp(s1_log_stop)
                s2_log_stop = (np.log(s1_current_price) - (z_stop_level * std_spread) - mean_spread) / hedge_ratio
                details['s2_stop'] = np.exp(s2_log_stop)
                details['s1_entry'], details['s2_entry'] = s1_current_price, s2_current_price
                return details

            time_stop_candles = int(2.5 * pair_info['half_life'])
            
            if pair in open_positions:
                position = open_positions[pair]
                is_closing, exit_reason = False, ""
                if (position['direction'] == 'LONG' and current_z_score >= Z_SCORE_EXIT) or (position['direction'] == 'SHORT' and current_z_score <= Z_SCORE_EXIT):
                    is_closing, exit_reason = True, f"PROFIT TARGET (Z-Score crossed {Z_SCORE_EXIT:.1f})"
                elif (position['direction'] == 'LONG' and current_z_score <= -Z_SCORE_STOP_LOSS) or (position['direction'] == 'SHORT' and current_z_score >= Z_SCORE_STOP_LOSS):
                    is_closing, exit_reason = True, f"STATISTICAL STOP (Z-Score hit {Z_SCORE_STOP_LOSS:.1f})"
                # FIX: Correct time-stop logic using bar index
                elif 'entry_index' in position:
                    bars_held = len(s1_df) - position['entry_index']
                    if bars_held > time_stop_candles:
                        is_closing, exit_reason = True, f"TIME STOP ({bars_held} bars > {time_stop_candles})"
                
                if is_closing:
                    signals.append(PairSignal(pair, f"EXIT_{position['direction']}", exit_reason, current_z_score, 
                                              hedge_ratio, pair_info['half_life'], "", trade_details=position['trade_details']))
                    del open_positions[pair]
                else:
                     signals.append(PairSignal(pair, f"HOLD_{position['direction']}", "Position Open", current_z_score, 
                                              hedge_ratio, pair_info['half_life'], "", trade_details=position['trade_details']))
            else:
                if current_z_score > Z_SCORE_ENTRY:
                    trade_details = get_trade_details(Z_SCORE_STOP_LOSS)
                    signals.append(PairSignal(pair, "ENTER_SHORT", f"Z-Score > {Z_SCORE_ENTRY:.1f}", current_z_score, 
                                              hedge_ratio, pair_info['half_life'], "", trade_details=trade_details))
                    # FIX: Store entry_index for time-stop
                    open_positions[pair] = {"direction": "SHORT", "entry_index": len(s1_df), "trade_details": trade_details}
                elif current_z_score < -Z_SCORE_ENTRY:
                    trade_details = get_trade_details(-Z_SCORE_STOP_LOSS)
                    signals.append(PairSignal(pair, "ENTER_LONG", f"Z-Score < -{Z_SCORE_ENTRY:.1f}", current_z_score, 
                                              hedge_ratio, pair_info['half_life'], "", trade_details=trade_details))
                    # FIX: Store entry_index for time-stop
                    open_positions[pair] = {"direction": "LONG", "entry_index": len(s1_df), "trade_details": trade_details}
        except Exception as e:
            logger.warning(f"Could not generate signal for pair ({pair_info['pair'][0]}, {pair_info['pair'][1]}) due to an error: {e}. Skipping pair for this cycle.")
            continue
    
    return signals

def generate_signals(all_symbols_data, num_candles):
    """Main orchestrator function for the Stat Arb strategy."""
    logger.info("--- Starting Statistical Arbitrage Strategy Cycle ---")

    if strategy_cache["cointegrated_pairs"] is None:
        logger.info("First run: Finding cointegrated pairs. This may take a while...")
        start_time = time.time()
        found_pairs = find_cointegrated_pairs(all_symbols_data, num_candles)
        strategy_cache["cointegrated_pairs"] = found_pairs
        logger.info(f"Pair finding process took {time.time() - start_time:.2f} seconds.")
    
    pairs_to_trade = strategy_cache["cointegrated_pairs"]
    if not pairs_to_trade:
        logger.warning("No cointegrated pairs found to trade. Ending cycle.")
        return

    logger.info(f"Generating signals for {len(pairs_to_trade)} pairs using latest data...")
    signals = generate_pair_signals(all_symbols_data, pairs_to_trade)

    if not signals:
        logger.info("No new signals or open positions in this cycle.")
        return
        
    logger.info("\n==================== STATISTICAL ARBITRAGE SIGNAL SUMMARY ====================")
    for sig in signals:
        s1, s2 = sig.pair
        details = sig.trade_details
        time_stop = int(2.5 * sig.half_life)
        
        color_map = {"ENTER": "\033[92m", "EXIT": "\033[93m", "HOLD": "\033[96m"}
        signal_color = next((color for key, color in color_map.items() if key in sig.signal_type), "\033[0m")
        
        log_message = (
            f"\n----------- PAIR: {s1} / {s2} -----------\n"
            f"  [STATUS]:           {signal_color}{sig.signal_type.replace('_', ' '):<15}{sig.reason}\033[0m\n"
            f"  [Z-Score]:          {sig.z_score:.2f}\n"
        )
        
        if "ENTER" in sig.signal_type:
            leg1, leg2 = "", ""
            if sig.signal_type == "ENTER_LONG":
                leg1 = f"  > Enter LONG  {s1:<12} at {details['s1_entry']:>8.2f}, Target: {details['s1_target']:>8.2f}, Stop: {details['s1_stop']:>8.2f}"
                leg2 = f"  > Enter SHORT {s2:<12} at {details['s2_entry']:>8.2f}, Target: {details['s2_target']:>8.2f}, Stop: {details['s2_stop']:>8.2f}"
            elif sig.signal_type == "ENTER_SHORT":
                leg1 = f"  > Enter SHORT {s1:<12} at {details['s1_entry']:>8.2f}, Target: {details['s1_target']:>8.2f}, Stop: {details['s1_stop']:>8.2f}"
                leg2 = f"  > Enter LONG  {s2:<12} at {details['s2_entry']:>8.2f}, Target: {details['s2_target']:>8.2f}, Stop: {details['s2_stop']:>8.2f}"
            
            log_message += (
                f"  [ACTION PLAN]:\n{leg1}\n{leg2}\n"
                f"  > Time Stop: Hold for max {time_stop} candles.\n"
                f"  > Hedge Ratio (β): {sig.hedge_ratio:.3f}"
            )
        else: # For HOLD or EXIT signals
            orig_details = sig.trade_details
            leg1_action, leg2_action = ("LONG", "SHORT") if "LONG" in sig.signal_type else ("SHORT", "LONG")

            log_message += (
                f"  [ORIGINAL PLAN]:\n"
                f"  > {leg1_action:<6} {s1:<12} at {orig_details['s1_entry']:>8.2f}, Target: {orig_details['s1_target']:>8.2f}, Stop: {orig_details['s1_stop']:>8.2f}\n"
                f"  > {leg2_action:<6} {s2:<12} at {orig_details['s2_entry']:>8.2f}, Target: {orig_details['s2_target']:>8.2f}, Stop: {orig_details['s2_stop']:>8.2f}"
             )
        
        logger.info(log_message)
    logger.info("\n==============================================================================")