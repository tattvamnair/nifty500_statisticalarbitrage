# stat_arb_trader_dhan/main.py

import sys
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.logger_setup import logger
from core.dhan_client import DhanClient
from data_feeds.instrument_manager import InstrumentManager
from data_feeds.data_fetcher import DataFetcher

# Import all available strategy modules
from strategy_logic.ema_crossover import generate_signals as ema_crossover_strategy
from strategy_logic.rsi_divergence import generate_signals as rsi_divergence_strategy

def run_strategy_cycle(strategy_function, symbols, timeframe, num_candles, instrument_mgr, data_fetcher):
    """
    Executes one full cycle of data fetching and signal generation for all symbols.
    """
    logger.info("--- Starting New Strategy Cycle ---")
    actionable_signals_found = 0

    for symbol in symbols:
        logger.info(f"================== Processing {symbol} ==================")
        
        isin = instrument_mgr.get_isin_for_nse_symbol(symbol)
        if not isin:
            logger.warning(f"Could not find ISIN for {symbol}. Skipping.")
            continue

        historical_df = data_fetcher.fetch_data(symbol, isin, timeframe, num_candles)
        
        if historical_df.empty or 'ADX_14' not in historical_df.columns:
            logger.warning(f"Could not fetch or process sufficient historical data for {symbol}. Skipping analysis.")
            continue

        signals_df = strategy_function(historical_df, use_trend_filter=True)
        
        latest_signal_row = signals_df.iloc[-1]
        latest_signal = latest_signal_row['signal']
        
        # --- FINALIZED "DASHBOARD" LOGGING WITH TIMEZONE CORRECTION ---
        
        # FIX: Convert Timestamp from UTC to India Standard Time (IST) for correct display
        latest_timestamp_utc = latest_signal_row.name
        latest_timestamp_ist = latest_timestamp_utc.tz_localize('UTC').tz_convert('Asia/Kolkata')
        
        is_intraday = timeframe.upper() not in ['D', 'W', '1D', '1W']
        ts_format = '%Y-%m-%d %H:%M IST' if is_intraday else '%Y-%m-%d' # Add 'IST' to format

        # 1. Gather all the data points for the dashboard
        last_close = latest_signal_row['close']
        ema_200 = latest_signal_row['EMA_200']
        adx_14 = latest_signal_row['ADX_14']
        adx_threshold = 25

        # 2. Determine the market state based on the strategy's rules
        market_state = "SIDEWAYS / CHOP"
        if adx_14 >= adx_threshold:
            if last_close > ema_200:
                market_state = "UPTREND"
            else:
                market_state = "DOWNTREND"
        
        # 3. Determine the signal text and color code for clarity
        signal_color_map = {
            "BUY": "\033[92m",  "SELL": "\033[91m", "EXIT_LONG": "\033[93m",
            "EXIT_SHORT": "\033[93m", "HOLD_LONG": "\033[96m", "HOLD_SHORT": "\033[96m",
            "HOLD": "\033[0m"
        }
        END_COLOR = "\033[0m"
        color = signal_color_map.get(latest_signal, "\033[0m")
        colored_signal = f"{color}{latest_signal.replace('_', ' ')}{END_COLOR}"

        # 4. Build the detailed log message using the corrected IST timestamp
        log_message = (
            f"\n"
            f"-------------------- ANALYSIS & SIGNAL: {symbol} --------------------\n"
            f"  [MARKET STATE]: {market_state}\n"
            f"    - Trend Strength (ADX): {adx_14:.2f} (Threshold: {adx_threshold})\n"
            f"    - Long-Term Trend (EMA200): {ema_200:.2f} (Current Price: {last_close:.2f})\n"
            f"\n"
            f"  [LATEST CANDLE DATA]:\n"
            f"    - Timestamp:        {latest_timestamp_ist.strftime(ts_format)}\n"
            f"    - Price (Close):    {last_close:.2f}\n"
            f"    - EMA(9) / EMA(15):   {latest_signal_row.get('EMA_9', 0):.2f} / {latest_signal_row.get('EMA_15', 0):.2f}\n"
            f"    - RSI(14):          {latest_signal_row.get('RSI_14', 0):.2f}\n"
            f"\n"
            f"  [FINAL SIGNAL]:\n"
            f"    - Decision:         {colored_signal}\n"
            f"    - Position Status:  {int(latest_signal_row['position'])} (1=Long, -1=Short, 0=Flat)\n"
            f"----------------------------------------------------------------------"
        )
        logger.info(log_message)

        if latest_signal in ['BUY', 'SELL', 'EXIT_LONG', 'EXIT_SHORT']:
            actionable_signals_found += 1
            print(f"\n"
                  f"  **********************************************************\n"
                  f"  *** ACTIONABLE ALERT: {symbol} -> {color}{latest_signal.replace('_', ' ')}{END_COLOR} ***\n"
                  f"  **********************************************************\n")
    
    logger.info(f"--- Strategy Cycle Finished. Found {actionable_signals_found} actionable signal(s). ---")

def main():
    """
    Main control script that runs the bot in a continuous loop.
    """
    # =================================================================================
    # --- 1. YOUR INPUTS: CONFIGURE YOUR STRATEGY AND DATA HERE ---
    # =================================================================================
    SELECTED_STRATEGY = 1
    SYMBOLS_TO_TRACK = ['TATAMOTORS', 'ITC', 'HDFCBANK']
    TIMEFRAME = '1M'
    NUM_CANDLES = 252
    CYCLE_INTERVAL_SECONDS = 60
    # =================================================================================

    logger.info("--- Initializing Trading Bot ---")
    
    dhan_connection = DhanClient()
    dhan_api_sdk = dhan_connection.get_api_client()
    if not dhan_api_sdk: sys.exit(1)
    
    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    if data_fetcher.instrument_master_df is None: sys.exit(1)

    strategy_map = { 1: ema_crossover_strategy, 2: rsi_divergence_strategy }
    strategy_function = strategy_map.get(SELECTED_STRATEGY)
    if not strategy_function:
        logger.error(f"Invalid strategy number: {SELECTED_STRATEGY}. Exiting.")
        sys.exit(1)
    
    logger.info(
        f"\n"
        f"====================== BOT CONFIGURATION ======================\n"
        f"STRATEGY:          #{SELECTED_STRATEGY} ({strategy_function.__name__})\n"
        f"SYMBOLS:           {', '.join(SYMBOLS_TO_TRACK)}\n"
        f"TIMEFRAME:         {TIMEFRAME}\n"
        f"CANDLES TO ANALYZE: {NUM_CANDLES}\n"
        f"CYCLE INTERVAL:    {CYCLE_INTERVAL_SECONDS} seconds\n"
        f"==============================================================="
    )
    
    logger.info("--- Initialization Complete. Starting main loop. ---")
    
    while True:
        try:
            run_strategy_cycle(
                strategy_function=strategy_function,
                symbols=SYMBOLS_TO_TRACK,
                timeframe=TIMEFRAME,
                num_candles=NUM_CANDLES,
                instrument_mgr=instrument_manager,
                data_fetcher=data_fetcher
            )
            
            logger.info(f"Cycle complete. Waiting for {CYCLE_INTERVAL_SECONDS} seconds...\n")
            time.sleep(CYCLE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.critical(f"An unhandled exception in the main loop: {e}", exc_info=True)
            logger.info(f"Attempting to continue after a {CYCLE_INTERVAL_SECONDS}-second delay...")
            time.sleep(CYCLE_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
    logger.info("--- Trading Bot Shutting Down ---")