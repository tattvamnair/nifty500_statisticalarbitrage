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
        
        if historical_df.empty:
            logger.warning(f"Could not fetch historical data for {symbol}. Skipping analysis.")
            continue

        signals_df = strategy_function(historical_df, use_trend_filter=True)
        
        latest_signal_row = signals_df.iloc[-1]
        latest_signal = latest_signal_row['signal']
        
        # --- LOGGING ENHANCEMENT: DETAILED SIGNAL OUTPUT ---
        is_intraday = timeframe.upper() not in ['D', 'W', '1D', '1W']
        ts_format = '%Y-%m-%d %H:%M' if is_intraday else '%Y-%m-%d'
        
        # Prepare a detailed, formatted string for the log
        log_message = (
            f"\n"
            f"-------------------- LATEST SIGNAL: {symbol} --------------------\n"
            f"  > Timestamp:         {latest_signal_row.name.strftime(ts_format)}\n"
            f"  > Signal:            {latest_signal} (Current Position: {int(latest_signal_row['position'])})\n"
            f"  > Last Close:        {latest_signal_row['close']:.2f}\n"
            f"  > EMA(9) / EMA(15):    {latest_signal_row.get('EMA_9', 0):.2f} / {latest_signal_row.get('EMA_15', 0):.2f}\n"
            f"  > RSI(14):           {latest_signal_row.get('RSI_14', 0):.2f}\n"
            f"-----------------------------------------------------------------"
        )
        logger.info(log_message)

        if latest_signal != 'HOLD':
            actionable_signals_found += 1
            print(f"\n"
                  f"  **********************************************************\n"
                  f"  *** ACTIONABLE SIGNAL: {symbol} -> {latest_signal} ***\n"
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
    TIMEFRAME = 'W'  
    # --- SET YOUR TIMEFRAME ---
    # Valid options:
    # Intraday: '1M', '5M', '15M', '30M', '1H', '4H'
    # Daily:    'D'
    # Weekly:   'W'
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
    
    # --- LOGGING ENHANCEMENT: PRINT CONFIGURATION ---
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