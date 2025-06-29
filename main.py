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
    for symbol in symbols:
        logger.info(f"================== Processing {symbol} ==================")
        
        isin = instrument_mgr.get_isin_for_nse_symbol(symbol)
        if not isin:
            logger.warning(f"Could not find ISIN for {symbol}. Skipping.")
            continue

        historical_df = data_fetcher.fetch_data(symbol, isin, timeframe, num_candles)
        
        if historical_df.empty:
            logger.warning(f"Could not fetch historical data for {symbol}.")
            continue

        signals_df = strategy_function(historical_df, use_trend_filter=True)
        
        latest_signal_row = signals_df.iloc[-1]
        latest_signal = latest_signal_row['signal']
        
        logger.info(f"LATEST SIGNAL for {symbol} on {latest_signal_row.name.date()}: {latest_signal} (Position: {latest_signal_row['position']})")

        if latest_signal != 'HOLD':
            print(f"  > ACTIONABLE SIGNAL: {symbol} -> {latest_signal}")
    
    logger.info("--- Strategy Cycle Finished ---")

def main():
    """
    Main control script that runs the bot in a continuous loop.
    """
    # =================================================================================
    # --- 1. YOUR INPUTS: CONFIGURE YOUR STRATEGY AND DATA HERE ---
    # =================================================================================
    SELECTED_STRATEGY = 2
    SYMBOLS_TO_TRACK = ['RELIANCE', 'TCS', 'HDFCBANK']
    TIMEFRAME = 'D'
    NUM_CANDLES = 252
    CYCLE_INTERVAL_SECONDS = 30  # <-- BOT WILL RUN EVERY 30 SECONDS
    # =================================================================================

    logger.info("--- Initializing Services ---")
    
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
    
    logger.info(f"--- Initialization Complete. Strategy #{SELECTED_STRATEGY} selected. Starting main loop. ---")
    
    # --- THIS IS THE CONTINUOUS LOOP ---
    while True:
        try:
            # You can add a market hours check here if you only want to run during trading hours
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