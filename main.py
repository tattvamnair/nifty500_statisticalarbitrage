# stat_arb_trader_dhan/main.py

import sys
import pandas as pd
from core.logger_setup import logger
from core.dhan_client import DhanClient
from data_feeds.instrument_manager import InstrumentManager
from data_feeds.data_fetcher import DataFetcher

# Import all available strategy modules
from strategy_logic.ema_crossover import generate_signals as ema_crossover_strategy
from strategy_logic.rsi_divergence import generate_signals as rsi_divergence_strategy

def main():
    """
    Main control script to fetch data and generate trading signals based on a selected strategy.
    """
    # =================================================================================
    # --- 1. YOUR INPUTS: CONFIGURE YOUR STRATEGY AND DATA HERE ---
    # =================================================================================
    
    # Choose which strategy to run by its number
    # 1: 9/15 EMA Crossover Strategy
    # 2: RSI Divergence Swing Strategy
    SELECTED_STRATEGY = 2
    
    # --- Common Inputs ---
    SYMBOLS_TO_TRACK = ['RELIANCE', 'TCS', 'HDFCBANK']
    TIMEFRAME = 'D'
    NUM_CANDLES = 252
    # =================================================================================

    logger.info("--- Initializing Services ---")
    
    dhan_connection = DhanClient()
    dhan_api_sdk = dhan_connection.get_api_client()
    if not dhan_api_sdk: sys.exit(1)
    
    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    if data_fetcher.instrument_master_df is None: sys.exit(1)

    # --- Strategy Selection ---
    strategy_map = {
        1: ema_crossover_strategy,
        2: rsi_divergence_strategy
    }
    strategy_function = strategy_map.get(SELECTED_STRATEGY)
    if not strategy_function:
        logger.error(f"Invalid strategy number: {SELECTED_STRATEGY}. Exiting.")
        sys.exit(1)
    logger.info(f"--- Initialization Complete. Strategy #{SELECTED_STRATEGY} selected. ---")
    
    for symbol in SYMBOLS_TO_TRACK:
        logger.info(f"================== Processing {symbol} for Strategy #{SELECTED_STRATEGY} ==================")
        
        isin = instrument_manager.get_isin_for_nse_symbol(symbol)
        if not isin:
            logger.warning(f"Could not find ISIN for {symbol}. Skipping.")
            continue

        historical_df = data_fetcher.fetch_data(symbol, isin, TIMEFRAME, NUM_CANDLES)
        if historical_df.empty:
            logger.warning(f"Could not fetch historical data for {symbol}.")
            continue

        signals_df = strategy_function(historical_df, use_trend_filter=True)
        
        if not signals_df.empty:
            actionable_signals = signals_df[signals_df['signal'].str.contains('ENTRY|EXIT')]
            
            print(f"\n--- ACTIONABLE SIGNALS FOR {symbol} (Timeframe: {TIMEFRAME}) ---")
            if not actionable_signals.empty:
                pd.set_option('display.width', 1000)
                print(actionable_signals[['close', 'RSI_14', 'signal', 'position']])
            else:
                print("No new entry or exit signals found in the historical data.")
            print("-" * 70)
            
    logger.info("--- Strategy Signal Generation Script Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)