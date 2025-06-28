# stat_arb_trader_dhan/main.py

import sys
import pandas as pd
from core.logger_setup import logger
from core.dhan_client import DhanClient
from data_feeds.instrument_manager import InstrumentManager
from data_feeds.data_fetcher import DataFetcher

def main():
    """
    This is the main control script for fetching and analyzing historical data.
    """
    # =================================================================================
    # --- 1. YOUR INPUTS: CONFIGURE YOUR DATA REQUEST HERE ---
    # =================================================================================
    # Add one or more stock symbols to this list
    SYMBOLS_TO_TRACK = ['RELIANCE', 'INFY', 'TCS', 'HDFCBANK']
    
    # Set the desired candle timeframe
    # Options: '1', '5', '15', '60' (for minutes), 'D' (Daily), 'W' (Weekly)
    TIMEFRAME = 'D'
    
    # Set the number of historical candles you want to look back on
    NUM_CANDLES = 200
    # =================================================================================

    logger.info("--- Initializing Data Services ---")
    
    # --- Initialization (runs once) ---
    dhan_connection = DhanClient()
    dhan_api_sdk = dhan_connection.get_api_client()
    if not dhan_api_sdk:
        logger.critical("Failed to initialize Dhan Client. Exiting.")
        sys.exit(1)
    
    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    if data_fetcher.instrument_master_df is None:
        logger.critical("Failed to load instrument master file. Exiting.")
        sys.exit(1)

    logger.info("--- Initialization Complete. Fetching historical data. ---")
    
    # This loop will process each stock you defined above.
    for symbol in SYMBOLS_TO_TRACK:
        logger.info(f"================== Processing: {symbol} ==================")
        
        isin = instrument_manager.get_isin_for_nse_symbol(symbol)
        if not isin:
            logger.warning(f"Could not find ISIN for {symbol}. Skipping.")
            continue

        # Fetch the historical data with indicators
        historical_df = data_fetcher.fetch_data(symbol, isin, TIMEFRAME, NUM_CANDLES)
        
        # This is where your strategy would use the data.
        # For now, we just print it to view the result.
        if not historical_df.empty:
            print(f"\n--- LATEST DATA FOR {symbol} ---")
            print(f"Showing last 5 of {len(historical_df)} candles for timeframe '{TIMEFRAME}'")
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_columns', 15)
            print(historical_df.tail())
            print("-" * 50)
        else:
            logger.warning(f"No historical data returned for {symbol}.")
            
    logger.info("--- Data Fetching Script Finished ---")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script stopped by user.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)