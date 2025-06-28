# stat_arb_trader_dhan/main.py

import sys
import time
import pandas as pd
from core.logger_setup import logger
from core.dhan_client import DhanClient
from data_feeds.instrument_manager import InstrumentManager
from data_feeds.data_fetcher import DataFetcher

def main():
    """
    This is the main control script for the trading bot.
    It initializes services and enters a continuous loop to fetch data
    for the specified instruments and timeframes.
    """
    # =================================================================================
    # --- 1. YOUR INPUTS: CONFIGURE YOUR DATA REQUEST HERE ---
    # =================================================================================
    # Add one or more stock symbols to this list
    SYMBOLS_TO_TRACK = ['RELIANCE', 'INFY', 'TCS']
    
    # Set the desired candle timeframe
    # Options: '1', '5', '15', '60' (for minutes), 'D' (Daily), 'W' (Weekly)
    TIMEFRAME = 'D'
    
    # Set the number of historical candles you want to look back on
    NUM_CANDLES = 200
    
    # Set how often the script should update (in seconds)
    CYCLE_INTERVAL_SECONDS = 30
    # =================================================================================

    logger.info("--- Initializing Trading Bot Services ---")
    
    # --- Initialization (runs once) ---
    dhan_connection = DhanClient()
    dhan_api_sdk = dhan_connection.get_api_client()
    if not dhan_api_sdk:
        logger.critical("Failed to initialize Dhan Client. Exiting.")
        sys.exit(1)
    
    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    if data_fetcher.instrument_master_df is None:
        logger.critical("Failed to load Dhan instrument master file. Exiting.")
        sys.exit(1)

    logger.info("--- Initialization Complete. Starting Live Data Cycle. ---")
    
    # --- The Main Loop (runs continuously) ---
    while True:
        try:
            for symbol in SYMBOLS_TO_TRACK:
                logger.info(f"================== Processing: {symbol} ==================")
                
                # A. Get the instrument's ISIN needed for the API calls
                isin = instrument_manager.get_isin_for_nse_symbol(symbol)
                if not isin:
                    logger.warning(f"Could not find ISIN for {symbol}. Skipping this symbol for the current cycle.")
                    continue

                # B. Fetch the latest LIVE (or last closed) quote
                # This is a snapshot of the most current price data.
                logger.info(f"--- Fetching Live Quote for {symbol} ---")
                live_quote_data = data_fetcher.get_live_quotes([(symbol, isin)])
                
                # VIEW THE LIVE DATA
                if live_quote_data and symbol in live_quote_data:
                    # The 'live_quote_data[symbol]' dictionary is what you'd use for live decisions
                    live_price = live_quote_data[symbol].get('last_price', 'N/A')
                    logger.info(f"LIVE DATA for {symbol}: Last Price = {live_price}")
                else:
                    logger.warning(f"Could not fetch live quote for {symbol}.")


                # C. Fetch the HISTORICAL data based on your settings
                # This fetches the historical candles and adds all technical indicators.
                logger.info(f"--- Fetching {NUM_CANDLES} Historical '{TIMEFRAME}' Candles for {symbol} ---")
                historical_df = data_fetcher.fetch_data(symbol, isin, TIMEFRAME, NUM_CANDLES)
                
                # VIEW THE HISTORICAL DATA
                if not historical_df.empty:
                    # The 'historical_df' DataFrame is the structure you will pass to your strategies
                    print("Latest historical data with indicators:")
                    print(historical_df.tail(3)) # Print the last 3 candles for a quick view
                else:
                    logger.warning(f"Could not fetch historical data for {symbol}.")

                # In a real strategy, you would now pass this data to a function:
                # e.g., generate_signals(symbol, live_quote_data[symbol], historical_df)
                
                # Short delay to respect API rate limits when tracking multiple symbols
                time.sleep(1)

            logger.info(f"\nCycle complete. Waiting for {CYCLE_INTERVAL_SECONDS} seconds...\n")
            time.sleep(CYCLE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.critical(f"An unhandled exception occurred in the main loop: {e}", exc_info=True)
            logger.info(f"Attempting to continue after a {CYCLE_INTERVAL_SECONDS}-second delay...")
            time.sleep(CYCLE_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
    logger.info("--- StatArb Trading Bot Shutting Down ---")