# stat_arb_trader_dhan/main.py

import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
from core.logger_setup import logger
from config import settings
from core.dhan_client import DhanClient
from data_feeds.instrument_manager import InstrumentManager
from data_feeds.data_fetcher import DataFetcher

# --- Global Service Instances ---
# These will be initialized once and used throughout the application's life.
dhan_connection = None
instrument_manager = None
data_fetcher = None

def initialize_services():
    """
    Initializes all core services required for the bot to run.
    Returns True on success, False on critical failure.
    """
    global dhan_connection, instrument_manager, data_fetcher
    
    logger.info("--- Initializing Trading Bot Services ---")
    
    # 1. Initialize Dhan Client and test connection
    dhan_connection = DhanClient()
    if not dhan_connection.get_api_client():
        logger.critical("Failed to initialize Dhan Client. Exiting.")
        return False
    
    if not dhan_connection.test_connection():
        # This is a warning, not a critical failure, as the API might be temporarily down.
        logger.warning("Dhan API connection test failed. Functionality may be limited.")

    # 2. Initialize Instrument and Data Fetching services
    dhan_api_sdk = dhan_connection.get_api_client()
    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    
    if data_fetcher.instrument_master_df is None:
        logger.critical("Failed to load Dhan instrument master file. Exiting.")
        return False
        
    logger.info("--- All services initialized successfully ---")
    return True

def run_trading_cycle():
    """
    Executes a single data fetching and processing cycle.
    """
    if not all([instrument_manager, data_fetcher]):
        logger.error("Critical services not available. Skipping trading cycle.")
        return

    # 1. Get the list of instruments to track
    symbols_to_track = instrument_manager.get_nse_symbols_with_isin()
    if not symbols_to_track:
        logger.error("No instruments found to track. Skipping cycle.")
        return
    logger.info(f"Starting trading cycle for {len(symbols_to_track)} Nifty 50 symbols.")
    
    # 2. Fetch Historical Data
    logger.info("\n--- STEP A: Fetching Historical Data (Last 90 Days) ---")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    historical_data = data_fetcher.get_bulk_historical_data(symbols_to_track, from_date, to_date)
    
    if not historical_data:
        logger.error("Failed to fetch sufficient historical data. Cycle aborted.")
        return
        
    # 3. Calculate Indicators
    logger.info("\n--- STEP B: Calculating Technical Indicators ---")
    historical_data_with_indicators = data_fetcher.calculate_indicators(historical_data)
    first_symbol = list(historical_data_with_indicators.keys())[0]
    logger.info(f"Sample data for {first_symbol} with indicators:")
    print(historical_data_with_indicators[first_symbol].tail())
    
    # 4. Fetch Live Data
    logger.info("\n--- STEP C: Fetching Live Quotes (Snapshot) ---")
    live_quotes = data_fetcher.get_live_quotes(symbols_to_track)
    if live_quotes:
        pd.set_option('display.width', 1000)
        live_df = pd.DataFrame.from_dict(live_quotes, orient='index')
        logger.info("Sample live quotes:")
        print(live_df[['last_price', 'volume', 'open', 'high', 'low', 'close']].head())
    else:
        logger.error("Failed to fetch any live quotes in this cycle.")
        
    logger.info("\n--- Trading Cycle Complete ---")

def main():
    """
    The main entry point for the trading bot.
    """
    if not initialize_services():
        sys.exit(1) # Exit if initialization fails

    while True:
        try:
            now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
            
            # Simplified market hours check
            # Note: This does not account for market holidays.
            is_market_hours = (now_ist.weekday() < 5) and \
                              (now_ist.time() >= datetime.strptime("09:15", "%H:%M").time()) and \
                              (now_ist.time() < datetime.strptime("15:30", "%H:%M").time())

            if is_market_hours:
                logger.info(f"Market is OPEN. Current IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
                run_trading_cycle()
            else:
                logger.info(f"Market is CLOSED. Current IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}. Bot is idle.")
            
            # Wait before the next cycle
            logger.info("Sleeping for 60 seconds...")
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Bot run interrupted by user (KeyboardInterrupt).")
            break
        except Exception as e:
            logger.critical(f"An unhandled exception occurred in the main loop: {e}", exc_info=True)
            logger.info("Attempting to continue after a 60-second delay...")
            time.sleep(60)

if __name__ == "__main__":
    main()
    logger.info("--- StatArb Trading Bot Shutting Down ---")