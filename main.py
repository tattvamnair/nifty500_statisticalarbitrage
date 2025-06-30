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
    
    long_entries, long_exits, short_entries, short_exits = [], [], [], []

    for symbol in symbols:
        logger.info(f"================== Processing {symbol} ==================")
        
        isin = instrument_mgr.get_isin_for_nse_symbol(symbol)
        if not isin:
            logger.warning(f"Could not find ISIN for {symbol}. Skipping.")
            continue

        historical_df = data_fetcher.fetch_data(symbol, isin, timeframe, num_candles)
        
        if historical_df.empty:
            logger.warning(f"No historical data for {symbol}. Skipping analysis.")
            continue

        # The strategy function is now solely responsible for signal generation
        signals_df = strategy_function(historical_df)
        
        # Defensive check in case the strategy returns an empty dataframe
        if signals_df.empty:
            logger.warning(f"Signal generation failed for {symbol}. Skipping analysis.")
            continue

        latest_signal_row = signals_df.iloc[-1]
        latest_signal = latest_signal_row['signal']
        
        if latest_signal == 'BUY': long_entries.append(symbol)
        elif latest_signal == 'SELL': short_entries.append(symbol)
        elif latest_signal == 'EXIT_LONG': long_exits.append(symbol)
        elif latest_signal == 'EXIT_SHORT': short_exits.append(symbol)

        # --- ADAPTIVE "DASHBOARD" LOGGING ---
        # This new logic intelligently adapts the dashboard to the timeframe.
        is_intraday = timeframe.upper() not in ['D', 'W', '1D', '1W']
        
        # Convert timestamp to IST for correct display
        latest_timestamp_ist = latest_signal_row.name.tz_localize('UTC').tz_convert('Asia/Kolkata')
        ts_format = '%Y-%m-%d %H:%M IST' if is_intraday else '%Y-%m-%d'
        
        last_close = latest_signal_row['close']
        adx_14 = latest_signal_row.get('ADX_14', 0)

        # Dynamically set parameters based on timeframe
        if is_intraday:
            trend_filter_val = latest_signal_row.get('VWAP', 0)
            trend_filter_name = "VWAP"
            adx_threshold = 22
            fast_ema, slow_ema = latest_signal_row.get('EMA_9', 0), latest_signal_row.get('EMA_15', 0)
            ema_label = "EMA(9) / EMA(15)"
        else:
            trend_filter_val = latest_signal_row.get('EMA_200', 0)
            trend_filter_name = "EMA(200)"
            adx_threshold = 25
            fast_ema, slow_ema = latest_signal_row.get('EMA_9', 0), latest_signal_row.get('EMA_15', 0)
            ema_label = "EMA(9) / EMA(15)"

        market_state = "SIDEWAYS / CHOP"
        if adx_14 >= adx_threshold and trend_filter_val > 0: # Ensure filter value is calculated
            market_state = "UPTREND" if last_close > trend_filter_val else "DOWNTREND"
        
        signal_color_map = {"BUY": "\033[92m", "SELL": "\033[91m", "EXIT_LONG": "\033[93m", "EXIT_SHORT": "\033[93m", "HOLD_LONG": "\033[96m", "HOLD_SHORT": "\033[96m", "HOLD": "\033[0m"}
        END_COLOR, color = "\033[0m", signal_color_map.get(latest_signal, "\033[0m")
        colored_signal = f"{color}{latest_signal.replace('_', ' ')}{END_COLOR}"

        log_message = (
            f"\n-------------------- ANALYSIS & SIGNAL: {symbol} --------------------\n"
            f"  [MARKET STATE]: {market_state}\n"
            f"    - Trend Strength (ADX): {adx_14:.2f} (Threshold: {adx_threshold})\n"
            f"    - Trend Filter ({trend_filter_name}): {trend_filter_val:.2f} (Current Price: {last_close:.2f})\n\n"
            f"  [LATEST CANDLE DATA]:\n"
            f"    - Timestamp:        {latest_timestamp_ist.strftime(ts_format)}\n"
            f"    - {ema_label}:   {fast_ema:.2f} / {slow_ema:.2f}\n"
            f"    - RSI(14):          {latest_signal_row.get('RSI_14', 0):.2f}\n\n"
            f"  [FINAL SIGNAL]:\n"
            f"    - Decision:         {colored_signal}\n"
            f"    - Position Status:  {int(latest_signal_row['position'])} (1=Long, -1=Short, 0=Flat)\n"
            f"----------------------------------------------------------------------"
        )
        logger.info(log_message)

        if latest_signal in ['BUY', 'SELL', 'EXIT_LONG', 'EXIT_SHORT']:
            print(f"\n"
                  f"  **********************************************************\n"
                  f"  *** ACTIONABLE ALERT: {symbol} -> {color}{latest_signal.replace('_', ' ')}{END_COLOR} ***\n"
                  f"  **********************************************************\n")
    
    logger.info("--- Strategy Cycle Finished ---")
    if not any([long_entries, long_exits, short_entries, short_exits]):
        logger.info("  > No new entry or exit signals generated this cycle.")
    else:
        summary_message = "\n==================== CYCLE SIGNAL SUMMARY ====================\n"
        if long_entries: summary_message += f"  > \033[92mNew Long Entries (BUY):\033[0m   {', '.join(long_entries)}\n"
        if short_entries: summary_message += f"  > \033[91mNew Short Entries (SELL):\033[0m  {', '.join(short_entries)}\n"
        if long_exits: summary_message += f"  > \033[93mPosition Exits (Long):\033[0m    {', '.join(long_exits)}\n"
        if short_exits: summary_message += f"  > \033[93mPosition Exits (Short):\033[0m   {', '.join(short_exits)}\n"
        summary_message += "============================================================"
        logger.info(summary_message)

def main():
    """
    Main control script that runs the bot in a continuous loop.
    """
    # =================================================================================
    # --- 1. YOUR INPUTS: CONFIGURE YOUR STRATEGY AND DATA HERE ---
    # =================================================================================
    SELECTED_STRATEGY = 1
    SYMBOLS_TO_TRACK = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ADANIENT', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'ITC', 'AXISBANK', 'SBIN', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'NESTLEIND', 'HCLTECH', 'M&M', 'POWERGRID', 'ULTRACEMCO', 'DRREDDY', 'TATASTEEL', 'SUNPHARMA', 'TITAN', 'TECHM', 'NTPC', 'WIPRO', 'BRITANNIA', 'BAJAJ-AUTO', 'CIPLA', 'ONGC', 'HEROMOTOCO', 'HINDALCO', 'ADANIPORTS', 'GRASIM', 'COALINDIA', 'INDUSINDBK', 'JSWSTEEL', 'TATAMOTORS', 'EICHERMOT', 'HDFCLIFE', 'BAJAJFINSV', 'DIVISLAB', 'BPCL', 'TATACONSUM', 'APOLLOHOSP', 'SHRIRAMFIN', 'SBILIFE', 'LTIM']
    TIMEFRAME = '1'
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
        f"SYMBOLS:           ({len(SYMBOLS_TO_TRACK)}) Nifty 50 Symbols\n"
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