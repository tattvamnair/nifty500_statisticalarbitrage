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
    hold_longs, hold_shorts = [], []

    for symbol in symbols:
        try:
            logger.info(f"================== Processing {symbol} ==================")
            
            isin = instrument_mgr.get_isin_for_nse_symbol(symbol)
            if not isin:
                logger.warning(f"Could not find ISIN for {symbol}. Skipping.")
                continue

            historical_df = data_fetcher.fetch_data(symbol, isin, timeframe, num_candles)
            
            if historical_df.empty:
                logger.warning(f"No historical data for {symbol}. Skipping analysis.")
                continue

            signals_df = strategy_function(historical_df)
            
            if signals_df.empty:
                logger.warning(f"Signal generation failed for {symbol} (as per strategy logic). Skipping analysis.")
                continue

            latest_signal_row = signals_df.iloc[-1]
            latest_signal = latest_signal_row['signal']
            
            if latest_signal == 'BUY': long_entries.append(symbol)
            elif latest_signal == 'SELL': short_entries.append(symbol)
            elif latest_signal == 'EXIT_LONG': long_exits.append(symbol)
            elif latest_signal == 'EXIT_SHORT': short_exits.append(symbol)
            elif latest_signal == 'HOLD_LONG': hold_longs.append(symbol)
            elif latest_signal == 'HOLD_SHORT': hold_shorts.append(symbol)

            is_intraday = timeframe.upper() not in ['D', 'W', '1D', '1W']
            latest_timestamp_ist = latest_signal_row.name.tz_localize('UTC').tz_convert('Asia/Kolkata')
            ts_format = '%Y-%m-%d %H:%M IST' if is_intraday else '%Y-%m-%d'
            last_close = latest_signal_row.get('close', 0.0)
            adx_14 = latest_signal_row.get('ADX_14', 0.0)

            if is_intraday:
                trend_filter_val = latest_signal_row.get('VWAP', 0.0)
                trend_filter_name = "VWAP"
                adx_threshold = 22
                fast_ema, slow_ema = latest_signal_row.get('EMA_9', 0.0), latest_signal_row.get('EMA_15', 0.0)
                ema_label = "EMA(9) / EMA(15)"
            else:
                trend_filter_val = latest_signal_row.get('EMA_200', 0.0)
                trend_filter_name = "EMA(200)"
                adx_threshold = 25
                fast_ema, slow_ema = latest_signal_row.get('EMA_9', 0.0), latest_signal_row.get('EMA_15', 0.0)
                ema_label = "EMA(9) / EMA(15)"

            market_state = "SIDEWAYS / CHOP"
            if adx_14 >= adx_threshold and trend_filter_val > 0:
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
                f"    - RSI(14):          {latest_signal_row.get('RSI_14', 0.0):.2f}\n\n"
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
        
        except Exception as e:
            logger.error(f"A critical error occurred while processing {symbol}. Error: {e}. Skipping.", exc_info=True)
            continue
    
    logger.info("--- Strategy Cycle Finished ---")
    
    if not any([long_entries, long_exits, short_entries, short_exits, hold_longs, hold_shorts]):
        logger.info("  > No new entry, exit, or active hold signals generated this cycle.")
    else:
        summary_message = "\n==================== CYCLE SIGNAL SUMMARY ====================\n"
        if long_entries: summary_message += f"  > \033[92mNew Long Entries (BUY):\033[0m      {', '.join(long_entries)}\n"
        if short_entries: summary_message += f"  > \033[91mNew Short Entries (SELL):\033[0m     {', '.join(short_entries)}\n"
        if long_exits: summary_message += f"  > \033[93mPosition Exits (Long):\033[0m       {', '.join(long_exits)}\n"
        if short_exits: summary_message += f"  > \033[93mPosition Exits (Short):\033[0m      {', '.join(short_exits)}\n"
        if hold_longs: summary_message += f"  > \033[96mCurrently Holding Long:\033[0m      {', '.join(hold_longs)}\n"
        if hold_shorts: summary_message += f"  > \033[96mCurrently Holding Short:\033[0m     {', '.join(hold_shorts)}\n"
        summary_message += "============================================================"
        logger.info(summary_message)

def main():
    """
    Main control script that runs the bot in a continuous loop.
    """
    SELECTED_STRATEGY = 1
    SYMBOLS_TO_TRACK = ['360ONE', '3MINDIA', 'ABB', 'ACC', 'ACMESOLAR', 'AIAENG', 'APLAPOLLO', 'AUBANK', 'AWL', 'AADHARHFC', 'AARTIIND', 'AAVAS', 'ABBOTINDIA', 'ACE', 'ADANIENSOL', 'ADANIENT', 'ADANIGREEN', 'ADANIPORTS', 'ADANIPOWER', 'ATGL', 'ABCAPITAL', 'ABFRL', 'ABREL', 'ABSLAMC', 'AEGISLOG', 'AFCONS', 'AFFLE', 'AJANTPHARM', 'AKUMS', 'APLLTD', 'ALIVUS', 'ALKEM', 'ALKYLAMINE', 'ALOKINDS', 'ARE&M', 'AMBER', 'AMBUJACEM', 'ANANDRATHI', 'ANANTRAJ', 'ANGELONE', 'APARINDS', 'APOLLOHOSP', 'APOLLOTYRE', 'APTUS', 'ASAHIINDIA', 'ASHOKLEY', 'ASIANPAINT', 'ASTERDM', 'ASTRAZEN', 'ASTRAL', 'ATUL', 'AUROPHARMA', 'AIIL', 'DMART', 'AXISBANK', 'BASF', 'BEML', 'BLS', 'BSE', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BAJAJHLDNG', 'BAJAJHFL', 'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA', 'MAHABANK', 'BATAINDIA', 'BAYERCROP', 'BERGEPAINT', 'BDL', 'BEL', 'BHARATFORG', 'BHEL', 'BPCL', 'BHARTIARTL', 'BHARTIHEXA', 'BIKAJI', 'BIOCON', 'BSOFT', 'BLUEDART', 'BLUESTARCO', 'BBTC', 'BOSCHLTD', 'FIRSTCRY', 'BRIGADE', 'BRITANNIA', 'MAPMYINDIA', 'CCL', 'CESC', 'CGPOWER', 'CRISIL', 'CAMPUS', 'CANFINHOME', 'CANBK', 'CAPLIPOINT', 'CGCL', 'CARBORUNIV', 'CASTROLIND', 'CEATLTD', 'CENTRALBK', 'CDSL', 'CENTURYPLY', 'CERA', 'CHALET', 'CHAMBLFERT', 'CHENNPETRO', 'CHOLAHLDNG', 'CHOLAFIN', 'CIPLA', 'CUB', 'CLEAN', 'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COHANCE', 'COLPAL', 'CAMS', 'CONCORDBIO', 'CONCOR', 'COROMANDEL', 'CRAFTSMAN', 'CREDITACC', 'CROMPTON', 'CUMMINSIND', 'CYIENT', 'DCMSHRIRAM', 'DLF', 'DOMS', 'DABUR', 'DALBHARAT', 'DATAPATTNS', 'DEEPAKFERT', 'DEEPAKNTR', 'DELHIVERY', 'DEVYANI', 'DIVISLAB', 'DIXON', 'LALPATHLAB', 'DRREDDY', 'DUMMYRAYMN', 'EIDPARRY', 'EIHOTEL', 'EICHERMOT', 'ELECON', 'ELGIEQUIP', 'EMAMILTD', 'EMCURE', 'ENDURANCE', 'ENGINERSIN', 'ERIS', 'ESCORTS', 'ETERNAL', 'EXIDEIND', 'NYKAA', 'FEDERALBNK', 'FACT', 'FINCABLES', 'FINPIPE', 'FSL', 'FIVESTAR', 'FORTIS', 'GAIL', 'GVT&D', 'GMRAIRPORT', 'GRSE', 'GICRE', 'GILLETTE', 'GLAND', 'GLAXO', 'GLENMARK', 'MEDANTA', 'GODIGIT', 'GPIL', 'GODFRYPHLP', 'GODREJAGRO', 'GODREJCP', 'GODREJIND', 'GODREJPROP', 'GRANULES', 'GRAPHITE', 'GRASIM', 'GRAVITA', 'GESHIP', 'FLUOROCHEM', 'GUJGASLTD', 'GMDCLTD', 'GNFC', 'GPPL', 'GSPL', 'HEG', 'HBLENGINE', 'HCLTECH', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HFCL', 'HAPPSTMNDS', 'HAVELLS', 'HEROMOTOCO', 'HSCL', 'HINDALCO', 'HAL', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC', 'POWERINDIA', 'HOMEFIRST', 'HONASA', 'HONAUT', 'HUDCO', 'HYUNDAI', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB', 'IFCI', 'IIFL', 'INOXINDIA', 'IRB', 'IRCON', 'ITC', 'ITI', 'INDGN', 'INDIACEM', 'INDIAMART', 'INDIANB', 'IEX', 'INDHOTEL', 'IOC', 'IOB', 'IRCTC', 'IRFC', 'IREDA', 'IGL', 'INDUSTOWER', 'INDUSINDBK', 'NAUKRI', 'INFY', 'INOXWIND', 'INTELLECT', 'INDIGO', 'IGIL', 'IKS', 'IPCALAB', 'JBCHEPHARM', 'JKCEMENT', 'JBMA', 'JKTYRE', 'JMFINANCIL', 'JSWENERGY', 'JSWHL', 'JSWINFRA', 'JSWSTEEL', 'JPPOWER', 'J&KBANK', 'JINDALSAW', 'JSL', 'JINDALSTEL', 'JIOFIN', 'JUBLFOOD', 'JUBLINGREA', 'JUBLPHARMA', 'JWL', 'JUSTDIAL', 'JYOTHYLAB', 'JYOTICNC', 'KPRMILL', 'KEI', 'KNRCON', 'KPITTECH', 'KAJARIACER', 'KPIL', 'KALYANKJIL', 'KANSAINER', 'KARURVYSYA', 'KAYNES', 'KEC', 'KFINTECH', 'KIRLOSBROS', 'KIRLOSENG', 'KOTAKBANK', 'KIMS', 'LTF', 'LTTS', 'LICHSGFIN', 'LTFOODS', 'LTIM', 'LT', 'LATENTVIEW', 'LAURUSLABS', 'LEMONTREE', 'LICI', 'LINDEINDIA', 'LLOYDSME', 'LUPIN', 'MMTC', 'MRF', 'LODHA', 'MGL', 'MAHSEAMLES', 'M&MFIN', 'M&M', 'MANAPPURAM', 'MRPL', 'MANKIND', 'MARICO', 'MARUTI', 'MASTEK', 'MFSL', 'MAXHEALTH', 'MAZDOCK', 'METROPOLIS', 'MINDACORP', 'MSUMI', 'MOTILALOFS', 'MPHASIS', 'MCX', 'MUTHOOTFIN', 'NATCOPHARM', 'NBCC', 'NCC', 'NHPC', 'NLCINDIA', 'NMDC', 'NSLNISP', 'NTPCGREEN', 'NTPC', 'NH', 'NATIONALUM', 'NAVA', 'NAVINFLUOR', 'NESTLEIND', 'NETWEB', 'NETWORK18', 'NEULANDLAB', 'NEWGEN', 'NAM-INDIA', 'NIVABUPA', 'NUVAMA', 'OBEROIRLTY', 'ONGC', 'OIL', 'OLAELEC', 'OLECTRA', 'PAYTM', 'OFSS', 'POLICYBZR', 'PCBL', 'PGEL', 'PIIND', 'PNBHOUSING', 'PNCINFRA', 'PTCIL', 'PVRINOX', 'PAGEIND', 'PATANJALI', 'PERSISTENT', 'PETRONET', 'PFIZER', 'PHOENIXLTD', 'PIDILITIND', 'PEL', 'PPLPHARMA', 'POLYMED', 'POLYCAB', 'POONAWALLA', 'PFC', 'POWERGRID', 'PRAJIND', 'PREMIERENE', 'PRESTIGE', 'PNB', 'RRKABEL', 'RBLBANK', 'RECLTD', 'RHIM', 'RITES', 'RADICO', 'RVNL', 'RAILTEL', 'RAINBOW', 'RKFORGE', 'RCF', 'RTNINDIA', 'RAYMONDLSL', 'RAYMOND', 'REDINGTON', 'RELIANCE', 'RPOWER', 'ROUTE', 'SBFC', 'SBICARD', 'SBILIFE', 'SJVN', 'SKFINDIA', 'SRF', 'SAGILITY', 'SAILIFE', 'SAMMAANCAP', 'MOTHERSON', 'SAPPHIRE', 'SARDAEN', 'SAREGAMA', 'SCHAEFFLER', 'SCHNEIDER', 'SCI', 'SHREECEM', 'RENUKA', 'SHRIRAMFIN', 'SHYAMMETL', 'SIEMENS', 'SIGNATURE', 'SOBHA', 'SOLARINDS', 'SONACOMS', 'SONATSOFTW', 'STARHEALTH', 'SBIN', 'SAIL', 'SWSOLAR', 'SUMICHEM', 'SUNPHARMA', 'SUNTV', 'SUNDARMFIN', 'SUNDRMFAST', 'SUPREMEIND', 'SUZLON', 'SWANENERGY', 'SWIGGY', 'SYNGENE', 'SYRMA', 'TBOTEK', 'TVSMOTOR', 'TANLA', 'TATACHEM', 'TATACOMM', 'TCS', 'TATACONSUM', 'TATAELXSI', 'TATAINVEST', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TATATECH', 'TTML', 'TECHM', 'TECHNOE', 'TEJASNET', 'NIACL', 'RAMCOCEM', 'THERMAX', 'TIMKEN', 'TITAGARH', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TARIL', 'TRENT', 'TRIDENT', 'TRIVENI', 'TRITURBINE', 'TIINDIA', 'UCOBANK', 'UNOMINDA', 'UPL', 'UTIAMC', 'ULTRACEMCO', 'UNIONBANK', 'UBL', 'UNITDSPR', 'USHAMART', 'VGUARD', 'DBREALTY', 'VTL', 'VBL', 'MANYAVAR', 'VEDL', 'VIJAYA', 'VMM', 'IDEA', 'VOLTAS', 'WAAREEENER', 'WELCORP', 'WELSPUNLIV', 'WESTLIFE', 'WHIRLPOOL', 'WIPRO', 'WOCKPHARMA', 'YESBANK', 'ZFCVINDIA', 'ZEEL', 'ZENTEC', 'ZENSARTECH', 'ZYDUSLIFE', 'ECLERX']      
    TIMEFRAME = 'D'
    NUM_CANDLES = 252
    CYCLE_INTERVAL_SECONDS = 60

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