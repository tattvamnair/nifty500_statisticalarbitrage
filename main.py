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
## ADDED: Import the new Statistical Arbitrage strategy ##
from strategy_logic.stat_arb import generate_signals as stat_arb_strategy

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
            
            # ** MODIFIED: Update lists to include RSI signal names **
            if latest_signal in ['BUY', 'BULLISH_DIVERGENCE_ENTRY']: long_entries.append(symbol)
            elif latest_signal in ['SELL', 'BEARISH_DIVERGENCE_ENTRY']: short_entries.append(symbol)
            elif latest_signal in ['EXIT_LONG', 'LONG_EXIT_RSI']: long_exits.append(symbol)
            elif latest_signal in ['EXIT_SHORT', 'SHORT_EXIT_RSI']: short_exits.append(symbol)
            elif latest_signal == 'HOLD_LONG': hold_longs.append(symbol)
            elif latest_signal == 'HOLD_SHORT': hold_shorts.append(symbol)

            # --- INTELLIGENT DASHBOARD LOGGING ---
            latest_timestamp_ist = latest_signal_row.name.tz_localize('UTC').tz_convert('Asia/Kolkata')
            ts_format = '%Y-%m-%d %H:%M IST' if timeframe.upper() not in ['D', 'W', '1D', '1W'] else '%Y-%m-%d'

            # ** MODIFIED: Update color map to include RSI signal names **
            signal_color_map = {"BUY": "\033[92m", "BULLISH_DIVERGENCE_ENTRY": "\033[92m",
                                "SELL": "\033[91m", "BEARISH_DIVERGENCE_ENTRY": "\033[91m",
                                "EXIT_LONG": "\033[93m", "LONG_EXIT_RSI": "\033[93m",
                                "EXIT_SHORT": "\033[93m", "SHORT_EXIT_RSI": "\033[93m",
                                "HOLD_LONG": "\033[96m", "HOLD_SHORT": "\033[96m", "HOLD": "\033[0m"}
            END_COLOR, color = "\033[0m", signal_color_map.get(latest_signal, "\033[0m")
            colored_signal = f"{color}{latest_signal.replace('_', ' ')}{END_COLOR}"

            # ** NEW: Check for the 'divergence_status' tag and display the correct dashboard **
            if 'divergence_status' in latest_signal_row.index:
                # --- RSI DIVERGENCE DASHBOARD ---
                divergence_status = latest_signal_row.get('divergence_status', 'N/A')
                log_message = (
                    f"\n----------- RSI DIVERGENCE ANALYSIS: {symbol} -----------\n"
                    f"  [DIVERGENCE STATUS]:  {divergence_status}\n\n"
                    f"  [LATEST CANDLE DATA]:\n"
                    f"    - Timestamp:          {latest_timestamp_ist.strftime(ts_format)}\n"
                    f"    - Price:              {latest_signal_row.get('close', 0.0):.2f}\n"
                    f"    - RSI(14):            {latest_signal_row.get('RSI_14', 0.0):.2f}\n\n"
                    f"  [FINAL SIGNAL]:\n"
                    f"    - Decision:           {colored_signal}\n"
                    f"    - Position Status:    {int(latest_signal_row['position'])} (1=Long, -1=Short, 0=Flat)\n"
                    f"----------------------------------------------------------"
                )
            else:
                # --- EMA CROSSOVER DASHBOARD (Your original logic) ---
                last_close = latest_signal_row.get('close', 0.0)
                adx_14 = latest_signal_row.get('ADX_14', 0.0)
                trend_filter_val = latest_signal_row.get('EMA_200', 0.0)
                trend_filter_name = "EMA(200)"
                adx_threshold = 25
                fast_ema, slow_ema = latest_signal_row.get('EMA_9', 0.0), latest_signal_row.get('EMA_15', 0.0)
                ema_label = "EMA(9) / EMA(15)"
                market_state = "SIDEWAYS / CHOP"
                if adx_14 >= adx_threshold and trend_filter_val > 0:
                    market_state = "UPTREND" if last_close > trend_filter_val else "DOWNTREND"
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

            # ** MODIFIED: More robust check for actionable alerts **
            if "ENTRY" in latest_signal or "EXIT" in latest_signal:
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
    #INPUT
    ## MODIFIED FOR STRATEGY 3 ##
    # Set to 1, 2, or 3 to choose your strategy
    SELECTED_STRATEGY = 3
    
    #NIFTY50
    SYMBOLS_TO_TRACK = ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BEL', 'BHARTIARTL', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'ETERNAL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'JIOFIN', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SHRIRAMFIN', 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO']
    #NIFTY500
    #SYMBOLS_TO_TRACK = ['360ONE', '3MINDIA', 'ABB', 'ACC', 'ACMESOLAR', 'AIAENG', 'APLAPOLLO', 'AUBANK', 'AWL', 'AADHARHFC', 'AARTIIND', 'AAVAS', 'ABBOTINDIA', 'ACE', 'ADANIENSOL', 'ADANIENT', 'ADANIGREEN', 'ADANIPORTS', 'ADANIPOWER', 'ATGL', 'ABCAPITAL', 'ABFRL', 'ABREL', 'ABSLAMC', 'AEGISLOG', 'AFCONS', 'AFFLE', 'AJANTPHARM', 'AKUMS', 'APLLTD', 'ALIVUS', 'ALKEM', 'ALKYLAMINE', 'ALOKINDS', 'ARE&M', 'AMBER', 'AMBUJACEM', 'ANANDRATHI', 'ANANTRAJ', 'ANGELONE', 'APARINDS', 'APOLLOHOSP', 'APOLLOTYRE', 'APTUS', 'ASAHIINDIA', 'ASHOKLEY', 'ASIANPAINT', 'ASTERDM', 'ASTRAZEN', 'ASTRAL', 'ATUL', 'AUROPHARMA', 'AIIL', 'DMART', 'AXISBANK', 'BASF', 'BEML', 'BLS', 'BSE', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BAJAJHLDNG', 'BAJAJHFL', 'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA', 'MAHABANK', 'BATAINDIA', 'BAYERCROP', 'BERGEPAINT', 'BDL', 'BEL', 'BHARATFORG', 'BHEL', 'BPCL', 'BHARTIARTL', 'BHARTIHEXA', 'BIKAJI', 'BIOCON', 'BSOFT', 'BLUEDART', 'BLUESTARCO', 'BBTC', 'BOSCHLTD', 'FIRSTCRY', 'BRIGADE', 'BRITANNIA', 'MAPMYINDIA', 'CCL', 'CESC', 'CGPOWER', 'CRISIL', 'CAMPUS', 'CANFINHOME', 'CANBK', 'CAPLIPOINT', 'CGCL', 'CARBORUNIV', 'CASTROLIND', 'CEATLTD', 'CENTRALBK', 'CDSL', 'CENTURYPLY', 'CERA', 'CHALET', 'CHAMBLFERT', 'CHENNPETRO', 'CHOLAHLDNG', 'CHOLAFIN', 'CIPLA', 'CUB', 'CLEAN', 'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COHANCE', 'COLPAL', 'CAMS', 'CONCORDBIO', 'CONCOR', 'COROMANDEL', 'CRAFTSMAN', 'CREDITACC', 'CROMPTON', 'CUMMINSIND', 'CYIENT', 'DCMSHRIRAM', 'DLF', 'DOMS', 'DABUR', 'DALBHARAT', 'DATAPATTNS', 'DEEPAKFERT', 'DEEPAKNTR', 'DELHIVERY', 'DEVYANI', 'DIVISLAB', 'DIXON', 'LALPATHLAB', 'DRREDDY', 'DUMMYRAYMN', 'EIDPARRY', 'EIHOTEL', 'EICHERMOT', 'ELECON', 'ELGIEQUIP', 'EMAMILTD', 'EMCURE', 'ENDURANCE', 'ENGINERSIN', 'ERIS', 'ESCORTS', 'ETERNAL', 'EXIDEIND', 'NYKAA', 'FEDERALBNK', 'FACT', 'FINCABLES', 'FINPIPE', 'FSL', 'FIVESTAR', 'FORTIS', 'GAIL', 'GVT&D', 'GMRAIRPORT', 'GRSE', 'GICRE', 'GILLETTE', 'GLAND', 'GLAXO', 'GLENMARK', 'MEDANTA', 'GODIGIT', 'GPIL', 'GODFRYPHLP', 'GODREJAGRO', 'GODREJCP', 'GODREJIND', 'GODREJPROP', 'GRANULES', 'GRAPHITE', 'GRASIM', 'GRAVITA', 'GESHIP', 'FLUOROCHEM', 'GUJGASLTD', 'GMDCLTD', 'GNFC', 'GPPL', 'GSPL', 'HEG', 'HBLENGINE', 'HCLTECH', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HFCL', 'HAPPSTMNDS', 'HAVELLS', 'HEROMOTOCO', 'HSCL', 'HINDALCO', 'HAL', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC', 'POWERINDIA', 'HOMEFIRST', 'HONASA', 'HONAUT', 'HUDCO', 'HYUNDAI', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB', 'IFCI', 'IIFL', 'INOXINDIA', 'IRB', 'IRCON', 'ITC', 'ITI', 'INDGN', 'INDIACEM', 'INDIAMART', 'INDIANB', 'IEX', 'INDHOTEL', 'IOC', 'IOB', 'IRCTC', 'IRFC', 'IREDA', 'IGL', 'INDUSTOWER', 'INDUSINDBK', 'NAUKRI', 'INFY', 'INOXWIND', 'INTELLECT', 'INDIGO', 'IGIL', 'IKS', 'IPCALAB', 'JBCHEPHARM', 'JKCEMENT', 'JBMA', 'JKTYRE', 'JMFINANCIL', 'JSWENERGY', 'JSWHL', 'JSWINFRA', 'JSWSTEEL', 'JPPOWER', 'J&KBANK', 'JINDALSAW', 'JSL', 'JINDALSTEL', 'JIOFIN', 'JUBLFOOD', 'JUBLINGREA', 'JUBLPHARMA', 'JWL', 'JUSTDIAL', 'JYOTHYLAB', 'JYOTICNC', 'KPRMILL', 'KEI', 'KNRCON', 'KPITTECH', 'KAJARIACER', 'KPIL', 'KALYANKJIL', 'KANSAINER', 'KARURVYSYA', 'KAYNES', 'KEC', 'KFINTECH', 'KIRLOSBROS', 'KIRLOSENG', 'KOTAKBANK', 'KIMS', 'LTF', 'LTTS', 'LICHSGFIN', 'LTFOODS', 'LTIM', 'LT', 'LATENTVIEW', 'LAURUSLABS', 'LEMONTREE', 'LICI', 'LINDEINDIA', 'LLOYDSME', 'LUPIN', 'MMTC', 'MRF', 'LODHA', 'MGL', 'MAHSEAMLES', 'M&MFIN', 'M&M', 'MANAPPURAM', 'MRPL', 'MANKIND', 'MARICO', 'MARUTI', 'MASTEK', 'MFSL', 'MAXHEALTH', 'MAZDOCK', 'METROPOLIS', 'MINDACORP', 'MSUMI', 'MOTILALOFS', 'MPHASIS', 'MCX', 'MUTHOOTFIN', 'NATCOPHARM', 'NBCC', 'NCC', 'NHPC', 'NLCINDIA', 'NMDC', 'NSLNISP', 'NTPCGREEN', 'NTPC', 'NH', 'NATIONALUM', 'NAVA', 'NAVINFLUOR', 'NESTLEIND', 'NETWEB', 'NETWORK18', 'NEULANDLAB', 'NEWGEN', 'NAM-INDIA', 'NIVABUPA', 'NUVAMA', 'OBEROIRLTY', 'ONGC', 'OIL', 'OLAELEC', 'OLECTRA', 'PAYTM', 'OFSS', 'POLICYBZR', 'PCBL', 'PGEL', 'PIIND', 'PNBHOUSING', 'PNCINFRA', 'PTCIL', 'PVRINOX', 'PAGEIND', 'PATANJALI', 'PERSISTENT', 'PETRONET', 'PFIZER', 'PHOENIXLTD', 'PIDILITIND', 'PEL', 'PPLPHARMA', 'POLYMED', 'POLYCAB', 'POONAWALLA', 'PFC', 'POWERGRID', 'PRAJIND', 'PREMIERENE', 'PRESTIGE', 'PNB', 'RRKABEL', 'RBLBANK', 'RECLTD', 'RHIM', 'RITES', 'RADICO', 'RVNL', 'RAILTEL', 'RAINBOW', 'RKFORGE', 'RCF', 'RTNINDIA', 'RAYMONDLSL', 'RAYMOND', 'REDINGTON', 'RELIANCE', 'RPOWER', 'ROUTE', 'SBFC', 'SBICARD', 'SBILIFE', 'SJVN', 'SKFINDIA', 'SRF', 'SAGILITY', 'SAILIFE', 'SAMMAANCAP', 'MOTHERSON', 'SAPPHIRE', 'SARDAEN', 'SAREGAMA', 'SCHAEFFLER', 'SCHNEIDER', 'SCI', 'SHREECEM', 'RENUKA', 'SHRIRAMFIN', 'SHYAMMETL', 'SIEMENS', 'SIGNATURE', 'SOBHA', 'SOLARINDS', 'SONACOMS', 'SONATSOFTW', 'STARHEALTH', 'SBIN', 'SAIL', 'SWSOLAR', 'SUMICHEM', 'SUNPHARMA', 'SUNTV', 'SUNDARMFIN', 'SUNDRMFAST', 'SUPREMEIND', 'SUZLON', 'SWANENERGY', 'SWIGGY', 'SYNGENE', 'SYRMA', 'TBOTEK', 'TVSMOTOR', 'TANLA', 'TATACHEM', 'TATACOMM', 'TCS', 'TATACONSUM', 'TATAELXSI', 'TATAINVEST', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TATATECH', 'TTML', 'TECHM', 'TECHNOE', 'TEJASNET', 'NIACL', 'RAMCOCEM', 'THERMAX', 'TIMKEN', 'TITAGARH', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TARIL', 'TRENT', 'TRIDENT', 'TRIVENI', 'TRITURBINE', 'TIINDIA', 'UCOBANK', 'UNOMINDA', 'UPL', 'UTIAMC', 'ULTRACEMCO', 'UNIONBANK', 'UBL', 'UNITDSPR', 'USHAMART', 'VGUARD', 'DBREALTY', 'VTL', 'VBL', 'MANYAVAR', 'VEDL', 'VIJAYA', 'VMM', 'IDEA', 'VOLTAS', 'WAAREEENER', 'WELCORP', 'WELSPUNLIV', 'WESTLIFE', 'WHIRLPOOL', 'WIPRO', 'WOCKPHARMA', 'YESBANK', 'ZFCVINDIA', 'ZEEL', 'ZENTEC', 'ZENSARTECH', 'ZYDUSLIFE', 'ECLERX']
    #NIFTY MIDSMALLCAP 400
    #SYMBOLS_TO_TRACK = ['360ONE', '3MINDIA', 'ACC', 'ACMESOLAR', 'AIAENG', 'APLAPOLLO', 'AUBANK', 'AWL', 'AADHARHFC', 'AARTIIND', 'AAVAS', 'ABBOTINDIA', 'ACE', 'ATGL', 'ABCAPITAL', 'ABFRL', 'ABREL', 'ABSLAMC', 'AEGISLOG', 'AFCONS', 'AFFLE', 'AJANTPHARM', 'AKUMS', 'APLLTD', 'ALIVUS', 'ALKEM', 'ALKYLAMINE', 'ALOKINDS', 'ARE&M', 'AMBER', 'ANANDRATHI', 'ANANTRAJ', 'ANGELONE', 'APARINDS', 'APOLLOTYRE', 'APTUS', 'ASAHIINDIA', 'ASHOKLEY', 'ASTERDM', 'ASTRAZEN', 'ASTRAL', 'ATUL', 'AUROPHARMA', 'AIIL', 'BASF', 'BEML', 'BLS', 'BSE', 'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKINDIA', 'MAHABANK', 'BATAINDIA', 'BAYERCROP', 'BERGEPAINT', 'BDL', 'BHARATFORG', 'BHEL', 'BHARTIHEXA', 'BIKAJI', 'BIOCON', 'BSOFT', 'BLUEDART', 'BLUESTARCO', 'BBTC', 'FIRSTCRY', 'BRIGADE', 'MAPMYINDIA', 'CCL', 'CESC', 'CRISIL', 'CAMPUS', 'CANFINHOME', 'CAPLIPOINT', 'CGCL', 'CARBORUNIV', 'CASTROLIND', 'CEATLTD', 'CENTRALBK', 'CDSL', 'CENTURYPLY', 'CERA', 'CHALET', 'CHAMBLFERT', 'CHENNPETRO', 'CHOLAHLDNG', 'CUB', 'CLEAN', 'COCHINSHIP', 'COFORGE', 'COHANCE', 'COLPAL', 'CAMS', 'CONCORDBIO', 'CONCOR', 'COROMANDEL', 'CRAFTSMAN', 'CREDITACC', 'CROMPTON', 'CUMMINSIND', 'CYIENT', 'DCMSHRIRAM', 'DOMS', 'DALBHARAT', 'DATAPATTNS', 'DEEPAKFERT', 'DEEPAKNTR', 'DELHIVERY', 'DEVYANI', 'DIXON', 'LALPATHLAB', 'EIDPARRY', 'EIHOTEL', 'ELECON', 'ELGIEQUIP', 'EMAMILTD', 'EMCURE', 'ENDURANCE', 'ENGINERSIN', 'ERIS', 'ESCORTS', 'EXIDEIND', 'NYKAA', 'FEDERALBNK', 'FACT', 'FINCABLES', 'FINPIPE', 'FSL', 'FIVESTAR', 'FORTIS', 'GVT&D', 'GMRAIRPORT', 'GRSE', 'GICRE', 'GILLETTE', 'GLAND', 'GLAXO', 'GLENMARK', 'MEDANTA', 'GODIGIT', 'GPIL', 'GODFRYPHLP', 'GODREJAGRO', 'GODREJIND', 'GODREJPROP', 'GRANULES', 'GRAPHITE', 'GRAVITA', 'GESHIP', 'FLUOROCHEM', 'GUJGASLTD', 'GMDCLTD', 'GNFC', 'GPPL', 'GSPL', 'HEG', 'HBLENGINE', 'HDFCAMC', 'HFCL', 'HAPPSTMNDS', 'HSCL', 'HINDCOPPER', 'HINDPETRO', 'HINDZINC', 'POWERINDIA', 'HOMEFIRST', 'HONASA', 'HONAUT', 'HUDCO', 'IDBI', 'IDFCFIRSTB', 'IFCI', 'IIFL', 'INOXINDIA', 'IRB', 'IRCON', 'ITI', 'INDGN', 'INDIACEM', 'INDIAMART', 'INDIANB', 'IEX', 'IOB', 'IRCTC', 'IREDA', 'IGL', 'INDUSTOWER', 'INOXWIND', 'INTELLECT', 'IGIL', 'IKS', 'IPCALAB', 'JBCHEPHARM', 'JKCEMENT', 'JBMA', 'JKTYRE', 'JMFINANCIL', 'JSWHL', 'JSWINFRA', 'JPPOWER', 'J&KBANK', 'JINDALSAW', 'JSL', 'JUBLFOOD', 'JUBLINGREA', 'JUBLPHARMA', 'JWL', 'JUSTDIAL', 'JYOTHYLAB', 'JYOTICNC', 'KPRMILL', 'KEI', 'KNRCON', 'KPITTECH', 'KAJARIACER', 'KPIL', 'KALYANKJIL', 'KANSAINER', 'KARURVYSYA', 'KAYNES', 'KEC', 'KFINTECH', 'KIRLOSBROS', 'KIRLOSENG', 'KIMS', 'LTF', 'LTTS', 'LICHSGFIN', 'LTFOODS', 'LATENTVIEW', 'LAURUSLABS', 'LEMONTREE', 'LINDEINDIA', 'LLOYDSME', 'LUPIN', 'MMTC', 'MRF', 'MGL', 'MAHSEAMLES', 'M&MFIN', 'MANAPPURAM', 'MRPL', 'MANKIND', 'MARICO', 'MASTEK', 'MFSL', 'MAXHEALTH', 'MAZDOCK', 'METROPOLIS', 'MINDACORP', 'MSUMI', 'MOTILALOFS', 'MPHASIS', 'MCX', 'MUTHOOTFIN', 'NATCOPHARM', 'NBCC', 'NCC', 'NHPC', 'NLCINDIA', 'NMDC', 'NSLNISP', 'NTPCGREEN', 'NH', 'NATIONALUM', 'NAVA', 'NAVINFLUOR', 'NETWEB', 'NETWORK18', 'NEULANDLAB', 'NEWGEN', 'NAM-INDIA', 'NIVABUPA', 'NUVAMA', 'OBEROIRLTY', 'OIL', 'OLAELEC', 'OLECTRA', 'PAYTM', 'OFSS', 'POLICYBZR', 'PCBL', 'PGEL', 'PIIND', 'PNBHOUSING', 'PNCINFRA', 'PTCIL', 'PVRINOX', 'PAGEIND', 'PATANJALI', 'PERSISTENT', 'PETRONET', 'PFIZER', 'PHOENIXLTD', 'PEL', 'PPLPHARMA', 'POLYMED', 'POLYCAB', 'POONAWALLA', 'PRAJIND', 'PREMIERENE', 'PRESTIGE', 'RRKABEL', 'RBLBANK', 'RHIM', 'RITES', 'RADICO', 'RVNL', 'RAILTEL', 'RAINBOW', 'RKFORGE', 'RCF', 'RTNINDIA', 'RAYMONDLSL', 'RAYMOND', 'RAYMONDREL', 'REDINGTON', 'RPOWER', 'ROUTE', 'SBFC', 'SBICARD', 'SJVN', 'SKFINDIA', 'SRF', 'SAGILITY', 'SAILIFE', 'SAMMAANCAP', 'SAPPHIRE', 'SARDAEN', 'SAREGAMA', 'SCHAEFFLER', 'SCHNEIDER', 'SCI', 'RENUKA', 'SHYAMMETL', 'SIGNATURE', 'SOBHA', 'SOLARINDS', 'SONACOMS', 'SONATSOFTW', 'STARHEALTH', 'SAIL', 'SWSOLAR', 'SUMICHEM', 'SUNTV', 'SUNDARMFIN', 'SUNDRMFAST', 'SUPREMEIND', 'SUZLON', 'SWANENERGY', 'SYNGENE', 'SYRMA', 'TBOTEK', 'TANLA', 'TATACHEM', 'TATACOMM', 'TATAELXSI', 'TATAINVEST', 'TATATECH', 'TTML', 'TECHNOE', 'TEJASNET', 'NIACL', 'RAMCOCEM', 'THERMAX', 'TIMKEN', 'TITAGARH', 'TORNTPOWER', 'TARIL', 'TRIDENT', 'TRIVENI', 'TRITURBINE', 'TIINDIA', 'UCOBANK', 'UNOMINDA', 'UPL', 'UTIAMC', 'UNIONBANK', 'UBL', 'USHAMART', 'VGUARD', 'DBREALTY', 'VTL', 'MANYAVAR', 'VIJAYA', 'VMM', 'IDEA', 'VOLTAS', 'WAAREEENER', 'WELCORP', 'WELSPUNLIV', 'WESTLIFE', 'WHIRLPOOL', 'WOCKPHARMA', 'YESBANK', 'ZFCVINDIA', 'ZEEL', 'ZENTEC', 'ZENSARTECH', 'ECLERX']
    #NIFTY microcap list
    #SYMBOLS_TO_TRACK = ['AGI', 'ASKAUTOLTD', 'AARTIDRUGS', 'AARTIPHARM', 'ACUTAAS', 'AVL', 'ADVENZYMES', 'AETHER', 'AHLUCONT', 'AKZOINDIA', 'ALLCARGO', 'ABDL', 'PARKHOTELS', 'ACI', 'ARVINDFASN', 'ARVIND', 'ASHOKA', 'ASTRAMICRO', 'AURIONPRO', 'AVALON', 'AVANTIFEED', 'AWFIS', 'AZAD', 'BAJAJHIND', 'BALAMINES', 'BALUFORGE', 'BANCOINDIA', 'BANSALWIRE', 'BEPL', 'BBL', 'BIRLACORPN', 'BLUEJET', 'BOMDYEING', 'BOROLTD', 'BORORENEW', 'CIEINDIA', 'CMSINFO', 'CSBBANK', 'CARTRADE', 'CEIGALL', 'CELLO', 'CHEMPLASTS', 'CHOICEIN', 'CIGNITITEC', 'CYIENTDLM', 'DCBBANK', 'DCXINDIA', 'DATAMATICS', 'DHANI', 'DBL', 'DCAL', 'DODLA', 'DUMMYSTLTE', 'DYNAMATECH', 'EPL', 'EASEMYTRIP', 'EDELWEISS', 'EMIL', 'ELECTCAST', 'EMBDL', 'ENTERO', 'EIEL', 'EPIGRAL', 'EQUITASBNK', 'ETHOSLTD', 'EUREKAFORB', 'FDC', 'FIEMIND', 'FINEORG', 'FCL', 'FORCEMOT', 'GRINFRA', 'GHCL', 'GMMPFAUDLR', 'GMRP&UI', 'GABRIEL', 'GANESHHOUC', 'GANECOS', 'GRWRHITECH', 'GARFIBRES', 'GATEWAY', 'GOKEX', 'GOPAL', 'GREAVESCOT', 'GREENPANEL', 'GREENPLY', 'GAEL', 'GSFC', 'GULFOILLUB', 'HGINFRA', 'HATHWAY', 'HCG', 'HEIDELBERG', 'HEMIPROP', 'HERITGFOOD', 'HIKAL', 'HCC', 'IFBIND', 'IIFLCAPS', 'ITDCEM', 'IMAGICAA', 'INDIAGLYCO', 'INDIASHLTR', 'IMFA', 'INDIGOPNTS', 'ICIL', 'INFIBEAM', 'INGERRAND', 'INNOVACAP', 'INOXGREEN', 'IONEXCHANG', 'ISGEC', 'JKIL', 'JKLAKSHMI', 'JKPAPER', 'JTLIND', 'JAIBALAJI', 'JAICORPLTD', 'JISLJALEQS', 'JAMNAAUTO', 'JSFB', 'JINDWORLD', 'JCHAC', 'KPIGREEN', 'KRBL', 'KSB', 'KSL', 'KTKBANK', 'KSCL', 'KIRLPNU', 'LMW', 'LXCHEM', 'IXIGO', 'LLOYDSENGG', 'LLOYDSENT', 'LUXIND', 'MOIL', 'MSTCLTD', 'MTARTECH', 'MAHSCOOTER', 'MAHLIFE', 'MANINFRA', 'MARKSANS', 'MAXESTATES', 'MEDPLUS', 'MIDHANI', 'BECTORFOOD', 'NEOGEN', 'NESCO', 'NOCIL', 'NFL', 'NAZARA', 'NUVOCO', 'OPTIEMUS', 'ORCHPHARMA', 'ORIENTCEM', 'ORISSAMINE', 'PNGJL', 'PCJEWELLER', 'PTC', 'PAISALO', 'PARADEEP', 'PARAS', 'PATELENG', 'PGIL', 'POLYPLEX', 'POWERMECH', 'PRICOLLTD', 'PRINCEPIPE', 'PRSMJOHNSN', 'PRUDENT', 'RAIN', 'RAJESHEXPO', 'RALLIS', 'RATEGAIN', 'RTNPOWER', 'REDTAPE', 'REFEX', 'RELINFRA', 'RELIGARE', 'RESPONIND', 'RBA', 'ROSSARI', 'SAFARI', 'SAMHI', 'SANOFICONR', 'SANOFI', 'SANSERA', 'SENCO', 'SEQUENT', 'SHAILY', 'SHAKTIPUMP', 'SHARDACROP', 'SHAREINDIA', 'SFL', 'SHILPAMED', 'SBCL', 'SHOPERSTOP', 'SHRIPISTON', 'SKIPPER', 'SOUTHBANK', 'SPANDANA', 'STARCEMENT', 'STLTECH', 'STAR', 'STYLAMIND', 'SUBROS', 'SUDARSCHEM', 'SULA', 'SPARC', 'SUNFLAG', 'SUNTECK', 'SUPRAJIT', 'SUPRIYA', 'SURYAROSNI', 'SYMPHONY', 'TARC', 'TDPOWERSYS', 'TVSSCS', 'TEAMLEASE', 'TIIL', 'TEGA', 'TEXRAIL', 'THANGAMAYL', 'ANUP', 'TIRUMALCHM', 'THOMASCOOK', 'TI', 'TIMETECHNO', 'TIPSMUSIC', 'TRANSRAILL', 'UJJIVANSFB', 'UNIMECH', 'VMART', 'VIPIND', 'VSTIND', 'WABAG', 'VAIBHAVGBL', 'VARROC', 'VENTIVE', 'VENUSPIPES', 'VESUVIUS', 'VOLTAMP', 'WEBELSOLAR', 'WELENT', 'WONDERLA', 'YATHARTH', 'ZAGGLE', 'BLACKBUCK', 'ZYDUSWELL', 'EMUDHRA']

    
    TIMEFRAME = '5M'
    NUM_CANDLES = 252
    CYCLE_INTERVAL_SECONDS = 3600  
    # =================================================================================

    logger.info("--- Initializing Trading Bot ---")
    
    dhan_connection = DhanClient()
    dhan_api_sdk = dhan_connection.get_api_client()
    if not dhan_api_sdk: sys.exit(1)
    
    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    if data_fetcher.instrument_master_df is None: sys.exit(1)

    ## ADDED: Add strategy 3 to the map ##
    strategy_map = { 
        1: ema_crossover_strategy, 
        2: rsi_divergence_strategy,
        3: stat_arb_strategy 
    }
    strategy_function = strategy_map.get(SELECTED_STRATEGY)
    if not strategy_function:
        logger.error(f"Invalid strategy number: {SELECTED_STRATEGY}. Exiting.")
        sys.exit(1)
    
    logger.info(
        f"\n"
        f"====================== BOT CONFIGURATION ======================\n"
        f"STRATEGY:          #{SELECTED_STRATEGY} ({strategy_function.__name__})\n"
        ## MODIFIED FOR STRATEGY 3 ##
        f"SYMBOLS:           ({len(SYMBOLS_TO_TRACK)}) Symbols Provided\n"
        f"TIMEFRAME:         {TIMEFRAME}\n"
        f"CANDLES TO ANALYZE: {NUM_CANDLES}\n"
        f"CYCLE INTERVAL:    {CYCLE_INTERVAL_SECONDS} seconds\n"
        f"==============================================================="
    )
    
    logger.info("--- Initialization Complete. Starting main loop. ---")
    
    while True:
        try:
            ## MODIFIED FOR STRATEGY 3: Branching logic to handle different strategy types ##
            if SELECTED_STRATEGY == 3:
                # This block is exclusively for the pairs trading strategy
                logger.info("--- Starting Pairs Strategy Cycle ---")
                logger.info(f"Fetching data for all {len(SYMBOLS_TO_TRACK)} symbols for analysis...")
                all_symbols_data = {}
                for symbol in SYMBOLS_TO_TRACK:
                    isin = instrument_manager.get_isin_for_nse_symbol(symbol)
                    if not isin:
                        logger.warning(f"Could not find ISIN for {symbol}. Skipping.")
                        continue
                    
                    df = data_fetcher.fetch_data(symbol, isin, TIMEFRAME, NUM_CANDLES)
                    if not df.empty:
                        all_symbols_data[symbol] = df
                    else:
                        logger.warning(f"No data fetched for {symbol}.")
                
                if len(all_symbols_data) > 1:
                    # Call the stat arb function with the complete dataset
                    strategy_function(all_symbols_data, NUM_CANDLES)
                else:
                    logger.error("Not enough data to run Statistical Arbitrage. Need data for at least 2 symbols.")
            
            else:
                # This block runs the original logic for single-instrument strategies (1 and 2)
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