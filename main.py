# stat_arb_trader_dhan/main.py

import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from core.logger_setup import logger
from core.dhan_client import DhanClient
from data_feeds.instrument_manager import InstrumentManager
from data_feeds.data_fetcher import DataFetcher

# Import all available single-symbol strategy modules
from strategy_logic.ema_crossover import generate_signals as ema_crossover_strategy
from strategy_logic.rsi_divergence import generate_signals as rsi_divergence_strategy

# --- CORRECTED IMPORTS FOR STAT ARB STRATEGY ---
# We import the necessary functions and configuration constants for calculations.
from strategy_logic.stat_arb import (
    find_cointegrated_pairs,
    generate_pair_signals,
    strategy_cache,
    ROLLING_WINDOW, Z_SCORE_EXIT, Z_SCORE_STOP_LOSS
)

def run_strategy_cycle(strategy_function, symbols, timeframe, num_candles, instrument_mgr, data_fetcher):
    """
    Executes one full cycle for SINGLE-INSTRUMENT strategies (e.g., EMA, RSI).
    This function remains unchanged and works for strategies 1 and 2.
    """
    logger.info("--- Starting New Single-Instrument Strategy Cycle ---")
    
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
            
            if signals_df is None or signals_df.empty:
                logger.warning(f"Signal generation failed for {symbol}. Skipping analysis.")
                continue

            latest_signal_row = signals_df.iloc[-1]
            latest_signal = latest_signal_row['signal']
            
            if latest_signal in ['BUY', 'BULLISH_DIVERGENCE_ENTRY']: long_entries.append(symbol)
            elif latest_signal in ['SELL', 'BEARISH_DIVERGENCE_ENTRY']: short_entries.append(symbol)
            elif latest_signal in ['EXIT_LONG', 'LONG_EXIT_RSI']: long_exits.append(symbol)
            elif latest_signal in ['EXIT_SHORT', 'SHORT_EXIT_RSI']: short_exits.append(symbol)
            elif latest_signal == 'HOLD_LONG': hold_longs.append(symbol)
            elif latest_signal == 'HOLD_SHORT': hold_shorts.append(symbol)

            latest_timestamp_ist = latest_signal_row.name.tz_localize('UTC').tz_convert('Asia/Kolkata')
            ts_format = '%Y-%m-%d %H:%M IST' if timeframe.upper() not in ['D', 'W', '1D', '1W'] else '%Y-%m-%d'

            signal_color_map = {"BUY": "\033[92m", "BULLISH_DIVERGENCE_ENTRY": "\033[92m",
                                "SELL": "\033[91m", "BEARISH_DIVERGENCE_ENTRY": "\033[91m",
                                "EXIT_LONG": "\033[93m", "LONG_EXIT_RSI": "\033[93m",
                                "EXIT_SHORT": "\033[93m", "SHORT_EXIT_RSI": "\033[93m",
                                "HOLD_LONG": "\033[96m", "HOLD_SHORT": "\033[96m", "HOLD": "\033[0m"}
            END_COLOR, color = "\033[0m", signal_color_map.get(latest_signal, "\033[0m")
            colored_signal = f"{color}{latest_signal.replace('_', ' ')}{END_COLOR}"

            if 'divergence_status' in latest_signal_row.index:
                divergence_status = latest_signal_row.get('divergence_status', 'N/A')
                log_message = (f"\n----------- RSI DIVERGENCE ANALYSIS: {symbol} -----------\n"
                               f"  [DIVERGENCE STATUS]:  {divergence_status}\n\n"
                               f"  [FINAL SIGNAL]:       {colored_signal}\n"
                               f"----------------------------------------------------------")
            else:
                last_close = latest_signal_row.get('close', 0.0)
                adx_14 = latest_signal_row.get('ADX_14', 0.0)
                trend_filter_val = latest_signal_row.get('EMA_200', 0.0)
                market_state = "SIDEWAYS / CHOP"
                if adx_14 >= 25 and trend_filter_val > 0:
                    market_state = "UPTREND" if last_close > trend_filter_val else "DOWNTREND"
                log_message = (f"\n-------------------- ANALYSIS & SIGNAL: {symbol} --------------------\n"
                               f"  [MARKET STATE]: {market_state} (ADX: {adx_14:.2f})\n\n"
                               f"  [FINAL SIGNAL]:     {colored_signal}\n"
                               f"----------------------------------------------------------------------")
            
            logger.info(log_message)

            if "ENTRY" in latest_signal or "EXIT" in latest_signal:
                print(f"\n"
                      f"  **********************************************************\n"
                      f"  *** ACTIONABLE ALERT: {symbol} -> {color}{latest_signal.replace('_', ' ')}{END_COLOR} ***\n"
                      f"  **********************************************************\n")
        
        except Exception as e:
            logger.error(f"A critical error occurred while processing {symbol}. Error: {e}. Skipping.", exc_info=True)
            continue
    
    logger.info("--- Single-Instrument Strategy Cycle Finished ---")

def _calculate_trade_plan_details(signal, s1_df, s2_df):
    """
    Calculates the detailed price targets and stops for a given stat arb signal.
    This logic now correctly resides in the execution engine (main.py).
    """
    details = {}
    try:
        alpha, beta = signal.hedge_ratio
        
        # Recalculate spread stats from the full available history to get mean and std
        s1_log = np.log(s1_df['close'])
        s2_log = np.log(s2_df['close'])
        spread = s1_log - (alpha + beta * s2_log)
        rolling_spread = spread.tail(ROLLING_WINDOW)
        mean_spread, std_spread = rolling_spread.mean(), rolling_spread.std()

        s1_current_price = s1_df['close'].iloc[-1]
        s2_current_price = s2_df['close'].iloc[-1]
        details['s1_entry'] = s1_current_price
        details['s2_entry'] = s2_current_price

        # Determine target and stop Z-scores based on trade direction
        target_z = Z_SCORE_EXIT if "LONG" in signal.signal_type else -Z_SCORE_EXIT
        stop_z = -Z_SCORE_STOP_LOSS if "LONG" in signal.signal_type else Z_SCORE_STOP_LOSS

        # Calculate price target for the primary leg (s1)
        # log(s1_target) = (target_z * std) + mean + (beta * log(s2_current)) + alpha
        s1_log_target = (target_z * std_spread) + mean_spread + (beta * np.log(s2_current_price)) + alpha
        details['s1_target'] = np.exp(s1_log_target)
        
        # Calculate price stop for the primary leg (s1)
        s1_log_stop = (stop_z * std_spread) + mean_spread + (beta * np.log(s2_current_price)) + alpha
        details['s1_stop'] = np.exp(s1_log_stop)
        
    except Exception as e:
        logger.error(f"Failed to calculate trade plan details for {signal.pair}: {e}")
        return None
        
    return details

def main():
    """Main control script that runs the bot in a continuous loop."""
    # ========================== INPUT CONFIGURATION ==========================
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
    NUM_CANDLES_SINGLE = 200
    NUM_CANDLES_STAT_ARB = 504 # Approx 2 trading months on a 30M chart
    CYCLE_INTERVAL_SECONDS = 900
    PAIR_RECALC_INTERVAL = timedelta(hours=4)
    # =========================================================================

    logger.info("--- Initializing Trading Bot ---")
    dhan_connection = DhanClient()
    dhan_api_sdk = dhan_connection.get_api_client()
    if not dhan_api_sdk: sys.exit(1)
    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    if data_fetcher.instrument_master_df is None: sys.exit(1)

    if not 1 <= SELECTED_STRATEGY <= 3:
        logger.error(f"Invalid strategy number: {SELECTED_STRATEGY}. Exiting.")
        sys.exit(1)

    logger.info(
        f"\n"
        f"====================== BOT CONFIGURATION ======================\n"
        f"STRATEGY:          #{SELECTED_STRATEGY}\n"
        f"SYMBOLS:           ({len(SYMBOLS_TO_TRACK)}) Symbols Provided\n"
        f"TIMEFRAME:         {TIMEFRAME}\n"
        f"CYCLE INTERVAL:    {CYCLE_INTERVAL_SECONDS} seconds\n"
        f"==============================================================="
    )
    logger.info("--- Initialization Complete. Starting main loop. ---")
    
    while True:
        try:
            if SELECTED_STRATEGY == 3:
                logger.info("--- Starting Statistical Arbitrage Strategy Cycle ---")
                logger.info(f"Fetching data for all {len(SYMBOLS_TO_TRACK)} symbols...")
                all_symbols_data = {}
                for symbol in SYMBOLS_TO_TRACK:
                    isin = instrument_manager.get_isin_for_nse_symbol(symbol)
                    if not isin: continue
                    df = data_fetcher.fetch_data(symbol, isin, TIMEFRAME, NUM_CANDLES_STAT_ARB + 60)
                    if df is not None and not df.empty:
                        all_symbols_data[symbol] = df
                
                if len(all_symbols_data) < 2:
                    logger.error("Not enough symbol data to run Stat Arb.")
                    time.sleep(CYCLE_INTERVAL_SECONDS)
                    continue

                if "last_recalc_time" not in strategy_cache or \
                   (datetime.now() - strategy_cache.get("last_recalc_time", datetime.min)) > PAIR_RECALC_INTERVAL:
                    logger.info(f"Recalculation interval passed. Finding new pairs...")
                    formation_data = {s: df.tail(NUM_CANDLES_STAT_ARB) for s, df in all_symbols_data.items()}
                    strategy_cache["cointegrated_pairs"] = find_cointegrated_pairs(formation_data, NUM_CANDLES_STAT_ARB)
                    strategy_cache["last_recalc_time"] = datetime.now()
                
                pairs_to_trade = strategy_cache.get("cointegrated_pairs")
                if not pairs_to_trade:
                    logger.warning("No cointegrated pairs currently identified. Waiting for next recalc.")
                    time.sleep(CYCLE_INTERVAL_SECONDS)
                    continue

                logger.info(f"Generating signals for {len(pairs_to_trade)} identified pairs...")
                actionable_signals = []
                for pair_info in pairs_to_trade:
                    s1, s2 = pair_info['pair']
                    signal = generate_pair_signals(
                        pair_data_slice={s1: all_symbols_data[s1], s2: all_symbols_data[s2]},
                        pair_info=pair_info,
                        open_position_state=None
                    )
                    if signal and "ENTER" in signal.signal_type:
                        actionable_signals.append(signal)
                
                if not actionable_signals:
                    logger.info("No new actionable ENTRY signals generated in this cycle.")
                else:
                    logger.info("\n==================== STATISTICAL ARBITRAGE SIGNAL SUMMARY ====================")
                    for sig in actionable_signals:
                        s1, s2 = sig.pair
                        
                        # --- Calculate the full trade plan using the helper ---
                        details = _calculate_trade_plan_details(sig, all_symbols_data[s1], all_symbols_data[s2])
                        if not details: continue

                        # --- Build the detailed action plan log message ---
                        leg1_action, leg2_action = ("", "")
                        if sig.signal_type == "ENTER_LONG":
                            color = "\033[92m" # Green
                            leg1_action = f"  > Enter LONG  {s1:<12} at {details['s1_entry']:>8.2f}, Target: {details['s1_target']:>8.2f}, Stop: {details['s1_stop']:>8.2f}"
                            leg2_action = f"  > Enter SHORT {s2:<12} at {details['s2_entry']:>8.2f}"
                        elif sig.signal_type == "ENTER_SHORT":
                            color = "\033[91m" # Red
                            leg1_action = f"  > Enter SHORT {s1:<12} at {details['s1_entry']:>8.2f}, Target: {details['s1_target']:>8.2f}, Stop: {details['s1_stop']:>8.2f}"
                            leg2_action = f"  > Enter LONG  {s2:<12} at {details['s2_entry']:>8.2f}"
                        
                        log_message = (
                            f"\n----------- PAIR: {s1} / {s2} -----------\n"
                            f"  [STATUS]:           {color}{sig.signal_type.replace('_', ' '):<15}{sig.reason}\033[0m\n"
                            f"  [Z-Score]:          {sig.z_score:.2f}\n"
                            f"  [ACTION PLAN]:\n{leg1_action}\n{leg2_action}\n"
                            f"  > Time Stop: Hold for max {int(2.5 * sig.half_life)} candles.\n"
                            f"  > Hedge Ratio (β):  {sig.hedge_ratio[1]:.4f}" # Display only beta
                        )
                        logger.info(log_message)
                    logger.info("\n==============================================================================")
            else:
                # This block for strategies 1 and 2 calls the original cycle runner
                strategy_function = ema_crossover_strategy if SELECTED_STRATEGY == 1 else rsi_divergence_strategy
                run_strategy_cycle(
                    strategy_function=strategy_function,
                    symbols=SYMBOLS_TO_TRACK,
                    timeframe=TIMEFRAME,
                    num_candles=NUM_CANDLES_SINGLE,
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
            time.sleep(30)

if __name__ == "__main__":
    main()