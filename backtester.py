# stat_arb_trader_dhan/backtester.py

import pandas as pd
import numpy as np
import csv
import os
from collections import namedtuple
from datetime import timedelta

from core.logger_setup import logger
# --- CORRECTED IMPORTS ---
# We now need the DhanClient to make a real API connection
from core.dhan_client import DhanClient
from data_feeds.instrument_manager import InstrumentManager
from data_feeds.data_fetcher import DataFetcher
from strategy_logic.stat_arb import (
    find_cointegrated_pairs, generate_pair_signals, strategy_cache, ROLLING_WINDOW
)

# ==============================================================================
# ========================= BACKTESTER CONFIGURATION ===========================
# ==============================================================================
#SYMBOLS_TO_TEST = ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BEL', 'BHARTIARTL', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'ETERNAL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'JIOFIN', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SHRIRAMFIN', 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO']
SYMBOLS_TO_TEST = ['360ONE', '3MINDIA', 'ABB', 'ACC', 'ACMESOLAR', 'AIAENG', 'APLAPOLLO', 'AUBANK', 'AWL', 'AADHARHFC', 'AARTIIND', 'AAVAS', 'ABBOTINDIA', 'ACE', 'ADANIENSOL', 'ADANIENT', 'ADANIGREEN', 'ADANIPORTS', 'ADANIPOWER', 'ATGL', 'ABCAPITAL', 'ABFRL', 'ABREL', 'ABSLAMC', 'AEGISLOG', 'AFCONS', 'AFFLE', 'AJANTPHARM', 'AKUMS', 'APLLTD', 'ALIVUS', 'ALKEM', 'ALKYLAMINE', 'ALOKINDS', 'ARE&M', 'AMBER', 'AMBUJACEM', 'ANANDRATHI', 'ANANTRAJ', 'ANGELONE', 'APARINDS', 'APOLLOHOSP', 'APOLLOTYRE', 'APTUS', 'ASAHIINDIA', 'ASHOKLEY', 'ASIANPAINT', 'ASTERDM', 'ASTRAZEN', 'ASTRAL', 'ATUL', 'AUROPHARMA', 'AIIL', 'DMART', 'AXISBANK', 'BASF', 'BEML', 'BLS', 'BSE', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BAJAJHLDNG', 'BAJAJHFL', 'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA', 'MAHABANK', 'BATAINDIA', 'BAYERCROP', 'BERGEPAINT', 'BDL', 'BEL', 'BHARATFORG', 'BHEL', 'BPCL', 'BHARTIARTL', 'BHARTIHEXA', 'BIKAJI', 'BIOCON', 'BSOFT', 'BLUEDART', 'BLUESTARCO', 'BBTC', 'BOSCHLTD', 'FIRSTCRY', 'BRIGADE', 'BRITANNIA', 'MAPMYINDIA', 'CCL', 'CESC', 'CGPOWER', 'CRISIL', 'CAMPUS', 'CANFINHOME', 'CANBK', 'CAPLIPOINT', 'CGCL', 'CARBORUNIV', 'CASTROLIND', 'CEATLTD', 'CENTRALBK', 'CDSL', 'CENTURYPLY', 'CERA', 'CHALET', 'CHAMBLFERT', 'CHENNPETRO', 'CHOLAHLDNG', 'CHOLAFIN', 'CIPLA', 'CUB', 'CLEAN', 'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COHANCE', 'COLPAL', 'CAMS', 'CONCORDBIO', 'CONCOR', 'COROMANDEL', 'CRAFTSMAN', 'CREDITACC', 'CROMPTON', 'CUMMINSIND', 'CYIENT', 'DCMSHRIRAM', 'DLF', 'DOMS', 'DABUR', 'DALBHARAT', 'DATAPATTNS', 'DEEPAKFERT', 'DEEPAKNTR', 'DELHIVERY', 'DEVYANI', 'DIVISLAB', 'DIXON', 'LALPATHLAB', 'DRREDDY', 'DUMMYRAYMN', 'EIDPARRY', 'EIHOTEL', 'EICHERMOT', 'ELECON', 'ELGIEQUIP', 'EMAMILTD', 'EMCURE', 'ENDURANCE', 'ENGINERSIN', 'ERIS', 'ESCORTS', 'ETERNAL', 'EXIDEIND', 'NYKAA', 'FEDERALBNK', 'FACT', 'FINCABLES', 'FINPIPE', 'FSL', 'FIVESTAR', 'FORTIS', 'GAIL', 'GVT&D', 'GMRAIRPORT', 'GRSE', 'GICRE', 'GILLETTE', 'GLAND', 'GLAXO', 'GLENMARK', 'MEDANTA', 'GODIGIT', 'GPIL', 'GODFRYPHLP', 'GODREJAGRO', 'GODREJCP', 'GODREJIND', 'GODREJPROP', 'GRANULES', 'GRAPHITE', 'GRASIM', 'GRAVITA', 'GESHIP', 'FLUOROCHEM', 'GUJGASLTD', 'GMDCLTD', 'GNFC', 'GPPL', 'GSPL', 'HEG', 'HBLENGINE', 'HCLTECH', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HFCL', 'HAPPSTMNDS', 'HAVELLS', 'HEROMOTOCO', 'HSCL', 'HINDALCO', 'HAL', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC', 'POWERINDIA', 'HOMEFIRST', 'HONASA', 'HONAUT', 'HUDCO', 'HYUNDAI', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB', 'IFCI', 'IIFL', 'INOXINDIA', 'IRB', 'IRCON', 'ITC', 'ITI', 'INDGN', 'INDIACEM', 'INDIAMART', 'INDIANB', 'IEX', 'INDHOTEL', 'IOC', 'IOB', 'IRCTC', 'IRFC', 'IREDA', 'IGL', 'INDUSTOWER', 'INDUSINDBK', 'NAUKRI', 'INFY', 'INOXWIND', 'INTELLECT', 'INDIGO', 'IGIL', 'IKS', 'IPCALAB', 'JBCHEPHARM', 'JKCEMENT', 'JBMA', 'JKTYRE', 'JMFINANCIL', 'JSWENERGY', 'JSWHL', 'JSWINFRA', 'JSWSTEEL', 'JPPOWER', 'J&KBANK', 'JINDALSAW', 'JSL', 'JINDALSTEL', 'JIOFIN', 'JUBLFOOD', 'JUBLINGREA', 'JUBLPHARMA', 'JWL', 'JUSTDIAL', 'JYOTHYLAB', 'JYOTICNC', 'KPRMILL', 'KEI', 'KNRCON', 'KPITTECH', 'KAJARIACER', 'KPIL', 'KALYANKJIL', 'KANSAINER', 'KARURVYSYA', 'KAYNES', 'KEC', 'KFINTECH', 'KIRLOSBROS', 'KIRLOSENG', 'KOTAKBANK', 'KIMS', 'LTF', 'LTTS', 'LICHSGFIN', 'LTFOODS', 'LTIM', 'LT', 'LATENTVIEW', 'LAURUSLABS', 'LEMONTREE', 'LICI', 'LINDEINDIA', 'LLOYDSME', 'LUPIN', 'MMTC', 'MRF', 'LODHA', 'MGL', 'MAHSEAMLES', 'M&MFIN', 'M&M', 'MANAPPURAM', 'MRPL', 'MANKIND', 'MARICO', 'MARUTI', 'MASTEK', 'MFSL', 'MAXHEALTH', 'MAZDOCK', 'METROPOLIS', 'MINDACORP', 'MSUMI', 'MOTILALOFS', 'MPHASIS', 'MCX', 'MUTHOOTFIN', 'NATCOPHARM', 'NBCC', 'NCC', 'NHPC', 'NLCINDIA', 'NMDC', 'NSLNISP', 'NTPCGREEN', 'NTPC', 'NH', 'NATIONALUM', 'NAVA', 'NAVINFLUOR', 'NESTLEIND', 'NETWEB', 'NETWORK18', 'NEULANDLAB', 'NEWGEN', 'NAM-INDIA', 'NIVABUPA', 'NUVAMA', 'OBEROIRLTY', 'ONGC', 'OIL', 'OLAELEC', 'OLECTRA', 'PAYTM', 'OFSS', 'POLICYBZR', 'PCBL', 'PGEL', 'PIIND', 'PNBHOUSING', 'PNCINFRA', 'PTCIL', 'PVRINOX', 'PAGEIND', 'PATANJALI', 'PERSISTENT', 'PETRONET', 'PFIZER', 'PHOENIXLTD', 'PIDILITIND', 'PEL', 'PPLPHARMA', 'POLYMED', 'POLYCAB', 'POONAWALLA', 'PFC', 'POWERGRID', 'PRAJIND', 'PREMIERENE', 'PRESTIGE', 'PNB', 'RRKABEL', 'RBLBANK', 'RECLTD', 'RHIM', 'RITES', 'RADICO', 'RVNL', 'RAILTEL', 'RAINBOW', 'RKFORGE', 'RCF', 'RTNINDIA', 'RAYMONDLSL', 'RAYMOND', 'REDINGTON', 'RELIANCE', 'RPOWER', 'ROUTE', 'SBFC', 'SBICARD', 'SBILIFE', 'SJVN', 'SKFINDIA', 'SRF', 'SAGILITY', 'SAILIFE', 'SAMMAANCAP', 'MOTHERSON', 'SAPPHIRE', 'SARDAEN', 'SAREGAMA', 'SCHAEFFLER', 'SCHNEIDER', 'SCI', 'SHREECEM', 'RENUKA', 'SHRIRAMFIN', 'SHYAMMETL', 'SIEMENS', 'SIGNATURE', 'SOBHA', 'SOLARINDS', 'SONACOMS', 'SONATSOFTW', 'STARHEALTH', 'SBIN', 'SAIL', 'SWSOLAR', 'SUMICHEM', 'SUNPHARMA', 'SUNTV', 'SUNDARMFIN', 'SUNDRMFAST', 'SUPREMEIND', 'SUZLON', 'SWANENERGY', 'SWIGGY', 'SYNGENE', 'SYRMA', 'TBOTEK', 'TVSMOTOR', 'TANLA', 'TATACHEM', 'TATACOMM', 'TCS', 'TATACONSUM', 'TATAELXSI', 'TATAINVEST', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TATATECH', 'TTML', 'TECHM', 'TECHNOE', 'TEJASNET', 'NIACL', 'RAMCOCEM', 'THERMAX', 'TIMKEN', 'TITAGARH', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TARIL', 'TRENT', 'TRIDENT', 'TRIVENI', 'TRITURBINE', 'TIINDIA', 'UCOBANK', 'UNOMINDA', 'UPL', 'UTIAMC', 'ULTRACEMCO', 'UNIONBANK', 'UBL', 'UNITDSPR', 'USHAMART', 'VGUARD', 'DBREALTY', 'VTL', 'VBL', 'MANYAVAR', 'VEDL', 'VIJAYA', 'VMM', 'IDEA', 'VOLTAS', 'WAAREEENER', 'WELCORP', 'WELSPUNLIV', 'WESTLIFE', 'WHIRLPOOL', 'WIPRO', 'WOCKPHARMA', 'YESBANK', 'ZFCVINDIA', 'ZEEL', 'ZENTEC', 'ZENSARTECH', 'ZYDUSLIFE', 'ECLERX']

# 5-minute
TIMEFRAME = '5M'
FORMATION_PERIOD_DAYS = 78   #78
PAIR_RECALC_PERIOD_DAYS = 15    #15


# 15-minute
#TIMEFRAME = '15M'
#FORMATION_PERIOD_DAYS = 90   #90
#PAIR_RECALC_PERIOD_DAYS = 21    #21


# 30-minute
#TIMEFRAME = '30M'
#FORMATION_PERIOD_DAYS = 90
#PAIR_RECALC_PERIOD_DAYS = 30


# 1-hour
#TIMEFRAME = '60M'
#FORMATION_PERIOD_DAYS = 126
#PAIR_RECALC_PERIOD_DAYS = 42

# 4-hour
#TIMEFRAME = '240M'
#FORMATION_PERIOD_DAYS = 252
#PAIR_RECALC_PERIOD_DAYS = 63

# Daily
#TIMEFRAME = '1440M'  # 1 day = 6.5 hours of trading
#FORMATION_PERIOD_DAYS = 252
#PAIR_RECALC_PERIOD_DAYS = 63



INITIAL_CAPITAL = 100_000_000.0
MAX_CONCURRENT_PAIRS = 4
TRADE_NOTIONAL_PER_PAIR = 25_000_000.0
FIXED_THEORETICAL_NOTIONAL = 10_000_000.0

TRANSACTION_COST_BPS = 0
ANNUAL_BORROW_COST_PERCENT = 0

OUTPUT_FILE_NAME = "trade_simulation_results_final.csv"
# ==============================================================================

def run_trade_simulator():
    # This function is unchanged and correct.
    logger.info("--- Starting Finalized Statistical Arbitrage Trade Simulator ---")

    logger.info("Initializing Dhan API connection for backtester data build...")
    dhan_connection = DhanClient()
    dhan_api_sdk = dhan_connection.get_api_client()
    if not dhan_api_sdk:
        logger.error("Failed to get Dhan API client. Check credentials. Exiting backtest.")
        return

    instrument_manager = InstrumentManager()
    data_fetcher = DataFetcher(dhan_api_sdk, instrument_manager)
    
    try:
        bars_per_hour = 60 // int(TIMEFRAME.replace('M',''))
    except (ValueError, ZeroDivisionError):
        logger.error(f"Invalid TIMEFRAME: {TIMEFRAME}. Must be in minutes (e.g., '5M', '30M').")
        return
        
    BARS_PER_DAY = int(6.5 * bars_per_hour)
    FORMATION_PERIOD_CANDLES = FORMATION_PERIOD_DAYS * BARS_PER_DAY
    PAIR_RECALC_PERIOD_CANDLES = PAIR_RECALC_PERIOD_DAYS * BARS_PER_DAY

    logger.info(f"Loading historical data for {len(SYMBOLS_TO_TEST)} symbols...")
    all_historical_data = {}
    total_candles_to_fetch = FORMATION_PERIOD_CANDLES + (PAIR_RECALC_PERIOD_DAYS * BARS_PER_DAY * 5)
    for symbol in SYMBOLS_TO_TEST:
        isin = instrument_manager.get_isin_for_nse_symbol(symbol)
        if not isin: continue
        df = data_fetcher.fetch_data(symbol, isin, TIMEFRAME, total_candles_to_fetch)
        if df is not None and not df.empty and all(c in df.columns for c in ['open','high','low','close']):
            all_historical_data[symbol] = df

    master_close_df = pd.DataFrame({s: df['close'] for s, df in all_historical_data.items()}).dropna(axis=0, how='any')

    if len(master_close_df) <= FORMATION_PERIOD_CANDLES:
        logger.error(f"Not enough aligned historical data for backtest. Found {len(master_close_df)}, need {FORMATION_PERIOD_CANDLES}. Exiting."); return

    logger.info(f"Data aligned. Sim will run from {master_close_df.index[FORMATION_PERIOD_CANDLES]} to {master_close_df.index[-1]}")
    
    realistic_portfolio = {"open_trades": {}, "closed_trades": [], "capital": INITIAL_CAPITAL, "trade_id": 1, "signals_skipped": 0}
    theoretical_portfolio = {"open_trades": {}, "closed_trades": [], "trade_id": 1}

    strategy_cache["cointegrated_pairs"] = None
    
    for i in range(FORMATION_PERIOD_CANDLES, len(master_close_df) - 1):
        current_timestamp = master_close_df.index[i]
        
        if (i - FORMATION_PERIOD_CANDLES) % PAIR_RECALC_PERIOD_CANDLES == 0:
            formation_slice = {s: df.loc[:current_timestamp].tail(FORMATION_PERIOD_CANDLES) for s, df in all_historical_data.items()}
            strategy_cache["cointegrated_pairs"] = find_cointegrated_pairs(formation_slice, FORMATION_PERIOD_CANDLES)
            
        if not strategy_cache["cointegrated_pairs"]: continue

        pairs_in_play = set(t['pair'] for t in realistic_portfolio['open_trades'].values()) | \
                        set(t['pair'] for t in theoretical_portfolio['open_trades'].values()) | \
                        set(p['pair'] for p in strategy_cache['cointegrated_pairs'])

        for pair_tuple in pairs_in_play:
            pair_info = next((p for p in strategy_cache["cointegrated_pairs"] if p['pair'] == pair_tuple), None)
            if not pair_info: continue

            s1, s2 = pair_tuple
            lookback_slice_len = ROLLING_WINDOW + 5
            if len(all_historical_data[s1].loc[:current_timestamp]) < lookback_slice_len: continue
            
            pair_data_slice = {s1: all_historical_data[s1].loc[:current_timestamp].tail(lookback_slice_len),
                               s2: all_historical_data[s2].loc[:current_timestamp].tail(lookback_slice_len)}

            realistic_pos_state = realistic_portfolio['open_trades'].get(pair_tuple)
            theoretical_pos_state = theoretical_portfolio['open_trades'].get(pair_tuple)

            open_pos_info_real = {"direction": realistic_pos_state['direction'], "bars_held": i - realistic_pos_state['entry_candle_index']} if realistic_pos_state else None
            signal_real = generate_pair_signals(pair_data_slice, pair_info, open_pos_info_real)
            
            open_pos_info_theo = {"direction": theoretical_pos_state['direction'], "bars_held": i - theoretical_pos_state['entry_candle_index']} if theoretical_pos_state else None
            signal_theo = generate_pair_signals(pair_data_slice, pair_info, open_pos_info_theo)

            next_timestamp = master_close_df.index[i + 1]
            exec_price_s1 = all_historical_data[s1].at[next_timestamp, 'open']
            exec_price_s2 = all_historical_data[s2].at[next_timestamp, 'open']
            
            if signal_real and "EXIT" in signal_real.signal_type and realistic_pos_state:
                _close_trade(realistic_pos_state, exec_price_s1, exec_price_s2, next_timestamp, signal_real, realistic_portfolio)
            if signal_theo and "EXIT" in signal_theo.signal_type and theoretical_pos_state:
                _close_trade(theoretical_pos_state, exec_price_s1, exec_price_s2, next_timestamp, signal_theo, theoretical_portfolio)
            
            if signal_real and "ENTER" in signal_real.signal_type and not realistic_pos_state:
                if len(realistic_portfolio['open_trades']) < MAX_CONCURRENT_PAIRS:
                    _open_trade(signal_real, exec_price_s1, exec_price_s2, i, next_timestamp, realistic_portfolio, is_realistic=True)
                else:
                    realistic_portfolio['signals_skipped'] += 1
            if signal_theo and "ENTER" in signal_theo.signal_type and not theoretical_pos_state:
                 _open_trade(signal_theo, exec_price_s1, exec_price_s2, i, next_timestamp, theoretical_portfolio, is_realistic=False)

    _generate_final_report(realistic_portfolio, theoretical_portfolio)

def _open_trade(signal, s1_price, s2_price, candle_idx, timestamp, portfolio, is_realistic):
    # This function is unchanged and correct.
    s1, s2 = signal.pair
    direction = 'LONG' if 'LONG' in signal.signal_type else 'SHORT'
    alpha, beta = signal.hedge_ratio
    notional = TRADE_NOTIONAL_PER_PAIR if is_realistic else FIXED_THEORETICAL_NOTIONAL
    total_hedge_parts = 1 + abs(beta)
    notional_s1 = notional / total_hedge_parts
    notional_s2 = (notional * abs(beta)) / total_hedge_parts
    qty_s1 = int(notional_s1 / s1_price) if s1_price > 0 else 0
    qty_s2 = int(notional_s2 / s2_price) if s2_price > 0 else 0
    if qty_s1 == 0 or qty_s2 == 0:
        logger.warning(f"SKIPPING trade for {s1}/{s2} due to zero quantity. Notional may be too small.")
        return
    trade_data = {"serial_number": portfolio['trade_id'], "pair": signal.pair, "entry_timestamp": timestamp, "entry_candle_index": candle_idx, "z_score_entry": signal.z_score, "half_life": signal.half_life, "hedge_ratio": signal.hedge_ratio, "direction": direction, "s1_symbol": s1, "s2_symbol": s2, "s1_pos": 'LONG' if direction == 'LONG' else 'SHORT', "s2_pos": 'SHORT' if direction == 'LONG' else 'LONG', "s1_entry_price": s1_price, "s2_entry_price": s2_price, "s1_qty": qty_s1, "s2_qty": qty_s2}
    portfolio['open_trades'][signal.pair] = trade_data
    portfolio['trade_id'] += 1
    logger.info(f"🔥 {'REAL' if is_realistic else 'THEO'}: OPEN {direction} {s1}/{s2} @ Z={signal.z_score:.2f} | Qty:({qty_s1}, {qty_s2})")

def _close_trade(trade_data, s1_price, s2_price, timestamp, signal, portfolio):
    # This function is unchanged and correct.
    trade_data['exit_timestamp'] = timestamp
    trade_data['exit_reason'] = signal.reason
    trade_data['z_score_exit'] = signal.z_score
    pnl_s1 = (s1_price - trade_data['s1_entry_price']) * trade_data['s1_qty'] if trade_data['s1_pos'] == 'LONG' else (trade_data['s1_entry_price'] - s1_price) * trade_data['s1_qty']
    pnl_s2 = (s2_price - trade_data['s2_entry_price']) * trade_data['s2_qty'] if trade_data['s2_pos'] == 'LONG' else (trade_data['s2_entry_price'] - s2_price) * trade_data['s2_qty']
    trade_data['gross_pnl'] = pnl_s1 + pnl_s2
    notional_s1_entry = trade_data['s1_entry_price'] * trade_data['s1_qty']
    notional_s2_entry = trade_data['s2_entry_price'] * trade_data['s2_qty']
    turnover = notional_s1_entry + (s1_price * trade_data['s1_qty']) + notional_s2_entry + (s2_price * trade_data['s2_qty'])
    trade_data['transaction_costs'] = turnover * (TRANSACTION_COST_BPS / 10000)
    days_held = max(1, (trade_data['exit_timestamp'] - trade_data['entry_timestamp']).days)
    short_leg_notional = notional_s2_entry if trade_data['s2_pos'] == 'SHORT' else notional_s1_entry
    trade_data['borrow_costs'] = (short_leg_notional * (ANNUAL_BORROW_COST_PERCENT / 100) / 365) * days_held
    trade_data['net_pnl'] = trade_data['gross_pnl'] - trade_data['transaction_costs'] - trade_data['borrow_costs']
    trade_data['days_held'] = days_held
    if "capital" in portfolio: portfolio['capital'] += trade_data['net_pnl']
    portfolio['closed_trades'].append(trade_data)
    del portfolio['open_trades'][trade_data['pair']]
    logger.info(f"✅ {'REAL' if 'capital' in portfolio else 'THEO'}: CLOSE {trade_data['s1_symbol']}/{trade_data['s2_symbol']} | Net PnL: {trade_data['net_pnl']:.2f}")

def _generate_final_report(realistic, theoretical):
    """
    Prints the final dual-summary report with added 'Mechanical vs Net' analysis,
    while preserving the original structure.
    """
    logger.info("\n--- Simulation Complete ---\n")
    
    output_headers = [
        'serial_number', 'pair', 'direction', 'entry_timestamp', 'exit_timestamp', 'days_held',
        'z_score_entry', 'z_score_exit', 'exit_reason', 's1_symbol', 's1_pos', 's1_entry_price',
        's1_qty', 's2_symbol', 's2_pos', 's2_entry_price', 's2_qty', 'gross_pnl',
        'transaction_costs', 'borrow_costs', 'net_pnl', 'hedge_ratio', 'half_life'
    ]
    if realistic['closed_trades']:
        trades_to_write = [t.copy() for t in realistic['closed_trades']]
        for trade in trades_to_write:
            trade['pair'] = '_'.join(trade['pair'])
        with open(OUTPUT_FILE_NAME, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(trades_to_write)
        logger.info(f"Detailed trade log for realistic portfolio saved to '{OUTPUT_FILE_NAME}'")

    def print_summary(portfolio_data, name):
        trades = portfolio_data['closed_trades']
        total_trades = len(trades)
        
        print("\n" + "="*60)
        print(f"       BACKTEST SUMMARY: {name.upper()}".center(60))
        print("="*60)

        if "capital" in portfolio_data:
            net_pnl = portfolio_data['capital'] - INITIAL_CAPITAL
            pnl_percent = (net_pnl / INITIAL_CAPITAL) * 100
            print("--- Portfolio Performance ---")
            print(f"  Initial Capital:        ₹{INITIAL_CAPITAL:,.2f}")
            print(f"  Final Capital:          ₹{portfolio_data['capital']:,.2f}")
            print(f"  Net PnL:                ₹{net_pnl:,.2f} ({pnl_percent:+.2f}%)")
            print("\n--- Trade Execution & Outcomes ---")
            print(f"  Total Trades Executed:  {total_trades}")
            print(f"  Signals Skipped:        {portfolio_data.get('signals_skipped', 0)}")
        else:
             print("--- Signal Quality ---")
             print(f"  Total Signals Generated: {total_trades}")

        if total_trades > 0:
            # --- CALCULATIONS (Original and New) ---
            wins = [t for t in trades if t['net_pnl'] > 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            
            tp_hits = len([t for t in trades if "PROFIT TARGET" in t['exit_reason']])
            sl_hits = len([t for t in trades if "STATISTICAL STOP" in t['exit_reason']])
            time_hits = len([t for t in trades if "TIME STOP" in t['exit_reason']])
            
            # --- NEW: Calculation for Mechanical Win Rate ---
            total_resolved = tp_hits + sl_hits
            mechanical_win_rate = (tp_hits / total_resolved) * 100 if total_resolved > 0 else 0
            
            avg_pnl = np.mean([t['net_pnl'] for t in trades]) if trades else 0
            
            # --- PRINTING (Preserving original structure with additions) ---
            print(f"\n  Win Rate (Net PnL > 0): {win_rate:.2f}% ({len(wins)} Trades)")
            print(f"  Loss Rate (Net PnL <= 0): {100-win_rate:.2f}% ({total_trades - len(wins)} Trades)")
            print(f"  Average Net PnL/Trade:  ₹{avg_pnl:,.2f}")
            
            # --- NEW: Added Mechanical Win Rate display for direct comparison ---
            print(f"  Mechanical Win Rate:      {mechanical_win_rate:.2f}% (Based on TP vs SL hits)")

            print("\n  Breakdown by Exit Reason:")
            print(f"    - Profit Target Hits: \033[92m{tp_hits}\033[0m")
            print(f"    - Stop Loss Hits:     \033[91m{sl_hits}\033[0m")
            print(f"    - Time Stop Hits:     \033[93m{time_hits}\033[0m")
        else:
            print("\n  No trades were closed during the simulation.")
        print("="*60)
        
    print_summary(realistic, "Realistic Portfolio")
    print_summary(theoretical, "Theoretical Signals")

if __name__ == "__main__":
    run_trade_simulator()