# stat_arb_trader_dhan/data_feeds/data_fetcher.py

import pandas as pd
import time
from collections import defaultdict
from core.logger_setup import logger
from config import settings
from data_feeds.instrument_manager import InstrumentManager

class DataFetcher:
    """
    Handles fetching historical and live market data from DhanHQ.
    """
    def __init__(self, dhan_api_sdk, instrument_manager: InstrumentManager):
        self.dhanhq_sdk = dhan_api_sdk
        self.instrument_mgr = instrument_manager
        self.instrument_master_df = None
        self._fetch_and_cache_instrument_master()

    def _fetch_and_cache_instrument_master(self):
        try:
            url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
            dtype_spec = {'SECURITY_ID': str, 'ISIN': str}
            self.instrument_master_df = pd.read_csv(url, low_memory=False, dtype=dtype_spec)
            logger.info(f"Successfully fetched and cached Dhan instrument master. Shape: {self.instrument_master_df.shape}")
        except Exception as e:
            logger.error(f"Failed to fetch or process Dhan instrument master CSV: {e}", exc_info=True)
            self.instrument_master_df = None

    def get_dhan_details_by_isin(self, isin: str):
        if self.instrument_master_df is None:
            logger.error("Instrument master not loaded. Cannot look up details by ISIN.")
            return None
        match = self.instrument_master_df[
            (self.instrument_master_df['ISIN'] == isin) &
            (self.instrument_master_df['EXCH_ID'] == 'NSE') &
            (self.instrument_master_df['INSTRUMENT'] == 'EQUITY') &
            (self.instrument_master_df['SERIES'] == 'EQ')
        ]
        if match.empty:
            logger.debug(f"ISIN {isin} not found with strict 'EQ' series filter. Broadening search...")
            match = self.instrument_master_df[
                (self.instrument_master_df['ISIN'] == isin) &
                (self.instrument_master_df['EXCH_ID'] == 'NSE') &
                (self.instrument_master_df['INSTRUMENT'] == 'EQUITY')
            ]
        if not match.empty:
            if len(match) > 1:
                logger.warning(f"Found {len(match)} entries for ISIN '{isin}'. Using first one.")
            return match.iloc[0]['SECURITY_ID']
        logger.warning(f"No Dhan Security ID found for ISIN '{isin}' in NSE Equity segment after all attempts.")
        return None

    def get_bulk_historical_data(self, symbol_isin_list: list, from_date: str, to_date: str):
        all_data = {}
        for symbol, isin in symbol_isin_list:
            security_id = self.get_dhan_details_by_isin(isin)
            if not security_id:
                logger.warning(f"Skipping historical data for {symbol}: Could not resolve security ID from ISIN {isin}.")
                time.sleep(0.21)
                continue
            try:
                response = self.dhanhq_sdk.historical_daily_data(str(security_id), settings.DHAN_SEGMENT_NSE_EQ, settings.DHAN_INSTRUMENT_EQUITY, from_date, to_date)
                if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                    data = response.get('data', {})
                    if data and data.get('timestamp'):
                        df = pd.DataFrame(data)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df.set_index('timestamp', inplace=True)
                        all_data[symbol] = df
                        logger.info(f"Fetched {len(df)} historical records for {symbol} (SecID: {security_id}).")
                    else:
                        logger.warning(f"Historical data success, but no data payload for {symbol} (SecID: {security_id}).")
                else:
                    logger.error(f"Failed to fetch historical data for {symbol} (SecID: {security_id}). Remarks: {response.get('remarks')}")
            except Exception as e:
                logger.error(f"Exception fetching historical data for {symbol}: {e}", exc_info=True)
            time.sleep(0.21)
        return all_data

    def get_live_quotes(self, symbol_isin_list: list):
        if not self.dhanhq_sdk:
            logger.error("Cannot fetch live quotes: Dhan SDK client not initialized.")
            return {}
        payload = defaultdict(list)
        security_id_to_symbol_map = {}
        for symbol, isin in symbol_isin_list:
            security_id = self.get_dhan_details_by_isin(isin)
            if security_id:
                payload[settings.DHAN_SEGMENT_NSE_EQ].append(security_id)
                security_id_to_symbol_map[security_id] = symbol
            else:
                logger.warning(f"Skipping live quote for {symbol}: Could not resolve its Dhan Security ID.")
        if not payload:
            logger.error("Could not resolve any instruments to fetch live quotes for.")
            return {}
        try:
            logger.debug(f"Sending live quote request payload: {dict(payload)}")
            # --- THE CORRECTED LINE ---
            response = self.dhanhq_sdk.quote_data(dict(payload))
            # --- END OF CORRECTION ---
            if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                data = response.get('data', {})
                live_quotes_by_symbol = {}
                nse_eq_data = data.get(settings.DHAN_SEGMENT_NSE_EQ, {})
                for sec_id, quote_details in nse_eq_data.items():
                    symbol = security_id_to_symbol_map.get(str(sec_id)) # Match as string
                    if symbol:
                        live_quotes_by_symbol[symbol] = quote_details
                logger.info(f"Successfully fetched live quotes for {len(live_quotes_by_symbol)} symbols.")
                return live_quotes_by_symbol
            else:
                logger.error(f"Live quote API call failed. Status: {response.get('status', 'N/A')}, Remarks: {response.get('remarks', 'N/A')}")
                return {}
        except Exception as e:
            logger.error(f"Exception during live quote fetch: {e}", exc_info=True)
            return {}

    def calculate_indicators(self, historical_data: dict):
        logger.info("Calculating indicators on historical data...")
        for symbol, df in historical_data.items():
            if not df.empty:
                try:
                    import pandas_ta as ta
                    df['SMA_20'] = ta.sma(df['close'], length=20)
                    df['RSI_14'] = ta.rsi(df['close'], length=14)
                    historical_data[symbol] = df
                    logger.debug(f"Calculated indicators for {symbol}. Last SMA: {df['SMA_20'].iloc[-1]:.2f}, Last RSI: {df['RSI_14'].iloc[-1]:.2f}")
                except ImportError:
                    logger.warning("`pandas-ta` is not installed. Skipping indicator calculation. Install with `pip install pandas-ta`")
                    break
                except Exception as e:
                    logger.error(f"Error calculating indicators for {symbol}: {e}")
        return historical_data