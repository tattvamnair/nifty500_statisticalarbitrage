# stat_arb_trader_dhan/data_feeds/data_fetcher.py

import pandas as pd
import time
import requests
import json
from collections import defaultdict
from datetime import datetime, timedelta
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

    def get_dhan_details_by_isin(self, isin: str):
        if self.instrument_master_df is None:
            return None
        match = self.instrument_master_df[(self.instrument_master_df['ISIN'] == isin) & (self.instrument_master_df['EXCH_ID'] == 'NSE') & (self.instrument_master_df['INSTRUMENT'] == 'EQUITY')]
        if not match.empty:
            return match.iloc[0]['SECURITY_ID']
        return None

    def fetch_data(self, symbol: str, isin: str, timeframe: str, num_candles: int):
        logger.info(f"Data request for {symbol}: {num_candles} candles of timeframe '{timeframe}'.")
        security_id = self.get_dhan_details_by_isin(isin)
        if not security_id:
            logger.error(f"Cannot fetch data for {symbol}: Could not resolve Security ID.")
            return pd.DataFrame()

        SUPPORTED_INTRADAY = ['1', '5', '15', '60']
        to_date = datetime.now()
        
        try:
            if timeframe in SUPPORTED_INTRADAY:
                from_date = to_date - timedelta(days=89)
                response = self.dhanhq_sdk.intraday_minute_data(security_id, settings.DHAN_SEGMENT_NSE_EQ, settings.DHAN_INSTRUMENT_EQUITY, timeframe, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))
            else: # Daily or Weekly
                days_to_fetch = int(num_candles * 1.8) if timeframe == 'D' else int(num_candles * 7 * 1.8)
                from_date = to_date - timedelta(days=days_to_fetch)
                response = self.dhanhq_sdk.historical_daily_data(security_id, settings.DHAN_SEGMENT_NSE_EQ, settings.DHAN_INSTRUMENT_EQUITY, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))

            if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                data = response.get('data', {})
                if data and data.get('timestamp'):
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    if timeframe == 'W':
                        df = df.resample('W-FRI').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
                    df = df.tail(num_candles).copy()
                    return self.calculate_indicators(df, symbol)
        except Exception as e:
            logger.error(f"Exception in fetch_data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

    def get_live_quotes(self, symbol_isin_list: list):
        """
        Fetches live quotes directly using the requests library for maximum reliability.
        """
        if not self.dhanhq_sdk:
            logger.error("Cannot fetch live quotes: Dhan SDK client not initialized.")
            return {}

        security_id_to_symbol_map = {}
        payload = defaultdict(list)
        for symbol, isin in symbol_isin_list:
            security_id = self.get_dhan_details_by_isin(isin)
            if security_id:
                payload[settings.DHAN_SEGMENT_NSE_EQ].append(security_id)
                security_id_to_symbol_map[security_id] = symbol
        
        if not payload:
            logger.error("Could not resolve any instruments for live quotes.")
            return {}

        try:
            api_url = "https://api.dhan.co/v2/marketfeed/quote"
            headers = {
                'access-token': self.dhanhq_sdk.dhan_http.access_token,
                'client-id': self.dhanhq_sdk.dhan_http.client_id,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            logger.debug(f"Sending DIRECT live quote request to {api_url}...")
            response = requests.post(api_url, headers=headers, json=dict(payload), timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            response_json = response.json()
            if response_json.get('status', '').lower() == 'success':
                data = response_json.get('data', {})
                live_quotes_by_symbol = {}
                nse_eq_data = data.get(settings.DHAN_SEGMENT_NSE_EQ, {})
                for sec_id, quote_details in nse_eq_data.items():
                    symbol = security_id_to_symbol_map.get(str(sec_id))
                    if symbol:
                        live_quotes_by_symbol[symbol] = quote_details
                return live_quotes_by_symbol
            else:
                logger.error(f"Live quote API returned failure status: {response_json}")
                return {}
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTPError during direct live quote call: {http_err}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from live quote response. Raw text: {response.text}")
        except Exception as e:
            logger.error(f"Exception during direct live quote fetch: {e}", exc_info=True)
        return {}

    def calculate_indicators(self, df: pd.DataFrame, symbol: str):
        if df.empty: return df
        try:
            import pandas_ta as ta
            df['EMA_9'] = ta.ema(df['close'], length=9)
            df['EMA_15'] = ta.ema(df['close'], length=15)
            df['EMA_65'] = ta.ema(df['close'], length=65)
            df['EMA_100'] = ta.ema(df['close'], length=100)
            df['EMA_200'] = ta.ema(df['close'], length=200)
            df['RSI_14'] = ta.rsi(df['close'], length=14)
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
        return df