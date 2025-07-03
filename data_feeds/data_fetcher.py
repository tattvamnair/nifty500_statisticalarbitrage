# stat_arb_trader_dhan/data_feeds/data_fetcher.py

import pandas as pd
import time
from datetime import datetime, timedelta
import os # ADDED: For file path operations
from core.logger_setup import logger
from config import settings
from data_feeds.instrument_manager import InstrumentManager

class DataFetcher:
    """
    Handles fetching historical market data from DhanHQ, now with a
    persistent local cache to speed up subsequent runs.
    """
    def __init__(self, dhan_api_sdk, instrument_manager: InstrumentManager):
        self.dhanhq_sdk = dhan_api_sdk
        self.instrument_mgr = instrument_manager
        self.instrument_master_df = None
        ## ADDED: Setup the cache directory on initialization ##
        self.cache_dir = "data_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created data cache directory at: ./{self.cache_dir}")
        self._fetch_and_cache_instrument_master()

    def _fetch_and_cache_instrument_master(self):
        # UNCHANGED: This function is preserved exactly as it was.
        try:
            url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
            dtype_spec = {'SECURITY_ID': str, 'ISIN': str}
            self.instrument_master_df = pd.read_csv(url, low_memory=False, dtype=dtype_spec)
        except Exception as e:
            logger.error(f"Failed to fetch or process Dhan instrument master CSV: {e}", exc_info=True)

    def get_dhan_details_by_isin(self, isin: str):
        # UNCHANGED: This function is preserved exactly as it was.
        if self.instrument_master_df is None:
            return None
        match = self.instrument_master_df[(self.instrument_master_df['ISIN'] == isin) & (self.instrument_master_df['EXCH_ID'] == 'NSE') & (self.instrument_master_df['INSTRUMENT'] == 'EQUITY')]
        if not match.empty:
            return match.iloc[0]['SECURITY_ID']
        return None

    ## REWRITTEN: This is the new cache-aware fetch_data method. ##
    def fetch_data(self, symbol: str, isin: str, timeframe: str, num_candles: int):
        security_id = self.get_dhan_details_by_isin(isin)
        if not security_id:
            return pd.DataFrame()

        # Sanitize symbol for filename and create unique cache path per timeframe
        safe_symbol = symbol.replace('&', '_').replace('-', '_')
        cache_path = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.parquet")
        cached_df = None
        
        # Step 1: Check for and load existing cache
        if os.path.exists(cache_path):
            try:
                cached_df = pd.read_parquet(cache_path)
                logger.info(f"Loaded {len(cached_df)} candles for {symbol} from local cache.")
            except Exception as e:
                logger.error(f"Could not read cache file for {symbol} at {cache_path}. Will fetch full history. Error: {e}")
                cached_df = None

        # Step 2: Determine the date range for the API call
        to_date = datetime.now()
        if cached_df is not None and not cached_df.empty:
            # Incremental fetch: start from the last timestamp in our cache
            from_date = cached_df.index[-1].to_pydatetime()
        else:
            # Full fetch: get enough data to satisfy num_candles
            timeframe_upper = timeframe.upper()
            is_daily = timeframe_upper in ['D', '1D', 'W', '1W']
            days_to_fetch = int(num_candles * (2.0 if is_daily else 1.5))
            if not is_daily:
                 # For intraday, Dhan API limit is 90 days, so respect that
                 days_to_fetch = min(days_to_fetch, 89)
            from_date = to_date - timedelta(days=days_to_fetch)
        
        from_date_str = from_date.strftime("%Y-%m-%d")
        to_date_str = to_date.strftime("%Y-%m-%d")

        # Step 3: Call the API (using a helper to keep this function clean)
        new_df = self._fetch_from_dhan_api(symbol, security_id, timeframe, from_date_str, to_date_str)
        
        # If no new data was fetched, we can just use the cache
        if new_df.empty and cached_df is not None:
            logger.info(f"No new candles found for {symbol}. Using cached data.")
            return self.calculate_indicators(cached_df.tail(num_candles).copy(), symbol)
        
        # Step 4: Combine, clean, and update the cache
        if cached_df is not None and not cached_df.empty:
            combined_df = pd.concat([cached_df, new_df])
            # This is critical: remove duplicates, keeping the latest data (handles overlaps)
            updated_df = combined_df[~combined_df.index.duplicated(keep='last')]
        else:
            updated_df = new_df

        # Step 5: Save to cache and return the final dataframe
        if not updated_df.empty:
            try:
                updated_df.to_parquet(cache_path)
                logger.info(f"Cache for {symbol} updated. Total candles: {len(updated_df)}.")
            except Exception as e:
                logger.error(f"Failed to write to cache for {symbol}: {e}")
            
            return self.calculate_indicators(updated_df.tail(num_candles).copy(), symbol)

        return pd.DataFrame()

    def _fetch_from_dhan_api(self, symbol, security_id, timeframe, from_date, to_date):
        """
        Helper function to contain the actual API call and data processing logic.
        This isolates the caching logic from the API interaction logic.
        """
        df = pd.DataFrame()
        timeframe_upper = timeframe.upper()
        is_intraday = timeframe_upper not in ['D', '1D', 'W', '1W']

        try:
            time.sleep(0.3)
            response = None

            if not is_intraday:
                days_to_fetch = int(252 * (2.0 if 'D' in timeframe_upper else 8.0)) # A safe default
                from_date_obj = datetime.strptime(from_date, "%Y-%m-%d")
                to_date_obj = datetime.strptime(to_date, "%Y-%m-%d")
                
                # Check if we need to adjust from_date for daily calls
                if (to_date_obj - from_date_obj).days < days_to_fetch:
                     from_date = (to_date_obj - timedelta(days=days_to_fetch)).strftime("%Y-%m-%d")

                response = self.dhanhq_sdk.historical_daily_data(
                    security_id=str(security_id), exchange_segment=settings.DHAN_SEGMENT_NSE_EQ,
                    instrument_type=settings.DHAN_INSTRUMENT_EQUITY, from_date=from_date, to_date=to_date
                )
            else:
                response = self.dhanhq_sdk.intraday_minute_data(
                    security_id=str(security_id), exchange_segment=settings.DHAN_SEGMENT_NSE_EQ,
                    instrument_type=settings.DHAN_INSTRUMENT_EQUITY, from_date=from_date, to_date=to_date
                )
            
            if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                data = response.get('data', {})
                if not data: return pd.DataFrame()

                df = pd.DataFrame(data)
                if df.empty: return pd.DataFrame()

                df.rename(columns={'start_Time': 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                # Resampling logic - preserved from your original code
                if 'W' in timeframe_upper:
                    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    df = df.resample('W-FRI').agg(ohlc_dict).dropna()
                elif is_intraday:
                    resample_map = {'1': '1min', '1M': '1min', '5': '5min', '5M': '5min', '15': '15min', '15M': '15min', '30': '30min', '30M': '30min', '60': '60min', '1H': '60min', '240': '240min', '4H': '240min'}
                    resample_rule = resample_map.get(timeframe)
                    if resample_rule and resample_rule != '1min':
                        ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                        df = df.resample(resample_rule).agg(ohlc_dict).dropna()

                return df
            else:
                logger.error(f"API call for historical data for {symbol} failed. Remarks: {response.get('remarks')}")
        except Exception as e:
            logger.error(f"Exception during data fetch for {symbol}: {e}", exc_info=True)
            
        return pd.DataFrame()

    def get_live_quotes(self, symbol_isin_list: list):
        # UNCHANGED: This function is preserved exactly as it was.
        if not self.dhanhq_sdk: return {}
        live_quotes_by_symbol = {}
        for symbol, isin in symbol_isin_list:
            security_id = self.get_dhan_details_by_isin(isin)
            if not security_id: continue
            try:
                payload = {settings.DHAN_SEGMENT_NSE_EQ: [security_id]}
                response = self.dhanhq_sdk.quote_data(securities=payload)
                if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                    data = response.get('data', {}).get(settings.DHAN_SEGMENT_NSE_EQ, {})
                    if data and security_id in data:
                        live_quotes_by_symbol[symbol] = data[security_id]
                else:
                    logger.error(f"Live quote API call failed for {symbol}. Remarks: {response.get('remarks', 'N/A')}")
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Exception during live quote fetch for {symbol}: {e}", exc_info=True)
        if live_quotes_by_symbol:
            logger.info(f"Fetched live quotes for {len(live_quotes_by_symbol)}/{len(symbol_isin_list)} symbols.")
        return live_quotes_by_symbol

    def calculate_indicators(self, df: pd.DataFrame, symbol: str):
        """
        Calculates all indicators needed for both intraday and daily strategies
        to ensure the adaptive strategy logic has the data it needs.
        """
        # UNCHANGED: This function is preserved exactly as it was.
        if df.empty: return df
        try:
            import pandas_ta as ta
            
            # --- CALCULATE ALL INDICATORS ---
            # Intraday-focused indicators
            df['EMA_5'] = ta.ema(df['close'], length=5)
            df['EMA_10'] = ta.ema(df['close'], length=10)
            if 'volume' in df.columns:
                df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

            # Daily/Positional-focused indicators
            df['EMA_9'] = ta.ema(df['close'], length=9)
            df['EMA_15'] = ta.ema(df['close'], length=15)
            
            # Indicators used by both strategies
            df['EMA_200'] = ta.ema(df['close'], length=200)
            df['RSI_14'] = ta.rsi(df['close'], length=14)
            df['SMA_50'] = ta.sma(df['close'], length=50)
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_data is not None and not adx_data.empty:
                df['ADX_14'] = adx_data['ADX_14']

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
        return df