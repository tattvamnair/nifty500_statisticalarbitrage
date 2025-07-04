# stat_arb_trader_dhan/data_feeds/data_fetcher.py

import pandas as pd
import time
from datetime import datetime, timedelta
import os
from core.logger_setup import logger
from config import settings
from data_feeds.instrument_manager import InstrumentManager

class DataFetcher:
    """
    Handles fetching historical market data from DhanHQ, with a
    persistent local cache to speed up subsequent runs. This version has been
    made more robust to build deep historical data for backtesting.
    """
    def __init__(self, dhan_api_sdk, instrument_manager: InstrumentManager):
        # This function is unchanged.
        self.dhanhq_sdk = dhan_api_sdk
        self.instrument_mgr = instrument_manager
        self.instrument_master_df = None
        self.cache_dir = "data_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created data cache directory at: ./{self.cache_dir}")
        self._fetch_and_cache_instrument_master()

    def _fetch_and_cache_instrument_master(self):
        # This function is unchanged.
        try:
            url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
            dtype_spec = {'SECURITY_ID': str, 'ISIN': str}
            self.instrument_master_df = pd.read_csv(url, low_memory=False, dtype=dtype_spec)
        except Exception as e:
            logger.error(f"Failed to fetch or process Dhan instrument master CSV: {e}", exc_info=True)

    def get_dhan_details_by_isin(self, isin: str):
        # This function is unchanged.
        if self.instrument_master_df is None:
            return None
        match = self.instrument_master_df[(self.instrument_master_df['ISIN'] == isin) & (self.instrument_master_df['EXCH_ID'] == 'NSE') & (self.instrument_master_df['INSTRUMENT'] == 'EQUITY')]
        if not match.empty:
            return match.iloc[0]['SECURITY_ID']
        return None

    def fetch_data(self, symbol: str, isin: str, timeframe: str, num_candles: int):
        """
        REWRITTEN: This is the new, robust, cache-aware fetch method.
        It can now build a deep historical database by fetching data in chunks.
        """
        security_id = self.get_dhan_details_by_isin(isin)
        if not security_id:
            return pd.DataFrame()

        safe_symbol = symbol.replace('&', '_').replace('-', '_')
        cache_path = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.parquet")
        
        # Step 1: Load existing data from cache if it exists.
        cached_df = pd.DataFrame()
        if os.path.exists(cache_path):
            try:
                cached_df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {len(cached_df)} candles for {symbol} from local cache.")
            except Exception as e:
                logger.error(f"Could not read cache file for {symbol}. Will refetch. Error: {e}")

        # Step 2: Fetch any new data since the last cache update (for live bot).
        # And fetch historical data if cache is insufficient (for backtester).
        now = datetime.now()
        
        # --- Fetch NEW data ---
        # If cache exists, get all new data from the last timestamp to now.
        from_date_new = cached_df.index[-1].to_pydatetime() if not cached_df.empty else now - timedelta(days=89)
        new_data_df = self._fetch_from_dhan_api(symbol, security_id, timeframe, from_date_new.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        
        # Combine new data with cache
        if not new_data_df.empty:
            cached_df = pd.concat([cached_df, new_data_df])
            cached_df = cached_df[~cached_df.index.duplicated(keep='last')]
            cached_df.sort_index(inplace=True)

        # Step 3: Check if we have enough data for the backtester. If not, fetch it.
        if len(cached_df) < num_candles:
            logger.info(f"Insufficient history for {symbol}. Required: {num_candles}, Have: {len(cached_df)}. Building cache...")
            
            earliest_date_in_cache = cached_df.index[0] if not cached_df.empty else now
            
            while len(cached_df) < num_candles:
                to_date_historical = earliest_date_in_cache - timedelta(days=1)
                from_date_historical = to_date_historical - timedelta(days=89) # Fetch in 90-day chunks
                
                historical_chunk = self._fetch_from_dhan_api(symbol, security_id, timeframe, from_date_historical.strftime("%Y-%m-%d"), to_date_historical.strftime("%Y-%m-%d"))
                
                if historical_chunk.empty:
                    logger.warning(f"No more historical data available for {symbol} before {to_date_historical.date()}. Stopping history build.")
                    break
                
                cached_df = pd.concat([historical_chunk, cached_df])
                cached_df = cached_df[~cached_df.index.duplicated(keep='last')]
                cached_df.sort_index(inplace=True)
                
                earliest_date_in_cache = cached_df.index[0]
                logger.info(f"Fetched historical chunk for {symbol}. Total candles now: {len(cached_df)}.")

        # Step 4: Save the potentially updated/built dataframe to cache and return
        if not cached_df.empty:
            try:
                cached_df.to_parquet(cache_path)
                logger.debug(f"Cache for {symbol} updated/saved. Total candles: {len(cached_df)}.")
            except Exception as e:
                logger.error(f"Failed to write to cache for {symbol}: {e}")
            
            # Finally, return the requested number of candles from the end of our complete DataFrame
            return self.calculate_indicators(cached_df.tail(num_candles).copy(), symbol)

        return pd.DataFrame()


    def _fetch_from_dhan_api(self, symbol, security_id, timeframe, from_date, to_date):
        # This function's internal logic is preserved exactly as it was.
        df = pd.DataFrame()
        timeframe_upper = timeframe.upper()
        is_intraday = timeframe_upper not in ['D', '1D', 'W', '1W']

        try:
            time.sleep(0.3)
            response = None
            if not is_intraday:
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
                remarks = response.get('remarks', 'N/A') if isinstance(response, dict) else "Invalid response"
                logger.error(f"API call for historical data for {symbol} failed. Remarks: {remarks}")
        except Exception as e:
            logger.error(f"Exception during data fetch for {symbol}: {e}", exc_info=True)
            
        return pd.DataFrame()

    def get_live_quotes(self, symbol_isin_list: list):
        # This function is unchanged.
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
        # This function is unchanged.
        if df.empty: return df
        try:
            import pandas_ta as ta
            df['EMA_5'] = ta.ema(df['close'], length=5)
            df['EMA_10'] = ta.ema(df['close'], length=10)
            if 'volume' in df.columns:
                df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            df['EMA_9'] = ta.ema(df['close'], length=9)
            df['EMA_15'] = ta.ema(df['close'], length=15)
            df['EMA_200'] = ta.ema(df['close'], length=200)
            df['RSI_14'] = ta.rsi(df['close'], length=14)
            df['SMA_50'] = ta.sma(df['close'], length=50)
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_data is not None and not adx_data.empty:
                df['ADX_14'] = adx_data['ADX_14']
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
        return df