# stat_arb_trader_dhan/data_feeds/data_fetcher.py

import pandas as pd
import time
from datetime import datetime, timedelta
from core.logger_setup import logger
from config import settings
from data_feeds.instrument_manager import InstrumentManager

class DataFetcher:
    """
    Handles fetching historical market data from DhanHQ.
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
        security_id = self.get_dhan_details_by_isin(isin)
        if not security_id:
            return pd.DataFrame()

        to_date = datetime.now()
        df = pd.DataFrame()
        timeframe = timeframe.upper()
        is_intraday = timeframe not in ['D', '1D', 'W', '1W']

        try:
            time.sleep(0.3)
            response = None

            if not is_intraday:
                days_to_fetch = int(num_candles * (2.0 if 'D' in timeframe else 8.0))
                from_date = to_date - timedelta(days=days_to_fetch)
                response = self.dhanhq_sdk.historical_daily_data(
                    security_id=str(security_id), exchange_segment=settings.DHAN_SEGMENT_NSE_EQ,
                    instrument_type=settings.DHAN_INSTRUMENT_EQUITY, from_date=from_date.strftime("%Y-%m-%d"), to_date=to_date.strftime("%Y-%m-%d")
                )
            else:
                from_date = to_date - timedelta(days=89)
                response = self.dhanhq_sdk.intraday_minute_data(
                    security_id=str(security_id), exchange_segment=settings.DHAN_SEGMENT_NSE_EQ,
                    instrument_type=settings.DHAN_INSTRUMENT_EQUITY, from_date=from_date.strftime("%Y-%m-%d"), to_date=to_date.strftime("%Y-%m-%d")
                )
            
            if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                data = response.get('data', {})
                df = pd.DataFrame(data)
                if df.empty: return pd.DataFrame()

                if 'start_Time' in df.columns:
                    df.rename(columns={'start_Time': 'timestamp'}, inplace=True)
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                if 'W' in timeframe:
                    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    df = df.resample('W-FRI').agg(ohlc_dict).dropna()
                elif is_intraday:
                    resample_map = {'1': '1min', '1M': '1min', '5': '5min', '5M': '5min', '15': '15min', '15M': '15min', '30': '30min', '30M': '30min', '60': '60min', '1H': '60min', '240': '240min', '4H': '240min'}
                    resample_rule = resample_map.get(timeframe)
                    if resample_rule and resample_rule != '1min':
                        ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                        df = df.resample(resample_rule).agg(ohlc_dict).dropna()

                if df.empty: return pd.DataFrame()
                
                return self.calculate_indicators(df.tail(num_candles).copy(), symbol)
            else:
                logger.error(f"API call for historical data for {symbol} failed. Remarks: {response.get('remarks')}")
        except Exception as e:
            logger.error(f"Exception during data fetch for {symbol}: {e}", exc_info=True)
            
        return pd.DataFrame()

    def get_live_quotes(self, symbol_isin_list: list):
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