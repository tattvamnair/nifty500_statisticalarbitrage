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
            logger.info(f"Successfully fetched and cached Dhan instrument master. Shape: {self.instrument_master_df.shape}")
        except Exception as e:
            logger.error(f"Failed to fetch or process Dhan instrument master CSV: {e}", exc_info=True)

    def get_dhan_details_by_isin(self, isin: str):
        if self.instrument_master_df is None:
            return None
        match = self.instrument_master_df[(self.instrument_master_df['ISIN'] == isin) & (self.instrument_master_df['EXCH_ID'] == 'NSE') & (self.instrument_master_df['INSTRUMENT'] == 'EQUITY')]
        if not match.empty:
            return match.iloc[0]['SECURITY_ID']
        logger.warning(f"No Dhan Security ID found for ISIN '{isin}' in NSE Equity segment.")
        return None

    def fetch_data(self, symbol: str, isin: str, timeframe: str, num_candles: int):
        security_id = self.get_dhan_details_by_isin(isin)
        if not security_id:
            logger.error(f"Cannot fetch data for {symbol}: Could not resolve Security ID.")
            return pd.DataFrame()

        SUPPORTED_INTRADAY = ['1', '5', '15', '60']
        to_date = datetime.now()
        df = pd.DataFrame()

        try:
            time.sleep(0.3)
            
            if timeframe in SUPPORTED_INTRADAY:
                from_date = to_date - timedelta(days=89)
                response = self.dhanhq_sdk.intraday_minute_data(
                    security_id=str(security_id),
                    exchange_segment=settings.DHAN_SEGMENT_NSE_EQ,
                    instrument_type=settings.DHAN_INSTRUMENT_EQUITY,
                    from_date=from_date.strftime("%Y-%m-%d"),
                    to_date=to_date.strftime("%Y-%m-%d")
                )
            else: # Daily or Weekly
                days_to_fetch = int(num_candles * 1.8) if timeframe == 'D' else int(num_candles * 7 * 1.8)
                from_date = to_date - timedelta(days=days_to_fetch)
                # --- THIS IS THE CORRECTED CALL ---
                response = self.dhanhq_sdk.historical_daily_data(
                    security_id=str(security_id),
                    exchange_segment=settings.DHAN_SEGMENT_NSE_EQ,
                    instrument_type=settings.DHAN_INSTRUMENT_EQUITY,
                    from_date=from_date.strftime("%Y-%m-%d"),
                    to_date=to_date.strftime("%Y-%m-%d")
                )
                # --- END OF CORRECTION ---

            if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                data = response.get('data', {})
                if data and data.get('timestamp'):
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    if timeframe == 'W':
                        ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                        df = df.resample('W-FRI').agg(ohlc_dict).dropna()
                    df = df.tail(num_candles).copy()
                    return self.calculate_indicators(df, symbol)
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
        if df.empty: return df
        try:
            import pandas_ta as ta
            df['EMA_9'] = ta.ema(df['close'], length=9)
            df['EMA_15'] = ta.ema(df['close'], length=15)
            df['EMA_65'] = ta.ema(df['close'], length=65)
            df['EMA_100'] = ta.ema(df['close'], length=100)
            df['EMA_200'] = ta.ema(df['close'], length=200)
            df['RSI_14'] = ta.rsi(df['close'], length=14)
            # --- THIS IS THE CORRECTED LINE ---
            df['VMA_20'] = ta.sma(df['volume'], length=20) # Calculate 20-period Volume Moving Average
            df['SMA_200'] = ta.sma(df['close'], length=200)
            # --- END OF CORRECTION ---
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
        return df