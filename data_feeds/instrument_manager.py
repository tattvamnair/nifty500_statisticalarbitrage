# stat_arb_trader_dhan/data_feeds/instrument_manager.py

import pandas as pd
from core.logger_setup import logger

# This list now contains the full, accurate list of Nifty 50 stocks with corrected ISINs.
NIFTY50_INTERNAL_LIST = [
    {"CompanyName": "Reliance Industries", "NSESymbol": "RELIANCE", "Sector": "Oil Gas & Consumer Goods", "ISIN": "INE002A01018"},
    {"CompanyName": "Tata Consultancy Services", "NSESymbol": "TCS", "Sector": "Information Technology", "ISIN": "INE467B01029"},
    {"CompanyName": "HDFC Bank", "NSESymbol": "HDFCBANK", "Sector": "Financial Services", "ISIN": "INE040A01034"},
    {"CompanyName": "Infosys", "NSESymbol": "INFY", "Sector": "Information Technology", "ISIN": "INE009A01021"},
    {"CompanyName": "ICICI Bank", "NSESymbol": "ICICIBANK", "Sector": "Financial Services", "ISIN": "INE090A01021"},
    {"CompanyName": "Hindustan Unilever", "NSESymbol": "HINDUNILVR", "Sector": "Fast Moving Consumer Goods", "ISIN": "INE030A01027"},
    {"CompanyName": "Adani Enterprises", "NSESymbol": "ADANIENT", "Sector": "Metals & Mining", "ISIN": "INE423A01024"},
    {"CompanyName": "Bharti Airtel", "NSESymbol": "BHARTIARTL", "Sector": "Telecommunications", "ISIN": "INE397D01024"},
    {"CompanyName": "Kotak Mahindra Bank", "NSESymbol": "KOTAKBANK", "Sector": "Financial Services", "ISIN": "INE237A01028"},
    {"CompanyName": "Larsen & Toubro", "NSESymbol": "LT", "Sector": "Construction", "ISIN": "INE018A01030"},
    {"CompanyName": "ITC", "NSESymbol": "ITC", "Sector": "Fast Moving Consumer Goods", "ISIN": "INE154A01025"},
    {"CompanyName": "Axis Bank", "NSESymbol": "AXISBANK", "Sector": "Financial Services", "ISIN": "INE238A01034"},
    {"CompanyName": "State Bank of India", "NSESymbol": "SBIN", "Sector": "Financial Services", "ISIN": "INE062A01020"},
    {"CompanyName": "Bajaj Finance", "NSESymbol": "BAJFINANCE", "Sector": "Financial Services", "ISIN": "INE296A01032"}, # Corrected ISIN
    {"CompanyName": "Asian Paints", "NSESymbol": "ASIANPAINT", "Sector": "Consumer Durables", "ISIN": "INE021A01026"},
    {"CompanyName": "Maruti Suzuki", "NSESymbol": "MARUTI", "Sector": "Automobile and Auto Components", "ISIN": "INE585B01010"},
    {"CompanyName": "Nestle India", "NSESymbol": "NESTLEIND", "Sector": "Fast Moving Consumer Goods", "ISIN": "INE239A01024"}, # Corrected ISIN
    {"CompanyName": "HCL Technologies", "NSESymbol": "HCLTECH", "Sector": "Information Technology", "ISIN": "INE860A01027"},
    {"CompanyName": "Mahindra & Mahindra", "NSESymbol": "M&M", "Sector": "Automobile and Auto Components", "ISIN": "INE101A01026"},
    {"CompanyName": "Power Grid Corporation of India", "NSESymbol": "POWERGRID", "Sector": "Power", "ISIN": "INE752E01010"},
    {"CompanyName": "UltraTech Cement", "NSESymbol": "ULTRACEMCO", "Sector": "Construction Materials", "ISIN": "INE481G01011"},
    {"CompanyName": "Dr Reddy's Laboratories", "NSESymbol": "DRREDDY", "Sector": "Healthcare", "ISIN": "INE089A01031"}, # Corrected ISIN
    {"CompanyName": "Tata Steel", "NSESymbol": "TATASTEEL", "Sector": "Metals & Mining", "ISIN": "INE081A01020"}, # Corrected ISIN
    {"CompanyName": "Sun Pharmaceutical", "NSESymbol": "SUNPHARMA", "Sector": "Healthcare", "ISIN": "INE044A01036"},
    {"CompanyName": "Titan Company", "NSESymbol": "TITAN", "Sector": "Consumer Durables", "ISIN": "INE280A01028"},
    {"CompanyName": "Tech Mahindra", "NSESymbol": "TECHM", "Sector": "Information Technology", "ISIN": "INE669C01036"},
    {"CompanyName": "NTPC", "NSESymbol": "NTPC", "Sector": "Power", "ISIN": "INE733E01010"},
    {"CompanyName": "Wipro", "NSESymbol": "WIPRO", "Sector": "Information Technology", "ISIN": "INE075A01022"},
    {"CompanyName": "Britannia Industries", "NSESymbol": "BRITANNIA", "Sector": "Fast Moving Consumer Goods", "ISIN": "INE216A01030"},
    {"CompanyName": "Bajaj Auto", "NSESymbol": "BAJAJ-AUTO", "Sector": "Automobile and Auto Components", "ISIN": "INE917I01010"},
    {"CompanyName": "Cipla", "NSESymbol": "CIPLA", "Sector": "Healthcare", "ISIN": "INE059A01026"},
    {"CompanyName": "Oil & Natural Gas Corporation", "NSESymbol": "ONGC", "Sector": "Oil Gas & Consumer Goods", "ISIN": "INE213A01029"},
    {"CompanyName": "Hero MotoCorp", "NSESymbol": "HEROMOTOCO", "Sector": "Automobile and Auto Components", "ISIN": "INE158A01026"},
    {"CompanyName": "Hindalco Industries", "NSESymbol": "HINDALCO", "Sector": "Metals & Mining", "ISIN": "INE038A01020"},
    {"CompanyName": "Adani Ports & SEZ", "NSESymbol": "ADANIPORTS", "Sector": "Services", "ISIN": "INE742F01042"},
    {"CompanyName": "Grasim Industries", "NSESymbol": "GRASIM", "Sector": "Construction Materials", "ISIN": "INE047A01021"},
    {"CompanyName": "Coal India", "NSESymbol": "COALINDIA", "Sector": "Oil Gas & Consumer Goods", "ISIN": "INE522F01014"},
    {"CompanyName": "IndusInd Bank", "NSESymbol": "INDUSINDBK", "Sector": "Financial Services", "ISIN": "INE095A01012"},
    {"CompanyName": "JSW Steel", "NSESymbol": "JSWSTEEL", "Sector": "Metals & Mining", "ISIN": "INE019A01038"},
    {"CompanyName": "Tata Motors", "NSESymbol": "TATAMOTORS", "Sector": "Automobile and Auto Components", "ISIN": "INE155A01022"},
    {"CompanyName": "Eicher Motors", "NSESymbol": "EICHERMOT", "Sector": "Automobile and Auto Components", "ISIN": "INE066A01021"}, # Corrected ISIN
    {"CompanyName": "HDFC Life Insurance", "NSESymbol": "HDFCLIFE", "Sector": "Financial Services", "ISIN": "INE795G01014"},
    {"CompanyName": "Bajaj Finserv", "NSESymbol": "BAJAJFINSV", "Sector": "Financial Services", "ISIN": "INE918I01026"}, # Corrected ISIN
    {"CompanyName": "Divi's Laboratories", "NSESymbol": "DIVISLAB", "Sector": "Healthcare", "ISIN": "INE361B01024"},
    {"CompanyName": "Bharat Petroleum Corporation", "NSESymbol": "BPCL", "Sector": "Oil Gas & Consumer Goods", "ISIN": "INE029A01011"},
    {"CompanyName": "Tata Consumer Products", "NSESymbol": "TATACONSUM", "Sector": "Fast Moving Consumer Goods", "ISIN": "INE192A01025"},
    {"CompanyName": "Apollo Hospitals", "NSESymbol": "APOLLOHOSP", "Sector": "Healthcare", "ISIN": "INE437A01024"},
    {"CompanyName": "Shriram Finance", "NSESymbol": "SHRIRAMFIN", "Sector": "Financial Services", "ISIN": "INE721A01047"}, # Corrected ISIN
    {"CompanyName": "SBI Life Insurance", "NSESymbol": "SBILIFE", "Sector": "Financial Services", "ISIN": "INE123W01016"},
    {"CompanyName": "LTI Mindtree", "NSESymbol": "LTIM", "Sector": "Information Technology", "ISIN": "INE214T01019"}
]

class InstrumentManager:
    """Manages the universe of instruments for trading."""
    def __init__(self):
        self.nifty50_df = pd.DataFrame(NIFTY50_INTERNAL_LIST)
        logger.info(f"InstrumentManager initialized with {len(self.nifty50_df)} internal Nifty 50 records.")

    def get_nse_symbols_with_isin(self):
        """Returns a list of tuples, where each tuple is (NSESymbol, ISIN)."""
        if self.nifty50_df.empty:
            logger.warning("Nifty50 DataFrame is empty.")
            return []
        return list(zip(self.nifty50_df['NSESymbol'], self.nifty50_df['ISIN']))