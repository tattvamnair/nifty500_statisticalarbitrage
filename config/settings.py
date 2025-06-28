# stat_arb_trader_dhan/config/settings.py

import os
from dotenv import load_dotenv

# --- Project Path Configuration ---
# Dynamically determine the absolute path to the project's root directory.
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Environment Variable Loading ---
ENV_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, '.env')

if os.path.exists(ENV_FILE_PATH):
    load_dotenv(ENV_FILE_PATH)
else:
    print(f"WARNING: .env file not found at {ENV_FILE_PATH}. Ensure it's created with your credentials.")

# --- Dhan API Credentials ---
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

# --- Dhan API Specific Constants (from docs) ---
# Exchange Segments
DHAN_SEGMENT_NSE_EQ = "NSE_EQ"
DHAN_SEGMENT_NSE_FNO = "NSE_FNO"
DHAN_SEGMENT_IDX_I = "IDX_I" # For underlying index lookups

# Instrument Types for API calls
DHAN_INSTRUMENT_EQUITY = "EQUITY"
DHAN_INSTRUMENT_OPTSTK = "OPTSTK"
DHAN_INSTRUMENT_OPTIDX = "OPTIDX"
DHAN_INSTRUMENT_FUTSTK = "FUTSTK"
DHAN_INSTRUMENT_FUTIDX = "FUTIDX"

# --- Logging Configuration ---
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "logs")
LOG_FILENAME = "trading_bot.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILENAME)
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR

# --- Initial Setup: Ensure Log Directory Exists ---
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create log directory '{LOG_DIR}'. OS error: {e}")

# --- Sanity Checks ---
if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
    print("CRITICAL WARNING: DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN is not set. Please check your .env file.")