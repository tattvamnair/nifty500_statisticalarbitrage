# stat_arb_trader_dhan/core/dhan_client.py

import sys
from dhanhq import dhanhq, DhanContext
from config import settings
from core.logger_setup import logger

class DhanClient:
    """
    A wrapper for the DhanHQ API client to handle initialization and connection testing.
    Uses the modern DhanContext for authentication.
    """
    def __init__(self):
        self.client_id = settings.DHAN_CLIENT_ID
        self.access_token = settings.DHAN_ACCESS_TOKEN
        self.dhan_sdk_instance = None

        if not self.client_id or not self.access_token:
            logger.critical("Dhan Client ID or Access Token is missing in settings. Cannot initialize API client.")
            return

        try:
            dhan_context = DhanContext(client_id=self.client_id, access_token=self.access_token)
            self.dhan_sdk_instance = dhanhq(dhan_context)
            logger.info("DhanHQ API client successfully initialized using DhanContext.")
        except Exception as e:
            logger.error(f"Failed to initialize DhanHQ API client: {e}", exc_info=True)
            self.dhan_sdk_instance = None

    def get_api_client(self):
        """Returns the initialized dhanhq SDK instance."""
        if self.dhan_sdk_instance is None:
            logger.warning("Attempted to get API client, but it was not initialized (or initialization failed).")
        return self.dhan_sdk_instance

    def test_connection(self):
        """Tests the API connection by fetching fund limits."""
        client = self.get_api_client()
        if not client:
            logger.error("Cannot test connection: Dhan API client not initialized.")
            return False
        
        try:
            logger.info("Attempting to fetch fund limits to test Dhan API connection...")
            response = client.get_fund_limits()
            
            if isinstance(response, dict) and response.get('status', '').lower() == 'success':
                data = response.get('data', {})
                balance = data.get('availabelBalance', data.get('availableBalance', 'N/A'))
                logger.info(f"Dhan API connection test successful. Available Balance: {balance}")
                return True
            else:
                remarks = response.get('remarks', 'No remarks provided.')
                logger.error(f"Dhan API connection test failed. Status: {response.get('status', 'N/A')}, Remarks: {remarks}")
                return False
        except Exception as e:
            logger.error(f"Exception during Dhan API connection test: {e}", exc_info=True)
            return False