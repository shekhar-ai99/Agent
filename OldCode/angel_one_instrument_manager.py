# angel_one_instrument_manager.py

import pandas as pd
import datetime as dt
import logging
import os
import traceback
from typing import Optional, List

import pytz

logger = logging.getLogger(__name__)

class InstrumentManager:
    def __init__(self, config):
        self.config = config
        self.instrument_df = None
        # Set to match OpenAPIScripMaster.json expiry format (e.g., '25JUN2026')
        self.expiry_date_format = '%d%b%Y'
        self.load_instrument_list()

    def load_instrument_list(self):
        path = self.config.get("instrument_cache_path", "data/instruments/OpenAPIScripMaster.json")
        if os.path.exists(path):
            self.instrument_df = pd.read_json(path)
            print("Loaded instrument columns:", self.instrument_df.columns.tolist())  # <---- ADD THIS LINE
            # Standardize columns
            if 'symbol' in self.instrument_df.columns:
                self.instrument_df.rename(columns={'symbol': 'tradingsymbol'}, inplace=True)
            if 'exch_seg' in self.instrument_df.columns:
                self.instrument_df.rename(columns={'exch_seg': 'exchange'}, inplace=True)
            print("After renaming, instrument columns:", self.instrument_df.columns.tolist())  # <---- ADD THIS LINE
        else:
            logger.error(f"Instrument file missing: {path}")


    def get_instrument_token(self, tradingsymbol, exchange="NFO"):
        if self.instrument_df is None:
            logger.error("Instrument list not loaded.")
            return None
        mask = (self.instrument_df["tradingsymbol"] == tradingsymbol) & (self.instrument_df["exchange"] == exchange)
        if mask.any():
            return str(self.instrument_df[mask]["token"].iloc[0])
        logger.error(f"Token not found for {tradingsymbol} on {exchange}")
        return None
    def refresh_instrument_list(self):
        """Placeholder for refreshing the instrument list (API method unavailable)."""
        cache_path = self.config.get("instrument_cache_path", "data/instruments/OpenAPIScripMaster.json")
        logger.warning(f"Cannot refresh instrument list via API. Ensure {cache_path} is up-to-date.")
        self.load_instrument_list()

    def get_atm_strike(self, underlying: str, expiry_date: str, current_price: float) -> Optional[float]:
        """Get the ATM strike price for an option."""
        if self.instrument_df is None:
            logger.error("Instrument list not loaded.")
            return None
        try:
            expiry_dt = pd.to_datetime(expiry_date, format='%d%b%Y')
            mask = (
                (self.instrument_df['name'] == underlying) &
                (self.instrument_df['exchange'] == 'NFO') &
                (self.instrument_df['instrumenttype'].isin(['OPTIDX'])) &
                (pd.to_datetime(self.instrument_df['expiry'], format=self.expiry_date_format) == expiry_dt)
            )
            options = self.instrument_df[mask]
            if options.empty:
                logger.error(f"No options found for {underlying} with expiry {expiry_date} on NFO")
                available_expiries = self.get_valid_expiries(underlying)
                logger.info(f"Available expiries for {underlying}: {sorted(available_expiries)}")
                return None
            strikes = options['strike'].astype(float) / 100  # Adjust for API scaling
            if strikes.empty:
                logger.error(f"No strike prices available for {underlying} with expiry {expiry_date}")
                return None
            atm_strike = strikes.iloc[(strikes - current_price).abs().argmin()]
            logger.info(f"Calculated ATM strike: {atm_strike} for price {current_price}")
            return atm_strike
        except Exception as e:
            logger.error(f"Error calculating ATM strike for {underlying} with expiry {expiry_date}: {e}\n{traceback.format_exc()}")
            return None

    def get_valid_expiries(self, underlying: str, exchange: str = 'NFO') -> List[str]:
        """Get a list of valid expiry dates for the underlying."""
        if self.instrument_df is None:
            logger.error("Instrument list not loaded.")
            return []
        try:
            mask = (
                (self.instrument_df['name'] == underlying) &
                (self.instrument_df['exchange'] == exchange) &
                (self.instrument_df['instrumenttype'].isin(['FUTIDX', 'OPTIDX']))
            )
            expiries = pd.to_datetime(self.instrument_df[mask]['expiry'], format=self.expiry_date_format).dt.strftime('%d%b%Y').unique()
            current_date = dt.datetime.now(pytz.timezone('Asia/Kolkata')).date()
            valid_expiries = [exp for exp in expiries if pd.to_datetime(exp, format='%d%b%Y').date() >= current_date]
            logger.debug(f"Valid expiries for {underlying} on {exchange}: {valid_expiries}")
            return sorted(valid_expiries)
        except ValueError as e:
            logger.error(f"Error fetching valid expiries for {underlying}: {e}\n{traceback.format_exc()}")
            # Fallback to mixed format parsing
            try:
                expiries = pd.to_datetime(self.instrument_df[mask]['expiry'], format='mixed', dayfirst=True).dt.strftime('%d%b%Y').unique()
                valid_expiries = [exp for exp in expiries if pd.to_datetime(exp, format='%d%b%Y').date() >= current_date]
                logger.warning(f"Fallback to mixed format parsing succeeded. Valid expiries: {valid_expiries}")
                return sorted(valid_expiries)
            except Exception as e2:
                logger.error(f"Fallback parsing failed: {e2}\n{traceback.format_exc()}")
                return []
        except Exception as e:
            logger.error(f"Error fetching valid expiries for {underlying}: {e}\n{traceback.format_exc()}")
            return []