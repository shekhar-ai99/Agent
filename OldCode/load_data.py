

# File: data_loader.py
import pandas as pd
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import traceback
import requests
import numpy as np
from typing import List, Optional, Tuple
from historical_data import HistoricalDataFetcher  # Assumes in historical_data.py
from compute_indicators import compute_indicators  # Assumes in compute_indicators.py
from config import Config, load_config
from angel_one.angel_one_api import AngelOneAPI
from angel_one.angel_one_instrument_manager import InstrumentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    def __init__(
        self,
        symbol: str,
        date_columns: List[str] = None,
        required_columns: List[str] = None,
        logger: logging.Logger = None
    ):
        """
        Initialize the HistoricalDataLoader.

        Args:
            symbol (str): The symbol for the data (e.g., 'NIFTY').
            date_columns (List[str], optional): List of possible date column names.
            required_columns (List[str], optional): List of required OHLCV columns.
            logger (logging.Logger, optional): Logger instance.
        """
        self.symbol = symbol.upper()
        self.date_columns = date_columns or ['date', 'datetime', 'time', 'timestamp']
        self.required_columns = required_columns or ['date', 'open', 'high', 'low', 'close', 'volume']
        self.logger = logger or logging.getLogger(__name__)

    def load_historical_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and process historical data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Processed DataFrame with 'date' as index.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns or date column are missing.
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            self.logger.info(f"Loading {self.symbol} data from {filepath}")
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()

            date_column = None
            for col in df.columns:
                if col.lower() in [dc.lower() for dc in self.date_columns]:
                    date_column = col
                    df.rename(columns={col: 'date'}, inplace=True)
                    break

            if date_column is None:
                raise ValueError(
                    f"No datetime column found in {filepath}. Expected one of {self.date_columns}"
                )

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if df['date'].isna().all():
                raise ValueError(f"Failed to parse 'date' column in {filepath}")

            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"CSV at {filepath} missing required columns: {missing_cols}. "
                    f"Expected columns: {self.required_columns}"
                )

            df.set_index('date', inplace=True)
            self.logger.info(f"Successfully loaded {len(df)} rows of {self.symbol} data from {filepath}")
            return df

        except FileNotFoundError as e:
            self.logger.error(f"File not found for {self.symbol}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading {self.symbol} data from {filepath}: {str(e)}")
            raise

    def fill_missing_ohlcv_bars(self, input_file_path: str, timeframe: str, output_dir: str = 'filled_data') -> Optional[pd.DataFrame]:
        """
        Process OHLCV data by day, fill missing bars, and save to a clean CSV.

        Args:
            input_file_path (str): Path to the input CSV file.
            timeframe (str): Timeframe for data (e.g., '5min').
            output_dir (str): Directory to save the filled CSV (default: 'filled_data').

        Returns:
            pd.DataFrame or None: Filled DataFrame if successful, None otherwise.
        """
        self.logger.info(f"--- Processing file: {input_file_path} for {self.symbol} ---")
        
        # Load data
        try:
            df = pd.read_csv(input_file_path, parse_dates=['datetime'])
        except Exception as e:
            self.logger.error(f"Error reading {input_file_path}: {e}")
            return None
        
        self.logger.info(f"Original row count in file: {len(df)}")
        
        # Ensure datetime is timezone-aware
        tz = pytz.timezone('Asia/Kolkata')
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize(tz)
        else:
            df['datetime'] = df['datetime'].dt.tz_convert(tz)
        
        # Validate timeframe
        timeframe_map = {'1min': 60, '3min': 180, '5min': 300, '15min': 900}
        if timeframe not in timeframe_map:
            self.logger.error(f"Invalid timeframe: {timeframe}. Supported: {list(timeframe_map.keys())}")
            return None
        
        timeframe_seconds = timeframe_map[timeframe]
        start_time = '09:15:00'
        end_time = '15:30:00'
        timezone = 'Asia/Kolkata'
        
        # Calculate expected bars per day
        start_dt = datetime.strptime(start_time, '%H:%M:%S')
        end_dt = datetime.strptime(end_time, '%H:%M:%S')
        session_seconds = int((end_dt - start_dt).total_seconds())
        expected_bars = session_seconds // timeframe_seconds + 1
        self.logger.info(f"Expected bars per day for {timeframe}: {expected_bars}")
        
        # Group data by date
        df['date'] = df['datetime'].dt.date
        dates = df['date'].unique()
        
        filled_rows = []
        total_new_rows = 0
        
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            date_data = df[df['date'] == date].set_index('datetime')
            existing_timestamps = date_data.index
            
            # Generate expected timestamps
            expected_timestamps = []
            current_dt = tz.localize(datetime.strptime(f"{date_str} {start_time}", '%Y-%m-%d %H:%M:%S'))
            end_dt = tz.localize(datetime.strptime(f"{date_str} {end_time}", '%Y-%m-%d %H:%M:%S'))
            while current_dt <= end_dt:
                expected_timestamps.append(current_dt)
                current_dt += timedelta(seconds=timeframe_seconds)
            
            # Log details for first day
            if i == 0:
                self.logger.info(f"\n  Detailed trace for first processed day: {date_str}")
                self.logger.info(f"    Expected number of bars for this day ({timeframe}): {len(expected_timestamps)}")
                self.logger.info(f"    Actual rows found in input for this day: {len(date_data)}")
                self.logger.info(f"      First expected_time generated: {expected_timestamps[0]} (TZ: {timezone})")
                if len(date_data) > 0:
                    self.logger.info(f"      Sample key from existing_rows_dict: {existing_timestamps[0]} (TZ: {timezone})")
                self.logger.info(f"    --- Checking each expected timestamp for {date_str} ---")
            
            # Identify missing timestamps
            missing_timestamps = [ts for ts in expected_timestamps if ts not in existing_timestamps]
            if missing_timestamps:
                self.logger.info(f"    Missing timestamps for {date_str}: {[ts.strftime('%Y-%m-%d %H:%M:%S%z') for ts in missing_timestamps]}")
            
            new_rows_count = 0
            for ts in expected_timestamps:
                if ts in existing_timestamps:
                    if i == 0:
                        self.logger.info(f"      - Expected: {ts.strftime('%Y-%m-%d %H:%M:%S%z')} -> FOUND in original data")
                    filled_rows.append(date_data.loc[ts].to_dict())
                else:
                    if i == 0:
                        self.logger.info(f"      - Expected: {ts.strftime('%Y-%m-%d %H:%M:%S%z')} -> NOT FOUND in original data - will attempt to fill")
                        self.logger.info(f"        Attempting to fill missing bar for: {ts.strftime('%Y-%m-%d %H:%M:%S%z')}")
                    
                    prev_row = date_data[date_data.index < ts].tail(1)
                    next_row = date_data[date_data.index > ts].head(1)
                    
                    filled_bar = {}
                    if not prev_row.empty and not next_row.empty:
                        filled_bar = {
                            'open': round((prev_row.iloc[0]['open'] + next_row.iloc[0]['open']) / 2, 2),
                            'close': round((prev_row.iloc[0]['close'] + next_row.iloc[0]['close']) / 2, 2),
                            'high': round((prev_row.iloc[0]['high'] + next_row.iloc[0]['high']) / 2, 2),
                            'low': round((prev_row.iloc[0]['low'] + next_row.iloc[0]['low']) / 2, 2),
                            'volume': int((prev_row.iloc[0]['volume'] + next_row.iloc[0]['volume']) / 2)
                        }
                    elif not prev_row.empty:
                        filled_bar = prev_row.iloc[0].to_dict()
                    elif not next_row.empty:
                        filled_bar = next_row.iloc[0].to_dict()
                    else:
                        filled_bar = {'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0}
                    filled_bar['datetime'] = ts
                    filled_rows.append(filled_bar)
                    new_rows_count += 1
            
            if i > 0:
                self.logger.info(f"Date {date_str}: Found {len(date_data)} existing rows, Created {new_rows_count} new rows.")
            total_new_rows += new_rows_count
        
        # Create output DataFrame
        filled_df = pd.DataFrame(filled_rows)
        filled_df['datetime'] = pd.to_datetime(filled_df['datetime'])
        filled_df = filled_df.sort_values('datetime').reset_index(drop=True)
        
        # Ensure correct column order
        columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        filled_df = filled_df[columns]
        
        # Round prices to 2 decimals, volume to integer
        filled_df[['open', 'high', 'low', 'close']] = filled_df[['open', 'high', 'low', 'close']].round(2)
        filled_df['volume'] = filled_df['volume'].astype(int, errors='ignore')
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_file_path)
        output_file = os.path.join(output_dir, filename.replace('.csv', '_filled.csv'))
        try:
            filled_df.to_csv(output_file, index=False)
            self.logger.info(f"Filled CSV saved to: {output_file}")
            self.logger.info(f"  Summary for {filename}:")
            self.logger.info(f"    Total original rows read from input: {len(df)}")
            self.logger.info(f"    Total existing rows included in output: {len(df)}")
            self.logger.info(f"    Total new rows created for missing bars: {total_new_rows}")
            self.logger.info(f"    Total rows in filled file: {len(filled_df)}")
            self.logger.info(f"--- Finished processing: {input_file_path} ---")
            return filled_df
        except Exception as e:
            self.logger.error(f"Error writing filled CSV to {output_file}: {e}")
            return None
    # NEW METHOD: Integrated download_instrument_list
    def download_instrument_list(self, output_path: str = "data/instruments/OpenAPIScripMaster.json") -> bool:
        """
        Download the Angel One instrument list if it does not exist.

        Args:
            output_path (str): Path to save the instrument file (default: 'data/instruments/OpenAPIScripMaster.json').

        Returns:
            bool: True if successful, False otherwise.
        """
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        try:
            if os.path.exists(output_path):
                self.logger.info(f"Instrument file already exists at {output_path}")
                return True
            
            self.logger.info(f"Downloading instrument list from {url}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            self.logger.info("✅ Instrument list downloaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to download instrument list: {str(e)}")
            return False

def load_all_data(
    symbol: str,
    timeframe: str,
    expiry_date: str,
    strike_price: str,
    output_dir: str = "data",
    days: int = 30
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load or fetch NIFTY index, CE, and PE option DataFrames for a given timeframe.

    Args:
        symbol (str): The base symbol (e.g., 'NIFTY').
        timeframe (str): Timeframe for data (e.g., '5min').
        expiry_date (str): Expiry date in 'YYYY-MM-DD' format (e.g., '2025-05-29').
        strike_price (str): Strike price (e.g., '24000').
        output_dir (str): Base directory for data storage (default: 'data').
        days (int): Number of days to fetch if data is missing (default: 60).

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            Tuple of (nifty_df, ce_df, pe_df). Returns None for any DataFrame that fails to load/fetch.
    """
    # Map timeframes to API interval keys
    interval_mapping = {
        "1min": "ONE_MINUTE",
        "3min": "THREE_MINUTE",
        "5min": "FIVE_MINUTE",
        "15min": "FIFTEEN_MINUTE"
    }

    # Validate timeframe
    if timeframe not in interval_mapping:
        logger.error(f"Invalid timeframe: {timeframe}. Supported: {list(interval_mapping.keys())}")
        return None, None, None

    interval_key = interval_mapping[timeframe]

    # Initialize API
    try:
        config = load_config()
        api_client = AngelOneAPI(config)
        instrument_manager = api_client.get_instrument_manager()
        smart_api = api_client.get_smart_connect_object()
        if instrument_manager is None or smart_api is None:
            logger.error("Failed to initialize AngelOne API or InstrumentManager.")
            return None, None, None
    except Exception as e:
        logger.error(f"API initialization failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

    # Define directories
    base_dir = Path(output_dir)
    raw_data_dir = base_dir / "raw"
    option_data_dir = base_dir / "option"
    indicator_data_dir = base_dir / "datawithindicator"
    filled_data_dir = base_dir / "filled_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    option_data_dir.mkdir(parents=True, exist_ok=True)
    indicator_data_dir.mkdir(parents=True, exist_ok=True)
    filled_data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fetcher
    fetcher = HistoricalDataFetcher(interval=interval_key)

    # Load or fetch NIFTY data
    nifty_loader = HistoricalDataLoader(symbol=symbol)
    # NEW: Download instrument list if missing
    nifty_loader.download_instrument_list()
    nifty_data_path = raw_data_dir / f"{symbol.lower()}_{timeframe}_historical.csv"
    nifty_df = None
    try:
        nifty_df = nifty_loader.load_historical_data(nifty_data_path)
        logger.info(f"Loaded Nifty data for {timeframe}. Shape: {nifty_df.shape}")
    except FileNotFoundError:
        logger.warning(f"Nifty {timeframe} data file not found at {nifty_data_path}. Fetching last {days} days.")
        try:
            nifty_df = fetcher.fetch_historical_data(smart_api, days=days)
            nifty_df.to_csv(nifty_data_path)
            logger.info(f"Fetched and saved Nifty {timeframe} data to {nifty_data_path}")
        except Exception as e:
            logger.error(f"Error fetching Nifty data for {timeframe}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            nifty_df = None
    except Exception as e:
        logger.error(f"Error loading Nifty data for {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        nifty_df = None

    # Compute indicators for NIFTY data if loaded
    if nifty_df is not None:
        output_path = indicator_data_dir / f"nifty_{timeframe}_with_indicators.csv"
        try:
            compute_indicators(nifty_df, output_path)
            nifty_df = nifty_loader.load_historical_data(output_path)
            logger.info(f"Computed indicators and loaded enriched Nifty data for {timeframe}. Shape: {nifty_df.shape}")
        except Exception as e:
            logger.error(f"Error computing indicators for Nifty {timeframe}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            nifty_df = None

    # Load or fetch CE option data
    expiry_formatted = datetime.strptime(expiry_date, '%Y-%m-%d').strftime('%d%b%y').upper()
    ce_instrument_name = f"{symbol}{strike_price}CE{expiry_formatted}"
    ce_data_path = option_data_dir / f"{ce_instrument_name}_{timeframe}.csv"
    ce_filled_path = filled_data_dir / f"{ce_instrument_name}_{timeframe}_filled.csv"
    ce_loader = HistoricalDataLoader(symbol=ce_instrument_name)
    ce_df = None
    try:
        # NEW: Check for filled data first
        if os.path.exists(ce_filled_path):
            ce_df = ce_loader.load_historical_data(ce_filled_path)
            logger.info(f"Loaded filled CE option data for {timeframe}. Shape: {ce_df.shape}")
        elif os.path.exists(ce_data_path):
            ce_df = ce_loader.load_historical_data(ce_data_path)
            logger.info(f"Loaded CE option data for {timeframe}. Shape: {ce_df.shape}")
            # NEW: Fill missing bars
            ce_df = ce_loader.fill_missing_ohlcv_bars(ce_data_path, timeframe, filled_data_dir)
            if ce_df is not None:
                logger.info(f"Filled CE option data for {timeframe}. Shape: {ce_df.shape}")
        else:
            logger.warning(f"CE option data file not found at {ce_data_path}. Fetching data.")
            ce_token = instrument_manager.get_instrument_token(f"{symbol}{expiry_formatted}{strike_price}CE", "NFO")
            if ce_token:
                try:
                    ce_df = fetcher.fetch_historical_candles(
                        smart_api,
                        symboltoken=ce_token,
                        exchange="NFO",
                        interval=interval_key,
                        fromdate=(datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(days=days)).strftime('%Y-%m-%d %H:%M'),
                        todate=datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M'),
                        instrument_name=ce_instrument_name
                    )
                    if ce_df is not None and not ce_df.empty:
                        ce_df.to_csv(ce_data_path)
                        logger.info(f"Fetched and saved CE option data to {ce_data_path}")
                        # NEW: Fill missing bars
                        ce_df = ce_loader.fill_missing_ohlcv_bars(ce_data_path, timeframe, filled_data_dir)
                        if ce_df is not None:
                            logger.info(f"Filled CE option data for {timeframe}. Shape: {ce_df.shape}")
                    else:
                        logger.warning(f"No data fetched for CE option: {ce_instrument_name}")
                        ce_df = None
                except Exception as e:
                    logger.error(f"Error fetching CE option data for {timeframe}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    ce_df = None
            else:
                logger.warning(f"Could not retrieve token for CE option: {ce_instrument_name}")
                ce_df = None
    except Exception as e:
        logger.error(f"Error loading CE option data for {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        ce_df = None

    # Load or fetch PE option data
    pe_instrument_name = f"{symbol}{strike_price}PE{expiry_formatted}"
    pe_data_path = option_data_dir / f"{pe_instrument_name}_{timeframe}.csv"
    pe_filled_path = filled_data_dir / f"{pe_instrument_name}_{timeframe}_filled.csv"
    pe_loader = HistoricalDataLoader(symbol=pe_instrument_name)
    pe_df = None
    try:
        # NEW: Check for filled data first
        if os.path.exists(pe_filled_path):
            pe_df = pe_loader.load_historical_data(pe_filled_path)
            logger.info(f"Loaded filled PE option data for {timeframe}. Shape: {pe_df.shape}")
        elif os.path.exists(pe_data_path):
            pe_df = pe_loader.load_historical_data(pe_data_path)
            logger.info(f"Loaded PE option data for {timeframe}. Shape: {pe_df.shape}")
            # NEW: Fill missing bars
            pe_df = pe_loader.fill_missing_ohlcv_bars(pe_data_path, timeframe, filled_data_dir)
            if pe_df is not None:
                logger.info(f"Filled PE option data for {timeframe}. Shape: {pe_df.shape}")
        else:
            logger.warning(f"PE option data file not found at {pe_data_path}. Fetching data.")
            pe_token = instrument_manager.get_instrument_token(f"{symbol}{expiry_formatted}{strike_price}PE", "NFO")
            if pe_token:
                try:
                    pe_df = fetcher.fetch_historical_candles(
                        smart_api,
                        symboltoken=pe_token,
                        exchange="NFO",
                        interval=interval_key,
                        fromdate=(datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(days=days)).strftime('%Y-%m-%d %H:%M'),
                        todate=datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M'),
                        instrument_name=pe_instrument_name
                    )
                    if pe_df is not None and not pe_df.empty:
                        pe_df.to_csv(pe_data_path)
                        logger.info(f"Fetched and saved PE option data to {pe_data_path}")
                        # NEW: Fill missing bars
                        pe_df = pe_loader.fill_missing_ohlcv_bars(pe_data_path, timeframe, filled_data_dir)
                        if pe_df is not None:
                            logger.info(f"Filled PE option data for {timeframe}. Shape: {pe_df.shape}")
                    else:
                        logger.warning(f"No data fetched for PE option: {pe_instrument_name}")
                        pe_df = None
                except Exception as e:
                    logger.error(f"Error fetching PE option data for {timeframe}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    pe_df = None
            else:
                logger.warning(f"Could not retrieve token for PE option: {pe_instrument_name}")
                pe_df = None
    except Exception as e:
        logger.error(f"Error loading PE option data for {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        pe_df = None

    return nifty_df, ce_df, pe_df