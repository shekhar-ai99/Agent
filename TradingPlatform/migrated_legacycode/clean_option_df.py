# import os
# import glob
# import pandas as pd
# import numpy as np

# def fill_missing_ohlcv_bars(csv_path, timeframe, output_csv_path=None):
#     """
#     Fills missing OHLCV bars in a CSV file for a given timeframe within a trading session.
#     Includes detailed diagnostics for the first day processed in each file.
#     """
#     ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
#     print(f"--- Processing file: {csv_path} for timeframe: {timeframe} ---")

#     try:
#         df = pd.read_csv(csv_path, parse_dates=['datetime'])
#     except FileNotFoundError:
#         print(f"Error: File not found at {csv_path}")
#         return
#     except Exception as e:
#         print(f"Error reading CSV {csv_path}: {e}")
#         return

#     if 'datetime' not in df.columns:
#         print(f"Error: 'datetime' column not found in {csv_path}")
#         return
        
#     if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
#         df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
#         df.dropna(subset=['datetime'], inplace=True)

#     if df.empty:
#         print(f"DataFrame is empty after initial load or datetime conversion for {csv_path}.")
#         return

#     df = df.sort_values('datetime').reset_index(drop=True)
#     original_row_count = len(df)
#     print(f"Original row count in file: {original_row_count}")

#     for col in ohlcv_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         else:
#             print(f"Warning: Column '{col}' not found in {csv_path}. Creating it.")
#             df[col] = np.nan if col != 'volume' else 0

#     try:
#         interval_str = timeframe.lower().replace('min', '')
#         if not interval_str.isdigit():
#             raise ValueError(f"Invalid timeframe format: {timeframe}.")
#         interval = int(interval_str)
#         if interval <= 0:
#             raise ValueError("Timeframe interval must be positive.")
#     except ValueError as e:
#         print(f"Error processing timeframe '{timeframe}': {e}")
#         return

#     session_start_time_obj = pd.Timestamp('09:15:00').time()
#     session_end_time_obj = pd.Timestamp('15:30:00').time()

#     all_filled_rows = []
#     total_existing_rows_added_file = 0
#     total_new_rows_created_file = 0
    
#     original_tz = df['datetime'].dt.tz
    
#     first_day_processed_for_file = True # Flag for detailed first-day logging

#     for date_obj_val, day_df in df.groupby(df['datetime'].dt.date):
#         if day_df.empty:
#             continue
        
#         daily_existing_rows_added = 0
#         daily_new_rows_created = 0

#         day_start_timestamp_naive = pd.Timestamp.combine(pd.to_datetime(date_obj_val), session_start_time_obj)
        
#         if original_tz:
#             day_start_timestamp = day_start_timestamp_naive.tz_localize(original_tz)
#         else:
#             day_start_timestamp = day_start_timestamp_naive
        
#         session_duration_seconds = (pd.Timestamp.combine(pd.Timestamp('1970-01-01'), session_end_time_obj) - 
#                                     pd.Timestamp.combine(pd.Timestamp('1970-01-01'), session_start_time_obj)).total_seconds()
#         interval_seconds = interval * 60
#         num_expected_bars_for_day = int(session_duration_seconds // interval_seconds) + 1
        
#         expected_session_times = [day_start_timestamp + pd.Timedelta(minutes=interval * i) for i in range(num_expected_bars_for_day)]
        
#         # Use .to_dict('records') for robust row data, keys are original datetimes from day_df
#         existing_rows_dict = {dt_val: row for dt_val, row in zip(day_df['datetime'], day_df.to_dict('records'))}

#         if first_day_processed_for_file:
#             print(f"\n  Detailed trace for first processed day: {date_obj_val}")
#             print(f"    Expected number of bars for this day ({timeframe}): {num_expected_bars_for_day}")
#             print(f"    Actual rows found in input for this day: {len(day_df)}")
#             if expected_session_times:
#                  print(f"      First expected_time generated: {expected_session_times[0]} (TZ: {expected_session_times[0].tzinfo})")
#             if existing_rows_dict:
#                 sample_key = next(iter(existing_rows_dict.keys()))
#                 print(f"      Sample key from existing_rows_dict: {sample_key} (TZ: {sample_key.tzinfo})")
#             else:
#                 print(f"      existing_rows_dict is empty for this day (no original rows).")
#             print(f"    --- Checking each expected timestamp for {date_obj_val} ---")

#         day_filled_rows = []
#         for i, expected_time in enumerate(expected_session_times):
#             is_found = expected_time in existing_rows_dict

#             if first_day_processed_for_file:
#                 status_msg = "FOUND in original data" if is_found else "NOT FOUND in original data - will attempt to fill"
#                 print(f"      - Expected: {expected_time} -> {status_msg}")

#             if is_found:
#                 day_filled_rows.append(existing_rows_dict[expected_time])
#                 daily_existing_rows_added += 1
#             else:
#                 if first_day_processed_for_file: # Print only for the first day's detailed trace
#                      print(f"        Attempting to fill missing bar for: {expected_time}") 
                
#                 prev_data_row = None
#                 for k in range(i - 1, -1, -1): 
#                     if expected_session_times[k] in existing_rows_dict:
#                         prev_data_row = existing_rows_dict[expected_session_times[k]]
#                         break
                
#                 next_data_row = None
#                 for k in range(i + 1, len(expected_session_times)): 
#                     if expected_session_times[k] in existing_rows_dict:
#                         next_data_row = existing_rows_dict[expected_session_times[k]]
#                         break
                
#                 new_filled_row = {'datetime': expected_time}
                
#                 for col in ohlcv_cols:
#                     values_for_interp = []
#                     if prev_data_row is not None and col in prev_data_row and not pd.isnull(prev_data_row[col]):
#                         values_for_interp.append(prev_data_row[col])
#                     if next_data_row is not None and col in next_data_row and not pd.isnull(next_data_row[col]):
#                         values_for_interp.append(next_data_row[col])
                    
#                     if len(values_for_interp) == 2:
#                         interpolated_value = np.mean(values_for_interp)
#                         new_filled_row[col] = int(round(interpolated_value)) if col == 'volume' else round(interpolated_value, 2)
#                     elif len(values_for_interp) == 1:
#                         new_filled_row[col] = values_for_interp[0]
#                     else:
#                         new_filled_row[col] = 0 if col == 'volume' else np.nan
#                 day_filled_rows.append(new_filled_row)
#                 daily_new_rows_created +=1
        
#         if first_day_processed_for_file:
#              print(f"    --- End of detailed trace for {date_obj_val} ---")
#              first_day_processed_for_file = False # Turn off detailed logging for subsequent days in this file

#         if daily_existing_rows_added > 0 or daily_new_rows_created > 0: 
#             if not first_day_processed_for_file: # Avoid double printing for the first day
#                  print(f"  Date {date_obj_val}: Found {daily_existing_rows_added} existing rows, Created {daily_new_rows_created} new rows.")
#         all_filled_rows.extend(day_filled_rows)
#         total_existing_rows_added_file += daily_existing_rows_added
#         total_new_rows_created_file += daily_new_rows_created

#     if not all_filled_rows:
#         print(f"No data processed or generated for {csv_path} after daily aggregation.")
#         return

#     filled_df = pd.DataFrame(all_filled_rows)
    
#     final_cols_ordered = ['datetime'] 
#     present_ohlcv_cols = [col for col in ohlcv_cols if col in filled_df.columns]
#     final_cols_ordered.extend(present_ohlcv_cols)
    
#     original_other_cols = [col for col in df.columns if col not in final_cols_ordered]
#     final_cols_ordered.extend(original_other_cols)
    
#     filled_df = filled_df.reindex(columns=final_cols_ordered) 
#     filled_df = filled_df.sort_values('datetime').reset_index(drop=True)

#     if output_csv_path is None:
#         dir_name = os.path.dirname(csv_path)
#         base_name = os.path.basename(csv_path)
#         file_name, file_ext = os.path.splitext(base_name)
#         output_dir = os.path.join(dir_name, "filled_data") 
#         os.makedirs(output_dir, exist_ok=True)
#         output_csv_path = os.path.join(output_dir, f"{file_name}_filled{file_ext}")
    
#     try:
#         filled_df.to_csv(output_csv_path, index=False, float_format='%.2f')
#         print(f"\nFilled CSV saved to: {output_csv_path}")
#         print(f"  Summary for {os.path.basename(csv_path)}:")
#         print(f"    Total original rows read from input: {original_row_count}")
#         print(f"    Total existing rows included in output: {total_existing_rows_added_file}")
#         print(f"    Total new rows created for missing bars: {total_new_rows_created_file}")
#         print(f"    Total rows in filled file: {len(filled_df)}")
#         print(f"--- Finished processing: {csv_path} ---\n")
#     except Exception as e:
#         print(f"Error writing filled CSV to {output_csv_path}: {e}")

# if __name__ == "__main__":
#     option_data_dir = "data/option"  
#     if not os.path.isdir(option_data_dir):
#         print(f"Error: Data directory '{option_data_dir}' not found. Please update the path.")
#     else:
#         all_csvs = glob.glob(os.path.join(option_data_dir, "*.csv"))
#         if not all_csvs:
#             print(f"No CSV files found in '{option_data_dir}'.")
#         for csv_file_path in all_csvs:
#             base_name = os.path.basename(csv_file_path)
#             file_part_for_check, ext_part = os.path.splitext(base_name)
#             if file_part_for_check.endswith("_filled"): 
#                 print(f"Skipping already filled file: {csv_file_path}")
#                 continue
            
#             parts = base_name.replace('.csv', '').split('_')
#             timeframe_str = None
#             for part in reversed(parts): 
#                 if 'min' in part.lower() and part.lower().replace('min','').isdigit():
#                     timeframe_str = part.lower()
#                     break
            
#             if timeframe_str:
#                 fill_missing_ohlcv_bars(csv_file_path, timeframe_str) 
#             else:
#                 print(f"\nCould not automatically detect timeframe for file: {csv_file_path}.")
#                 print(f"  Ensure filename contains timeframe like '_5min.csv', '_1min.csv', etc.")
import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
import glob

def get_trading_session_timeframe():
    """Define trading session and timeframe details."""
    return {
        'start_time': '09:15:00',
        'end_time': '15:30:00',
        'timezone': 'Asia/Kolkata',
        'timeframes': {
            '1min': 60,
            '3min': 180,
            '5min': 300,
            '15min': 900
        }
    }

def generate_expected_timestamps(date, timeframe_seconds, start_time, end_time, timezone):
    """Generate expected timestamps for a given date and timeframe."""
    tz = pytz.timezone(timezone)
    start_dt = tz.localize(datetime.strptime(f"{date} {start_time}", '%Y-%m-%d %H:%M:%S'))
    end_dt = tz.localize(datetime.strptime(f"{date} {end_time}", '%Y-%m-%d %H:%M:%S'))
    timestamps = []
    current_dt = start_dt
    while current_dt <= end_dt:
        timestamps.append(current_dt)
        current_dt += timedelta(seconds=timeframe_seconds)
    return timestamps

def fill_missing_bar(prev_row, next_row):
    """Fill missing bar by interpolating or copying nearest available row."""
    if prev_row is not None and next_row is not None:
        return {
            'open': round((prev_row['open'] + next_row['open']) / 2, 2),
            'high': round((prev_row['high'] + next_row['high']) / 2, 2),
            'low': round((prev_row['low'] + next_row['low']) / 2, 2),
            'close': round((prev_row['close'] + next_row['close']) / 2, 2),
            'volume': int((prev_row['volume'] + next_row['volume']) / 2)
        }
    elif prev_row is not None:
        return prev_row.copy()
    elif next_row is not None:
        return next_row.copy()
    else:
        return {'open': float('nan'), 'high': float('nan'), 'low': float('nan'), 'close': float('nan'), 'volume': 0}

def fill_missing_ohlcv_bars(input_file_path, output_dir='filled_data'):
    """Process OHLCV data by day, fill missing bars, and save to a clean CSV."""
    print(f"--- Processing file: {input_file_path} ---")
    
    # Load data
    try:
        df = pd.read_csv(input_file_path, parse_dates=['datetime'])
    except Exception as e:
        print(f"Error reading {input_file_path}: {e}")
        return
    
    print(f"Original row count in file: {len(df)}")
    
    # Ensure datetime is timezone-aware
    tz = pytz.timezone('Asia/Kolkata')
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize(tz)
    else:
        df['datetime'] = df['datetime'].dt.tz_convert(tz)
    
    # Extract timeframe from filename
    filename = os.path.basename(input_file_path)
    timeframe = None
    for tf in ['1min', '3min', '5min', '15min']:
        if tf in filename.lower():
            timeframe = tf
            break
    if not timeframe:
        print(f"Could not detect timeframe in filename: {filename}. Expected '1min', '3min', '5min', or '15min'.")
        return
    print(f"Detected timeframe: {timeframe}")
    
    session_info = get_trading_session_timeframe()
    timeframe_seconds = session_info['timeframes'][timeframe]
    start_time = session_info['start_time']
    end_time = session_info['end_time']
    timezone = session_info['timezone']
    
    # Calculate expected bars per day
    start_dt = datetime.strptime(start_time, '%H:%M:%S')
    end_dt = datetime.strptime(end_time, '%H:%M:%S')
    session_seconds = int((end_dt - start_dt).total_seconds())
    expected_bars = session_seconds // timeframe_seconds + 1
    print(f"Expected bars per day for {timeframe}: {expected_bars}")
    
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
        expected_timestamps = generate_expected_timestamps(date_str, timeframe_seconds, start_time, end_time, timezone)
        
        # Log details for first day
        if i == 0:
            print(f"\n  Detailed trace for first processed day: {date_str}")
            print(f"    Expected number of bars for this day ({timeframe}): {len(expected_timestamps)}")
            print(f"    Actual rows found in input for this day: {len(date_data)}")
            print(f"      First expected_time generated: {expected_timestamps[0]} (TZ: {timezone})")
            if len(date_data) > 0:
                print(f"      Sample key from existing_rows_dict: {existing_timestamps[0]} (TZ: {timezone})")
            print(f"    --- Checking each expected timestamp for {date_str} ---")
        
        # Identify missing timestamps
        missing_timestamps = [ts for ts in expected_timestamps if ts not in existing_timestamps]
        if missing_timestamps:
            print(f"    Missing timestamps for {date_str}: {[ts.strftime('%Y-%m-%d %H:%M:%S%z') for ts in missing_timestamps]}")
        
        new_rows_count = 0
        for ts in expected_timestamps:
            if ts in existing_timestamps:
                if i == 0:
                    print(f"      - Expected: {ts.strftime('%Y-%m-%d %H:%M:%S%z')} -> FOUND in original data")
                filled_rows.append(date_data.loc[ts].to_dict())
            else:
                if i == 0:
                    print(f"      - Expected: {ts.strftime('%Y-%m-%d %H:%M:%S%z')} -> NOT FOUND in original data - will attempt to fill")
                    print(f"        Attempting to fill missing bar for: {ts.strftime('%Y-%m-%d %H:%M:%S%z')}")
                
                prev_row = date_data[date_data.index < ts].tail(1)
                next_row = date_data[date_data.index > ts].head(1)
                prev_row = prev_row.iloc[0].to_dict() if not prev_row.empty else None
                next_row = next_row.iloc[0].to_dict() if not next_row.empty else None
                
                filled_bar = fill_missing_bar(prev_row, next_row)
                filled_bar['datetime'] = ts
                filled_rows.append(filled_bar)
                new_rows_count += 1
        
        if i == 0:
            print(f"    --- End of detailed trace for {date_str} ---")
        print(f"  Date {date_str}: Found {len(date_data)} existing rows, Created {new_rows_count} new rows.")
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
    output_file = os.path.join(output_dir, filename.replace('.csv', '__filled.csv'))
    filled_df.to_csv(output_file, index=False)
    print(f"\nFilled CSV saved to: {output_file}")
    print(f"  Summary for {filename}:")
    print(f"    Total original rows read from input: {len(df)}")
    print(f"    Total existing rows included in output: {len(df)}")
    print(f"    Total new rows created for missing bars: {total_new_rows}")
    print(f"    Total rows in filled file: {len(filled_df)}")
    print(f"--- Finished processing: {input_file_path} ---")

if __name__ == "__main__":
    option_data_dir = "data/option"
    if not os.path.isdir(option_data_dir):
        print(f"Error: Data directory '{option_data_dir}' not found. Please update the path.")
    else:
        all_csvs = glob.glob(os.path.join(option_data_dir, "*.csv"))
        if not all_csvs:
            print(f"No CSV files found in '{option_data_dir}'.")
        else:
            for csv_file_path in all_csvs:
                base_name = os.path.basename(csv_file_path)
                file_part_for_check, ext_part = os.path.splitext(base_name)
                if file_part_for_check.endswith(("_filled", "_clean")):
                    print(f"Skipping already processed file: {csv_file_path}")
                    continue
                
                print(f"Processing file: {csv_file_path}")
                fill_missing_ohlcv_bars(csv_file_path, output_dir="data/filled_data")