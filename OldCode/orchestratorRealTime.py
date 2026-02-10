import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from real_time_data import initialize_api, fetch_real_time_data, continuous_data_fetch

class TradingOrchestrator:
    def __init__(self, realtime=False):
        self.realtime_mode = realtime
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.api = initialize_api()
        
    def run_analysis(self, df):
        """Run your full analysis pipeline"""
        # 1. Indicators
        from indicators import calculate_indicators
        df = calculate_indicators(df)
        
        # 2. Signals
        from signals import generate_signals
        df = generate_signals(df)
        
        # 3. Trading Logic
        from mylogicsignal import apply_logic
        df = apply_logic(df)
        
        # 4. Signal Analysis
        from signal_analyzer import analyze
        df = analyze(df)
        
        return df
    
    def process_historical(self):
        """Process historical data"""
        from historical_data import load_historical_data
        df = load_historical_data()
        df = self.run_analysis(df)
        df.to_csv(self.data_dir / "historical_analysis.csv")
        print("Historical analysis complete")
    
    def process_realtime(self):
        """Process real-time data"""
        print("Starting real-time analysis...")
        print("Timestamp\t| Close\t| Signal\t| Position")
        print("-" * 50)
        
        # Start continuous data fetching in background
        from threading import Thread
        data_thread = Thread(target=continuous_data_fetch, args=(self.api,))
        data_thread.daemon = True
        data_thread.start()
        
        while True:
            try:
                # Load latest data
                df = pd.read_csv(self.data_dir / "nifty_realtime_data.csv", 
                                parse_dates=['datetime'], 
                                index_col='datetime')
                
                # Run analysis on latest data point
                latest = df.iloc[-1:].copy()
                analyzed = self.run_analysis(latest)
                
                # Print to console
                last_row = analyzed.iloc[-1]
                print(
                    f"{last_row.name.strftime('%H:%M:%S')}\t| "
                    f"{last_row['close']:.2f}\t| "
                    f"{last_row.get('signal', 'HOLD')}\t| "
                    f"{last_row.get('position', '')}"
                )
                
                # Save to cumulative CSV
                output_file = self.data_dir / "realtime_analysis.csv"
                if not output_file.exists():
                    analyzed.to_csv(output_file)
                else:
                    analyzed.to_csv(output_file, mode='a', header=False)
                
                time.sleep(60)  # Update every minute
            
            except KeyboardInterrupt:
                print("\nStopping real-time analysis...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)
    
    def run(self):
        """Main entry point"""
        if self.realtime_mode:
            self.process_realtime()
        else:
            self.process_historical()

if __name__ == "__main__":
    # Check for realtime flag
    realtime = "--realtime" in sys.argv
    
    # Run pipeline
    orchestrator = TradingOrchestrator(realtime=realtime)
    orchestrator.run()