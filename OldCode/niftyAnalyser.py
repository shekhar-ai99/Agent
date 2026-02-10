import os
import pandas as pd
import numpy as np
from SmartApi import SmartConnect
import pyotp
from logzero import logger
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

class NiftyAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('ANGELONE_API_KEY')
        self.client_code = os.getenv('ANGELONE_CLIENT_CODE')
        self.password = os.getenv('ANGELONE_PASSWORD')
        self.totp_secret = os.getenv('ANGELONE_TOTP_SECRET')
        
        if not all([self.api_key, self.client_code, self.password, self.totp_secret]):
            raise ValueError("Missing required environment variables")
            
        self.smart_api = SmartConnect(self.api_key)
        self.login()
    
    def login(self):
        """Authenticate with Angel One API"""
        try:
            totp = pyotp.TOTP(self.totp_secret).now()
            login_data = self.smart_api.generateSession(self.client_code, self.password, totp)
            
            if not login_data['status']:
                logger.error(f"Login failed: {login_data}")
                raise Exception(login_data['message'])
            
            self.auth_token = login_data['data']['jwtToken']
            self.refresh_token = login_data['data']['refreshToken']
            self.feed_token = self.smart_api.getfeedToken()
            logger.info("Login successful")
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            raise
    
    def get_3min_data(self):
        """Fetch 3-minute candle data"""
        try:
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)
            
            params = {
                "exchange": "NSE",
                "symboltoken": "99926000",
                "interval": "THREE_MINUTE",
                "fromdate": now.strftime('%Y-%m-%d 09:15'),
                "todate": now.strftime('%Y-%m-%d %H:%M')
            }
            
            response = self.smart_api.getCandleData(params)
            
            if not response['status'] or not response.get('data'):
                logger.error(f"API Error: {response.get('message', 'No data')}")
                return None
            
            df = pd.DataFrame(
                response['data'],
                columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators and trading signals"""
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # RSI (14 period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (12,26,9)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands (20,2)
        df['upper_bb'] = df['sma_20'] + (2 * df['close'].rolling(window=20).std())
        df['lower_bb'] = df['sma_20'] - (2 * df['close'].rolling(window=20).std())
        
        # Trading Signals
        df['signal'] = 'Hold'
        
        # Long Entry Signals
        long_conditions = (
            (df['close'] > df['ema_9']) & 
            (df['close'] > df['ema_21']) & 
            (df['rsi'] > 50) & 
            (df['macd'] > df['macd_signal'])
        )
        
        # Short Entry Signals
        short_conditions = (
            (df['close'] < df['ema_9']) & 
            (df['close'] < df['ema_21']) & 
            (df['rsi'] < 50) & 
            (df['macd'] < df['macd_signal'])
        )
        
        # Exit Signals
        exit_long = (df['close'] < df['ema_9']) | (df['rsi'] > 70)
        exit_short = (df['close'] > df['ema_9']) | (df['rsi'] < 30)
        
        # Generate signals
        df.loc[long_conditions, 'signal'] = 'Long'
        df.loc[short_conditions, 'signal'] = 'Short'
        df.loc[exit_long & (df['signal'].shift(1) == 'Long'), 'signal'] = 'Exit Long'
        df.loc[exit_short & (df['signal'].shift(1) == 'Short'), 'signal'] = 'Exit Short'
        
        # Position State
        df['position'] = 'Hold'
        current_position = 'Hold'
        
        for i in range(1, len(df)):
            if df['signal'].iloc[i] == 'Long' and current_position != 'Long':
                df['position'].iloc[i] = 'Enter Long'
                current_position = 'Long'
            elif df['signal'].iloc[i] == 'Short' and current_position != 'Short':
                df['position'].iloc[i] = 'Enter Short'
                current_position = 'Short'
            elif df['signal'].iloc[i] == 'Exit Long' and current_position == 'Long':
                df['position'].iloc[i] = 'Exit Long'
                current_position = 'Hold'
            elif df['signal'].iloc[i] == 'Exit Short' and current_position == 'Short':
                df['position'].iloc[i] = 'Exit Short'
                current_position = 'Hold'
            else:
                df['position'].iloc[i] = current_position
        
        return df
    
    def logout(self):
        """Terminate session"""
        try:
            self.smart_api.terminateSession(self.client_code)
            logger.info("Logout successful")
        except Exception as e:
            logger.error(f"Logout failed: {str(e)}")

if __name__ == "__main__":
    try:
        analyzer = NiftyAnalyzer()
        nifty_data = analyzer.get_3min_data()
        
        if nifty_data is not None:
            analyzed_data = analyzer.calculate_indicators(nifty_data)
            
            # Print important columns
            print(analyzed_data[['close', 'ema_9', 'ema_21', 'rsi', 'macd', 
                               'macd_signal', 'upper_bb', 'lower_bb', 
                               'signal', 'position']].tail(20))
            
            # Save to CSV
            filename = f"NIFTY_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            analyzed_data.to_csv(filename)
            logger.info(f"Data saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        analyzer.logout()