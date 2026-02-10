import logging
import pandas as pd
import numpy as np
from datetime import time
import os

from app.option_trade_executor import OptionTradeExecutor

logger = logging.getLogger(__name__)
pece_csv_path = "data/filled_data/"


def save_trades_to_csv(trades, run_id, output_dir=None):
    """Save trade details to a CSV file."""
    if not trades:
        logger.warning("No trades to save.")
        return
    if output_dir is None:
        output_dir = os.path.join('reports', f'run_{run_id}', 'trade_logs')

    os.makedirs(output_dir, exist_ok=True)
    df_trades = pd.DataFrame(trades)
    
    columns = [
        'strategy', 'timeframe', 'symbol', 'position', 'entry_timestamp', 'exit_timestamp',
        'entry_price', 'exit_price', 'pnl', 'profit_or_loss', 'exit_reason', 'regime', 'volatility', 'session', 'day', 'is_expiry'
    ]
    
    df_trades = df_trades[[col for col in columns if col in df_trades.columns]]
    
    output_path = os.path.join(output_dir, f"trades_{run_id}.csv")
    df_trades.to_csv(output_path, index=False)
    logger.info(f"Saved {len(trades)} trades to {output_path}")

class SimpleBacktester:
    def __init__(self, df, strategy_name, strategy_func, params, timeframe, symbol, exchange, ce_premiums_df=None, pe_premiums_df=None, run_id=None, expiry_date=None,
            strike_price=None):
        self.df = df.copy()
        self.strategy_name = strategy_name
        self.strategy_func = strategy_func
        self.params = params
        self.timeframe = timeframe
        self.symbol = symbol
        self.exchange = exchange
        self.ce_premiums_df = ce_premiums_df
        self.pe_premiums_df = pe_premiums_df
        self.sim_logger = logging.getLogger(f"{__name__}.{strategy_name}")
        self.trades = []
        self.trade_executor = OptionTradeExecutor(
            lot_size=75,
            entry_premium=100,
            premium_change_rate=0.1,
            points_per_change=10,
            pece_csv_path=pece_csv_path,
            timeframe=timeframe,
            run_id=run_id or "default_run",
            symbol="NIFTY",
            expiry=expiry_date,
            strike=strike_price,
            
        )

        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'timestamp' in self.df.columns:
                self.df.index = pd.to_datetime(self.df['timestamp'])
            else:
                raise ValueError("DataFrame must have a datetime index or 'timestamp' column")

        self.atr_percentiles = self.df['atr_14'].quantile([0.25, 0.75])

    def get_session(self, timestamp):
        """Classify timestamp into trading session."""
        t = timestamp.time()
        if time(9, 15) <= t < time(11, 30):
            return 'Session 1 (9:15-11:30)'
        elif time(11, 30) <= t < time(13, 45):
            return 'Session 2 (11:30-13:45)'
        elif time(13, 45) <= t <= time(15, 30):
            return 'Session 3 (13:45-15:30)'
        return 'Outside Session'

    def get_regime(self, row):
        """Classify market regime based on ADX and ATR."""
        adx = row.get('adx_14', 0)
        atr = row['atr_14']
        atr_median = self.df['atr_14'].median()
        
        if adx > 25:
            return 'Trending'
        elif atr > atr_median * 1.5:
            return 'Choppy'
        else:
            return 'Ranging'

    def get_volatility(self, atr):
        """Classify volatility based on ATR percentiles."""
        if atr < self.atr_percentiles[0.25]:
            return 'Low'
        elif atr < self.atr_percentiles[0.75]:
            return 'Medium'
        else:
            return 'High'
    
    def run_simulation(self):
        if not all(col in self.df.columns for col in ['close', 'high', 'low', 'atr_14']):
            error_msg = "Missing required columns: ['close', 'high', 'low', 'atr_14']"
            self.sim_logger.error(error_msg)
            return {
                'total_pnl': 0, 'total_trades': 0, 'profitable_trades': 0, 'losing_trades': 0, 'win_rate': 0,
                'performance_score': 0, 'buy_trades': 0, 'sell_trades': 0,
                'exit_reasons': {'sl': 0, 'tsl': 0, 'tp': 0, 'signal': 0, 'session_end': 0},
                'error': error_msg,
                'day_wise': {}, 'session_wise': {}, 'regime_wise': {}, 'volatility_wise': {}, 'expiry_wise': {}, 'trades': []
            }

        trade_id_counter = 0
        position = None
        entry_price = 0
        sl_price = 0
        tp_price = 0
        tsl_price = 0
        buy_trades = 0
        sell_trades = 0
        trade_id = None
        entry_timestamp = None
        entry_regime = None
        entry_volatility = None
        last_session_end_date = None  # Track last processed session-end date

        # Configuration
        same_day_exit = self.params.get('same_day_exit', True)  # Default to same-day exits
        session_end_time = time(15, 30)  # Market close at 15:30 IST
        latest_entry_time = time(15, 20)  # No entries after 15:20 IST if same_day_exit=True
        self.sim_logger.info(f"Running simulation with same_day_exit={same_day_exit}")

        exit_reasons = {'sl': 0, 'tsl': 0, 'tp': 0, 'signal': 0, 'session_end': 0}
        signal_count = {'buy_potential': 0, 'sell_potential': 0, 'none': 0}

        day_wise = {d: {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0} for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        session_wise = {s: {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0} for s in ['Session 1 (9:15-11:30)', 'Session 2 (11:30-13:45)', 'Session 3 (13:45-15:30)', 'Outside Session']}
        regime_wise = {r: {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0} for r in ['Trending', 'Choppy', 'Ranging']}
        volatility_wise = {v: {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0} for v in ['Low', 'Medium', 'High']}
        expiry_wise = {'Expiry Thursday': {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0}}

        for idx, row in self.df.iterrows():
            #current_bar_time = idx.time()
            if pd.isna(idx):
            # Option 1: Skip this row entirely (recommended for market data)
                continue
            try:
                current_bar_time = idx.time()
            except Exception:
                # Option 2: Handle/log and skip if .time() fails for any reason
                continue

            # Session-End Closure at 15:30 IST (only if same_day_exit=True)
            if same_day_exit and current_bar_time >= session_end_time and (last_session_end_date is None or idx.date() != last_session_end_date):
                session_end_timestamp = idx.replace(hour=15, minute=30, second=0, microsecond=0)
                self.sim_logger.info(f"Processing session end closure at {session_end_timestamp}")
                self.trade_executor.close_open_trades_at_session_end(
                    session_end_time=session_end_timestamp,
                    index_price=row['close'],
                    strategy_name=self.strategy_name,
                    timeframe=self.timeframe
                )
                if position is not None:
                    exit_reason = 'session_end'
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) if position == 'long' else (entry_price - exit_price)
                    trade = {
                        'strategy': self.strategy_name, 'timeframe': self.timeframe, 'symbol': self.symbol,
                        'position': position, 'entry_timestamp': entry_timestamp, 'exit_timestamp': idx,
                        'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl,
                        'profit_or_loss': 'Profit' if pnl > 0 else 'Loss', 'exit_reason': exit_reason,
                        'regime': entry_regime, 'volatility': entry_volatility,
                        'session': self.get_session(entry_timestamp), 'day': entry_timestamp.strftime('%A'),
                        'is_expiry': entry_timestamp.strftime('%A') == 'Thursday', 'trade_id': trade_id
                    }
                    self.trades.append(trade)
                    self.sim_logger.info(f"[TRADE EXIT] {position.upper()} | Entry={entry_price:.2f} | Exit={exit_price:.2f} | PnL={pnl:.2f} | Reason={exit_reason} | Time={idx} | Result={'Profit' if pnl > 0 else 'Loss'}")
                    exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1

                    # Update stats
                    day_name = idx.strftime('%A')
                    session = self.get_session(idx)
                    is_profitable = pnl > 0
                    day_wise[day_name]['total'] += 1
                    day_wise[day_name]['profitable' if is_profitable else 'losing'] += 1
                    day_wise[day_name]['accuracy'] = (day_wise[day_name]['profitable'] / day_wise[day_name]['total'] * 100) if day_wise[day_name]['total'] > 0 else 0.0
                    if session in session_wise:
                        session_wise[session]['total'] += 1
                        session_wise[session]['profitable' if is_profitable else 'losing'] += 1
                        session_wise[session]['accuracy'] = (session_wise[session]['profitable'] / session_wise[session]['total'] * 100) if session_wise[session]['total'] > 0 else 0.0
                    regime_wise[entry_regime]['total'] += 1
                    regime_wise[entry_regime]['profitable' if is_profitable else 'losing'] += 1
                    regime_wise[entry_regime]['accuracy'] = (regime_wise[entry_regime]['profitable'] / regime_wise[entry_regime]['total'] * 100) if regime_wise[entry_regime]['total'] > 0 else 0.0
                    volatility_wise[entry_volatility]['total'] += 1
                    volatility_wise[entry_volatility]['profitable' if is_profitable else 'losing'] += 1
                    volatility_wise[entry_volatility]['accuracy'] = (volatility_wise[entry_volatility]['profitable'] / volatility_wise[entry_volatility]['total'] * 100) if volatility_wise[entry_volatility]['total'] > 0 else 0.0
                    if entry_timestamp.strftime('%A') == 'Thursday':
                        expiry_wise['Expiry Thursday']['total'] += 1
                        expiry_wise['Expiry Thursday']['profitable' if is_profitable else 'losing'] += 1
                        expiry_wise['Expiry Thursday']['accuracy'] = (expiry_wise['Expiry Thursday']['profitable'] / expiry_wise['Expiry Thursday']['total'] * 100) if expiry_wise['Expiry Thursday']['total'] > 0 else 0.0

                    position = None
                    trade_id = None
                last_session_end_date = idx.date()

            result = self.strategy_func(row, self.df[:idx], self.params)
            signal = result['signal']

            mapped_signal = 'none'
            if signal == 1 or signal == 'buy_potential':
                mapped_signal = 'buy_potential'
            elif signal == -1 or signal == 'sell_potential':
                mapped_signal = 'sell_potential'
            signal_count[mapped_signal] += 1

            sl_val = result.get('sl', None)
            tp_val = result.get('tp', None)
            tsl_val = result.get('tsl', None)

            if isinstance(sl_val, pd.Series): sl_val = sl_val.iloc[-1] if not sl_val.empty else np.nan
            if isinstance(tp_val, pd.Series): tp_val = tp_val.iloc[-1] if not tp_val.empty else np.nan
            if isinstance(tsl_val, pd.Series): tsl_val = tsl_val.iloc[-1] if not tsl_val.empty else np.nan

            atr = row['atr_14']

            # Exit Logic
            if position:
                high, low, close = row['high'], row['low'], row['close']
                exit_reason, exit_price, pnl = None, None, None

                if position == 'long':
                    new_tsl = max(tsl_price, close - self.params['tsl_atr_mult'] * atr)
                    if low <= sl_price:
                        pnl = sl_price - entry_price
                        exit_reason = 'sl'
                        exit_price = sl_price
                    elif low <= new_tsl:
                        pnl = new_tsl - entry_price
                        exit_reason = 'tsl'
                        exit_price = new_tsl
                    elif high >= tp_price:
                        pnl = tp_price - entry_price
                        exit_reason = 'tp'
                        exit_price = tp_price
                    elif mapped_signal == 'sell_potential':
                        pnl = close - entry_price
                        exit_reason = 'signal'
                        exit_price = close
                    else:
                        continue
                else:  # short
                    new_tsl = min(tsl_price, close + self.params['tsl_atr_mult'] * atr)
                    if high >= sl_price:
                        pnl = entry_price - sl_price
                        exit_reason = 'sl'
                        exit_price = sl_price
                    elif high >= new_tsl:
                        pnl = entry_price - new_tsl
                        exit_reason = 'tsl'
                        exit_price = new_tsl
                    elif low <= tp_price:
                        pnl = entry_price - tp_price
                        exit_reason = 'tp'
                        exit_price = tp_price
                    elif mapped_signal == 'buy_potential':
                        pnl = entry_price - close
                        exit_reason = 'signal'
                        exit_price = close
                    else:
                        continue

                trade = {
                    'strategy': self.strategy_name, 'timeframe': self.timeframe, 'symbol': self.symbol,
                    'position': position, 'entry_timestamp': entry_timestamp, 'exit_timestamp': idx,
                    'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl,
                    'profit_or_loss': 'Profit' if pnl > 0 else 'Loss', 'exit_reason': exit_reason,
                    'regime': entry_regime, 'volatility': entry_volatility,
                    'session': self.get_session(entry_timestamp), 'day': entry_timestamp.strftime('%A'),
                    'is_expiry': entry_timestamp.strftime('%A') == 'Thursday', 'trade_id': trade_id
                }
                self.trades.append(trade)
                self.sim_logger.info(f"[TRADE EXIT] {position.upper()} | Entry={entry_price:.2f} | Exit={exit_price:.2f} | PnL={pnl:.2f} | Reason={exit_reason} | Time={idx} | Result={'Profit' if pnl > 0 else 'Loss'}")
                self.trade_executor.exit_option_trade(
                    trade_id=trade_id,
                    timestamp=idx,
                    index_price=close,
                    exit_reason=exit_reason,
                    strategy_name=self.strategy_name,
                    timeframe=self.timeframe
                )

                day_name = idx.strftime('%A')
                session = self.get_session(idx)
                is_profitable = pnl > 0
                exit_reasons[exit_reason] += 1
                day_wise[day_name]['total'] += 1
                day_wise[day_name]['profitable' if is_profitable else 'losing'] += 1
                day_wise[day_name]['accuracy'] = (day_wise[day_name]['profitable'] / day_wise[day_name]['total'] * 100) if day_wise[day_name]['total'] > 0 else 0.0
                if session in session_wise:
                    session_wise[session]['total'] += 1
                    session_wise[session]['profitable' if is_profitable else 'losing'] += 1
                    session_wise[session]['accuracy'] = (session_wise[session]['profitable'] / session_wise[session]['total'] * 100) if session_wise[session]['total'] > 0 else 0.0
                regime_wise[entry_regime]['total'] += 1
                regime_wise[entry_regime]['profitable' if is_profitable else 'losing'] += 1
                regime_wise[entry_regime]['accuracy'] = (regime_wise[entry_regime]['profitable'] / regime_wise[entry_regime]['total'] * 100) if regime_wise[entry_regime]['total'] > 0 else 0.0
                volatility_wise[entry_volatility]['total'] += 1
                volatility_wise[entry_volatility]['profitable' if is_profitable else 'losing'] += 1
                volatility_wise[entry_volatility]['accuracy'] = (volatility_wise[entry_volatility]['profitable'] / volatility_wise[entry_volatility]['total'] * 100) if volatility_wise[entry_volatility]['total'] > 0 else 0.0
                if entry_timestamp.strftime('%A') == 'Thursday':
                    expiry_wise['Expiry Thursday']['total'] += 1
                    expiry_wise['Expiry Thursday']['profitable' if is_profitable else 'losing'] += 1
                    expiry_wise['Expiry Thursday']['accuracy'] = (expiry_wise['Expiry Thursday']['profitable'] / expiry_wise['Expiry Thursday']['total'] * 100) if expiry_wise['Expiry Thursday']['total'] > 0 else 0.0

                position = None
                trade_id = None

            # Entry Logic
            can_enter = position is None
            if same_day_exit:
                can_enter = can_enter and current_bar_time < latest_entry_time
                if not can_enter and mapped_signal != 'none':
                    self.sim_logger.info(f"Skipping entry at {idx}: Time={current_bar_time} is after {latest_entry_time} with same_day_exit=True")

            if can_enter:
                if mapped_signal == 'buy_potential':
                    position = 'long'
                    entry_price = round(row['close'], 2)
                    sl_price = round(sl_val if sl_val is not None else entry_price - self.params['sl_atr_mult'] * atr, 2)
                    tp_price = round(tp_val if tp_val is not None else entry_price + self.params['tp_atr_mult'] * atr, 2)
                    tsl_price = round(tsl_val if tsl_val is not None else entry_price - self.params['tsl_atr_mult'] * atr, 2)
                    entry_timestamp = idx
                    entry_regime = self.get_regime(row)
                    entry_volatility = self.get_volatility(atr)
                    buy_trades += 1
                    trade_id_counter += 1
                    trade_id = f"TRADE_{trade_id_counter}"
                    self.sim_logger.info(f"[TRADE ENTRY] LONG | ID={trade_id} | Entry={entry_price} | SL={sl_price} | TP={tp_price} | TSL={tsl_price} | Time={entry_timestamp} | Regime={entry_regime} | Volatility={entry_volatility}")
                    self.trade_executor.enter_option_trade('buy_potential', idx, row['close'], trade_id=trade_id)
                elif mapped_signal == 'sell_potential':
                    position = 'short'
                    entry_price = round(row['close'], 2)
                    sl_price = round(sl_val if sl_val is not None else entry_price + self.params['sl_atr_mult'] * atr, 2)
                    tp_price = round(tp_val if tp_val is not None else entry_price - self.params['tp_atr_mult'] * atr, 2)
                    tsl_price = round(tsl_val if tsl_val is not None else entry_price + self.params['tsl_atr_mult'] * atr, 2)
                    entry_timestamp = idx
                    entry_regime = self.get_regime(row)
                    entry_volatility = self.get_volatility(atr)
                    sell_trades += 1
                    trade_id_counter += 1
                    trade_id = f"TRADE_{trade_id_counter}"
                    self.sim_logger.info(f"[TRADE ENTRY] SHORT | ID={trade_id} | Entry={entry_price} | SL={sl_price} | TP={tp_price} | TSL={tsl_price} | Time={entry_timestamp} | Regime={entry_regime} | Volatility={entry_volatility}")
                    self.trade_executor.enter_option_trade('sell_potential', idx, row['close'], trade_id=trade_id)

        # Final safeguard for open trades at dataset end
        if position is not None:
            final_idx = self.df.index[-1]
            final_close = self.df.iloc[-1]['close']
            exit_reason = 'session_end'
            pnl = (final_close - entry_price) if position == 'long' else (entry_price - final_close)
            self.sim_logger.info(f"Forcing final session end exit for open {position} trade at {final_idx}")
            trade = {
                'strategy': self.strategy_name, 'timeframe': self.timeframe, 'symbol': self.symbol,
                'position': position, 'entry_timestamp': entry_timestamp, 'exit_timestamp': final_idx,
                'entry_price': entry_price, 'exit_price': final_close, 'pnl': pnl,
                'profit_or_loss': 'Profit' if pnl > 0 else 'Loss', 'exit_reason': exit_reason,
                'regime': entry_regime, 'volatility': entry_volatility,
                'session': self.get_session(entry_timestamp), 'day': entry_timestamp.strftime('%A'),
                'is_expiry': entry_timestamp.strftime('%A') == 'Thursday', 'trade_id': trade_id
            }
            self.trades.append(trade)
            exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
            self.trade_executor.exit_option_trade(
                trade_id=trade_id,
                timestamp=final_idx,
                index_price=final_close,
                exit_reason=exit_reason,
                strategy_name=self.strategy_name,
                timeframe=self.timeframe
            )

        total_pnl = sum(t['pnl'] for t in self.trades)
        total_trades = len(self.trades)
        profitable_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in self.trades if t['pnl'] <= 0)
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        performance_score = total_pnl * (win_rate / 100) if total_trades > 0 else 0

        result = {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'performance_score': performance_score,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'exit_reasons': exit_reasons,
            'day_wise': day_wise,
            'session_wise': session_wise,
            'regime_wise': regime_wise,
            'volatility_wise': volatility_wise,
            'expiry_wise': expiry_wise,
            'trades': self.trades,
            'signal_count': signal_count
        }
        self.sim_logger.info(f"Backtest result for {self.strategy_name}: PnL={result['total_pnl']:.2f}, Trades={result['total_trades']}, WinRate={result['win_rate']:.2f}%")
        return result