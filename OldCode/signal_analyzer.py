from timeit import main
import pandas as pd
import logging
from datetime import datetime, timedelta
import argparse
from collections import defaultdict
import sys
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SignalAnalyzer:
    def __init__(self):
        self.trade_history = []
        self.summary_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_pnl_percent': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'expectancy': 0
        }
        self.entry_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'sl_hit': 0,
            'target_hit': 0,
            'avg_risk_reward': 0
        })
        self.exit_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'avg_pnl': 0
        })
        self.vwap_stats = {
            'above_vwap_entries': 0,
            'below_vwap_entries': 0,
            'above_vwap_success': 0,
            'below_vwap_success': 0
        }
        self.streak_stats = {
            'max_consec_wins': 0,
            'max_consec_losses': 0,
            'current_streak': 0,
            'current_streak_type': None
        }
        self.time_stats = {
            'avg_win_duration': 0,
            'avg_loss_duration': 0,
            'win_durations': [],
            'loss_durations': []
        }

    def analyze_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze trades and add results to DataFrame"""
        df['profit_loss'] = 0.0
        df['pct_change'] = 0.0
        df['trade_outcome'] = ''
        current_trade = None
        running_balance = 0
        peak_balance = 0
        win_pnls = []
        loss_pnls = []

        for i, row in df.iterrows():
            if 'Enter' in str(row['signal_type']):
                signal_type = row['signal_type'].split()[1]
                self.entry_stats[signal_type]['total'] += 1
                
                if pd.notna(row['vwap']):
                    if row['close'] > row['vwap']:
                        self.vwap_stats['above_vwap_entries'] += 1
                    else:
                        self.vwap_stats['below_vwap_entries'] += 1
                
                # Calculate risk/reward ratio
                risk = abs(row['close'] - row['stop_loss'])
                reward = abs(row['target_price'] - row['close'])
                risk_reward = reward / risk if risk > 0 else 0
                
                current_trade = {
                    'entry_index': i,
                    'entry_price': row['close'],
                    'position': signal_type,
                    'stop_loss': row['stop_loss'],
                    'target_price': row['target_price'],
                    'entry_signal': signal_type,
                    'entry_vwap_relation': 'above' if pd.notna(row['vwap']) and row['close'] > row['vwap'] else 'below',
                    'risk_reward': risk_reward,
                    'start_time': i if isinstance(i, datetime) else None
                }
            
            elif current_trade and 'Exit' in str(row['signal_type']):
                if current_trade['position'] == 'Long':
                    pnl = row['close'] - current_trade['entry_price']
                    pct_change = (pnl / current_trade['entry_price']) * 100
                else:
                    pnl = current_trade['entry_price'] - row['close']
                    pct_change = (pnl / current_trade['entry_price']) * 100
                
                outcome = 'Profit' if pnl > 0 else 'Loss'
                exit_type = row['signal_type'].split('(')[1].split(')')[0] if '(' in str(row['signal_type']) else 'Unknown'
                duration = (i - current_trade['entry_index']).total_seconds()/60 if isinstance(i, datetime) else 0
                
                # Update running balance for drawdown calculation
                running_balance += pnl
                peak_balance = max(peak_balance, running_balance)
                drawdown = peak_balance - running_balance
                self.summary_stats['max_drawdown'] = max(self.summary_stats['max_drawdown'], drawdown)
                
                # Track PnLs for expectancy calculation
                if outcome == 'Profit':
                    win_pnls.append(pnl)
                else:
                    loss_pnls.append(abs(pnl))
                
                # Update trade history with additional metrics
                trade_record = {
                    'entry_price': current_trade['entry_price'],
                    'exit_price': row['close'],
                    'pnl': pnl,
                    'pct_change': pct_change,
                    'outcome': outcome,
                    'duration': duration,
                    'exit_type': exit_type,
                    'risk_reward': current_trade['risk_reward']
                }
                self.trade_history.append(trade_record)
                
                # Update all statistics
                self._update_stats(current_trade, outcome, exit_type, duration, pnl)
                
                current_trade = None

        # Calculate advanced metrics after all trades processed
        self._calculate_advanced_metrics(win_pnls, loss_pnls)
        return df

    def _update_stats(self, current_trade, outcome, exit_type, duration, pnl):
        """Update all statistics for a completed trade"""
        # Update summary stats
        self.summary_stats['total_trades'] += 1
        self.summary_stats['total_pnl'] += pnl
        self.summary_stats['total_pnl_percent'] += (pnl / current_trade['entry_price']) * 100
        if outcome == 'Profit':
            self.summary_stats['winning_trades'] += 1
            self.time_stats['win_durations'].append(duration)
        else:
            self.summary_stats['losing_trades'] += 1
            self.time_stats['loss_durations'].append(duration)
        
        # Update entry stats
        if outcome == 'Profit':
            self.entry_stats[current_trade['entry_signal']]['success'] += 1
            if current_trade['entry_vwap_relation'] == 'above':
                self.vwap_stats['above_vwap_success'] += 1
            else:
                self.vwap_stats['below_vwap_success'] += 1
        
        # Update exit stats
        self.exit_stats[exit_type]['total'] += 1
        self.exit_stats[exit_type]['avg_pnl'] += pnl
        if outcome == 'Profit':
            self.exit_stats[exit_type]['success'] += 1
        
        # Update streaks
        if outcome == self.streak_stats['current_streak_type']:
            self.streak_stats['current_streak'] += 1
        else:
            self.streak_stats['current_streak'] = 1
            self.streak_stats['current_streak_type'] = outcome
        
        if outcome == 'Profit':
            self.streak_stats['max_consec_wins'] = max(
                self.streak_stats['max_consec_wins'],
                self.streak_stats['current_streak']
            )
        else:
            self.streak_stats['max_consec_losses'] = max(
                self.streak_stats['max_consec_losses'],
                self.streak_stats['current_streak']
            )

    def _calculate_advanced_metrics(self, win_pnls, loss_pnls):
        """Calculate advanced performance metrics"""
        # Profit factor
        gross_profit = sum(win_pnls) if win_pnls else 0
        gross_loss = sum(loss_pnls) if loss_pnls else 0
        self.summary_stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        win_prob = len(win_pnls) / self.summary_stats['total_trades'] if self.summary_stats['total_trades'] > 0 else 0
        loss_prob = len(loss_pnls) / self.summary_stats['total_trades'] if self.summary_stats['total_trades'] > 0 else 0
        self.summary_stats['expectancy'] = (avg_win * win_prob) - (avg_loss * loss_prob)
        
        # Time stats
        self.time_stats['avg_win_duration'] = np.mean(self.time_stats['win_durations']) if self.time_stats['win_durations'] else 0
        self.time_stats['avg_loss_duration'] = np.mean(self.time_stats['loss_durations']) if self.time_stats['loss_durations'] else 0
        
        # Entry stats risk/reward
        for signal in self.entry_stats:
            trades = [t for t in self.trade_history if t['outcome'] and t['entry_signal'] == signal]
            if trades:
                self.entry_stats[signal]['avg_risk_reward'] = np.mean([t['risk_reward'] for t in trades])
        
        # Exit stats avg pnl
        for exit_type in self.exit_stats:
            if self.exit_stats[exit_type]['total'] > 0:
                self.exit_stats[exit_type]['avg_pnl'] /= self.exit_stats[exit_type]['total']

    def print_summary(self):
        """Print comprehensive performance statistics"""
        # Basic calculations
        win_rate = (self.summary_stats['winning_trades'] / self.summary_stats['total_trades'] * 100 
                  if self.summary_stats['total_trades'] > 0 else 0)
        avg_pnl = (self.summary_stats['total_pnl'] / self.summary_stats['total_trades']
                  if self.summary_stats['total_trades'] > 0 else 0)
        avg_pnl_pct = (self.summary_stats['total_pnl_percent'] / self.summary_stats['total_trades']
                      if self.summary_stats['total_trades'] > 0 else 0)
        
        # Create summary sections
        sections = []
        
        # 1. Core Performance
        sections.extend([
            "\n=== CORE PERFORMANCE ===",
            f"Total Trades: {self.summary_stats['total_trades']}",
            f"Winning Trades: {self.summary_stats['winning_trades']} ({win_rate:.1f}%)",
            f"Losing Trades: {self.summary_stats['losing_trades']}",
            f"Total P&L: {self.summary_stats['total_pnl']:.2f}",
            f"Total P&L %: {self.summary_stats['total_pnl_percent']:.2f}%",
            f"Avg P&L per Trade: {avg_pnl:.2f} ({avg_pnl_pct:.2f}%)",
            f"Profit Factor: {self.summary_stats['profit_factor']:.2f}",
            f"Expectancy: {self.summary_stats['expectancy']:.2f}",
            f"Max Drawdown: {self.summary_stats['max_drawdown']:.2f}"
        ])
        
        # 2. Entry Statistics
        entry_lines = [
            "\n=== ENTRY STATISTICS ===",
            "Signal | Total | Success % | SL % | Target % | Avg R:R",
            "------------------------------------------------------"
        ]
        for signal, stats in self.entry_stats.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            sl_rate = (stats['sl_hit'] / stats['total'] * 100) if stats['total'] > 0 else 0
            target_rate = (stats['target_hit'] / stats['total'] * 100) if stats['total'] > 0 else 0
            entry_lines.append(
                f"{signal:6} | {stats['total']:5} | {success_rate:8.1f}% | {sl_rate:4.1f}% | {target_rate:6.1f}% | {stats['avg_risk_reward']:6.2f}:1"
            )
        sections.extend(entry_lines)
        
        # 3. Exit Statistics
        exit_lines = [
            "\n=== EXIT STATISTICS ===",
            "Exit Type     | Total | Success % | Avg P&L",
            "-------------------------------------------"
        ]
        for exit_type, stats in self.exit_stats.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            exit_lines.append(
                f"{exit_type:12} | {stats['total']:5} | {success_rate:8.1f}% | {stats['avg_pnl']:7.2f}"
            )
        sections.extend(exit_lines)
        
        # 4. VWAP Statistics
        if self.vwap_stats['above_vwap_entries'] > 0 or self.vwap_stats['below_vwap_entries'] > 0:
            above_success = (self.vwap_stats['above_vwap_success'] / self.vwap_stats['above_vwap_entries'] * 100
                            if self.vwap_stats['above_vwap_entries'] > 0 else 0)
            below_success = (self.vwap_stats['below_vwap_success'] / self.vwap_stats['below_vwap_entries'] * 100
                            if self.vwap_stats['below_vwap_entries'] > 0 else 0)
            sections.extend([
                "\n=== VWAP STATISTICS ===",
                f"Entries Above VWAP: {self.vwap_stats['above_vwap_entries']} ({above_success:.1f}% success)",
                f"Entries Below VWAP: {self.vwap_stats['below_vwap_entries']} ({below_success:.1f}% success)"
            ])
        
        # 5. Time Statistics
        if self.time_stats['win_durations'] or self.time_stats['loss_durations']:
            sections.extend([
                "\n=== TIME STATISTICS ===",
                f"Avg Win Duration: {self.time_stats['avg_win_duration']:.1f} min",
                f"Avg Loss Duration: {self.time_stats['avg_loss_duration']:.1f} min",
                f"Win/Loss Duration Ratio: {self.time_stats['avg_win_duration']/self.time_stats['avg_loss_duration']:.2f}" 
                if self.time_stats['avg_loss_duration'] else "N/A"
            ])
        
        # 6. Streak Statistics
        sections.extend([
            "\n=== STREAK STATISTICS ===",
            f"Max Consecutive Wins: {self.streak_stats['max_consec_wins']}",
            f"Max Consecutive Losses: {self.streak_stats['max_consec_losses']}"
        ])
        
        # 7. Strategy Health
        health_score = min(10, max(0, 
            (win_rate * 0.3) + 
            (min(self.summary_stats['profit_factor'], 3) * 2.5 + 
            (1 - (self.time_stats['avg_loss_duration']/self.time_stats['avg_win_duration'] 
                 if self.time_stats['avg_win_duration'] else 0) * 2
        ))))
        sections.extend([
            "\n=== STRATEGY HEALTH ===",
            f"Health Score: {health_score:.1f}/10",
            "⚠️ Warning: Sample size too small (<30 trades)" if self.summary_stats['total_trades'] < 30 else "✅ Good sample size"
        ])
        
        # Print all sections
        print("\n".join(sections))
        logger.info("\n".join(sections))

    def main(input_file: str, output_file: str):
        """Main analysis function with forced output"""
        try:
            logger.info(f"Loading data from {input_file}")
            df = pd.read_csv(input_file, parse_dates=['datetime'], index_col='datetime')
            
            analyzer = SignalAnalyzer()
            analyzed_df = analyzer.analyze_trades(df)
            
            logger.info(f"Saving results to {output_file}")
            analyzed_df.to_csv(output_file)
            
            analyzer.print_summary()
            logger.info("Analysis completed")
            return True
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return False

if __name__ == "__main__":
        # Force unbuffered output
        sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
        
        parser = argparse.ArgumentParser(description='Analyze trading signals')
        parser.add_argument("--input", required=True, help="Input CSV file path")
        parser.add_argument("--output", required=True, help="Output CSV file path")
        args = parser.parse_args()
        
        success = main(args.input, args.output)
        sys.exit(0 if success else 1)
   