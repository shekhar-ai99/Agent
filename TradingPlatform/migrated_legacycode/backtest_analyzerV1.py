# # backtest_analyzer.py
# import pandas as pd
# import numpy as np
# import logging
# from pathlib import Path
# from datetime import datetime
# import matplotlib.pyplot as plt
# from typing import Dict, Optional, List, Any

# # Use the same logger configuration or define a new one
# logger = logging.getLogger(__name__) # Assumes logger is configured elsewhere if run standalone

# class BacktestAnalyzerReporter:
#     """
#     Analyzes the results DataFrame from EnhancedMultiStrategyBacktester,
#     calculates performance metrics, and generates reports (CSV, plots, HTML).
#     """
#     def __init__(self, results_df: pd.DataFrame, strategies: List[str], initial_capital: float = 100000):
#         """
#         Initialize the analyzer/reporter.

#         Args:
#             results_df: The DataFrame output from EnhancedMultiStrategyBacktester.run_backtest.
#             strategies: A list of strategy names that were run (keys from the strategies dict).
#             initial_capital: Initial capital (can be used for scaling metrics if needed).
#         """
#         if results_df is None or results_df.empty:
#             raise ValueError("Input results_df cannot be None or empty.")
#         if not strategies:
#             raise ValueError("Strategies list cannot be empty.")

#         self.results_df = results_df
#         self.strategies = strategies
#         self.initial_capital = initial_capital # Store for potential future use
#         self.performance_metrics = None # Calculated by analyze_all_strategies
#         self.trades_dict = {} # Store extracted trades per strategy

#         # Analyze results upon initialization
#         self.analyze_all_strategies()

#     def analyze_all_strategies(self) -> Optional[Dict[str, Dict[str, Any]]]:
#         """Calculate performance metrics for all strategies."""
#         logger.info("Calculating performance metrics for all strategies...")
#         metrics = {}
#         self.trades_dict = {} # Reset trades

#         for name in self.strategies:
#             logger.debug(f"Analyzing strategy: {name}")
#             try:
#                 # Extract trades first
#                 trades_df = self._extract_trades(name)
#                 self.trades_dict[name] = trades_df # Store for potential later use

#                 if trades_df.empty:
#                     logger.warning(f"No trades extracted for strategy '{name}'. Skipping metrics.")
#                     metrics[name] = {'total_trades': 0}
#                     continue

#                 # Use PnL points for calculations
#                 pnl_points = trades_df['pnl_points']
#                 wins = pnl_points[pnl_points > 0]
#                 losses = pnl_points[pnl_points <= 0]

#                 # Get equity curve (cumulative points PnL) directly from results_df
#                 equity_curve_points = self.results_df[f'{name}_cumulative_pnl_points'].dropna().fillna(0)

#                 total_trades = len(trades_df)
#                 metrics[name] = {
#                     'total_trades': total_trades,
#                     'win_rate': (len(wins) / total_trades * 100) if total_trades > 0 else 0,
#                     'avg_win_points': wins.mean() if not wins.empty else 0,
#                     'avg_loss_points': losses.mean() if not losses.empty else 0,
#                     'total_pnl_points': pnl_points.sum(),
#                     'profit_factor': self._calculate_profit_factor(pnl_points),
#                     'max_drawdown_points': self._calculate_max_drawdown_points(equity_curve_points),
#                     # Add more metrics as desired
#                 }
#                 # Example: Expectancy in points
#                 avg_win = metrics[name]['avg_win_points']
#                 avg_loss = metrics[name]['avg_loss_points'] # Note: avg_loss is typically negative
#                 win_rate_dec = metrics[name]['win_rate'] / 100.0
#                 metrics[name]['expectancy_points'] = (avg_win * win_rate_dec) + (avg_loss * (1 - win_rate_dec)) if total_trades > 0 else 0

#             except KeyError as e:
#                  logger.error(f"Missing expected column for strategy '{name}': {e}. Skipping metrics.", exc_info=True)
#                  metrics[name] = {'total_trades': f'Error - Missing Column {e}'}
#             except Exception as e:
#                 logger.error(f"Failed to calculate metrics for strategy '{name}': {e}", exc_info=True)
#                 metrics[name] = {'total_trades': 'Error'}

#         self.performance_metrics = metrics
#         logger.info("Finished calculating performance metrics.")
#         return metrics

#     def _extract_trades(self, strategy_name: str) -> pd.DataFrame:
#         """Extract completed trades for a strategy, using trade_id."""
#         # (Implementation remains the same as the previous version)
#         df = self.results_df; entry_col=f'{strategy_name}_entry_price'; exit_col=f'{strategy_name}_exit_price'; id_col=f'{strategy_name}_trade_id'; pnl_col=f'{strategy_name}_pnl_points'; pos_col=f'{strategy_name}_position'; cum_pnl_col=f'{strategy_name}_cumulative_pnl_points'
#         exit_rows = df[df[exit_col].notna()].copy(); trades = []; processed_trade_ids = set()
#         if exit_rows.empty: return pd.DataFrame()
#         for exit_idx, exit_row in exit_rows.iterrows():
#             trade_id = exit_row[id_col];
#             if trade_id in processed_trade_ids or trade_id == 0: continue
#             entry_rows = df[(df[id_col] == trade_id) & df[entry_col].notna()]
#             if entry_rows.empty: logger.warning(f"No entry for trade_id {trade_id} ('{strategy_name}') ending {exit_idx}. Skipping."); continue
#             entry_row = entry_rows.iloc[0]; entry_idx = entry_row.name
#             try: duration_bars = df.index.get_loc(exit_idx) - df.index.get_loc(entry_idx)
#             except KeyError: duration_bars = np.nan # Handle cases where index might not be unique/monotonic if data is unusual
#             trade = {'trade_id': trade_id, 'entry_time': entry_idx, 'exit_time': exit_idx, 'position': entry_row[pos_col], 'entry_price': entry_row[entry_col], 'exit_price': exit_row[exit_col], 'duration_bars': duration_bars, 'pnl_points': exit_row[pnl_col], 'cumulative_pnl_points': exit_row[cum_pnl_col]}
#             trades.append(trade); processed_trade_ids.add(trade_id)
#         return pd.DataFrame(trades)


#     def _calculate_profit_factor(self, pnl_points: pd.Series) -> float:
#         """Calculate profit factor (gross wins / gross losses in points)"""
#         # (Implementation remains the same as the previous version)
#         wins = pnl_points[pnl_points > 0].sum(); losses = abs(pnl_points[pnl_points <= 0].sum())
#         if losses == 0: return np.inf if wins > 0 else 1.0
#         return wins / losses

#     def _calculate_max_drawdown_points(self, cumulative_pnl_points: pd.Series) -> float:
#         """Calculate maximum drawdown from peak in points."""
#         # (Implementation remains the same as the previous version)
#         if cumulative_pnl_points.empty or cumulative_pnl_points.isnull().all(): return 0.0
#         # Ensure starts from 0 if first value is NaN or positive
#         equity = cumulative_pnl_points.fillna(0)
#         if equity.iloc[0] > 0: equity = pd.concat([pd.Series([0], index=[equity.index[0] - pd.Timedelta(seconds=1)]), equity]) # Prepend 0
#         peak = equity.cummax()
#         drawdown = peak - equity
#         return drawdown.max()

#     # --- Reporting ---
#     def generate_full_report(self, output_dir: str = 'analysis_report') -> Optional[Dict[str, Path]]:
#         """Generate all report components: CSVs, plots, and HTML summary."""
#         if self.results_df is None:
#              logger.error("Cannot generate report: Backtest results DataFrame is missing.")
#              return None
#         if self.performance_metrics is None:
#              logger.warning("Performance metrics not calculated. Analyzing strategies first.")
#              self.analyze_all_strategies()
#              if self.performance_metrics is None: # Check again
#                   logger.error("Cannot generate report: Metric calculation failed.")
#                   return None

#         logger.info(f"Generating full analysis report in directory: {output_dir}")
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
#         report_files = {}

#         try:
#             # --- 1. Save Data ---
#             # Detailed results (already includes signals, prices, pnl etc.)
#             results_detail_path = output_path / 'backtest_results_detailed.csv'
#             self.results_df.to_csv(results_detail_path)
#             report_files['results_detailed_csv'] = results_detail_path
#             logger.info(f"Saved detailed backtest results to {results_detail_path}")

#             # Performance metrics summary
#             metrics_summary_path = output_path / 'performance_metrics_summary.csv'
#             pd.DataFrame(self.performance_metrics).T.to_csv(metrics_summary_path)
#             report_files['metrics_summary_csv'] = metrics_summary_path
#             logger.info(f"Saved performance metrics summary to {metrics_summary_path}")

#             # Extracted trades per strategy (optional, but useful for deep dive)
#             for name, trades_df in self.trades_dict.items():
#                 if not trades_df.empty:
#                      trade_list_path = output_path / f'{name}_tradelist.csv'
#                      trades_df.to_csv(trade_list_path, index=False)
#                      report_files[f'{name}_tradelist_csv'] = trade_list_path
#             logger.info("Saved individual strategy trade lists.")

#             # --- 2. Generate Visualizations ---
#             logger.info("Generating equity curve plots...")
#             equity_plot_path = self._plot_equity_curves(output_path)
#             if equity_plot_path: report_files['equity_plot'] = equity_plot_path

#             logger.info("Generating trade analysis plots...")
#             trade_plot_paths = self._plot_trade_analysis(output_path)
#             report_files.update(trade_plot_paths) # Add individual plot paths

#             # --- 3. Generate HTML Report ---
#             logger.info("Generating HTML summary report...")
#             html_report_content = self._render_html_report()
#             html_path = output_path / 'backtest_summary_report.html'
#             with open(html_path, 'w', encoding='utf-8') as f:
#                 f.write(html_report_content)
#             report_files['html_report'] = html_path
#             logger.info(f"HTML report saved to {html_path}")

#             logger.info(f"Report generation successful in {output_path}")
#             return report_files

#         except Exception as e:
#             logger.error(f"Failed to generate full report: {e}", exc_info=True)
#             return None


#     def _plot_equity_curves(self, output_path: Path) -> Optional[Path]:
#         """Plot equity curves (cumulative PnL points) for all strategies"""
#         # (Implementation remains the same as the previous version)
#         try:
#             fig, ax = plt.subplots(figsize=(14, 7)); has_data = False
#             for name in self.strategies:
#                 cum_pnl_col = f'{name}_cumulative_pnl_points'
#                 if cum_pnl_col in self.results_df.columns:
#                     equity = self.results_df[cum_pnl_col].ffill().fillna(0)
#                     if not equity.empty: ax.plot(equity.index, equity, label=name, linewidth=1.5); has_data = True
#             if not has_data: logger.warning("No PnL data for equity curves."); plt.close(fig); return None
#             ax.set_title('Strategy Equity Curves (Cumulative PnL Points)'); ax.set_xlabel('Date'); ax.set_ylabel('Cumulative PnL (Points)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
#             equity_path = output_path / 'equity_curves.png'; plt.savefig(equity_path, dpi=150); plt.close(fig); logger.info(f"Saved equity plot: {equity_path}"); return equity_path
#         except Exception as e: logger.error(f"Equity plot failed: {e}", exc_info=True); plt.close(); return None


#     def _plot_trade_analysis(self, output_path: Path) -> Dict[str, Path]:
#         """Generate PnL distribution and cumulative PnL plots per strategy."""
#         # (Implementation remains the same as the previous version)
#         plot_paths = {}
#         for name in self.strategies:
#             try:
#                 trades_df = self.trades_dict.get(name) # Get trades from stored dict
#                 if trades_df is None or trades_df.empty: continue
#                 # PnL Dist
#                 fig1, ax1 = plt.subplots(figsize=(10, 5)); ax1.hist(trades_df['pnl_points'], bins=30, edgecolor='black'); ax1.set_title(f'{name} - Trade PnL (Points) Distribution'); ax1.set_xlabel('PnL (Points)'); ax1.set_ylabel('Frequency'); ax1.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); pnl_dist_path = output_path / f'{name}_pnl_distribution.png'; plt.savefig(pnl_dist_path, dpi=100); plt.close(fig1); plot_paths[f'{name}_pnl_distribution_plot'] = pnl_dist_path
#                 # Cum PnL
#                 fig2, ax2 = plt.subplots(figsize=(12, 5)); ax2.plot(trades_df['exit_time'], trades_df['cumulative_pnl_points'], marker='.', linestyle='-', markersize=4); ax2.set_title(f'{name} - Cumulative PnL (Points) Over Time'); ax2.set_xlabel('Exit Date'); ax2.set_ylabel('Cumulative PnL (Points)'); ax2.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); cum_pnl_path = output_path / f'{name}_cumulative_pnl_plot.png'; plt.savefig(cum_pnl_path, dpi=100); plt.close(fig2); plot_paths[f'{name}_cumulative_pnl_plot'] = cum_pnl_path
#                 logger.info(f"Generated trade plots for '{name}'.")
#             except Exception as e: logger.error(f"Trade plot failed for '{name}': {e}", exc_info=True); plt.close() # Close any open figures on error
#         return plot_paths


#     def _render_html_report(self) -> str:
#         """Generate comprehensive HTML report string."""
#         # (Implementation remains the same as the previous version)
#         if not self.performance_metrics: return "<html><body><h2>Error: No metrics.</h2></body></html>"
#         try:
#             metrics_df = pd.DataFrame(self.performance_metrics).T.fillna(0); metrics_to_format = {'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}', 'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}', 'max_drawdown_points': '{:,.2f}'}; metrics_display_df = metrics_df.copy()
#             for col, fmt in metrics_to_format.items():
#                  if col in metrics_display_df.columns: metrics_display_df[col] = metrics_display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
#             if 'total_trades' in metrics_display_df.columns: metrics_display_df['total_trades'] = metrics_display_df['total_trades'].astype(int)
#             metrics_html = metrics_display_df.to_html(classes='performance-table', border=1, justify='right')
#         except Exception as e: logger.error(f"HTML metrics table error: {e}"); metrics_html = "<p>Metrics table error.</p>"
#         html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Multi-Strategy Backtest Report</title><style>body{{font-family:'Segoe UI',sans-serif;margin:20px;background-color:#f4f4f4;color:#333}}.container{{max-width:1200px;margin:auto;background-color:#fff;padding:25px;box-shadow:0 0 10px rgba(0,0,0,0.1);border-radius:8px}}h1,h2,h3{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:5px;margin-top:30px}}h1{{text-align:center;margin-bottom:20px}}table.performance-table{{border-collapse:collapse;width:100%;margin-bottom:25px;font-size:0.9em}}th,td{{border:1px solid #ddd;padding:10px;text-align:right}}th{{background-color:#3498db;color:white;text-align:center;font-weight:bold}}tr:nth-child(even){{background-color:#f9f9f9}}.metric-card{{border:1px solid #e0e0e0;border-radius:6px;padding:15px;margin-bottom:20px;background-color:#fdfdfd;box-shadow:0 1px 3px rgba(0,0,0,0.05)}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px}}img{{max-width:95%;height:auto;border-radius:4px;margin-top:10px;border:1px solid #eee;display:block;margin-left:auto;margin-right:auto}}.positive{{color:#27ae60;font-weight:bold}}.negative{{color:#c0392b;font-weight:bold}}.neutral{{color:#7f8c8d}}.timestamp{{text-align:center;color:#7f8c8d;margin-bottom:30px;font-size:0.9em}}details>summary{{cursor:pointer;font-weight:bold;color:#3498db}}</style></head><body><div class="container"><h1>Multi-Strategy Backtest Report</h1><p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p><h2>Performance Summary</h2><div class="metric-card">{metrics_html}<p style="font-size:0.8em;color:#555;">Note: PnL & Drawdown in points.</p></div><h2>Equity Curves (Cumulative PnL Points)</h2><div class="metric-card"><img src="equity_curves.png" alt="Equity Curves Comparison"></div><h2>Individual Strategy Analysis</h2><div class="metric-grid">"""
#         for name in self.strategies:
#             if name not in self.performance_metrics or self.performance_metrics[name]['total_trades']==0 or isinstance(self.performance_metrics[name]['total_trades'], str): html += f"""<div class="metric-card"><h3>{name}</h3><p>No trades or error.</p></div>"""; continue
#             metrics=self.performance_metrics[name]; win_rate_str=f"{metrics.get('win_rate',0):.2f}%"; pf_str=f"{metrics.get('profit_factor',0):.2f}"; pnl_str=f"{metrics.get('total_pnl_points',0):,.2f}"
#             html+=f"""<div class="metric-card"><h3>{name}</h3><p><strong>Total Trades:</strong> {metrics.get('total_trades',0)}</p><p><strong>Win Rate:</strong> {win_rate_str}</p><p><strong>Profit Factor:</strong> {pf_str}</p><p><strong>Total PnL (Points):</strong> {pnl_str}</p><details><summary>Show Plots ({name})</summary><img src="{name}_pnl_distribution.png" alt="{name} PnL Dist"><img src="{name}_cumulative_pnl_plot.png" alt="{name} Cum PnL"></details></div>"""
#         html += """</div></div></body></html>"""
#         return html


# backtest_analyzer.py (with enhanced reporting)
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional, List, Any, Tuple
import io # Required for embedding plots
import base64 # Required for embedding plots

log_file = 'backtest_analyser.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger() # Get root logger
logger.info(f"Logging setup complete. Log file: {log_file}")

class BacktestAnalyzerReporter:
    """
    Analyzes results from EnhancedMultiStrategyBacktester, calculates metrics,
    and generates reports with embedded plots and best strategy identification.
    """
    def __init__(self, results_df: pd.DataFrame, strategies: List[str], initial_capital: float = 100000):
        """
        Initialize the analyzer/reporter.

        Args:
            results_df: The DataFrame output from EnhancedMultiStrategyBacktester.run_backtest.
            strategies: A list of strategy names that were run.
            initial_capital: Initial capital (can be used for scaling metrics if needed).
        """
        if results_df is None or results_df.empty:
            raise ValueError("Input results_df cannot be None or empty.")
        if not strategies:
            raise ValueError("Strategies list cannot be empty.")

        self.results_df = results_df
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.performance_metrics = None
        self.trades_dict = {}
        self.best_strategy_info = None # To store info about the best strategy

        # Analyze results upon initialization
        self.analyze_all_strategies()
        self._identify_best_strategy() # Identify best strategy after analysis

    def analyze_all_strategies(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Calculate performance metrics for all strategies."""
        # (Implementation remains the same as the previous version)
        logger.info("Calculating performance metrics for all strategies...")
        metrics = {}
        self.trades_dict = {}
        for name in self.strategies:
            logger.debug(f"Analyzing strategy: {name}")
            try:
                trades_df = self._extract_trades(name); self.trades_dict[name] = trades_df
                if trades_df.empty: metrics[name] = {'total_trades': 0}; continue
                pnl_points = trades_df['pnl_points']; wins = pnl_points[pnl_points > 0]; losses = pnl_points[pnl_points <= 0]
                equity_curve_points = self.results_df[f'{name}_cumulative_pnl_points'].dropna().fillna(0)
                total_trades = len(trades_df)
                metrics[name] = {
                    'total_trades': total_trades,
                    'win_rate': (len(wins) / total_trades * 100) if total_trades > 0 else 0,
                    'avg_win_points': wins.mean() if not wins.empty else 0,
                    'avg_loss_points': losses.mean() if not losses.empty else 0,
                    'total_pnl_points': pnl_points.sum(),
                    'profit_factor': self._calculate_profit_factor(pnl_points),
                    'max_drawdown_points': self._calculate_max_drawdown_points(equity_curve_points),
                }
                avg_win = metrics[name]['avg_win_points']; avg_loss = metrics[name]['avg_loss_points']
                win_rate_dec = metrics[name]['win_rate'] / 100.0
                metrics[name]['expectancy_points'] = (avg_win * win_rate_dec) + (avg_loss * (1 - win_rate_dec)) if total_trades > 0 else 0
            except KeyError as e: logger.error(f"Missing column for '{name}': {e}.", exc_info=True); metrics[name] = {'total_trades': f'Error - Missing Column {e}'}
            except Exception as e: logger.error(f"Metrics failed for '{name}': {e}", exc_info=True); metrics[name] = {'total_trades': 'Error'}
        self.performance_metrics = metrics; logger.info("Finished calculating metrics."); return metrics

    def _extract_trades(self, strategy_name: str) -> pd.DataFrame:
        """Extract completed trades for a strategy, using trade_id."""
        # (Implementation remains the same as the previous version)
        df = self.results_df; entry_col=f'{strategy_name}_entry_price'; exit_col=f'{strategy_name}_exit_price'; id_col=f'{strategy_name}_trade_id'; pnl_col=f'{strategy_name}_pnl_points'; pos_col=f'{strategy_name}_position'; cum_pnl_col=f'{strategy_name}_cumulative_pnl_points'
        exit_rows = df[df[exit_col].notna()].copy(); trades = []; processed_trade_ids = set()
        if exit_rows.empty: return pd.DataFrame()
        for exit_idx, exit_row in exit_rows.iterrows():
            trade_id = exit_row[id_col];
            if trade_id in processed_trade_ids or trade_id == 0: continue
            entry_rows = df[(df[id_col] == trade_id) & df[entry_col].notna()]
            if entry_rows.empty: logger.warning(f"No entry for trade_id {trade_id} ('{strategy_name}') ending {exit_idx}. Skipping."); continue
            entry_row = entry_rows.iloc[0]; entry_idx = entry_row.name
            try: duration_bars = df.index.get_loc(exit_idx) - df.index.get_loc(entry_idx)
            except KeyError: duration_bars = np.nan
            trade = {'trade_id': trade_id, 'entry_time': entry_idx, 'exit_time': exit_idx, 'position': entry_row[pos_col], 'entry_price': entry_row[entry_col], 'exit_price': exit_row[exit_col], 'duration_bars': duration_bars, 'pnl_points': exit_row[pnl_col], 'cumulative_pnl_points': exit_row[cum_pnl_col]}
            trades.append(trade); processed_trade_ids.add(trade_id)
        return pd.DataFrame(trades)

    def _calculate_profit_factor(self, pnl_points: pd.Series) -> float:
        """Calculate profit factor (gross wins / gross losses in points)"""
        # (Implementation remains the same as the previous version)
        wins = pnl_points[pnl_points > 0].sum(); losses = abs(pnl_points[pnl_points <= 0].sum())
        if losses == 0: return np.inf if wins > 0 else 1.0
        return wins / losses

    def _calculate_max_drawdown_points(self, cumulative_pnl_points: pd.Series) -> float:
        """Calculate maximum drawdown from peak in points."""
        # (Implementation remains the same as the previous version)
        if cumulative_pnl_points.empty or cumulative_pnl_points.isnull().all(): return 0.0
        equity = cumulative_pnl_points.fillna(0)
        if not equity.empty and equity.iloc[0] > 0: equity = pd.concat([pd.Series([0], index=[equity.index[0] - pd.Timedelta(seconds=1)]), equity])
        peak = equity.cummax(); drawdown = peak - equity; return drawdown.max()

    def _identify_best_strategy(self, metric: str = 'total_pnl_points', min_trades: int = 5) -> None:
        """Identifies the best strategy based on a specified metric."""
        if not self.performance_metrics:
            self.best_strategy_info = None
            return

        best_strategy = None
        best_score = -np.inf # Initialize for maximization

        valid_strategies = {
            name: metrics for name, metrics in self.performance_metrics.items()
            if isinstance(metrics.get('total_trades'), (int, float)) and metrics.get('total_trades', 0) >= min_trades
               and pd.notna(metrics.get(metric))
        }

        if not valid_strategies:
             logger.warning(f"No strategies found with >= {min_trades} trades to determine 'best' based on '{metric}'.")
             # Optional: Fallback to best among all strategies regardless of trade count
             valid_strategies = {name: metrics for name, metrics in self.performance_metrics.items() if pd.notna(metrics.get(metric))}
             if not valid_strategies:
                 self.best_strategy_info = None
                 return


        if metric in ['max_drawdown_points', 'avg_loss_points']: # Metrics where lower is better
            best_score = np.inf
            for name, metrics in valid_strategies.items():
                if metrics[metric] < best_score:
                    best_score = metrics[metric]
                    best_strategy = name
        else: # Metrics where higher is better (default)
            for name, metrics in valid_strategies.items():
                if metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_strategy = name

        if best_strategy:
            self.best_strategy_info = {
                'name': best_strategy,
                'metric': metric,
                'score': best_score
            }
            logger.info(f"Identified best strategy based on '{metric}': {best_strategy} (Score: {best_score:.2f})")
        else:
            self.best_strategy_info = None
            logger.warning(f"Could not determine best strategy based on '{metric}'.")

    # --- Reporting ---
    def generate_full_report(self, output_dir: str = 'analysis_report') -> Optional[Dict[str, Path]]:
        """Generate all report components: CSVs, plots (as base64), and HTML summary."""
        # (Modified to handle plot embedding)
        if self.results_df is None or self.performance_metrics is None:
             logger.error("Cannot generate report: Results or metrics missing.")
             return None

        logger.info(f"Generating full analysis report in directory: {output_dir}")
        output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
        report_files = {}
        plot_data_uris = {} # To store base64 encoded plots for HTML

        try:
            # --- 1. Save Data (CSVs) ---
            results_detail_path = output_path / 'backtest_results_detailed.csv'; self.results_df.to_csv(results_detail_path); report_files['results_detailed_csv'] = results_detail_path; logger.info(f"Saved detailed results to {results_detail_path}")
            metrics_summary_path = output_path / 'performance_metrics_summary.csv'; pd.DataFrame(self.performance_metrics).T.to_csv(metrics_summary_path); report_files['metrics_summary_csv'] = metrics_summary_path; logger.info(f"Saved metrics summary to {metrics_summary_path}")
            for name, trades_df in self.trades_dict.items():
                if not trades_df.empty: trade_list_path = output_path / f'{name}_tradelist.csv'; trades_df.to_csv(trade_list_path, index=False); report_files[f'{name}_tradelist_csv'] = trade_list_path
            logger.info("Saved individual strategy trade lists.")

            # --- 2. Generate Visualizations (as base64 strings) ---
            logger.info("Generating equity curve plot data...")
            plot_data_uris['equity_curves'] = self._plot_to_base64(self._plot_equity_curves)

            logger.info("Generating trade analysis plot data...")
            trade_plot_uris = self._plot_trade_analysis_to_base64()
            plot_data_uris.update(trade_plot_uris)

            # --- 3. Generate HTML Report ---
            logger.info("Generating HTML summary report...")
            html_report_content = self._render_html_report(plot_data_uris) # Pass plot data
            html_path = output_path / 'backtest_summary_report.html';
            with open(html_path, 'w', encoding='utf-8') as f: f.write(html_report_content)
            report_files['html_report'] = html_path; logger.info(f"HTML report saved to {html_path}")

            logger.info(f"Report generation successful in {output_path}")
            return report_files
        except Exception as e: logger.error(f"Failed to generate full report: {e}", exc_info=True); return None

    def _plot_to_base64(self, plot_func: Callable, *args, **kwargs) -> Optional[str]:
         """Executes a plotting function and returns the plot as a base64 encoded string."""
         try:
             buffer = io.BytesIO()
             plot_func(buffer=buffer, *args, **kwargs) # Pass buffer to save fig
             buffer.seek(0)
             image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
             buffer.close()
             return f"data:image/png;base64,{image_base64}"
         except Exception as e:
             logger.error(f"Failed to generate plot {plot_func.__name__} as base64: {e}", exc_info=True)
             return None

    def _plot_equity_curves(self, buffer: io.BytesIO) -> None:
        """Plot equity curves and save to buffer."""
        # (Modified to save to buffer instead of file)
        fig, ax = plt.subplots(figsize=(14, 7)); has_data = False
        for name in self.strategies:
            cum_pnl_col = f'{name}_cumulative_pnl_points'
            if cum_pnl_col in self.results_df.columns:
                equity = self.results_df[cum_pnl_col].ffill().fillna(0)
                if not equity.empty: ax.plot(equity.index, equity, label=name, linewidth=1.5); has_data = True
        if not has_data: logger.warning("No PnL data for equity curves."); plt.close(fig); raise ValueError("No data to plot")
        ax.set_title('Strategy Equity Curves (Cumulative PnL Points)'); ax.set_xlabel('Date'); ax.set_ylabel('Cumulative PnL (Points)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight') # Save to buffer
        plt.close(fig)


    def _plot_trade_analysis_to_base64(self) -> Dict[str, Optional[str]]:
        """Generate PnL distribution and cumulative PnL plots per strategy as base64."""
        # (Modified to return dict of base64 strings)
        plot_data_uris = {}
        for name in self.strategies:
            trades_df = self.trades_dict.get(name)
            if trades_df is None or trades_df.empty: continue

            # PnL Distribution Plot
            plot_data_uris[f'{name}_pnl_distribution'] = self._plot_to_base64(self._plot_pnl_distribution, trades_df, name)
            # Cumulative PnL Plot
            plot_data_uris[f'{name}_cumulative_pnl'] = self._plot_to_base64(self._plot_cumulative_pnl, trades_df, name)

        return plot_data_uris

    def _plot_pnl_distribution(self, trades_df: pd.DataFrame, name: str, buffer: io.BytesIO) -> None:
         """Plots PnL distribution to a buffer."""
         fig, ax = plt.subplots(figsize=(10, 5))
         ax.hist(trades_df['pnl_points'], bins=30, edgecolor='black')
         ax.set_title(f'{name} - Trade PnL (Points) Distribution')
         ax.set_xlabel('PnL (Points)'); ax.set_ylabel('Frequency')
         ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
         plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight')
         plt.close(fig)

    def _plot_cumulative_pnl(self, trades_df: pd.DataFrame, name: str, buffer: io.BytesIO) -> None:
         """Plots Cumulative PnL to a buffer."""
         fig, ax = plt.subplots(figsize=(12, 5))
         ax.plot(trades_df['exit_time'], trades_df['cumulative_pnl_points'], marker='.', linestyle='-', markersize=4)
         ax.set_title(f'{name} - Cumulative PnL (Points) Over Time')
         ax.set_xlabel('Exit Date'); ax.set_ylabel('Cumulative PnL (Points)')
         ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
         plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight')
         plt.close(fig)


    def _render_html_report(self, plot_data_uris: Dict[str, Optional[str]]) -> str:
        """Generate comprehensive HTML report string with embedded plots and best strategy highlight."""
        # (Modified to embed plots and highlight best strategy)
        if not self.performance_metrics: return "<html><body><h2>Error: No metrics.</h2></body></html>"
        try:
            metrics_df = pd.DataFrame(self.performance_metrics).T.fillna(0); metrics_to_format = {'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}', 'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}', 'max_drawdown_points': '{:,.2f}'}; metrics_display_df = metrics_df.copy()
            for col, fmt in metrics_to_format.items():
                 if col in metrics_display_df.columns: metrics_display_df[col] = metrics_display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
            if 'total_trades' in metrics_display_df.columns: metrics_display_df['total_trades'] = metrics_display_df['total_trades'].astype(int)

            # Add highlighting for the best strategy row
            best_strategy_name = self.best_strategy_info['name'] if self.best_strategy_info else None
            def highlight_best(s):
                 return ['background-color: #e0ffe0' if s.name == best_strategy_name else '' for v in s]

            metrics_html = metrics_display_df.style.apply(highlight_best, axis=1).to_html(classes='performance-table', border=1, justify='right')

        except Exception as e: logger.error(f"HTML metrics table error: {e}"); metrics_html = "<p>Metrics table error.</p>"

        # --- Best Strategy Section ---
        best_strategy_html = ""
        if self.best_strategy_info:
             bs_name = self.best_strategy_info['name']
             bs_metric = self.best_strategy_info['metric'].replace('_', ' ').title()
             bs_score = self.best_strategy_info['score']
             score_format = metrics_to_format.get(self.best_strategy_info['metric'], '{:.2f}') # Use specific format if available
             bs_score_str = score_format.format(bs_score) if pd.notna(bs_score) else 'N/A'

             best_strategy_html = f"""
             <h2>Best Performing Strategy</h2>
             <div class="metric-card" style="border-left: 5px solid #27ae60;">
                 <p>Based on maximizing <strong>'{bs_metric}'</strong> (min. 5 trades):</p>
                 <h3>{bs_name}</h3>
                 <p><strong>Score:</strong> {bs_score_str}</p>
                 <p><i>(Refer to the table and individual analysis below for full details.)</i></p>
             </div>
             """

        # --- HTML Structure ---
        # (Updated CSS, added Best Strategy section, embed plots using data URIs)
        html = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Multi-Strategy Backtest Report</title>
        <style>
            body{{font-family:'Segoe UI',sans-serif;margin:20px;background-color:#f4f4f4;color:#333;line-height:1.6;}}
            .container{{max-width:1300px;margin:auto;background-color:#fff;padding:30px;box-shadow:0 2px 15px rgba(0,0,0,0.1);border-radius:8px}}
            h1,h2,h3{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;margin-top:35px;margin-bottom:20px}}
            h1{{text-align:center;margin-bottom:25px;border-bottom:none;}}
            table.performance-table{{border-collapse:collapse;width:100%;margin-bottom:25px;font-size:0.9em;}}
            th,td{{border:1px solid #ddd;padding:10px 12px;text-align:right;}}
            th{{background-color:#3498db;color:white;text-align:center;font-weight:600;}}
            tr:nth-child(even){{background-color:#f9f9f9;}} tr:hover {{background-color: #f1f1f1;}}
            .metric-card{{border:1px solid #e0e0e0;border-radius:6px;padding:20px;margin-bottom:25px;background-color:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
            .metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:25px;}}
            img{{max-width:100%;height:auto;border-radius:4px;margin-top:15px;border:1px solid #eee;display:block;margin-left:auto;margin-right:auto;background-color:#fafafa;}}
            .positive{{color:#27ae60;font-weight:bold;}} .negative{{color:#c0392b;font-weight:bold;}} .neutral{{color:#7f8c8d;}}
            .timestamp{{text-align:center;color:#7f8c8d;margin-bottom:35px;font-size:0.9em;}}
            details{{margin-top:15px;border:1px solid #eee;padding:10px;border-radius:4px;background-color:#f9f9f9;}}
            details>summary{{cursor:pointer;font-weight:bold;color:#3498db;padding:5px;display:inline-block;}}
            details[open]>summary {{ border-bottom: 1px solid #ddd; margin-bottom: 10px; }}
        </style></head><body><div class="container">
            <h1>Multi-Strategy Backtest Report</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {best_strategy_html}
            <h2>Performance Summary</h2><div class="metric-card">{metrics_html}<p style="font-size:0.8em;color:#555;text-align:center;">Note: PnL & Drawdown in points.</p></div>
            <h2>Equity Curves (Cumulative PnL Points)</h2><div class="metric-card"><img src="{plot_data_uris.get('equity_curves', '')}" alt="Equity Curves Comparison"></div>
            <h2>Individual Strategy Analysis</h2><div class="metric-grid">"""
        # Add individual strategy cards
        for name in self.strategies:
            if name not in self.performance_metrics or isinstance(self.performance_metrics[name].get('total_trades'), str) or self.performance_metrics[name].get('total_trades', 0) == 0:
                 html += f"""<div class="metric-card"><h3>{name}</h3><p>No trades or error during analysis.</p></div>"""; continue
            metrics=self.performance_metrics[name]; win_rate_str=f"{metrics.get('win_rate',0):.2f}%"; pf_str=f"{metrics.get('profit_factor',0):.2f}"; pnl_str=f"{metrics.get('total_pnl_points',0):,.2f}"
            # Get plot URIs, default to empty string if plot failed/missing
            pnl_dist_uri = plot_data_uris.get(f'{name}_pnl_distribution', '')
            cum_pnl_uri = plot_data_uris.get(f'{name}_cumulative_pnl', '')
            html+=f"""<div class="metric-card"{' style="border-left: 5px solid #27ae60;"' if name == best_strategy_name else ''}><h3>{name}{' (Best)' if name == best_strategy_name else ''}</h3><p><strong>Total Trades:</strong> {metrics.get('total_trades',0)}</p><p><strong>Win Rate:</strong> {win_rate_str}</p><p><strong>Profit Factor:</strong> {pf_str}</p><p><strong>Total PnL (Points):</strong> {pnl_str}</p>
            <details><summary>Show Plots ({name})</summary>
            {'<img src="' + pnl_dist_uri + '" alt="' + name + ' PnL Dist">' if pnl_dist_uri else '<p>PnL Distribution plot unavailable.</p>'}
            {'<img src="' + cum_pnl_uri + '" alt="' + name + ' Cum PnL">' if cum_pnl_uri else '<p>Cumulative PnL plot unavailable.</p>'}
            </details></div>"""
        html += """</div></div></body></html>"""
        return html