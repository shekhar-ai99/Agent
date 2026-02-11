# backtest_analyzer.py (Corrected for Consolidated Reporting)

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Any, Tuple, Callable
import io
import base64

logger = logging.getLogger(__name__) # Use logger defined in main_workflow or configure here

# Ensure matplotlib backend is suitable for non-interactive use
import matplotlib
matplotlib.use('Agg') # Use Agg backend to prevent GUI issues when run as subprocess


class BacktestAnalyzerReporter:
    """
    Analyzes results DataFrame, calculates performance metrics,
    and provides data/plots for consolidated reporting.
    """
    def __init__(self, results_df: pd.DataFrame, strategies: List[str], initial_capital: float = 100000):
        if results_df is None or results_df.empty: raise ValueError("Input results_df cannot be None or empty.")
        if not strategies: raise ValueError("Strategies list cannot be empty.")
        self.results_df = results_df
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.performance_metrics = None
        self.trades_dict = {}
        self.best_strategy_info = None
        # Analyze results upon initialization
        self.analyze_all_strategies()
        self._identify_best_strategy()

    def analyze_all_strategies(self) -> Optional[Dict[str, Dict[str, Any]]]:
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
            except KeyError as e: logger.error(f"Missing column for '{name}': {e}.", exc_info=False); metrics[name] = {'total_trades': f'Error - Missing Column {e}'}
            except Exception as e: logger.error(f"Metrics failed for '{name}': {e}", exc_info=False); metrics[name] = {'total_trades': 'Error'}
        self.performance_metrics = metrics; logger.info("Finished calculating metrics."); return metrics

    def _extract_trades(self, strategy_name: str) -> pd.DataFrame:
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
        wins = pnl_points[pnl_points > 0].sum(); losses = abs(pnl_points[pnl_points <= 0].sum())
        if losses == 0: return np.inf if wins > 0 else 1.0
        return wins / losses

    def _calculate_max_drawdown_points(self, cumulative_pnl_points: pd.Series) -> float:
        if cumulative_pnl_points.empty or cumulative_pnl_points.isnull().all(): return 0.0
        equity = cumulative_pnl_points.fillna(0)
        if not equity.empty and equity.iloc[0] > 0: equity = pd.concat([pd.Series([0], index=[equity.index[0] - pd.Timedelta(seconds=1)]), equity])
        peak = equity.cummax(); drawdown = peak - equity; return drawdown.max()

    def _identify_best_strategy(self, metric: str = 'total_pnl_points', min_trades: int = 5) -> None:
        if not self.performance_metrics: self.best_strategy_info = None; return
        best_strategy = None; best_score = -np.inf
        valid_strategies = { name: metrics for name, metrics in self.performance_metrics.items() if isinstance(metrics.get('total_trades'), (int, float)) and metrics.get('total_trades', 0) >= min_trades and pd.notna(metrics.get(metric)) }
        if not valid_strategies:
             logger.warning(f"No strategies found with >= {min_trades} trades to determine 'best' based on '{metric}'. Falling back.")
             valid_strategies = {name: metrics for name, metrics in self.performance_metrics.items() if pd.notna(metrics.get(metric)) and isinstance(metrics.get('total_trades'), (int, float)) and metrics.get('total_trades', 0) > 0}
             if not valid_strategies: self.best_strategy_info = None; return
        if metric in ['max_drawdown_points', 'avg_loss_points']: best_score = np.inf; compare = lambda score, best: score < best
        else: compare = lambda score, best: score > best
        for name, metrics in valid_strategies.items():
            if compare(metrics[metric], best_score): best_score = metrics[metric]; best_strategy = name
        if best_strategy: self.best_strategy_info = {'name': best_strategy, 'metric': metric, 'score': best_score}; logger.info(f"Best strategy ({metric}): {best_strategy} (Score: {best_score:.2f})")
        else: self.best_strategy_info = None; logger.warning(f"Could not determine best strategy based on '{metric}'.")

    # --- Methods to RETURN data for consolidated report ---
    def get_performance_metrics_df(self) -> Optional[pd.DataFrame]:
        """Returns the calculated performance metrics as a DataFrame."""
        if self.performance_metrics:
             try: return pd.DataFrame(self.performance_metrics).T.fillna(0)
             except Exception as e: logger.error(f"Error converting metrics to DataFrame: {e}"); return None
        return None

    def generate_plot_data_uris(self) -> Dict[str, Optional[str]]:
        """Generates plots as base64 data URIs."""
        plot_data_uris = {}
        logger.info("Generating equity curve plot data URI...")
        plot_data_uris['equity_curves'] = self._plot_to_base64(self._plot_equity_curves)
        logger.info("Generating trade analysis plot data URIs...")
        trade_plot_uris = self._plot_trade_analysis_to_base64()
        plot_data_uris.update(trade_plot_uris)
        return plot_data_uris

    # --- Plotting helpers ---
    def _plot_to_base64(self, plot_func: Callable, *args, **kwargs) -> Optional[str]:
         try:
             buffer = io.BytesIO(); plot_func(buffer=buffer, *args, **kwargs)
             buffer.seek(0); image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8'); buffer.close()
             return f"data:image/png;base64,{image_base64}"
         except Exception as e: logger.error(f"Plot failed ({plot_func.__name__}): {e}", exc_info=False); return None

    def _plot_equity_curves(self, buffer: io.BytesIO) -> None:
        fig, ax = plt.subplots(figsize=(14, 7)); has_data = False
        for name in self.strategies:
            cum_pnl_col = f'{name}_cumulative_pnl_points'
            if cum_pnl_col in self.results_df.columns:
                equity = self.results_df[cum_pnl_col].ffill().fillna(0)
                if not equity.empty: ax.plot(equity.index, equity, label=name, linewidth=1.5); has_data = True
        if not has_data: plt.close(fig); raise ValueError("No data to plot equity curves") # Raise error if no data
        ax.set_title('Strategy Equity Curves (Cumulative PnL Points)'); ax.set_xlabel('Date'); ax.set_ylabel('Cumulative PnL (Points)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight'); plt.close(fig) # Lower dpi slightly

    def _plot_trade_analysis_to_base64(self) -> Dict[str, Optional[str]]:
        plot_data_uris = {}
        for name in self.strategies:
            trades_df = self.trades_dict.get(name)
            if trades_df is None or trades_df.empty: continue
            plot_data_uris[f'{name}_pnl_distribution'] = self._plot_to_base64(self._plot_pnl_distribution, trades_df, name)
            plot_data_uris[f'{name}_cumulative_pnl'] = self._plot_to_base64(self._plot_cumulative_pnl, trades_df, name)
        return plot_data_uris

    def _plot_pnl_distribution(self, trades_df: pd.DataFrame, name: str, buffer: io.BytesIO) -> None:
         fig, ax = plt.subplots(figsize=(10, 5)); ax.hist(trades_df['pnl_points'], bins=30, edgecolor='black'); ax.set_title(f'{name} - Trade PnL (Points) Distribution'); ax.set_xlabel('PnL (Points)'); ax.set_ylabel('Frequency'); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight'); plt.close(fig)

    def _plot_cumulative_pnl(self, trades_df: pd.DataFrame, name: str, buffer: io.BytesIO) -> None:
         fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(trades_df['exit_time'], trades_df['cumulative_pnl_points'], marker='.', linestyle='-', markersize=4); ax.set_title(f'{name} - Cumulative PnL (Points) Over Time'); ax.set_xlabel('Exit Date'); ax.set_ylabel('Cumulative PnL (Points)'); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight'); plt.close(fig)