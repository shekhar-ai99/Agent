from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import importlib
import inspect
from backtester_engine import EnhancedMultiStrategyBacktester
from backtester_analyzer import BacktestAnalyzerReporter
from strategy import Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class run_full_backtest_workflow:
    @staticmethod
    def get_strategies() -> List[Strategy]:
        """Load all strategy classes from strategy.py that subclass Strategy and are marked for inclusion."""
        strategies = []
        try:
            strategy_module = importlib.import_module('strategy')
            for name, obj in inspect.getmembers(strategy_module, inspect.isclass):
                if issubclass(obj, Strategy) and obj != Strategy:
                    include = getattr(obj, 'include', True)
                    if include:
                        strategies.append(obj)
            logger.info(f"Loaded {len(strategies)} strategies: {[s.__name__ for s in strategies]}")
            return strategies
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}", exc_info=True)
            return []

    @staticmethod
    def validate_indicators(data: pd.DataFrame, strategies: List[Strategy]) -> bool:
        """Validate that all required indicators are present in the DataFrame."""
        required_indicators = set()
        for strategy in strategies:
            indicators = getattr(strategy, 'required_indicators', [])
            if not isinstance(indicators, (list, set)):
                logger.warning(f"Strategy {strategy.__name__} has invalid required_indicators: {indicators}")
                continue
            required_indicators.update(indicators)
        
        # Comprehensive list of indicators from strategy.py
        expected_indicators = {
            'atr', 'ema_9', 'ema_21', 'rsi', 'SUPERT_10_3.0', 'atr_sma_5', 'bollinger_mid',
            'SUPERTd_10_3.0', 'SUPERTr_10_3.0', 'adx', 'volume', 'vol_ma', 'macd', 'macd_signal',
            'macd_hist', 'bollinger_bandwidth', 'bollinger_upper', 'bollinger_lower', 'plus_di',
            'minus_di', 'ema_50', 'support', 'resistance', 'breakout', 'max_atr', 'sma10', 'sma50',
            'upper', 'lower', 'rsi_slope', 'momentum', 'stochastic_k', 'htf_high_shifted',
            'htf_low_shifted', 'mss_high_shifted', 'mss_low_shifted', 'highest_10_shifted',
            'lowest_10_shifted', 'zlema_8', 'zlema_21'
        }
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        if missing_indicators:
            logger.error(f"Missing required indicators: {missing_indicators}")
            return False
        
        logger.info(f"Validated indicators: {sorted(list(required_indicators))}")
        return True

    @staticmethod
    def generate_consolidated_html_report(
        all_metrics: Dict[str, Optional[pd.DataFrame]],
        all_plot_uris: Dict[str, Dict[str, Optional[str]]],
        all_best_strategies: Dict[str, Optional[Dict]],
        output_file: Path):
        """Generate a consolidated HTML report for all timeframes."""
        logger.info(f"Generating consolidated HTML report at {output_file}...")
        timeframes = list(all_metrics.keys())
        if not timeframes:
            logger.warning("No metrics data found to generate consolidated report.")
            return

        # HTML header with CSS and JavaScript for tabs
        html_start = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Consolidated Backtest Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: auto;
            background-color: #fff;
            padding: 30px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin-top: 35px;
            margin-bottom: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 25px;
            border-bottom: none;
        }}
        .tab-buttons button {{
            background-color: #eee;
            border: 1px solid #ccc;
            padding: 10px 15px;
            cursor: pointer;
            transition: background-color 0.3s;
            border-radius: 4px 4px 0 0;
            margin-right: 2px;
            font-size: 1em;
        }}
        .tab-buttons button.active {{
            background-color: #3498db;
            color: white;
            border-bottom: 1px solid #3498db;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 4px 4px;
            background-color: #fff;
            animation: fadeIn 0.5s;
        }}
        .tab-content.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{opacity: 0;}}
            to {{opacity: 1;}}
        }}
        table.performance-table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 25px;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: right;
        }}
        th {{
            background-color: #3498db;
            color: white;
            text-align: center;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .metric-card {{
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 25px;
            background-color: #fff;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin-top: 15px;
            border: 1px solid #eee;
            display: block;
            margin-left: auto;
            margin-right: auto;
            background-color: #fafafa;
        }}
        .positive {{
            color: #27ae60;
            font-weight: bold;
        }}
        .negative {{
            color: #c0392b;
            font-weight: bold;
        }}
        .neutral {{
            color: #7f8c8d;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 35px;
            font-size: 0.9em;
        }}
        details {{
            margin-top: 15px;
            border: 1px solid #eee;
            padding: 10px;
            border-radius: 4px;
            background-color: #f9f9f9;
        }}
        details > summary {{
            cursor: pointer;
            font-weight: bold;
            color: #3498db;
            padding: 5px;
            display: inline-block;
        }}
        details[open] > summary {{
            border-bottom: 1px solid #ddd;
            margin-bottom: 10px;
        }}
    </style>
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>Consolidated Multi-Timeframe Backtest Report</h1>
        <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="tab-buttons">
"""
        # Add tab buttons for each timeframe
        for i, tf in enumerate(timeframes):
            active_class = 'active' if i == 0 else ''
            html_start += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{tf}\')">{tf}</button>'
        html_start += "</div>"

        # Generate content for each timeframe tab
        html_content = ""
        for i, tf in enumerate(timeframes):
            active_class = 'active' if i == 0 else ''
            html_content += f'<div id="{tf}" class="tab-content {active_class}">\n<h2>Results for Timeframe: {tf}</h2>\n'

            # Best strategy highlight
            metrics_df = all_metrics.get(tf)
            plot_uris = all_plot_uris.get(tf, {})
            best_strat_info = all_best_strategies.get(tf)
            if best_strat_info:
                bs_name = best_strat_info['name']
                bs_metric = best_strat_info['metric'].replace('_', ' ').title()
                bs_score = best_strat_info['score']
                score_format = {
                    'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}',
                    'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}',
                    'max_drawdown_points': '{:,.2f}'
                }.get(best_strat_info['metric'], '{:.2f}')
                bs_score_str = score_format.format(bs_score) if pd.notna(bs_score) else 'N/A'
                html_content += f'<div class="metric-card" style="border-left: 5px solid #27ae60;"><p>Best Strategy ({tf}, based on max. \'{bs_metric}\'): <strong>{bs_name}</strong> (Score: {bs_score_str})</p></div>'
            else:
                html_content += '<div class="metric-card"><p>Could not determine best strategy (requires >= 5 trades).</p></div>'

            # Performance summary table
            html_content += "<h3>Performance Summary</h3><div class=\"metric-card\">"
            if metrics_df is not None and not metrics_df.empty:
                try:
                    metrics_to_format = {
                        'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}',
                        'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}',
                        'max_drawdown_points': '{:,.2f}'
                    }
                    metrics_display_df = metrics_df.copy()
                    for col, fmt in metrics_to_format.items():
                        if col in metrics_display_df.columns:
                            metrics_display_df[col] = metrics_display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
                    if 'total_trades' in metrics_display_df.columns:
                        metrics_display_df['total_trades'] = metrics_display_df['total_trades'].astype(int)
                    best_strategy_name = best_strat_info['name'] if best_strat_info else None
                    def highlight_best(s):
                        return ['background-color: #e0ffe0' if s.name == best_strategy_name else '' for v in s]
                    metrics_table_html = metrics_display_df.style.apply(highlight_best, axis=1).to_html(classes='performance-table', border=1, justify='right')
                    html_content += metrics_table_html
                except Exception as e:
                    logger.error(f"HTML metrics table error for {tf}: {e}")
                    html_content += "<p>Error displaying metrics table.</p>"
            else:
                html_content += "<p>No metrics data available.</p>"
            html_content += '<p style="font-size: 0.8em; color: #555; text-align: center;">Note: PnL & Drawdown in points.</p></div>'

            # Equity curves
            html_content += "<h3>Equity Curves (Cumulative PnL Points)</h3><div class=\"metric-card\">"
            equity_uri = plot_uris.get('equity_curves', '')
            html_content += f'<img src="{equity_uri}" alt="Equity Curves {tf}">' if equity_uri else "<p>Equity curve plot unavailable.</p>"
            html_content += "</div>"

            # Individual strategy analysis
            html_content += "<h3>Individual Strategy Analysis</h3><div class=\"metric-grid\">"
            strategies_in_metrics = metrics_df.index if metrics_df is not None else []
            for name in strategies_in_metrics:
                if metrics_df is None or name not in metrics_df.index or pd.isna(metrics_df.loc[name, 'total_trades']) or metrics_df.loc[name, 'total_trades'] == 0:
                    html_content += f'<div class="metric-card"><h4>{name}</h4><p>No trades or error.</p></div>'
                    continue
                metrics = metrics_df.loc[name]
                win_rate_str = f"{metrics.get('win_rate', 0):.2f}%"
                pf_str = f"{metrics.get('profit_factor', 0):.2f}"
                pnl_str = f"{metrics.get('total_pnl_points', 0):,.2f}"
                best_style = ' style="border-left: 5px solid #27ae60;"' if best_strat_info and name == best_strat_info['name'] else ''
                html_content += f'<div class="metric-card"{best_style}><h4>{name}</h4>'
                html_content += '<details><summary>Key Metrics</summary>'
                html_content += f'<p><strong>Total Trades:</strong> {int(metrics.get("total_trades", 0))}</p>'
                html_content += f'<p><strong>Total PnL:</strong> <span class="{"positive" if metrics.get("total_pnl_points", 0) > 0 else "negative" if metrics.get("total_pnl_points", 0) < 0 else "neutral"}">{pnl_str}</span></p>'
                html_content += f'<p><strong>Win Rate:</strong> {win_rate_str}</p>'
                html_content += f'<p><strong>Profit Factor:</strong> {pf_str}</p>'
                html_content += f'<p><strong>Expectancy (Points):</strong> {metrics.get("expectancy_points", 0):,.2f}</p>'
                html_content += f'<p><strong>Max Drawdown (Points):</strong> {metrics.get("max_drawdown_points", 0):,.2f}</p>'
                html_content += '</details>'
                # PnL Distribution Plot
                pnl_dist_uri = plot_uris.get(f'{name}_pnl_distribution', '')
                if pnl_dist_uri:
                    html_content += f'<details><summary>PnL Distribution</summary><img src="{pnl_dist_uri}" alt="{name} PnL Distribution"></details>'
                # Cumulative PnL Plot
                cum_pnl_uri = plot_uris.get(f'{name}_cumulative_pnl', '')
                if cum_pnl_uri:
                    html_content += f'<details><summary>Cumulative PnL</summary><img src="{cum_pnl_uri}" alt="{name} Cumulative PnL"></details>'
                html_content += '</div>'
            html_content += '</div></div>'

        # Combine and write HTML
        html_end = "</div></body></html>"
        final_html = html_start + html_content + html_end
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_html)
            logger.info(f"Consolidated HTML report successfully written to {output_file}")
        except Exception as e:
            logger.error(f"Failed to write HTML report to {output_file}: {e}")

    @staticmethod
    def run_full_backtest_workflow(data_paths: Dict[str, str], output_base_dir: str) -> None:
        """Run the full backtesting workflow for multiple timeframes."""
        # Load strategies
        strategy_classes = run_full_backtest_workflow.get_strategies()
        if not strategy_classes:
            logger.error("No strategies loaded. Exiting.")
            return

        all_strategy_names = [cls.__name__.lower() for cls in strategy_classes]
        strategies_config = {name: {} for name in all_strategy_names}
        strategies_config['voting_ensemble'] = {
            'constituent_strategies': all_strategy_names,
            'min_votes_entry': 2,
            'exit_vote_percentage': 0.40
        }

        # Initialize backtester and analyzer
        backtester = EnhancedMultiStrategyBacktester()
        analyzer = BacktestAnalyzerReporter()

        all_metrics = {}
        all_plot_uris = {}
        all_best_strategies = {}

        # Process each timeframe
        for tf, data_path in data_paths.items():
            logger.info(f"Processing timeframe: {tf}")
            data_path = Path(data_path)
            if not data_path.exists():
                logger.warning(f"Data file for {tf} not found at {data_path}. Skipping.")
                continue

            # Load data
            try:
                data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
                logger.info(f"Loaded data for {tf}: {len(data)} rows from {data.index.min()} to {data.index.max()}")
            except Exception as e:
                logger.error(f"Failed to load data for {tf}: {e}")
                continue

            # Validate indicators
            if not run_full_backtest_workflow.validate_indicators(data, strategy_classes):
                logger.error(f"Indicator validation failed for {tf}. Skipping.")
                continue

            # Run backtest
            try:
                results = backtester.run_backtest(
                    data=data,
                    strategies=strategy_classes,
                    strategies_config=strategies_config,
                    initial_capital=100000,
                    commission=0.001,
                    leverage=10
                )
                logger.info(f"Backtest completed for {tf}")
            except Exception as e:
                logger.error(f"Backtest failed for {tf}: {e}")
                continue

            # Analyze results
            try:
                metrics, plots = analyzer.analyze_results(results, tf)
                all_metrics[tf] = metrics
                all_plot_uris[tf] = plots
                all_best_strategies[tf] = analyzer.get_best_strategy(metrics, min_trades=5)
                logger.info(f"Analysis completed for {tf}")
            except Exception as e:
                logger.error(f"Analysis failed for {tf}: {e}")
                continue

        # Generate consolidated report
        output_dir = Path(output_base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_html = output_dir / 'consolidated_backtest_report.html'
        run_full_backtest_workflow.generate_consolidated_html_report(
            all_metrics=all_metrics,
            all_plot_uris=all_plot_uris,
            all_best_strategies=all_best_strategies,
            output_file=output_html
        )

if __name__ == "__main__":
    # For standalone testing, use default data paths
    data_paths = {
        '1h': 'data/input/market_data_1h.csv',
        '4h': 'data/input/market_data_4h.csv',
        '1d': 'data/input/market_data_1d.csv'
    }
    run_full_backtest_workflow(
        data_paths=data_paths,
        output_base_dir='backtest_results'
    )