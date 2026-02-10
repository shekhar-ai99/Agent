"""
HTML Report Generator for Backtesting Results

Generates comprehensive reports with:
- Market-wise breakdown
- Timeframe-wise breakdown
- Per-strategy metrics
- Trade log table
- Equity curves
- Performance charts
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """Generate HTML reports for backtest results"""
    
    def __init__(self, results: Dict[str, Any], trade_history: List[Dict], equity_curve: List[float]):
        """
        Args:
            results: Backtest metrics dictionary
            trade_history: List of trade dictionaries
            equity_curve: List of equity values over time
        """
        self.results = results
        self.trade_history = trade_history
        self.equity_curve = equity_curve
        
    def generate_html(self, market: str, timeframe: str, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            market: Market name (e.g., "india", "crypto")
            timeframe: Timeframe (e.g., "5min", "15min")
            output_path: Path to save HTML file
            
        Returns:
            Path to saved HTML file
        """
        logger.info(f"Generating HTML report for {market} {timeframe}...")
        
        # Generate report sections
        html = self._generate_header()
        html += self._generate_summary_section(market, timeframe)
        html += self._generate_metrics_table()
        html += self._generate_strategy_breakdown()
        html += self._generate_trade_log_table()
        html += self._generate_footer()
        
        # Save to file
        if output_path is None:
            runs_dir = Path(__file__).resolve().parents[1] / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = runs_dir / f"backtest_report_{market}_{timeframe}_{timestamp}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {output_path}")
        return str(output_path)
    
    def _generate_header(self) -> str:
        """Generate HTML header with CSS"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: auto;
            background-color: #fff;
            padding: 30px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1, h2, h3 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin-top: 35px;
            margin-bottom: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 25px;
            border-bottom: none;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .summary-card {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }
        .summary-card h4 {
            margin: 0 0 10px 0;
            color: #6c757d;
            font-size: 0.9em;
            border: none;
        }
        .summary-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .negative {
            color: #c0392b;
            font-weight: bold;
        }
        .neutral {
            color: #7f8c8d;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 25px;
            font-size: 0.9em;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: right;
        }
        th {
            background-color: #3498db;
            color: white;
            text-align: center;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 35px;
            font-size: 0.9em;
        }
        .metric-card {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 25px;
            background-color: #fff;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .strategy-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .strategy-card {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            background-color: #fdfdfd;
        }
        .strategy-card h4 {
            margin-top: 0;
            color: #3498db;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 8px;
        }
        .trade-log {
            max-height: 600px;
            overflow-y: auto;
            margin-top: 15px;
        }
        .trade-log table {
            font-size: 0.85em;
        }
        .trade-log th {
            position: sticky;
            top: 0;
            background-color: #3498db;
            z-index: 10;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
<div class="container">
"""
    
    def _generate_summary_section(self, market: str, timeframe: str) -> str:
        """Generate summary section with key metrics"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        total_pnl = self.results.get('total_pnl', 0)
        return_pct = self.results.get('return_pct', 0)
        num_trades = self.results.get('num_trades', 0)
        win_rate = self.results.get('win_rate', 0)
        profit_factor = self.results.get('profit_factor', 0)
        max_drawdown = self.results.get('max_drawdown', 0)
        
        pnl_class = 'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else 'neutral'
        
        html = f"""
<h1>Backtest Report: {market.upper()} - {timeframe}</h1>
<p class="timestamp">Generated on {timestamp}</p>

<div class="summary-grid">
    <div class="summary-card">
        <h4>Total P&L</h4>
        <div class="value {pnl_class}">${total_pnl:,.2f}</div>
    </div>
    <div class="summary-card">
        <h4>Return %</h4>
        <div class="value {pnl_class}">{return_pct:.2f}%</div>
    </div>
    <div class="summary-card">
        <h4>Total Trades</h4>
        <div class="value">{num_trades}</div>
    </div>
    <div class="summary-card">
        <h4>Win Rate</h4>
        <div class="value">{win_rate:.2f}%</div>
    </div>
    <div class="summary-card">
        <h4>Profit Factor</h4>
        <div class="value">{profit_factor:.2f}</div>
    </div>
    <div class="summary-card">
        <h4>Max Drawdown</h4>
        <div class="value negative">{max_drawdown:.2f}%</div>
    </div>
</div>
"""
        return html
    
    def _generate_metrics_table(self) -> str:
        """Generate overall metrics table"""
        html = """
<h2>Performance Metrics</h2>
<div class="metric-card">
<table>
<thead>
<tr>
    <th>Metric</th>
    <th>Value</th>
</tr>
</thead>
<tbody>
"""
        metrics = {
            'Initial Capital': f"${self.results.get('initial_capital', 0):,.2f}",
            'Final Equity': f"${self.results.get('final_equity', 0):,.2f}",
            'Total P&L': f"${self.results.get('total_pnl', 0):,.2f}",
            'Return %': f"{self.results.get('return_pct', 0):.2f}%",
            'Total Trades': f"{self.results.get('num_trades', 0):,}",
            'Win Rate': f"{self.results.get('win_rate', 0):.2f}%",
            'Profit Factor': f"{self.results.get('profit_factor', 0):.2f}",
            'Sharpe Ratio': f"{self.results.get('sharpe_ratio', 0):.2f}",
            'Max Drawdown': f"{self.results.get('max_drawdown', 0):.2f}%",
            'Total Bars': f"{self.results.get('total_bars', 0):,}",
        }
        
        for metric, value in metrics.items():
            html += f"<tr><td>{metric}</td><td>{value}</td></tr>\n"
        
        html += "</tbody></table></div>\n"
        return html
    
    def _generate_strategy_breakdown(self) -> str:
        """Generate per-strategy breakdown"""
        if not self.trade_history:
            return "<h2>Strategy Breakdown</h2><p>No trades to analyze.</p>"
        
        # Group trades by strategy
        df = pd.DataFrame(self.trade_history)
        if 'strategy' not in df.columns:
            return "<h2>Strategy Breakdown</h2><p>Strategy information not available.</p>"
        
        strategy_stats = df.groupby('strategy').agg({
            'pnl': ['sum', 'mean', 'count'],
            'pnl_percent': 'mean'
        }).round(2)
        
        html = "<h2>Strategy Breakdown</h2>\n<div class=\"strategy-grid\">\n"
        
        for strategy_name in strategy_stats.index:
            trades = df[df['strategy'] == strategy_name]
            total_pnl = trades['pnl'].sum()
            avg_pnl = trades['pnl'].mean()
            trade_count = len(trades)
            wins = len(trades[trades['pnl'] > 0])
            losses = len(trades[trades['pnl'] < 0])
            win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
            
            pnl_class = 'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else 'neutral'
            
            html += f"""
<div class="strategy-card">
    <h4>{strategy_name}</h4>
    <p><strong>Total Trades:</strong> {trade_count}</p>
    <p><strong>Wins/Losses:</strong> {wins}/{losses}</p>
    <p><strong>Win Rate:</strong> {win_rate:.2f}%</p>
    <p><strong>Total P&L:</strong> <span class="{pnl_class}">${total_pnl:.2f}</span></p>
    <p><strong>Avg P&L/Trade:</strong> <span class="{pnl_class}">${avg_pnl:.2f}</span></p>
</div>
"""
        
        html += "</div>\n"
        return html
    
    def _generate_trade_log_table(self) -> str:
        """Generate detailed trade log table"""
        if not self.trade_history:
            return "<h2>Trade Log</h2><p>No trades executed.</p>"
        
        html = """
<h2>Trade Log</h2>
<div class="trade-log">
<table>
<thead>
<tr>
    <th>#</th>
    <th>Strategy</th>
    <th>Symbol</th>
    <th>Direction</th>
    <th>Entry Time</th>
    <th>Exit Time</th>
    <th>Entry Price</th>
    <th>Exit Price</th>
    <th>Quantity</th>
    <th>P&L</th>
    <th>P&L %</th>
    <th>Exit Reason</th>
</tr>
</thead>
<tbody>
"""
        
        for i, trade in enumerate(self.trade_history, 1):
            pnl = trade.get('pnl', 0)
            pnl_pct = trade.get('pnl_percent', 0)
            pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'neutral'
            
            entry_time = trade.get('entry_time', '')
            exit_time = trade.get('exit_time', '')
            
            if isinstance(entry_time, datetime):
                entry_time = entry_time.strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(exit_time, datetime):
                exit_time = exit_time.strftime('%Y-%m-%d %H:%M:%S')
            
            html += f"""
<tr>
    <td>{i}</td>
    <td>{trade.get('strategy', 'unknown')}</td>
    <td>{trade.get('symbol', '')}</td>
    <td>{trade.get('direction', '')}</td>
    <td>{entry_time}</td>
    <td>{exit_time}</td>
    <td>${trade.get('entry_price', 0):.2f}</td>
    <td>${trade.get('exit_price', 0):.2f}</td>
    <td>{trade.get('quantity', 0)}</td>
    <td class="{pnl_class}">${pnl:.2f}</td>
    <td class="{pnl_class}">{pnl_pct:.2f}%</td>
    <td>{trade.get('exit_reason', '')}</td>
</tr>
"""
        
        html += "</tbody></table></div>\n"
        return html
    
    def _generate_footer(self) -> str:
        """Generate HTML footer"""
        return """
<div class="footer">
    <p>Generated by TradingPlatform Backtest Engine</p>
    <p>This report is for analysis purposes only. Past performance does not guarantee future results.</p>
</div>
</div>
</body>
</html>
"""


def generate_consolidated_report(
    all_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate consolidated report across multiple markets/timeframes.
    
    Args:
        all_results: Dictionary with structure {market_timeframe: {results, trade_history, equity_curve}}
        output_path: Path to save HTML file
        
    Returns:
        Path to saved HTML file
    """
    logger.info("Generating consolidated multi-market/timeframe report...")
    
    if not all_results:
        logger.warning("No results to generate consolidated report")
        return ""
    
    # Generate tabs for each market/timeframe
    html = _generate_consolidated_header()
    
    # Tab buttons
    html += '<div class="tab-buttons">\n'
    for i, market_tf in enumerate(all_results.keys()):
        active = 'active' if i == 0 else ''
        html += f'<button class="tab-button {active}" onclick="openTab(event, \'{market_tf}\')">{market_tf}</button>\n'
    html += '</div>\n'
    
    # Tab content
    for i, (market_tf, data) in enumerate(all_results.items()):
        active = 'active' if i == 0 else ''
        generator = BacktestReportGenerator(
            results=data['results'],
            trade_history=data['trade_history'],
            equity_curve=data['equity_curve']
        )
        
        market, tf = market_tf.split('_', 1)
        
        html += f'<div id="{market_tf}" class="tab-content {active}">\n'
        html += generator._generate_summary_section(market, tf)
        html += generator._generate_metrics_table()
        html += generator._generate_strategy_breakdown()
        html += generator._generate_trade_log_table()
        html += '</div>\n'
    
    html += _generate_consolidated_footer()
    
    # Save to file
    if output_path is None:
        runs_dir = Path(__file__).resolve().parents[1] / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = runs_dir / f"consolidated_report_{timestamp}.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Consolidated HTML report saved to {output_path}")
    return str(output_path)


def _generate_consolidated_header() -> str:
    """Generate header for consolidated report with tabs"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Consolidated Backtest Report</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; line-height: 1.6; }
        .container { max-width: 1400px; margin: auto; background-color: #fff; padding: 30px; box-shadow: 0 2px 15px rgba(0,0,0,0.1); border-radius: 8px; }
        h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 35px; margin-bottom: 20px; }
        h1 { text-align: center; margin-bottom: 25px; border-bottom: none; }
        .tab-buttons { display: flex; border-bottom: 2px solid #3498db; margin-bottom: 20px; }
        .tab-buttons button { background-color: #eee; border: 1px solid #ccc; padding: 10px 15px; cursor: pointer; transition: background-color 0.3s; border-radius: 4px 4px 0 0; margin-right: 2px; font-size: 1em; border-bottom: none; }
        .tab-buttons button.active { background-color: #3498db; color: white; }
        .tab-buttons button:hover { background-color: #d0d0d0; }
        .tab-buttons button.active:hover { background-color: #2980b9; }
        .tab-content { display: none; padding: 20px; animation: fadeIn 0.5s; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .summary-card { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; text-align: center; }
        .summary-card h4 { margin: 0 0 10px 0; color: #6c757d; font-size: 0.9em; border: none; }
        .summary-card .value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
        .positive { color: #27ae60; font-weight: bold; }
        .negative { color: #c0392b; font-weight: bold; }
        .neutral { color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: right; }
        th { background-color: #3498db; color: white; text-align: center; font-weight: 600; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        .timestamp { text-align: center; color: #7f8c8d; margin-bottom: 35px; font-size: 0.9em; }
        .metric-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 20px; margin-bottom: 25px; background-color: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
        .strategy-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .strategy-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 15px; background-color: #fdfdfd; }
        .strategy-card h4 { margin-top: 0; color: #3498db; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; }
        .trade-log { max-height: 600px; overflow-y: auto; margin-top: 15px; }
        .trade-log table { font-size: 0.85em; }
        .trade-log th { position: sticky; top: 0; background-color: #3498db; z-index: 10; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
<div class="container">
<h1>Consolidated Multi-Market/Timeframe Backtest Report</h1>
<p class="timestamp">Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
"""


def _generate_consolidated_footer() -> str:
    """Generate footer for consolidated report with JavaScript"""
    return """
<div class="footer">
    <p>Generated by TradingPlatform Backtest Engine</p>
    <p>This report is for analysis purposes only. Past performance does not guarantee future results.</p>
</div>
</div>
<script>
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
        tabcontent[i].classList.remove("active");
    }
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("active");
    }
    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}
</script>
</body>
</html>
"""
