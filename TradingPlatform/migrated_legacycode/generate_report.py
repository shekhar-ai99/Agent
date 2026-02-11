
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def generate_strategy_report(results, run_id, timeframes,option_pnl_analysis=None):
    """Generate a consolidated strategy performance report with one HTML file containing separate tables for each timeframe, plus per-timeframe CSV and suggestion files."""
    logger.info(f"Generating report for Run ID: {run_id}, Timeframes: {timeframes}")
    
    reports_dir = f"reports/run_{run_id}"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Strategy Performance Report - Run {run_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
            }}
            h1 {{
                text-align: center;
                color: #333;
            }}
            h2 {{
                color: #444;
                margin-top: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: #fff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px;
                text-align: right;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }}
            th:hover {{
                background-color: #45a049;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .strategy-column, .text-left {{
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <h1>Strategy Performance Report - Run {run_id}</h1>
        {tables}
        <script>
            function sortTable(tableId, n) {{
                let table = document.getElementById(tableId);
                let rows, switching = true;
                let i, shouldSwitch, dir = "asc";
                let switchcount = 0;
                while (switching) {{
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {{
                        shouldSwitch = false;
                        let x = rows[i].getElementsByTagName("TD")[n];
                        let y = rows[i + 1].getElementsByTagName("TD")[n];
                        let xVal = x.innerHTML;
                        let yVal = y.innerHTML;
                        if (!isNaN(parseFloat(xVal)) && !isNaN(parseFloat(yVal))) {{
                            xVal = parseFloat(xVal);
                            yVal = parseFloat(yVal);
                        }}
                        if (dir == "asc") {{
                            if (xVal > yVal) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xVal < yVal) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }}
                    if (shouldSwitch) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    }} else {{
                        if (switchcount == 0 && dir == "asc") {{
                            dir = "desc";
                            switching = true;
                        }}
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    tables_html = ""
    
    for timeframe in timeframes:
        tf_results = [r for r in results if r.get('timeframe') == timeframe]
        if not tf_results:
            logger.warning(f"No results for timeframe {timeframe}. Skipping...")
            continue
        
        summary_data = []
        
        for result in tf_results:
            strategy = result['strategy']
            row = {
                'Strategy': strategy,
                'PnL': round(result['pnl'], 2),
                'Win %': round(result['win_rate'], 2),
                'Performance Score': round(result['performance_score'], 2),
                'Total Trades': result['total_trades'],
                'Profitable Trades': result['profitable_trades'],
                'Losing Trades': result['losing_trades'],
                'Buy Trades': result['buy_trades'],
                'Sell Trades': result['sell_trades'],
                'SL Exits': result['exit_reasons']['sl'],
                'TSL Exits': result['exit_reasons']['tsl'],
                'TP Exits': result['exit_reasons']['tp'],
                'Signal Exits': result['exit_reasons']['signal']
            }
            
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                metrics = result['day_wise'].get(day, {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0})
                row[f'{day}_Trades'] = metrics['total']
                row[f'{day}_Profit'] = metrics['profitable']
                row[f'{day}_NonProfit'] = metrics['losing']
                row[f'{day}_Accuracy'] = round(metrics['accuracy'], 2)
            
            for session in ['Session 1 (9:15-11:30)', 'Session 2 (11:30-13:45)', 'Session 3 (13:45-15:30)']:
                metrics = result['session_wise'].get(session, {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0})
                session_key = session.split(' ')[0]
                row[f'{session_key}_Trades'] = metrics['total']
                row[f'{session_key}_Profit'] = metrics['profitable']
                row[f'{session_key}_NonProfit'] = metrics['losing']
                row[f'{session_key}_Accuracy'] = round(metrics['accuracy'], 2)
            
            for regime in ['Trending', 'Choppy', 'Ranging']:
                metrics = result['regime_wise'].get(regime, {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0})
                row[f'{regime}_Trades'] = metrics['total']
                row[f'{regime}_Profit'] = metrics['profitable']
                row[f'{regime}_NonProfit'] = metrics['losing']
                row[f'{regime}_Accuracy'] = round(metrics['accuracy'], 2)
            
            for volatility in ['Low', 'Medium', 'High']:
                metrics = result['volatility_wise'].get(volatility, {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0})
                row[f'{volatility}_Trades'] = metrics['total']
                row[f'{volatility}_Profit'] = metrics['profitable']
                row[f'{volatility}_NonProfit'] = metrics['losing']
                row[f'{volatility}_Accuracy'] = round(metrics['accuracy'], 2)
            
            metrics = result['expiry_wise'].get('Expiry Thursday', {'total': 0, 'profitable': 0, 'losing': 0, 'accuracy': 0.0})
            row['Expiry_Trades'] = metrics['total']
            row['Expiry_Profit'] = metrics['profitable']
            row['Expiry_NonProfit'] = metrics['losing']
            row['Expiry_Accuracy'] = round(metrics['accuracy'], 2)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        csv_path = f"{reports_dir}/strategy_summary_{run_id}_{timeframe}.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Consolidated CSV report for {timeframe} saved to {csv_path}")
        
        headers = ''.join(f'<th onclick="sortTable(\'strategyTable_{timeframe}\', {i})">{col}</th>' for i, col in enumerate(summary_df.columns))
        rows = ''
        for _, row in summary_df.iterrows():
            cells = []
            for i, val in enumerate(row):
                if i == 0:
                    cells.append(f'<td class="strategy-column">{val}</td>')
                elif isinstance(val, float):
                    cells.append(f'<td>{val:.2f}</td>')
                else:
                    cells.append(f'<td>{val}</td>')
            rows += '<tr>' + ''.join(cells) + '</tr>\n'
        
        table_html = f"""
        <h2>{timeframe} Performance</h2>
        <table id="strategyTable_{timeframe}">
            <thead>
                <tr>
                    {headers}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """
        tables_html += table_html
        
        trades = []
        for result in tf_results:
            trades.extend(result.get('trades', []))
        
        if trades:
            df_trades = pd.DataFrame(trades)
            # Format price columns
            price_columns = ['entry_price', 'exit_price', 'pnl']
            for col in price_columns:
                if col in df_trades.columns:
                    df_trades[col] = df_trades[col].round(2)
            
            # Format date columns
            date_columns = ['entry_timestamp', 'exit_timestamp']
            for col in date_columns:
                if col in df_trades.columns:
                    df_trades[col] = pd.to_datetime(df_trades[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            trade_csv_path = f"{reports_dir}/trade_log_{run_id}_{timeframe}.csv"
            df_trades.to_csv(trade_csv_path, index=False)
            logger.info(f"Trade log for {timeframe} saved to {trade_csv_path}")
            
            trade_stats = df_trades.groupby(['strategy']).agg({
                'pnl': ['sum', 'mean', 'std', 'count'],
                'exit_reason': lambda x: x.value_counts().to_dict(),
                'profit_or_loss': lambda x: x.value_counts().to_dict()
            }).reset_index()
            trade_stats.columns = ['Strategy', 'Total PNL', 'Avg PNL', 'PNL Std', 'Trade Count', 'Exit Reasons', 'Profit or Loss']
            trade_stats_path = f"{reports_dir}/trade_stats_{run_id}_{timeframe}.csv"
            trade_stats.to_csv(trade_stats_path, index=False)
            logger.info(f"Trade statistics for {timeframe} saved to {trade_stats_path}")
            
            trade_headers = ''.join(f'<th>{col}</th>' for col in df_trades.columns)
            trade_rows = ''
            for _, row in df_trades.head(10).iterrows():
                cells = []
                for col, val in row.items():
                    if col in price_columns:
                        cells.append(f'<td>{val:.2f}</td>')
                    elif col in date_columns:
                        cells.append(f'<td class="text-left">{val}</td>')
                    elif col == 'strategy':
                        cells.append(f'<td class="strategy-column">{val}</td>')
                    else:
                        cells.append(f'<td>{val}</td>')
                trade_rows += '<tr>' + ''.join(cells) + '</tr>\n'
            
            trade_table_html = f"""
            <h2>{timeframe} Trade Log (First 10 Trades)</h2>
            <table id="tradeTable_{timeframe}">
                <thead>
                    <tr>
                        {trade_headers}
                    </tr>
                </thead>
                <tbody>
                    {trade_rows}
                </tbody>
            </table>
            """
            tables_html += trade_table_html
        
        suggestions = []
        for result in tf_results:
            strategy = result['strategy']
            win_rate = result['win_rate']
            total_trades = result['total_trades']
            session_wise = result['session_wise']
            regime_wise = result['regime_wise']
            volatility_wise = result['volatility_wise']
            expiry_wise = result['expiry_wise']
            
            if total_trades == 0:
                suggestions.append(f"{strategy}: No trades executed. Check signal conditions or parameters in strategies.py.")
                continue
            
            best_session = max(session_wise.items(), key=lambda x: x[1]['accuracy'] if x[1]['total'] > 0 else 0, default=('None', {'total': 0}))[0]
            if session_wise[best_session]['total'] > 0:
                suggestions.append(f"{strategy}: Best in {best_session} (Win Rate: {session_wise[best_session]['accuracy']:.2f}%).")
            
            best_regime = max(regime_wise.items(), key=lambda x: x[1]['accuracy'] if x[1]['total'] > 0 else 0, default=('None', {'total': 0}))[0]
            if regime_wise[best_regime]['total'] > 0:
                suggestions.append(f"{strategy}: Excels in {best_regime} markets (Win Rate: {regime_wise[best_regime]['accuracy']:.2f}%).")
            
            best_volatility = max(volatility_wise.items(), key=lambda x: x[1]['accuracy'] if x[1]['total'] > 0 else 0, default=('None', {'total': 0}))[0]
            if volatility_wise[best_volatility]['total'] > 0:
                suggestions.append(f"{strategy}: Optimal in {best_volatility} volatility (Win Rate: {volatility_wise[best_volatility]['accuracy']:.2f}%).")
            
            if expiry_wise['Expiry Thursday']['total'] > 0:
                expiry_win_rate = expiry_wise['Expiry Thursday']['accuracy']
                suggestions.append(f"{strategy}: On Expiry Thursday, Win Rate: {expiry_win_rate:.2f}%. Consider adjusting atr_multiplier for high volatility.")
            
            if result['exit_reasons']['tp'] == 0:
                suggestions.append(f"{strategy}: No TP exits. Reduce tp_atr_mult (e.g., to 0.3) or disable TSL.")
            if win_rate < 50 and total_trades > 10:
                suggestions.append(f"{strategy}: Low win rate ({win_rate:.2f}%). Relax signal conditions or increase sl_atr_mult.")
        
        suggestions_path = f"{reports_dir}/suggestions_{run_id}_{timeframe}.txt"
        with open(suggestions_path, 'w') as f:
            for suggestion in suggestions:
                f.write(suggestion + '\n')
        logger.info(f"Suggestions for {timeframe} saved to {suggestions_path}")
    
    html_content = html_template.format(run_id=run_id, tables=tables_html)
    html_path = f"{reports_dir}/strategy_summary_{run_id}.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    logger.info(f"Consolidated HTML report saved to {html_path}")