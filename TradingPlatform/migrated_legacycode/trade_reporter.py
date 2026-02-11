import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TradeReporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_trade_report(self, trades, run_id):
        """
        Generate HTML trade report.
        Saves to output_dir/trade_report_{run_id}.html
        """
        if trades.empty:
            logger.warning("No trades to report")
            # Escape all CSS curly braces with double curlies {{ }}
            # And use keyword argument for .format
            html_content = """
            <html>
            <head><title>Trade Report</title>
            <style>
                body {{font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h2 {{ color: #333; }}
            </style>
            </head>
            <body>
            <h2>Trade Report (Run ID: {run_id})</h2>
            <p>No trades executed.</p>
            </body>
            </html>
            """.format(run_id=run_id) # Use keyword argument: run_id=run_id
            output_file = f"{self.output_dir}/trade_report_{run_id}.html"
            with open(output_file, 'w') as f:
                f.write(html_content)
            logger.info(f"Saved HTML report to {output_file}")
            return

        # This section is for when trades are NOT empty
        trade_columns = ['strategy', 'timeframe', 'entry_time', 'exit_time', 'option_type',
                        'strike_price', 'entry_premium', 'exit_premium', 'entry_index_price',
                        'exit_index_price', 'sl', 'tp', 'tsl', 'lot_size', 'pl', 'pl_percentage',
                        'exit_reason']
        trade_report = trades[trade_columns].copy()
        trade_report['entry_time'] = trade_report['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        trade_report['exit_time'] = trade_report['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        trade_report['pl'] = trade_report['pl'].round(2)
        trade_report['pl_percentage'] = trade_report['pl_percentage'].round(2)

        summary = trade_report.groupby(['strategy', 'timeframe']).agg({
            'pl': ['sum', 'count'],
            'pl_percentage': 'mean',
            'option_type': lambda x: x.value_counts().to_dict(),
            'exit_reason': lambda x: x.value_counts().to_dict()
        }).reset_index()
        summary.columns = ['strategy', 'timeframe', 'total_pl', 'total_trades', 'avg_pl_percentage',
                         'option_type_counts', 'exit_reason_counts']
        summary['total_pl'] = summary['total_pl'].round(2)
        summary['avg_pl_percentage'] = summary['avg_pl_percentage'].round(2)

        # Escape CSS curly braces here too
        html_content = """
        <html>
        <head>
            <title>Trade Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h2 {{ color: #333; }}
                .summary {{ background-color: #e6f3ff; }}
            </style>
        </head>
        <body>
            <h2>Trade Report (Run ID: {})</h2>
            <h3>Trade Details</h3>
            {}
            <h3>Summary</h3>
            {}
        </body>
        </html>
        """.format(run_id, trade_report.to_html(index=False, classes='trade-table'),
                  summary.to_html(index=False, classes='summary'))

        output_file = f"{self.output_dir}/trade_report_{run_id}.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {output_file}")