import os
import pandas as pd
from datetime import time
from jinja2 import Environment, FileSystemLoader

def classify_session(ts):
    t = ts.time()
    if time(9,15) <= t < time(11,30): return 'Session 1 (9:15-11:30)'
    if time(11,30) <= t < time(13,45): return 'Session 2 (11:30-13:45)'
    if time(13,45) <= t <= time(15,30): return 'Session 3 (13:45-15:30)'
    return 'Outside Session'

def analyze_option_trades(option_trades_csv):
    df = pd.read_csv(option_trades_csv)
    df['pnl'] = df['pl']
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    df['session'] = pd.to_datetime(df['entry_time']).apply(classify_session)
    # Aggregate
    summary = {
        'total_trades': len(df),
        'profitable_trades': (df['pnl'] > 0).sum(),
        'losing_trades': (df['pnl'] <= 0).sum(),
        'win_rate': 100 * (df['pnl'] > 0).sum() / max(1, len(df)),
        'total_option_pnl': df['pnl'].sum(),
        'pnl_by_strategy': {},
        'pnl_by_day_session': [],
        'pnl_by_option_type': df.groupby('option_type')['pnl'].sum().to_dict(),
        'pnl_by_timeframe': {},
        'trades': df.to_dict('records')
    }
    # Strategy breakdown
    for strat, sdf in df.groupby('strategy'):
        summary['pnl_by_strategy'][strat] = {
            'pnl': sdf['pnl'].sum(),
            'trades': len(sdf),
            'win_rate': 100 * (sdf['pnl'] > 0).sum() / max(1, len(sdf)),
        }
    # By day and session
    for (d, s), sdf in df.groupby(['date', 'session']):
        summary['pnl_by_day_session'].append({
            'date': str(d),
            'session': s,
            'pnl': sdf['pnl'].sum(),
            'trades': len(sdf),
        })
    # By timeframe
    for tf, sdf in df.groupby('timeframe'):
        summary['pnl_by_timeframe'][tf] = {
            'pnl': sdf['pnl'].sum(),
            'trades': len(sdf),
        }
    return summary

def generate_option_pnl_report(option_pnl_analysis, run_id, option_trades_file_path):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('option_report_template.html')  # Use the enhanced template from previous message!
    output_from_parsed_template = template.render(
        run_id=run_id,
        option_pnl_analysis=option_pnl_analysis,
        option_trades_file_path=option_trades_file_path
    )
    report_dir = os.path.join('reports', f'run_{run_id}')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'option_pnl_report_{run_id}.html')
    with open(report_path, 'w') as f:
        f.write(output_from_parsed_template)
    print(f"Option P&L report generated at: {report_path}")
# trade_option_report.py

def generate_option_report_from_csv(option_trades_csv, run_id, strategy=None, timeframe=None,template_path='option_report_template.html'):
    import pandas as pd
    from datetime import time
    from jinja2 import Environment, FileSystemLoader
    import os

    def classify_session(ts):
        t = ts.time()
        if time(9,15) <= t < time(11,30): return 'Session 1 (9:15-11:30)'
        if time(11,30) <= t < time(13,45): return 'Session 2 (11:30-13:45)'
        if time(13,45) <= t <= time(15,30): return 'Session 3 (13:45-15:30)'
        return 'Outside Session'

    df = pd.read_csv(option_trades_csv)
    df['pnl'] = df['pl']
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    df['session'] = pd.to_datetime(df['entry_time']).apply(classify_session)

    # Aggregate stats (same as before)
    option_pnl_analysis = {
        'total_trades': len(df),
        'profitable_trades': (df['pnl'] > 0).sum(),
        'losing_trades': (df['pnl'] <= 0).sum(),
        'win_rate': 100 * (df['pnl'] > 0).sum() / max(1, len(df)),
        'total_option_pnl': df['pnl'].sum(),
        'pnl_by_strategy': {},
        'pnl_by_day_session': [],
        'pnl_by_option_type': df.groupby('option_type')['pnl'].sum().to_dict(),
        'pnl_by_timeframe': {},
        'trades': df.to_dict('records')
    }
    for strat, sdf in df.groupby('strategy'):
        option_pnl_analysis['pnl_by_strategy'][strat] = {
            'pnl': sdf['pnl'].sum(),
            'trades': len(sdf),
            'win_rate': 100 * (sdf['pnl'] > 0).sum() / max(1, len(sdf)),
        }
    for (d, s), sdf in df.groupby(['date', 'session']):
        option_pnl_analysis['pnl_by_day_session'].append({
            'date': str(d),
            'session': s,
            'pnl': sdf['pnl'].sum(),
            'trades': len(sdf),
        })
    for tf, sdf in df.groupby('timeframe'):
        option_pnl_analysis['pnl_by_timeframe'][tf] = {
            'pnl': sdf['pnl'].sum(),
            'trades': len(sdf),
        }

    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path) or '.'))
    template = env.get_template(os.path.basename(template_path))
    # output_from_parsed_template = template.render(
    #     run_id=run_id,
    #     option_pnl_analysis=option_pnl_analysis,
    #     option_trades_file_path=option_trades_csv
    # )
    output_from_parsed_template = template.render(
        run_id=run_id,
        strategy=strategy,
        timeframe=timeframe,
        option_pnl_analysis=option_pnl_analysis,
        option_trades_file_path=option_trades_file_path
    )
    report_dir = os.path.join('reports', f'run_{run_id}')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'option_pnl_report_{run_id}.html')
    with open(report_path, 'w') as f:
        f.write(output_from_parsed_template)
    print(f"Option P&L report generated at: {report_path}")
    return report_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to option trades CSV")
    parser.add_argument("--run_id", required=True, help="Run ID for report output")
    args = parser.parse_args()

    # Relative path in report for showing the file
    option_trades_file_path = args.csv
    # Aggregate
    analysis = analyze_option_trades(args.csv)
    # Render report
    generate_option_pnl_report(analysis, args.run_id, option_trades_file_path)
