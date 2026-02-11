"""
TradingPlatform API Backend Stub
==================================

This is a reference implementation showing how to build Flask endpoints
that the UI expects. Adapt this to your actual backend structure.

Installation:
    pip install flask flask-cors python-dateutil

Usage:
    python backend_stub.py
    Then open http://localhost:5000/api/ui
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import random

app = Flask(__name__)
CORS(app)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sample_results(config):
    """Generate sample backtest results matching the config."""
    
    # Parse dates
    start_date = datetime.strptime(config.get('start_date', '2025-01-01'), '%Y-%m-%d')
    end_date = datetime.strptime(config.get('end_date', '2025-03-10'), '%Y-%m-%d')
    days = (end_date - start_date).days
    
    # Generate equity curve
    equity_curve = []
    current_equity = config.get('capital', 100000)
    current_date = start_date
    
    while current_date <= end_date:
        current_equity += random.uniform(-1, 1) * 2000  # Daily changes
        equity_curve.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'equity': max(current_equity, config.get('capital', 100000) * 0.5)
        })
        current_date += timedelta(days=1)
    
    # Generate trades
    trades = []
    strategies = ['EMACrossover', 'RSIMACDStrategy', 'BollingerMeanReversion', 'SuperTrendADX']
    trade_id = 1
    
    for _ in range(random.randint(30, 60)):
        entry_date = start_date + timedelta(days=random.randint(0, days))
        exit_date = entry_date + timedelta(hours=random.randint(1, 8))
        pnl = random.uniform(-2000, 3000)
        
        trades.append({
            'trade_id': trade_id,
            'strategy': random.choice(strategies),
            'entry_time': entry_date.isoformat(),
            'exit_time': exit_date.isoformat(),
            'entry_price': random.uniform(22000, 24000),
            'exit_price': random.uniform(22000, 24000),
            'quantity': random.randint(10, 100),
            'pnl': round(pnl, 2),
            'regime': random.choice(['TRENDING', 'RANGING', 'VOLATILE']),
            'volatility': random.choice(['LOW', 'MEDIUM', 'HIGH']),
            'session': random.choice(['MORNING', 'MIDDAY', 'AFTERNOON']),
            'day': entry_date.strftime('%A'),
            'status': 'closed'
        })
        trade_id += 1
    
    # Calculate metrics
    closed_trades = [t for t in trades if t.get('pnl')]
    wins = [t for t in closed_trades if t['pnl'] > 0]
    losses = [t for t in closed_trades if t['pnl'] <= 0]
    
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    
    max_equity = max(e['equity'] for e in equity_curve)
    min_equity = min(e['equity'] for e in equity_curve)
    max_drawdown = ((max_equity - min_equity) / max_equity * 100) if max_equity > 0 else 0
    
    return {
        'run_id': f"run-{datetime.now().timestamp()}",
        'status': 'completed',
        'market': config.get('market', 'india'),
        'exchange': config.get('exchange', 'nse'),
        'instrument': config.get('instrument', 'NIFTY50'),
        'timeframe': config.get('timeframe', '5m'),
        'mode': config.get('mode', 'backtest'),
        'capital': config.get('capital', 100000),
        'start_date': config.get('start_date', '2025-01-01'),
        'end_date': config.get('end_date', '2025-03-10'),
        'total_trades': len(closed_trades),
        'win_rate': round((len(wins) / len(closed_trades) * 100) if closed_trades else 0, 2),
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe_ratio': round(random.uniform(0.5, 2.0), 2),
        'equity_curve': equity_curve,
        'trades': trades
    }


def calculate_strategy_rankings(trades):
    """Calculate strategy performance rankings from trades."""
    
    strategy_metrics = {}
    
    for trade in trades:
        strategy = trade['strategy']
        if strategy not in strategy_metrics:
            strategy_metrics[strategy] = {'trades': []}
        strategy_metrics[strategy]['trades'].append(trade)
    
    rankings = []
    rank = 1
    
    for strategy, data in sorted(
        strategy_metrics.items(),
        key=lambda x: len([t for t in x[1]['trades'] if t['pnl'] > 0]) / max(len(x[1]['trades']), 1),
        reverse=True
    ):
        trades_list = data['trades']
        wins = [t for t in trades_list if t['pnl'] > 0]
        losses = [t for t in trades_list if t['pnl'] <= 0]
        
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        all_pnl = [t['pnl'] for t in trades_list]
        
        sample_trade = trades_list[0] if trades_list else {}
        rankings.append({
            'rank': rank,
            'strategy_name': strategy,
            'trades_count': len(trades_list),
            'win_rate': round((len(wins) / len(trades_list) * 100) if trades_list else 0, 2),
            'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'avg_pnl': round(sum(all_pnl) / len(all_pnl), 2) if all_pnl else 0,
            'max_pnl': max(all_pnl) if all_pnl else 0,
            'min_pnl': min(all_pnl) if all_pnl else 0,
            'market': 'india',
            'day_of_week': sample_trade.get('day', 'Monday'),
            'session': sample_trade.get('session', 'MORNING'),
            'volatility_bucket': sample_trade.get('volatility', 'MEDIUM'),
            'regime': sample_trade.get('regime', 'RANGING'),
            'timeframe': '5m'
        })
        rank += 1
    
    return rankings


def build_ui_results(results):
    """Normalize results into the UI summary/series schema."""
    equity_curve = results.get('equity_curve', [])
    trades = results.get('trades', [])

    series_equity = [
        {'label': point.get('date') or point.get('timestamp'), 'value': point.get('equity')}
        for point in equity_curve
    ]

    drawdown_curve = []
    peak = None
    for point in series_equity:
        value = point.get('value') or 0
        peak = value if peak is None else max(peak, value)
        drawdown = ((peak - value) / peak * 100) if peak else 0
        drawdown_curve.append({'label': point.get('label'), 'value': round(drawdown, 2)})

    trades_by_day = {}
    for trade in trades:
        entry_time = trade.get('entry_time') or trade.get('entryTime')
        if not entry_time:
            continue
        day_key = entry_time.split('T')[0]
        trades_by_day[day_key] = trades_by_day.get(day_key, 0) + 1

    trades_per_day = [
        {'label': day, 'value': count}
        for day, count in sorted(trades_by_day.items())
    ]

    pnl_values = [trade.get('pnl') for trade in trades if trade.get('pnl') is not None]
    pnl_distribution = []
    if pnl_values:
        min_pnl = min(pnl_values)
        max_pnl = max(pnl_values)
        bucket_count = 6
        bucket_size = (max_pnl - min_pnl) / bucket_count if max_pnl != min_pnl else 1
        buckets = [0] * bucket_count
        for pnl in pnl_values:
            idx = int((pnl - min_pnl) / bucket_size) if bucket_size else 0
            idx = min(idx, bucket_count - 1)
            buckets[idx] += 1
        pnl_distribution = [
            {
                'label': f"{round(min_pnl + i * bucket_size, 2)}",
                'value': buckets[i]
            }
            for i in range(bucket_count)
        ]

    gross_profit = results.get('gross_profit', 0)
    gross_loss = results.get('gross_loss', 0)
    net_pnl = results.get('net_pnl')
    if net_pnl is None:
        net_pnl = round(gross_profit - gross_loss, 2)

    summary = {
        'total_trades': results.get('total_trades', 0),
        'win_rate': results.get('win_rate', 0),
        'net_pnl': net_pnl,
        'max_drawdown': results.get('max_drawdown', 0),
        'sharpe_ratio': results.get('sharpe_ratio', 0)
    }

    return {
        'run_id': results.get('run_id'),
        'status': results.get('status', 'completed').title(),
        'summary': summary,
        'series': {
            'equity_curve': series_equity,
            'drawdown_curve': drawdown_curve,
            'trades_per_day': trades_per_day,
            'pnl_distribution': pnl_distribution
        },
        'meta': {
            'market': results.get('market'),
            'exchange': results.get('exchange'),
            'instrument': results.get('instrument'),
            'timeframe': results.get('timeframe'),
            'mode': results.get('mode')
        }
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/ui')
def ui():
    """Serve the UI (assumes it's in the ui/ folder)."""
    return app.send_static_file('index.html')


@app.route('/api/run/backtest', methods=['POST'])
@app.route('/run/backtest', methods=['POST'])
def submit_backtest():
    """
    Submit a backtest run.
    
    Expected JSON:
    {
        "market": "india|crypto",
        "exchange": "nse|bse|global",
        "instrument": "NIFTY50",
        "timeframe": "5m",
        "mode": "backtest",
        "capital": 100000,
        "risk_per_trade": 2.0,
        "start_date": "2025-01-01",
        "end_date": "2025-03-10"
    }
    """
    try:
        config = request.get_json()
        
        # Validate config
        required_fields = ['market', 'exchange', 'instrument', 'timeframe', 'capital']
        if not all(field in config for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Generate results (in production, this would call actual backtest engine)
        results = generate_sample_results(config)
        
        return jsonify(build_ui_results(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run/simulation', methods=['POST'])
@app.route('/run/simulation', methods=['POST'])
def submit_simulation():
    """
    Submit a simulation run (live data with paper trades).
    
    Same payload as backtest, but mode='simulation'.
    """
    try:
        config = request.get_json()
        config['mode'] = 'simulation'
        
        results = generate_sample_results(config)
        results['status'] = 'running'  # Simulations start running
        
        return jsonify(build_ui_results(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<run_id>')
@app.route('/results/<run_id>')
def get_results(run_id):
    """
    Get results for a specific run.
    
    Returns the full results object with equity curve and trades.
    """
    try:
        # In production, fetch from database
        # For now, generate sample data
        sample_config = {
            'market': 'india',
            'exchange': 'nse',
            'instrument': 'NIFTY50',
            'timeframe': '5m',
            'capital': 100000,
            'start_date': '2025-01-01',
            'end_date': '2025-03-10'
        }
        
        results = generate_sample_results(sample_config)
        results['run_id'] = run_id
        
        return jsonify(build_ui_results(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<run_id>/trades')
@app.route('/results/<run_id>/trades')
def get_trades(run_id):
    """
    Get detailed trade information for a run.
    
    Returns array of trade objects.
    """
    try:
        # In production, fetch from database
        sample_config = {
            'market': 'india',
            'exchange': 'nse',
            'instrument': 'NIFTY50',
            'timeframe': '5m',
            'capital': 100000,
            'start_date': '2025-01-01',
            'end_date': '2025-03-10'
        }
        
        results = generate_sample_results(sample_config)
        
        return jsonify({'trades': results['trades']}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/strategy_ranking.json')
@app.route('/results/strategy_ranking.json')
def get_strategy_ranking():
    """
    Get strategy ranking data.
    
    Supports optional filters:
    ?market=india&volatility=HIGH&regime=TRENDING
    """
    try:
        # Query parameters for filtering
        market = request.args.get('market')
        volatility = request.args.get('volatility')
        regime = request.args.get('regime')
        
        # Generate sample results
        sample_config = {
            'market': 'india',
            'exchange': 'nse',
            'instrument': 'NIFTY50',
            'timeframe': '5m',
            'capital': 100000,
            'start_date': '2025-01-01',
            'end_date': '2025-03-10'
        }
        
        results = generate_sample_results(sample_config)
        rankings = calculate_strategy_rankings(results['trades'])
        
        return jsonify({'strategies': rankings}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/runs/history')
@app.route('/runs/history')
def get_run_history():
    """
    Get history of all completed runs.
    
    Returns array of run summaries.
    """
    try:
        # In production, fetch from database
        sample_runs = [
            {
                'run_id': f'run-{1000 + i}',
                'instrument': 'NIFTY50',
                'timeframe': '5m',
                'total_trades': random.randint(20, 80),
                'win_rate': random.uniform(45, 75),
                'profit_factor': random.uniform(1.0, 3.0),
                'created_at': (datetime.now() - timedelta(days=i)).isoformat(),
                'status': 'completed'
            }
            for i in range(1, 6)
        ]
        
        return jsonify(sample_runs), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TradingPlatform API Backend Stub")
    print("="*60)
    print("\n✓ API running on http://localhost:5000")
    print("✓ Endpoints available:")
    print("  • POST   /api/run/backtest → Submit backtest")
    print("  • POST   /api/run/simulation → Submit simulation")
    print("  • GET    /api/results/{run_id} → Get results")
    print("  • GET    /api/results/{run_id}/trades → Get trades")
    print("  • GET    /api/results/strategy_ranking.json → Get rankings")
    print("  • GET    /api/runs/history → Get run history")
    print("  • POST   /run/backtest → Submit backtest")
    print("  • POST   /run/simulation → Submit simulation")
    print("  • GET    /results/{run_id} → Get results")
    print("  • GET    /results/{run_id}/trades → Get trades")
    print("  • GET    /results/strategy_ranking.json → Get rankings")
    print("  • GET    /runs/history → Get run history")
    print("\n✓ CORS enabled for all origins")
    print("\nNote: This is a stub with generated data.")
    print("Connect to actual TradingPlatform backtesting in production.\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
