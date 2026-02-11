"""
Simple Web UI for Trading Platform

Flask-based web interface for:
- Running backtests
- Viewing strategy performance
- Analyzing results

Run: python web_ui.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import sys
import json
import subprocess
import threading
import os

# Add TradingPlatform to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_loader import DatasetLoader
from strategies.strategy_registry import StrategyRegistry

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global state
backtest_running = False
backtest_results = None


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')


@app.route('/api/strategies')
def get_strategies():
    """Get all available strategies"""
    try:
        strategies = StrategyRegistry.get_all()
        strategy_list = []
        
        for class_name, strategy_class in sorted(strategies.items()):
            try:
                inst = strategy_class()
                strategy_list.append({
                    'class_name': class_name,
                    'name': inst.name,
                    'version': inst.version,
                    'enabled': inst.enabled
                })
            except Exception as e:
                strategy_list.append({
                    'class_name': class_name,
                    'name': class_name,
                    'version': 'unknown',
                    'enabled': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'count': len(strategy_list),
            'strategies': strategy_list
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset/info')
def dataset_info():
    """Get dataset information"""
    try:
        df = DatasetLoader.load_nifty(timeframe="5min")
        
        return jsonify({
            'success': True,
            'total_bars': len(df),
            'start_date': str(df.index.min()),
            'end_date': str(df.index.max()),
            'columns': df.columns.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """Run backtest (quick validation)"""
    global backtest_running, backtest_results
    
    if backtest_running:
        return jsonify({'success': False, 'error': 'Backtest already running'}), 400
    
    try:
        backtest_running = True
        
        # Run in background thread
        def run_backtest_thread():
            global backtest_running, backtest_results
            try:
                # Execute PHASE1_2_COMPLETE.py
                result = subprocess.run(
                    [sys.executable, 'PHASE1_2_COMPLETE.py'],
                    cwd=str(Path(__file__).parent),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                # Load results
                results_file = Path(__file__).parent / 'performance_data' / 'phase1_2_results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        backtest_results = json.load(f)
                else:
                    backtest_results = {
                        'error': 'Results file not found',
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
            except Exception as e:
                backtest_results = {'error': str(e)}
            finally:
                backtest_running = False
        
        thread = threading.Thread(target=run_backtest_thread)
        thread.start()
        
        return jsonify({'success': True, 'message': 'Backtest started'})
        
    except Exception as e:
        backtest_running = False
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/backtest/status')
def backtest_status():
    """Check backtest status"""
    return jsonify({
        'running': backtest_running,
        'has_results': backtest_results is not None
    })


@app.route('/api/backtest/results')
def backtest_results_api():
    """Get backtest results"""
    if backtest_results is None:
        # Try loading from file
        results_file = Path(__file__).parent / 'performance_data' / 'phase1_2_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify({'success': True, 'results': results})
        else:
            return jsonify({'success': False, 'error': 'No results available'}), 404
    
    return jsonify({'success': True, 'results': backtest_results})


@app.route('/api/performance/summary')
def performance_summary():
    """Get performance summary"""
    try:
        results_file = Path(__file__).parent / 'performance_data' / 'phase1_2_results.json'
        
        if not results_file.exists():
            return jsonify({'success': False, 'error': 'No performance data available'}), 404
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Calculate summary stats
        successful = [r for r in results if r.get('status') == 'SUCCESS']
        failed = [r for r in results if r.get('status') == 'FAILED']
        
        # Sort by signal rate
        successful_sorted = sorted(
            successful,
            key=lambda x: x.get('signal_rate', 0),
            reverse=True
        )
        
        return jsonify({
            'success': True,
            'summary': {
                'total_strategies': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'top_strategies': successful_sorted[:10]
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/templates/<path:filename>')
def serve_template(filename):
    """Serve template files"""
    templates_dir = Path(__file__).parent / 'templates'
    return send_from_directory(templates_dir, filename)


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 TRADING PLATFORM WEB UI")
    print("=" * 80)
    print("\nStarting Flask server...")
    print("Access the UI at: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    # Create templates directory if not exists
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
