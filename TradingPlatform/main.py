#!/usr/bin/env python3
"""
Unified Trading Platform - Main Entrypoint

Single entry point for all trading operations:
- Backtesting
- Simulation
- Long-run multi-month simulations
- Parallel market execution

Usage:
    python main.py --mode backtest --markets india --timeframes 5min
    python main.py --mode simulation --markets india,crypto --timeframes 5min,15min
    python main.py --mode long-run --markets india --duration-months 3
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PlatformConfig, load_from_yaml, load_from_json, save_to_yaml
from analytics.performance_engine import StrategyPerformanceEngine
from core.base_selector import StrategySelector
from execution.parallel_runner import ParallelMarketRunner
from execution.long_run_simulator import LongRunSimulationRunner
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


def setup_logging(config: PlatformConfig):
    """Configure logging system"""
    log_level = getattr(logging, config.log_level.upper())
    
    # Format
    log_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = []
    
    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(console_handler)
    
    # File handler
    if config.log_file:
        log_dir = Path(config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / config.log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    logger.info("=" * 80)
    logger.info("TRADING PLATFORM INITIALIZED")
    logger.info(f"Log level: {config.log_level}")
    if config.log_file:
        logger.info(f"Log file: {log_path}")
    logger.info("=" * 80)


def validate_environment(config: PlatformConfig):
    """Validate environment and dependencies"""
    logger.info("Validating environment...")
    
    # Check output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {output_dir}")
    
    # Check performance data directory
    perf_dir = Path(config.performance_engine_storage_dir)
    perf_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Performance directory: {perf_dir}")
    
    # Check strategies
    if config.strategy_auto_discovery:
        num_strategies = len(STRATEGY_REGISTRY)
        logger.info(f"✓ Discovered {num_strategies} strategies")
        
        if num_strategies == 0:
            logger.error("CRITICAL: No strategies found in registry!")
            if config.fail_fast_on_errors:
                sys.exit(1)
    
    # Check enabled markets
    for market in config.enabled_markets:
        symbols = config.get_symbols_for_market(market)
        logger.info(f"✓ Market '{market}' configured with {len(symbols)} symbols")
    
    logger.info("Environment validation complete ✅")


def run_backtest_mode(config: PlatformConfig, args):
    """Run backtest mode"""
    logger.info("=" * 80)
    logger.info("MODE: BACKTEST")
    logger.info("=" * 80)
    
    from execution.modes.backtest_mode import BacktestMode
    
    markets = args.markets.split(',') if args.markets else config.enabled_markets
    timeframes = args.timeframes.split(',') if args.timeframes else [config.default_timeframe]
    
    for market in markets:
        for timeframe in timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running backtest: {market} @ {timeframe}")
            logger.info(f"{'='*60}\n")
            
            backtest = BacktestMode(
                capital=config.initial_account_balance,
                market=market,
                timeframe=timeframe
            )
            
            # Get strategies
            strategies = args.strategies.split(',') if args.strategies else list(STRATEGY_REGISTRY.keys())
            
            results = backtest.run(
                strategies=strategies,
                bypass_selector=args.bypass_selector,
                force_close_positions=True,
                save_pnl=config.save_trade_history,
                save_strategy_pnl=True,
                generate_html_report=config.save_html_reports
            )
            
            logger.info(f"\n{'='*60}")
            logger.info(f"BACKTEST COMPLETE: {market} @ {timeframe}")
            logger.info(f"P&L: ${results['total_pnl']:.2f}")
            logger.info(f"Return: {results['return_pct']:.2f}%")
            logger.info(f"Trades: {results['num_trades']}")
            logger.info(f"Sharpe: {results['sharpe_ratio']:.2f}")
            if 'html_report' in results:
                logger.info(f"Report: {results['html_report']}")
            logger.info(f"{'='*60}\n")


def run_simulation_mode(config: PlatformConfig, args):
    """Run simulation mode (parallel markets)"""
    logger.info("=" * 80)
    logger.info("MODE: SIMULATION (PARALLEL)")
    logger.info("=" * 80)
    
    # Override config if args provided
    markets = args.markets.split(',') if args.markets else config.enabled_markets
    timeframes = args.timeframes.split(',') if args.timeframes else config.enabled_timeframes
    
    # Update config
    config.enabled_markets = markets
    config.enabled_timeframes = timeframes
    
    # Create parallel runner
    runner = ParallelMarketRunner(
        config=config,
        max_workers=args.workers
    )
    
    # Run all markets
    results = runner.run_all_markets(
        mode="simulation",
        timeframes=timeframes,
        duration_days=args.duration_days
    )
    
    # Display aggregate results
    metrics = runner.get_aggregate_metrics()
    
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATION COMPLETE - AGGREGATE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Markets run: {metrics['successful_markets']}/{metrics['total_markets']}")
    logger.info(f"Total P&L: ${metrics['total_pnl']:.2f}")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Markets: {', '.join(metrics['markets'])}")
    logger.info("=" * 80)


def run_long_run_mode(config: PlatformConfig, args):
    """Run long-run simulation mode"""
    logger.info("=" * 80)
    logger.info("MODE: LONG-RUN SIMULATION")
    logger.info("=" * 80)
    
    market = args.markets.split(',')[0] if args.markets else config.enabled_markets[0]
    timeframe = args.timeframes.split(',')[0] if args.timeframes else config.default_timeframe
    
    # Calculate date range
    start_date = args.start_date or config.simulation_start_date
    
    if args.duration_months:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=args.duration_months * 30)
        end_date = end_dt.strftime("%Y-%m-%d")
    else:
        end_date = args.end_date or config.simulation_end_date
    
    logger.info(f"Market: {market}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Create long-run simulator
    simulator = LongRunSimulationRunner(
        config=config,
        output_dir=f"{config.output_dir}/long_run_{market}_{timeframe}",
        checkpoint_frequency_days=config.long_run_checkpoint_frequency_days
    )
    
    # Run simulation
    summary = simulator.run_simulation(
        start_date=start_date,
        end_date=end_date,
        market=market,
        timeframe=timeframe,
        resume_from_checkpoint=args.resume
    )
    
    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("LONG-RUN SIMULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Duration: {summary['duration_days']} days")
    logger.info(f"Total P&L: ${summary['total_pnl']:.2f}")
    logger.info(f"Total Return: {summary['total_return_pct']:.2f}%")
    logger.info(f"Total Trades: {summary['total_trades']}")
    logger.info(f"Winning Days: {summary['winning_days']}/{summary['duration_days']}")
    logger.info(f"Reports: {summary['reports']}")
    logger.info("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest on India market with 5min data
  python main.py --mode backtest --markets india --timeframes 5min
  
  # Run parallel simulation on India + Crypto
  python main.py --mode simulation --markets india,crypto --timeframes 5min,15min
  
  # Long-run 3-month simulation
  python main.py --mode long-run --markets india --duration-months 3
  
  # Use custom config file
  python main.py --config myconfig.yaml --mode backtest
        """
    )
    
    # === General arguments ===
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['backtest', 'simulation', 'long-run'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--markets',
        type=str,
        help='Comma-separated list of markets (e.g., india,crypto)'
    )
    
    parser.add_argument(
        '--timeframes',
        type=str,
        help='Comma-separated list of timeframes (e.g., 5min,15min)'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        help='Comma-separated list of strategy names (empty = all)'
    )
    
    # === Mode-specific arguments ===
    parser.add_argument(
        '--bypass-selector',
        action='store_true',
        help='Bypass strategy selector and use all strategies'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers for simulation mode'
    )
    
    parser.add_argument(
        '--duration-months',
        type=int,
        help='Duration in months for long-run mode'
    )
    
    parser.add_argument(
        '--duration-days',
        type=int,
        help='Duration in days for simulation mode'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint (long-run mode)'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save current config to file (YAML path)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = load_from_yaml(args.config)
        elif args.config.endswith('.json'):
            config = load_from_json(args.config)
        else:
            print(f"Unknown config format: {args.config}")
            sys.exit(1)
    else:
        config = PlatformConfig()
    
    # Setup logging
    setup_logging(config)
    
    # Validate environment
    validate_environment(config)
    
    # Save config if requested
    if args.save_config:
        save_to_yaml(config, args.save_config)
        logger.info(f"Configuration saved to {args.save_config}")
    
    # Route to mode handler
    try:
        if args.mode == 'backtest':
            run_backtest_mode(config, args)
        elif args.mode == 'simulation':
            run_simulation_mode(config, args)
        elif args.mode == 'long-run':
            run_long_run_mode(config, args)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ EXECUTION COMPLETE")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ EXECUTION FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

