# Migrated Legacy Code from OldCode/

This directory contains a complete backup of all code from the legacy OldCode/ folder.
Migrated: 2026-02-10
Now that this backup exists, code can be safely integrated into TradingPlatform/.

## Organization

### Strategies Directory
- `strategies.py` - Main strategy collection
- `strategies1222.py` - Strategy variant
- `strategies copy.py` - Comprehensive strategy backup
- `trading_strategies.py` - Trading strategy implementations
- `strategy.py` - Base strategy framework
- `aiscore.py` - AI-driven signal generation
- `mylogicsignal.py` - Custom signal logic
- `ai_adaptive_strategy.py` - Adaptive market regime strategy
- `csAlpha.py`, `csalphav2.py`, `cs_alphatrend_modified.py` - AlphaTrend variants

### Backtesting
- `backtest_engine.py` - Backtesting engine
- `backtester_engine.py` - Alternative backtester
- `backtest_analyzerV1.py` - Backtest analysis
- `backtest_runner_india_5m.py` - NIFTY/India-specific backtest runner
- `backtester_analyzer.py` - Trade analysis
- `backtestData.py` - Backtest data utilities

### Simulation & Live Trading
- `simulation_engine.py` - Simulation engine
- `run_simulation_step.py` - Simulation runner
- `live_trading_engine.py` - Live trade execution
- `trade_management.py` - Trade management logic

### Data & Processing
- `processData.py` - Data transformation (729 lines - complex logic)
- `feature_engine.py` - Feature engineering
- `load_data.py` - Historical data loading
- `data_io.py` - I/O utilities
- `data_fetcher.py`, `fetcher_v2.py` - Data fetching
- `historical_data.py` - Historical data handler

### Analytics & Reporting
- `reporting.py` - Report generation
- `signal_analyzer.py` - Signal analysis
- `signal_processor.py` - Signal processing
- `signal_processor_with_stats.py` - Advanced signal stats
- `performance_logger_mongo.py` - MongoDB performance logging
- `backtest_analyzerV1.py` - Backtest analysis

### Broker Integration (Angel One)
- `angel_one_api.py` - Angel One API wrapper
- `angel_data_fetcher.py` - Angel One data fetching
- `angel_one_websocket.py` - WebSocket integration
- `angel_one_instrument_manager.py` - Instrument management
- `angel_one_market_data.py` - Market data handling
- `login.py` - Angel One login

### Infrastructure
- `orchestrator.py` - Pipeline orchestration (190 lines)
- `main_workflow.py` - Main workflow (452 lines)
- `pipeline_manager.py` - Pipeline management (481 lines)
- `strategy_utils.py` - Strategy utilities
- `utils.py` - General utilities

### Optimization & Tuning
- `optuna_tuner.py` - Hyperparameter tuning
- `llm_tuner.py` - LLM tuning

### Other Analyses
- `sentiment_analysis.py` - Sentiment analysis
- `context_data_analyzer.py` - Context analysis
- `realtime_analyzer.py` - Real-time analysis
- `option_trade_executor.py` - Option trading
- `option_trade_executor_wrapper.py` - Option executor wrapper

### Additional Files
- `niftyAnalyser.py` - NIFTY-specific analysis
- `vwap1.py` - VWAP indicator
- `moving_average_crossover.py` - MA strategy
- `trailing_sl_strategy.py` - Trailing SL logic
- `openaisignal.py` - OpenAI-based signals
- `mistrel.py` - Alternative signal source
- `cli.py` - Command-line interface
- `web_app.py` - Web application
- `run_backtest.py` - Backtest runner script
- `run_feature_engine.py` - Feature engine runner
- `run_strategy_tester.py` - Strategy tester runner
- `clean_option_df.py` - Option data cleaning
- `downlaodinstrumentlist.py` - Instrument list download
- Plus more...

### HTML/CSS/JS Templates
- `report_template.html`, `report_template_detailed.html` - Report templates
- `option_report_template.html` - Option report template
- `index.html` - Main template
- `report.css`, `report.js` - Styling and scripting

### Data Folders
- `btcData/` - Bitcoin data
- `instruments/` - Instrument definitions
- `raw/` - Raw data
- `datawithindicator/` - Processed data
- `output/` - Output files

## Integration Notes

When integrating files into TradingPlatform, ensure:
1. Fix relative imports (e.g., `from app.xxx` → appropriate TradingPlatform path)
2. Consolidate duplicate strategy implementations
3. Merge Angel One APIs with existing broker integrations
4. Test all data loading and backtest logic against TradingPlatform engines
5. Verify 100% parity of trading logic

## All Logic Preserved
✅ No code loss - 86 Python files + 4 data folders + HTML/JS/CSS assets
✅ Zero functionality lost
✅ Ready for integration at any time
