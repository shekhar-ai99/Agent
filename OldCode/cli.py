import argparse
import pandas as pd
from .utils import load_config, setup_logger
from .backtester import SimpleBacktester
from .strategies.moving_average_crossover import MovingAverageCrossover

def main():
    parser = argparse.ArgumentParser(description="Bitcoin Strategy Tester")
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(log_file="logs/run.log")

    df = pd.read_csv(config["data_source"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    strategy_name = config["strategy"]
    if strategy_name == "MovingAverageCrossover":
        strategy = MovingAverageCrossover(config["strategy_params"])
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    backtester = SimpleBacktester(strategy, df, config["initial_balance"], logger=logger)
    final_value, trades = backtester.run()
    print(f"Final Portfolio Value: {final_value}")
    for trade in trades:
        print(trade)
