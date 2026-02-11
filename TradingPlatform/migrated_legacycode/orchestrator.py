import sys
import subprocess
import logging
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import glob

# Import the backtesting workflow function from main_workflow.py.
# Adjust the import path as needed.
from main_workflow import run_full_backtest_workflow

class PipelineOrchestrator:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize all required directories
        self._setup_directories()
        self._setup_logging()
        
        # Define the scripts directory and set up pipeline phases.
        self.SCRIPT_DIR = self.BASE_DIR / "app"
        self.historical_script = "historical_data.py"
        self.indicator_script = "indicators.py"  # Runs the indicator calculator
        
        self.failed_scripts = []

    def _setup_directories(self):
        """Initialize all required directories."""
        self.RUN_DIR = self.BASE_DIR / "runs" / self.RUN_ID
        self.DATA_DIR = self.RUN_DIR / "data"
        self.LOGS_DIR = self.RUN_DIR / "logs"
        self.REPORTS_DIR = self.RUN_DIR / "reports"
        
        # Create directories if they don't exist
        self.RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        self.REPORTS_DIR.mkdir(exist_ok=True)

    def _setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.LOGS_DIR / 'orchestrator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_script(self, script: str, args_list: List[str], timeout: int) -> bool:
        """Run a given script with provided arguments and timeout.
        Logs output to a file named after the script.
        """
        log_file = self.LOGS_DIR / f"{script}.log"
        cmd = [sys.executable, str(self.SCRIPT_DIR / script)] + args_list
        self.logger.info(f"Running command: {' '.join(cmd)}")
        try:
            with open(log_file, 'w') as f:
                subprocess.run(
                    cmd,
                    check=True,
                    timeout=timeout,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.BASE_DIR
                )
            # Print log output
            with open(log_file) as f:
                self.logger.info(f"\n=== {script} Output ===\n" + f.read())
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{script} failed with code {e.returncode}")
            return False

    def run_historical_phase(self) -> bool:
        """Run the historical_data.py script to generate historical files.
        We pass an output directory so the script uses its default base name.
        """
        # Pass the DATA_DIR as the output so that historical_data.py writes its three files.
        output_arg = str(self.DATA_DIR)
        return self.run_script(
            script=self.historical_script,
            args_list=["--output", output_arg],
            timeout=300
        )

    def run_indicator_phase(self) -> bool:
        """For each historical data file generated, run the indicator calculator."""
        success = True
        pattern = str(self.DATA_DIR / "nifty_historical_data_*min.csv")
        historical_files = glob.glob(pattern)
        if not historical_files:
            self.logger.error("No historical data files found.")
            return False

        for hist_file in historical_files:
            hist_path = Path(hist_file)
            # Extract the timeframe suffix from the filename (e.g., "3min")
            parts = hist_path.stem.split("_")
            suffix = parts[-1] if parts else "default"

            indicator_output = self.DATA_DIR / f"nifty_indicators_{suffix}.csv"

            indicator_args = [
                "--input", str(hist_path),
                "--output", str(indicator_output)
            ]
            self.logger.info(f"Running indicator calculator for {hist_path.name}")
            if not self.run_script(
                script=self.indicator_script,
                args_list=indicator_args,
                timeout=180
            ):
                self.failed_scripts.append(f"{self.indicator_script} on {hist_path.name}")
                success = False

        return success

    def run_backtesting_phase(self) -> bool:
        """Run the full backtesting workflow using the indicator files generated.
        Builds a dictionary mapping timeframes to indicator file paths and calls
        the backtesting function from main_workflow.
        """
        # Build a dictionary: timeframe -> indicator file path.
        pattern = str(self.DATA_DIR / "nifty_indicators_*min.csv")
        indicator_files = glob.glob(pattern)
        if not indicator_files:
            self.logger.error("No indicator files found for backtesting.")
            return False

        data_files_by_timeframe: Dict[str, str] = {}
        for ind_file in indicator_files:
            ind_path = Path(ind_file)
            # Assume filename format: nifty_indicators_<timeframe>.csv.
            parts = ind_path.stem.split("_")
            timeframe = parts[-1] if parts else "default"
            data_files_by_timeframe[timeframe] = str(ind_path)

        # Define output directory for backtesting results within the current run folder.
        backtest_output_dir = str(self.RUN_DIR / "backtest_results")
        self.logger.info("Starting backtesting phase with the following data files:")
        for tf, path in data_files_by_timeframe.items():
            self.logger.info(f"  {tf}: {path}")
        
        try:
            # Call the backtesting workflow function from main_workflow.py.
            run_full_backtest_workflow(
                data_paths=data_files_by_timeframe,
                output_base_dir=backtest_output_dir
            )
            return True
        except Exception as e:
            self.logger.error(f"Backtesting phase failed: {e}", exc_info=True)
            return False

    def run(self) -> bool:
        """Execute the entire pipeline in three phases."""
        self.logger.info(f"Starting pipeline (Run ID: {self.RUN_ID})")
        
        if not self.run_historical_phase():
            self.logger.error("Historical data phase failed.")
            return False
        
        if not self.run_indicator_phase():
            self.logger.error("Indicator phase failed.")
            return False

        if not self.run_backtesting_phase():
            self.logger.error("Backtesting phase failed.")
            return False

        self.logger.info("Pipeline completed successfully")
        return True

if __name__ == "__main__":
    try:
        orchestrator = PipelineOrchestrator()
        success = orchestrator.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Fatal error in orchestrator: {e}", exc_info=True)
        sys.exit(1)
