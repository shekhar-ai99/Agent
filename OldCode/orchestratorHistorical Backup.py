import os
import subprocess
import sys
import logging
import shutil
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PipelineOrchestrator:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run-specific directories
        self.RUN_DIR = self.BASE_DIR / "runs" / self.RUN_ID
        self.DATA_DIR = self.RUN_DIR / "data"
        self.LOGS_DIR = self.RUN_DIR / "logs"
        self.REPORTS_DIR = self.RUN_DIR / "reports"
        
        # Script directory remains constant
        self.SCRIPT_DIR = self.BASE_DIR / "app"
        
        # Ensure directories exist
        self.RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        self.REPORTS_DIR.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.LOGS_DIR / 'orchestrator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Pipeline configuration
        self.SCRIPT_PIPELINE = [
            {
                'script': 'historical_data.py',
                'output_suffix': 'historical_data.csv',
                'timeout': 300
            },
            {
                'script': 'indicators.py',
                'input_suffix': 'historical_data.csv',
                'output_suffix': 'indicators.csv',
                'timeout': 180
            },
            {
                'script': 'signals.py',
                'input_suffix': 'indicators.csv',
                'output_suffix': 'signals.csv',
                'timeout': 240
            },
            # {
            #     'script': 'mylogicsignal.py',
            #     'input_suffix': 'signals.csv',
            #     'output_suffix': 'signals_final.csv',
            #     'timeout': 300
            # },
             {
                'script': 'signal_analyzer.py',
                'input_suffix': 'signals.csv',
                'output_suffix': 'signals_final.csv',
                'timeout': 300
            }
        ]
        
        self.execution_times = {}
        self.failed_scripts = []

    def cleanup_old_runs(self, max_runs_to_keep=5):
        """Keep only the most recent N runs"""
        runs_dir = self.BASE_DIR / "runs"
        if not runs_dir.exists():
            return
        
        # Get all run directories sorted by creation time
        run_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_ctime,
            reverse=True
        )
        
        # Remove older runs
        for old_run in run_dirs[max_runs_to_keep:]:
            try:
                shutil.rmtree(old_run)
                self.logger.info(f"Removed old run directory: {old_run}")
            except Exception as e:
                self.logger.error(f"Failed to remove {old_run}: {str(e)}")

    def find_matching_file(self, suffix: str) -> Path:
        """Find the newest matching file in data directory"""
        files = list(self.DATA_DIR.glob(f'*{suffix}'))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    def validate_script(self, script_name: str) -> bool:
        script_path = self.SCRIPT_DIR / script_name
        if not script_path.exists():
            self.logger.error(f"Script not found in example directory: {script_name}")
            self.logger.info(f"Expected location: {script_path}")
            return False
        return True

    def run_script(self, script_config: Dict) -> bool:
        """Execute a single script with proper file handling"""
        try:
            script_name = script_config['script']
            self.logger.info(f"Executing {script_name}")
            
            # Find input file if needed
            input_file = None
            if 'input_suffix' in script_config:
                input_file = self.find_matching_file(script_config['input_suffix'])
                if not input_file:
                    self.logger.error(f"No input file found matching *{script_config['input_suffix']}")
                    return False
                self.logger.info(f"Using input file: {input_file.name}")

            # Generate output filename
            output_file = self.DATA_DIR / f"nifty_{script_config['output_suffix']}"
            
            # Prepare command
            cmd = [sys.executable, str(self.SCRIPT_DIR / script_name)]
            if input_file:
                cmd.extend(["--input", str(input_file)])
            cmd.extend(["--output", str(output_file)])

            # Execute script
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                check=True,
                timeout=script_config.get('timeout', 300),
                capture_output=True,
                text=True,
                cwd=self.BASE_DIR
            )

            # Handle output
            self.execution_times[script_name] = (datetime.now() - start_time).total_seconds()
            
            if result.stderr:
                # Filter out common warnings
                filtered_err = [line for line in result.stderr.split('\n') 
                              if "NotOpenSSLWarning" not in line and line.strip()]
                if filtered_err:
                    self.logger.warning(f"Script output:\n" + '\n'.join(filtered_err))

            # Verify output file was created
            if not output_file.exists():
                self.logger.error(f"Output file not created: {output_file}")
                return False
            
            self.logger.info(f"Completed {script_name} in {self.execution_times[script_name]:.2f}s")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Script failed with code {e.returncode}:\n{e.stderr.strip()}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return False

    def run(self) -> bool:
        """Main execution method"""
        try:
            self.logger.info(f"Starting pipeline (Run ID: {self.RUN_ID})")
            
            # Setup data directory
            self.DATA_DIR.mkdir(exist_ok=True)

            # Verify all scripts exist first
            for script_config in self.SCRIPT_PIPELINE:
                if not self.validate_script(script_config['script']):
                    self.failed_scripts.append(script_config['script'])

            if self.failed_scripts:
                raise Exception("Missing required scripts")

            # Execute pipeline
            for script_config in self.SCRIPT_PIPELINE:
                if not self.run_script(script_config):
                    self.failed_scripts.append(script_config['script'])
                    # Continue to next script even if one fails

            # Generate report
            self.generate_report()
            
            if not self.failed_scripts:
                self.logger.info("Pipeline completed successfully")
                return True
            else:
                self.logger.error(f"Pipeline completed with errors in: {', '.join(self.failed_scripts)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False

    def generate_report(self) -> None:
        """Generate detailed execution report"""
        report_path = self.REPORTS_DIR / "pipeline_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Pipeline Execution Report\n{'='*30}\n")
            f.write(f"Run ID: {self.RUN_ID}\nTimestamp: {datetime.now()}\n\n")
            
            f.write("Execution Summary:\n")
            for script_config in self.SCRIPT_PIPELINE:
                script_name = script_config['script']
                status = "FAILED" if script_name in self.failed_scripts else "SUCCESS"
                duration = self.execution_times.get(script_name, 0)
                f.write(f"- {script_name}: {status} ({duration:.2f}s)\n")
            
            f.write("\nGenerated Files:\n")
            f.write(f"Data Directory: {self.DATA_DIR}\n")
            for script_config in self.SCRIPT_PIPELINE:
                if 'output_suffix' in script_config:
                    output_file = self.DATA_DIR / f"nifty_{script_config['output_suffix']}"
                    f.write(f"- {output_file.name}\n")

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    
    # Clean up old runs (keep last 5)
    orchestrator.cleanup_old_runs(max_runs_to_keep=5)
    
    # Run pipeline
    success = orchestrator.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)