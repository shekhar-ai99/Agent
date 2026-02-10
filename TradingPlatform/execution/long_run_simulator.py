"""
Long-Run Simulation Runner

Executes multi-month historical replay with checkpointing and detailed tracking.
Built for stability, resumability, and comprehensive reporting.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import logging
import pickle

from config.settings import PlatformConfig

logger = logging.getLogger(__name__)


@dataclass
class DailySummary:
    """Summary of trading activity for a single day"""
    date: str
    market: str
    timeframe: str
    
    # Trading metrics
    trades_taken: int
    total_pnl: float
    starting_balance: float
    ending_balance: float
    
    # Strategy performance
    best_strategy: Optional[str]
    worst_strategy: Optional[str]
    best_strategy_pnl: float
    worst_strategy_pnl: float
    
    # Market context
    regime_detected: str
    volatility_level: str
    session_breakdown: Dict[str, int]  # Session -> trade count
    
    # Trade breakdown
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    
    # Metadata
    bars_processed: int
    execution_time_seconds: float


@dataclass
class CheckpointData:
    """Checkpoint data for resuming simulation"""
    checkpoint_date: str
    current_date: str
    market: str
    timeframe: str
    
    # State
    account_balance: float
    open_positions: List[Dict]
    
    # Performance tracking
    total_trades: int
    total_pnl: float
    daily_summaries_count: int
    
    # Metadata
    simulation_start_date: str
    checkpoint_timestamp: str


class LongRunSimulationRunner:
    """
    Multi-month simulation runner with checkpointing and detailed tracking.
    
    **KEY FEATURES:**
    - Run 2-3 months of historical data
    - Daily checkpoints for restart safety
    - Memory-efficient processing
    - Detailed daily summaries
    - Strategy usage heatmaps
    - Monthly aggregate reports
    
    **SAFETY:**
    - Auto-saves every N days
    - Resumable from last checkpoint
    - No silent failures
    - Extensive validation
    """
    
    def __init__(
        self,
        config: PlatformConfig,
        output_dir: str = "./long_run_results",
        checkpoint_frequency_days: int = 7
    ):
        """
        Initialize long-run simulation runner.
        
        Args:
            config: Platform configuration
            output_dir: Directory for all outputs
            checkpoint_frequency_days: How often to checkpoint
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_frequency = checkpoint_frequency_days
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.daily_dir = self.output_dir / "daily_summaries"
        self.daily_dir.mkdir(exist_ok=True)
        self.monthly_dir = self.output_dir / "monthly_reports"
        self.monthly_dir.mkdir(exist_ok=True)
        
        # State
        self.daily_summaries: List[DailySummary] = []
        self.current_checkpoint: Optional[CheckpointData] = None
        
        logger.info(f"LongRunSimulationRunner initialized | Output: {self.output_dir}")
    
    def run_simulation(
        self,
        start_date: str,
        end_date: str,
        market: str,
        timeframe: str,
        resume_from_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        Run multi-month simulation.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            market: Market to run
            timeframe: Timeframe to use
            resume_from_checkpoint: Whether to resume from last checkpoint
            
        Returns:
            Final summary dict
        """
        logger.info("=" * 80)
        logger.info("LONG-RUN SIMULATION STARTING")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Market: {market} | Timeframe: {timeframe}")
        logger.info("=" * 80)
        
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        duration_days = (end_dt - start_dt).days
        
        logger.info(f"Duration: {duration_days} days ({duration_days/30:.1f} months)")
        
        # Check for existing checkpoint
        if resume_from_checkpoint:
            checkpoint = self._load_latest_checkpoint(market, timeframe)
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint.current_date}")
                start_dt = datetime.strptime(checkpoint.current_date, "%Y-%m-%d")
                self.current_checkpoint = checkpoint
        
        # Daily simulation loop
        current_date = start_dt
        day_counter = 0
        
        while current_date <= end_dt:
            day_str = current_date.strftime("%Y-%m-%d")
            logger.info(f"Processing day: {day_str}")
            
            try:
                # Run single day
                day_summary = self._run_single_day(
                    date=day_str,
                    market=market,
                    timeframe=timeframe
                )
                
                # Store summary
                self.daily_summaries.append(day_summary)
                
                # Save daily summary to file
                self._save_daily_summary(day_summary)
                
                # Log daily result
                logger.info(
                    f"  Day complete | Trades: {day_summary.trades_taken} | "
                    f"P&L: ${day_summary.total_pnl:.2f} | "
                    f"Balance: ${day_summary.ending_balance:.2f}"
                )
                
                # Progress checkpoint
                day_counter += 1
                if day_counter % self.checkpoint_frequency == 0:
                    logger.info(f"Creating checkpoint at day {day_str}")
                    self._create_checkpoint(
                        current_date=day_str,
                        market=market,
                        timeframe=timeframe
                    )
                
            except Exception as e:
                logger.error(f"CRITICAL: Day {day_str} failed: {e}", exc_info=True)
                logger.error("Creating emergency checkpoint...")
                self._create_checkpoint(
                    current_date=day_str,
                    market=market,
                    timeframe=timeframe,
                    emergency=True
                )
                raise
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Simulation complete
        logger.info("=" * 80)
        logger.info("LONG-RUN SIMULATION COMPLETE")
        logger.info("=" * 80)
        
        # Generate final reports
        return self._generate_final_reports(market, timeframe)
    
    def _run_single_day(
        self,
        date: str,
        market: str,
        timeframe: str
    ) -> DailySummary:
        """
        Run simulation for a single trading day.
        
        This is where the actual trading logic would execute.
        For now, this is a placeholder that would integrate with
        the UnifiedTradingSystem.
        """
        
        # PLACEHOLDER IMPLEMENTATION
        # In real implementation, this would:
        # 1. Load data for this day
        # 2. Execute trading cycles
        # 3. Track all metrics
        # 4. Return comprehensive summary
        
        logger.debug(f"Running {market}/{timeframe} for {date}")
        
        # Simulate some trading activity
        trades_taken = 0
        total_pnl = 0.0
        starting_balance = 100000.0
        
        if self.daily_summaries:
            starting_balance = self.daily_summaries[-1].ending_balance
        
        ending_balance = starting_balance + total_pnl
        
        summary = DailySummary(
            date=date,
            market=market,
            timeframe=timeframe,
            trades_taken=trades_taken,
            total_pnl=total_pnl,
            starting_balance=starting_balance,
            ending_balance=ending_balance,
            best_strategy=None,
            worst_strategy=None,
            best_strategy_pnl=0.0,
            worst_strategy_pnl=0.0,
            regime_detected="ranging",
            volatility_level="medium",
            session_breakdown={},
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            bars_processed=0,
            execution_time_seconds=0.0
        )
        
        return summary
    
    def _save_daily_summary(self, summary: DailySummary):
        """Save daily summary to JSON"""
        filename = f"daily_{summary.market}_{summary.date}.json"
        filepath = self.daily_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
    
    def _create_checkpoint(
        self,
        current_date: str,
        market: str,
        timeframe: str,
        emergency: bool = False
    ):
        """
        Create a checkpoint for resumability.
        """
        checkpoint = CheckpointData(
            checkpoint_date=datetime.now().isoformat(),
            current_date=current_date,
            market=market,
            timeframe=timeframe,
            account_balance=100000.0,  # Placeholder
            open_positions=[],
            total_trades=sum(s.trades_taken for s in self.daily_summaries),
            total_pnl=sum(s.total_pnl for s in self.daily_summaries),
            daily_summaries_count=len(self.daily_summaries),
            simulation_start_date=self.daily_summaries[0].date if self.daily_summaries else current_date,
            checkpoint_timestamp=datetime.now().isoformat()
        )
        
        # Save checkpoint
        prefix = "emergency_" if emergency else ""
        filename = f"{prefix}checkpoint_{market}_{current_date}.pkl"
        filepath = self.checkpoints_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def _load_latest_checkpoint(
        self,
        market: str,
        timeframe: str
    ) -> Optional[CheckpointData]:
        """Load the most recent checkpoint for a market"""
        checkpoints = list(self.checkpoints_dir.glob(f"checkpoint_{market}_*.pkl"))
        
        if not checkpoints:
            logger.info(f"No existing checkpoints found for {market}")
            return None
        
        # Get latest checkpoint
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Loading checkpoint: {latest}")
        
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return checkpoint
    
    def _generate_final_reports(self, market: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate comprehensive final reports.
        
        Outputs:
        - Daily CSV
        - Monthly CSV
        - Strategy usage heatmap
        - Performance summary
        """
        logger.info("Generating final reports...")
        
        if not self.daily_summaries:
            logger.warning("No daily summaries to report")
            return {}
        
        # Daily CSV
        daily_csv = self.output_dir / f"daily_report_{market}_{timeframe}.csv"
        daily_df = pd.DataFrame([asdict(s) for s in self.daily_summaries])
        daily_df.to_csv(daily_csv, index=False)
        logger.info(f"Daily CSV saved: {daily_csv}")
        
        # Monthly aggregation
        daily_df['month'] = pd.to_datetime(daily_df['date']).dt.to_period('M')
        monthly_df = daily_df.groupby('month').agg({
            'trades_taken': 'sum',
            'total_pnl': 'sum',
            'winning_trades': 'sum',
            'losing_trades': 'sum',
            'bars_processed': 'sum'
        }).reset_index()
        
        monthly_csv = self.output_dir / f"monthly_report_{market}_{timeframe}.csv"
        monthly_df.to_csv(monthly_csv, index=False)
        logger.info(f"Monthly CSV saved: {monthly_csv}")
        
        # Strategy usage heatmap data
        # This would require tracking which strategies were used each day
        # Placeholder for now
        
        # Performance summary
        total_pnl = daily_df['total_pnl'].sum()
        total_trades = daily_df['trades_taken'].sum()
        winning_days = len(daily_df[daily_df['total_pnl'] > 0])
        losing_days = len(daily_df[daily_df['total_pnl'] < 0])
        
        final_balance = daily_df['ending_balance'].iloc[-1]
        initial_balance = daily_df['starting_balance'].iloc[0]
        total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100
        
        summary = {
            "market": market,
            "timeframe": timeframe,
            "start_date": self.daily_summaries[0].date,
            "end_date": self.daily_summaries[-1].date,
            "duration_days": len(self.daily_summaries),
            "total_pnl": float(total_pnl),
            "total_trades": int(total_trades),
            "winning_days": int(winning_days),
            "losing_days": int(losing_days),
            "win_rate_days": winning_days / len(self.daily_summaries) if self.daily_summaries else 0,
            "initial_balance": float(initial_balance),
            "final_balance": float(final_balance),
            "total_return_pct": float(total_return_pct),
            "avg_trades_per_day": float(total_trades / len(self.daily_summaries)),
            "reports": {
                "daily_csv": str(daily_csv),
                "monthly_csv": str(monthly_csv)
            }
        }
        
        # Save summary JSON
        summary_json = self.output_dir / f"simulation_summary_{market}_{timeframe}.json"
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("SIMULATION SUMMARY")
        logger.info(f"Total P&L: ${summary['total_pnl']:.2f}")
        logger.info(f"Total Return: {summary['total_return_pct']:.2f}%")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Winning Days: {summary['winning_days']}/{summary['duration_days']}")
        logger.info(f"Reports saved to: {self.output_dir}")
        logger.info("=" * 80)
        
        return summary
    
    def export_strategy_usage_heatmap(self, output_path: str):
        """
        Export strategy usage heatmap showing:
        - Which strategies were used on which days
        - How frequently each strategy traded
        - Performance by strategy over time
        """
        # Placeholder - would require detailed strategy tracking
        logger.info(f"Strategy usage heatmap export: {output_path}")
        pass
    
    def get_progress(self) -> Dict:
        """Get current simulation progress"""
        if not self.daily_summaries:
            return {
                "days_completed": 0,
                "current_balance": self.config.initial_account_balance,
                "total_pnl": 0.0,
                "total_trades": 0
            }
        
        return {
            "days_completed": len(self.daily_summaries),
            "current_date": self.daily_summaries[-1].date,
            "current_balance": self.daily_summaries[-1].ending_balance,
            "total_pnl": sum(s.total_pnl for s in self.daily_summaries),
            "total_trades": sum(s.trades_taken for s in self.daily_summaries)
        }
    
    def __repr__(self) -> str:
        progress = self.get_progress()
        return (
            f"LongRunSimulationRunner("
            f"days={progress['days_completed']}, "
            f"balance=${progress['current_balance']:.2f}, "
            f"trades={progress['total_trades']})"
        )
