"""
Strategy Performance Engine

Tracks, analyzes, and persists strategy performance metrics across different market contexts.
Enables data-driven strategy selection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    """Defines the context in which a strategy operates"""
    strategy_name: str
    market: str  # "india" or "crypto"
    timeframe: str  # "1min", "5min", "15min", etc.
    regime: str  # "trending", "ranging", "volatile"
    session: Optional[str] = None  # Session name if applicable
    volatility_bucket: str = "medium"  # "low", "medium", "high"


@dataclass
class StrategyPerformanceMetrics:
    """Complete performance metrics for a strategy in a context"""
    # Context
    strategy_name: str
    market: str
    timeframe: str
    regime: str
    session: Optional[str]
    volatility_bucket: str
    
    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # P&L Metrics
    total_pnl: float
    avg_win: float
    avg_loss: float
    avg_pnl_per_trade: float
    
    # Performance Ratios
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Risk Metrics
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_duration_seconds: float
    
    # Consistency Metrics
    consistency_score: float  # 0-100
    reliability_score: float  # 0-100
    
    # Composite Score (for ranking)
    composite_score: float = 0.0
    
    # Metadata
    last_updated: str = ""
    first_trade_date: Optional[str] = None
    last_trade_date: Optional[str] = None
    
    def calculate_composite_score(self) -> float:
        """
        Calculate composite score for ranking strategies.
        
        Factors:
        - Win rate (25%)
        - Profit factor (25%)
        - Sharpe ratio (20%)
        - Consistency (15%)
        - Reliability (15%)
        
        Returns score 0-100
        """
        # Normalize win rate (0-1 -> 0-25)
        wr_score = self.win_rate * 25
        
        # Normalize profit factor (assume 2.0 is excellent)
        pf_score = min(self.profit_factor / 2.0, 1.0) * 25
        
        # Normalize Sharpe (assume 2.0 is excellent)
        sharpe_score = min(abs(self.sharpe_ratio) / 2.0, 1.0) * 20
        
        # Consistency and reliability are already 0-100
        consistency_score = (self.consistency_score / 100) * 15
        reliability_score = (self.reliability_score / 100) * 15
        
        # Penalty for insufficient trades
        trade_penalty = 1.0
        if self.total_trades < 10:
            trade_penalty = 0.5
        elif self.total_trades < 30:
            trade_penalty = 0.8
        
        self.composite_score = (wr_score + pf_score + sharpe_score + 
                               consistency_score + reliability_score) * trade_penalty
        
        return self.composite_score


class StrategyPerformanceEngine:
    """
    Analyzes strategy performance and maintains performance database.
    
    Responsibilities:
    1. Compute performance metrics from trade history
    2. Persist metrics to CSV/JSON
    3. Query performance by context
    4. Rank strategies for selection
    """
    
    def __init__(self, storage_dir: str = "./performance_data"):
        """
        Initialize the performance engine.
        
        Args:
            storage_dir: Directory to store performance data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.storage_dir / "strategy_performance.csv"
        self.json_path = self.storage_dir / "strategy_performance.json"
        
        # In-memory cache
        self.performance_db: Dict[str, StrategyPerformanceMetrics] = {}
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"StrategyPerformanceEngine initialized | Storage: {self.storage_dir}")
    
    def _load_existing_data(self):
        """Load existing performance data from disk"""
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                logger.info(f"Loaded {len(df)} performance records from {self.csv_path}")
                
                # Build in-memory cache
                for _, row in df.iterrows():
                    key = self._make_key(
                        row['strategy_name'],
                        row['market'],
                        row['timeframe'],
                        row['regime'],
                        row.get('session'),
                        row['volatility_bucket']
                    )
                    
                    metrics = StrategyPerformanceMetrics(
                        strategy_name=row['strategy_name'],
                        market=row['market'],
                        timeframe=row['timeframe'],
                        regime=row['regime'],
                        session=row.get('session'),
                        volatility_bucket=row['volatility_bucket'],
                        total_trades=int(row['total_trades']),
                        winning_trades=int(row['winning_trades']),
                        losing_trades=int(row['losing_trades']),
                        total_pnl=float(row['total_pnl']),
                        avg_win=float(row['avg_win']),
                        avg_loss=float(row['avg_loss']),
                        avg_pnl_per_trade=float(row['avg_pnl_per_trade']),
                        win_rate=float(row['win_rate']),
                        profit_factor=float(row['profit_factor']),
                        sharpe_ratio=float(row['sharpe_ratio']),
                        sortino_ratio=float(row['sortino_ratio']),
                        max_drawdown=float(row['max_drawdown']),
                        max_drawdown_pct=float(row['max_drawdown_pct']),
                        avg_trade_duration_seconds=float(row['avg_trade_duration_seconds']),
                        consistency_score=float(row['consistency_score']),
                        reliability_score=float(row['reliability_score']),
                        composite_score=float(row['composite_score']),
                        last_updated=row['last_updated'],
                        first_trade_date=row.get('first_trade_date'),
                        last_trade_date=row.get('last_trade_date')
                    )
                    
                    self.performance_db[key] = metrics
                    
            except Exception as e:
                logger.warning(f"Failed to load existing data: {e}")
    
    @staticmethod
    def _make_key(strategy: str, market: str, timeframe: str, 
                  regime: str, session: Optional[str], volatility: str) -> str:
        """Create unique key for context"""
        session_str = session or "N/A"
        return f"{strategy}|{market}|{timeframe}|{regime}|{session_str}|{volatility}"
    
    def analyze_trades(
        self,
        context: StrategyContext,
        trades: List[Dict]
    ) -> StrategyPerformanceMetrics:
        """
        Analyze trade history and compute performance metrics.
        
        Args:
            context: Strategy context
            trades: List of trade dicts with keys:
                   - pnl (float)
                   - duration (float, seconds)
                   - entry_time (datetime or str)
                   - exit_time (datetime or str)
                   
        Returns:
            StrategyPerformanceMetrics object
        """
        if not trades:
            logger.warning(f"No trades provided for {context.strategy_name}")
            return self._empty_metrics(context)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        
        # Basic counts
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        # P&L metrics
        total_pnl = df['pnl'].sum()
        avg_pnl_per_trade = df['pnl'].mean()
        
        wins = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        
        # Performance ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_wins = wins.sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Sharpe and Sortino ratios
        sharpe_ratio = self._calculate_sharpe(df['pnl'])
        sortino_ratio = self._calculate_sortino(df['pnl'])
        
        # Drawdown
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()
        
        # Calculate max drawdown percentage
        if running_max.max() > 0:
            max_dd_pct = (max_drawdown / running_max.max()) * 100
        else:
            max_dd_pct = 0.0
        
        # Trade duration
        if 'duration' in df.columns:
            avg_duration = df['duration'].mean()
        else:
            avg_duration = 0.0
        
        # Consistency score (0-100)
        consistency_score = self._calculate_consistency(df['pnl'])
        
        # Reliability score (0-100)
        reliability_score = self._calculate_reliability(win_rate, profit_factor, total_trades)
        
        # Trade dates
        first_trade_date = None
        last_trade_date = None
        if 'entry_time' in df.columns:
            first_trade_date = str(df['entry_time'].min())
            last_trade_date = str(df['entry_time'].max())
        
        # Create metrics object
        metrics = StrategyPerformanceMetrics(
            strategy_name=context.strategy_name,
            market=context.market,
            timeframe=context.timeframe,
            regime=context.regime,
            session=context.session,
            volatility_bucket=context.volatility_bucket,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_pnl_per_trade=avg_pnl_per_trade,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_dd_pct,
            avg_trade_duration_seconds=avg_duration,
            consistency_score=consistency_score,
            reliability_score=reliability_score,
            last_updated=datetime.now().isoformat(),
            first_trade_date=first_trade_date,
            last_trade_date=last_trade_date
        )
        
        # Calculate composite score
        metrics.calculate_composite_score()
        
        logger.info(
            f"Analyzed {context.strategy_name} in {context.market}/{context.regime}: "
            f"{total_trades} trades | Win%: {win_rate:.2%} | PF: {profit_factor:.2f} | "
            f"Score: {metrics.composite_score:.1f}"
        )
        
        return metrics
    
    def _empty_metrics(self, context: StrategyContext) -> StrategyPerformanceMetrics:
        """Create empty metrics for context with no trades"""
        return StrategyPerformanceMetrics(
            strategy_name=context.strategy_name,
            market=context.market,
            timeframe=context.timeframe,
            regime=context.regime,
            session=context.session,
            volatility_bucket=context.volatility_bucket,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_pnl_per_trade=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            avg_trade_duration_seconds=0.0,
            consistency_score=0.0,
            reliability_score=0.0,
            composite_score=0.0,
            last_updated=datetime.now().isoformat()
        )
    
    @staticmethod
    def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)  # Annualized
    
    @staticmethod
    def _calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    @staticmethod
    def _calculate_consistency(returns: pd.Series) -> float:
        """
        Calculate consistency score (0-100).
        Measures how consistent the returns are.
        Higher score = less variance in trade outcomes.
        """
        if len(returns) < 2:
            return 0.0
        
        # Coefficient of variation (inverted and normalized)
        mean_return = returns.mean()
        if mean_return == 0:
            return 0.0
        
        cv = abs(returns.std() / mean_return)
        
        # Lower CV = higher consistency
        # Normalize to 0-100 scale (assume CV of 5 is very inconsistent)
        consistency = max(0, 100 - (cv * 20))
        
        return min(consistency, 100.0)
    
    @staticmethod
    def _calculate_reliability(win_rate: float, profit_factor: float, 
                              total_trades: int) -> float:
        """
        Calculate reliability score (0-100).
        Combines win rate, profit factor, and sample size.
        """
        # Win rate component (0-40)
        wr_component = win_rate * 40
        
        # Profit factor component (0-40, assuming 2.0 is excellent)
        pf_component = min(profit_factor / 2.0, 1.0) * 40
        
        # Sample size component (0-20)
        sample_component = min(total_trades / 100, 1.0) * 20
        
        return wr_component + pf_component + sample_component
    
    def update_performance(
        self,
        context: StrategyContext,
        trades: List[Dict]
    ) -> StrategyPerformanceMetrics:
        """
        Update performance metrics for a strategy context.
        
        Args:
            context: Strategy context
            trades: Trade history
            
        Returns:
            Updated metrics
        """
        metrics = self.analyze_trades(context, trades)
        
        # Update in-memory cache
        key = self._make_key(
            context.strategy_name,
            context.market,
            context.timeframe,
            context.regime,
            context.session,
            context.volatility_bucket
        )
        
        self.performance_db[key] = metrics
        
        logger.debug(f"Updated performance for {context.strategy_name} | Score: {metrics.composite_score:.1f}")
        
        return metrics
    
    def get_performance(
        self,
        strategy_name: str,
        market: str,
        timeframe: str,
        regime: str,
        session: Optional[str] = None,
        volatility_bucket: str = "medium"
    ) -> Optional[StrategyPerformanceMetrics]:
        """Query performance for specific context"""
        key = self._make_key(strategy_name, market, timeframe, regime, session, volatility_bucket)
        return self.performance_db.get(key)
    
    def get_best_strategies(
        self,
        market: str,
        regime: str,
        timeframe: Optional[str] = None,
        volatility_bucket: Optional[str] = None,
        min_trades: int = 10,
        top_n: int = 3
    ) -> List[StrategyPerformanceMetrics]:
        """
        Get top N strategies for a market context.
        
        Args:
            market: Market type
            regime: Market regime
            timeframe: Optional timeframe filter
            volatility_bucket: Optional volatility filter
            min_trades: Minimum trades to consider
            top_n: Number of top strategies to return
            
        Returns:
            List of top strategy metrics, sorted by composite score
        """
        # Filter by context
        candidates = []
        for metrics in self.performance_db.values():
            if metrics.market != market:
                continue
            if metrics.regime != regime:
                continue
            if metrics.total_trades < min_trades:
                continue
            if timeframe and metrics.timeframe != timeframe:
                continue
            if volatility_bucket and metrics.volatility_bucket != volatility_bucket:
                continue
            
            candidates.append(metrics)
        
        # Sort by composite score
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        
        return candidates[:top_n]
    
    def persist(self):
        """Save performance database to disk (CSV + JSON)"""
        if not self.performance_db:
            logger.warning("No performance data to persist")
            return
        
        try:
            # Convert to DataFrame
            records = [asdict(metrics) for metrics in self.performance_db.values()]
            df = pd.DataFrame(records)
            
            # Save CSV
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Saved {len(df)} performance records to {self.csv_path}")
            
            # Save JSON (more detailed)
            with open(self.json_path, 'w') as f:
                json.dump(records, f, indent=2)
            logger.info(f"Saved performance data to {self.json_path}")
            
        except Exception as e:
            logger.error(f"Failed to persist performance data: {e}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the performance database"""
        if not self.performance_db:
            return {
                "total_records": 0,
                "unique_strategies": 0,
                "markets": [],
                "regimes": []
            }
        
        strategies = set(m.strategy_name for m in self.performance_db.values())
        markets = set(m.market for m in self.performance_db.values())
        regimes = set(m.regime for m in self.performance_db.values())
        
        avg_score = np.mean([m.composite_score for m in self.performance_db.values()])
        
        return {
            "total_records": len(self.performance_db),
            "unique_strategies": len(strategies),
            "markets": list(markets),
            "regimes": list(regimes),
            "avg_composite_score": avg_score,
            "strategies": list(strategies)
        }
    
    def __repr__(self) -> str:
        stats = self.get_summary_stats()
        return (
            f"StrategyPerformanceEngine("
            f"records={stats['total_records']}, "
            f"strategies={stats['unique_strategies']}, "
            f"avg_score={stats.get('avg_composite_score', 0):.1f})"
        )
