"""
Strategy Selection Engine

Selects the best strategies for the current market context based on historical performance.
Integrated with StrategyPerformanceEngine for data-driven selection.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from analytics.performance_engine import (
    StrategyPerformanceEngine,
    StrategyPerformanceMetrics,
    StrategyContext
)

logger = logging.getLogger(__name__)


@dataclass
class SelectedStrategy:
    """A strategy selected for trading with allocated weight"""
    strategy_name: str
    confidence_score: float
    capital_weight: float  # Percentage of capital allocated (0-1)
    performance_metrics: StrategyPerformanceMetrics


class StrategySelector:
    """
    Selects best strategies for current context using performance data.
    
    **CRITICAL: No hardcoded strategy lists. All selection is data-driven.**
    
    Responsibilities:
    1. Query performance engine for historical data
    2. Filter strategies by context compatibility
    3. Rank strategies by composite score
    4. Select TOP N strategies
    5. Allocate capital weights
    """

    def __init__(
        self,
        performance_engine: StrategyPerformanceEngine,
        min_trades_for_ranking: int = 10,
        default_top_n: int = 3
    ):
        """
        Args:
            performance_engine: Performance tracking engine
            min_trades_for_ranking: Minimum trades to consider strategy
            default_top_n: Default number of strategies to select
        """
        self.performance_engine = performance_engine
        self.min_trades_for_ranking = min_trades_for_ranking
        self.default_top_n = default_top_n
        self.logger = logging.getLogger("StrategySelector")
        
        self.logger.info(
            f"StrategySelector initialized | min_trades={min_trades_for_ranking} | "
            f"default_top_n={default_top_n}"
        )

    def select_strategies(
        self,
        market_type: str,
        regime: str,
        timeframe: str,
        session: Optional[str] = None,
        volatility: str = "medium",
        top_n: Optional[int] = None,
        available_strategies: Optional[List[str]] = None
    ) -> List[SelectedStrategy]:
        """
        Select top N strategies for the given context.
        
        **This is the CORE selection logic - fully dynamic.**
        
        Args:
            market_type: "india" or "crypto"
            regime: "trending", "ranging", "volatile"
            timeframe: "1min", "5min", "15min", etc.
            session: Optional session name
            volatility: "low", "medium", "high"
            top_n: Number of strategies to select (default: self.default_top_n)
            available_strategies: Optional filter list of strategy names
            
        Returns:
            List of SelectedStrategy objects with capital weights
        """
        top_n = top_n or self.default_top_n
        
        self.logger.info(
            f"Selecting strategies for: {market_type} | {regime} | {timeframe} | "
            f"{volatility} | session={session}"
        )
        
        # Get performance metrics from engine
        candidates = self.performance_engine.get_best_strategies(
            market=market_type,
            regime=regime,
            timeframe=timeframe,
            volatility_bucket=volatility,
            min_trades=self.min_trades_for_ranking,
            top_n=top_n * 2  # Get more candidates to filter
        )
        
        # Filter by available strategies if specified
        if available_strategies:
            candidates = [
                c for c in candidates 
                if c.strategy_name in available_strategies
            ]
        
        # Take top N
        top_candidates = candidates[:top_n]
        
        if not top_candidates:
            self.logger.warning(
                f"No strategies found for context. Falling back to general best."
            )
            # Fallback: get best strategies for this market regardless of context
            top_candidates = self.performance_engine.get_best_strategies(
                market=market_type,
                regime=regime,  # Still filter by regime
                min_trades=max(5, self.min_trades_for_ranking // 2),  # Lower threshold
                top_n=top_n
            )
        
        if not top_candidates:
            self.logger.error(
                f"CRITICAL: No strategies available for {market_type}. "
                "System cannot trade without strategy history."
            )
            return []
        
        # Allocate capital weights
        selected_strategies = self._allocate_weights(top_candidates)
        
        # Log selection
        self.logger.info(f"Selected {len(selected_strategies)} strategies:")
        for i, sel in enumerate(selected_strategies, 1):
            self.logger.info(
                f"  {i}. {sel.strategy_name} | Score: {sel.performance_metrics.composite_score:.1f} | "
                f"Weight: {sel.capital_weight:.1%} | "
                f"WR: {sel.performance_metrics.win_rate:.1%} | "
                f"PF: {sel.performance_metrics.profit_factor:.2f}"
            )
        
        return selected_strategies

    def _allocate_weights(
        self,
        metrics_list: List[StrategyPerformanceMetrics]
    ) -> List[SelectedStrategy]:
        """
        Allocate capital weights based on composite scores.
        
        Uses score-weighted allocation:
        - Higher score = higher weight
        - Min weight: 10%
        - Max weight: 50%
        """
        if not metrics_list:
            return []
        
        # Calculate total score
        total_score = sum(m.composite_score for m in metrics_list)
        
        if total_score == 0:
            # Equal weighting if no scores
            equal_weight = 1.0 / len(metrics_list)
            return [
                SelectedStrategy(
                    strategy_name=m.strategy_name,
                    confidence_score=m.composite_score,
                    capital_weight=equal_weight,
                    performance_metrics=m
                )
                for m in metrics_list
            ]
        
        # Score-weighted allocation
        selected = []
        for metrics in metrics_list:
            raw_weight = metrics.composite_score / total_score
            
            # Apply bounds
            capped_weight = max(0.1, min(0.5, raw_weight))
            
            selected.append(SelectedStrategy(
                strategy_name=metrics.strategy_name,
                confidence_score=metrics.composite_score,
                capital_weight=capped_weight,
                performance_metrics=metrics
            ))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(s.capital_weight for s in selected)
        for sel in selected:
            sel.capital_weight /= total_weight
        
        return selected

    def get_strategy_for_context(
        self,
        strategy_name: str,
        market_type: str,
        regime: str,
        timeframe: str,
        volatility: str = "medium"
    ) -> Optional[StrategyPerformanceMetrics]:
        """
        Get performance metrics for a specific strategy in a context.
        
        Useful for checking if a strategy is suitable before forcing its use.
        """
        return self.performance_engine.get_performance(
            strategy_name=strategy_name,
            market=market_type,
            timeframe=timeframe,
            regime=regime,
            volatility_bucket=volatility
        )

    def get_all_strategies_for_market(
        self,
        market_type: str,
        min_score: float = 0.0
    ) -> List[str]:
        """
        Get all unique strategy names that have traded in this market.
        
        Args:
            market_type: Market to query
            min_score: Minimum composite score filter
            
        Returns:
            List of strategy names
        """
        strategies = set()
        
        for metrics in self.performance_engine.performance_db.values():
            if metrics.market == market_type:
                if metrics.composite_score >= min_score:
                    strategies.add(metrics.strategy_name)
        
        return sorted(strategies)

    def get_performance_summary(self) -> Dict:
        """Get summary of performance database"""
        return self.performance_engine.get_summary_stats()

    def export_performance_table(self, filepath: str):
        """Export full performance table to CSV"""
        if not self.performance_engine.performance_db:
            self.logger.warning("No performance data to export")
            return
        
        records = []
        for metrics in self.performance_engine.performance_db.values():
            records.append({
                "strategy": metrics.strategy_name,
                "market": metrics.market,
                "timeframe": metrics.timeframe,
                "regime": metrics.regime,
                "session": metrics.session or "N/A",
                "volatility": metrics.volatility_bucket,
                "trades": metrics.total_trades,
                "win_rate": f"{metrics.win_rate:.2%}",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "sharpe": f"{metrics.sharpe_ratio:.2f}",
                "max_dd_pct": f"{metrics.max_drawdown_pct:.2f}%",
                "composite_score": f"{metrics.composite_score:.1f}",
                "total_pnl": f"${metrics.total_pnl:.2f}"
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("composite_score", ascending=False)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Performance table exported to {filepath}")

    def __str__(self) -> str:
        stats = self.performance_engine.get_summary_stats()
        return (
            f"StrategySelector("
            f"strategies={stats['unique_strategies']}, "
            f"records={stats['total_records']}, "
            f"avg_score={stats.get('avg_composite_score', 0):.1f})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()
